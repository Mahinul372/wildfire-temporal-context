"""Experiment orchestration: temporal (E1-E4) and modality (M1-M3) ablations.

Design:
  * One sample population for every configuration, defined by the
    5-day history requirement (see data/sequences.py). Only the input
    window (E) or the active-fire handling (M) changes.
  * Preprocessed days are read from ``data/processed/days/`` (written
    by scripts/prepare_data.py), never re-read from raw GeoTIFFs here.
  * Training rows are bounded: all positive pixels are kept, negatives
    subsampled per sample, and the assembled training matrix is capped
    at ``max_train_rows`` (positives always kept).
  * Thresholds: 0.5 fixed, plus an F1-optimal threshold selected on
    VALIDATION data only and then applied to test.
  * Splits are by fire event; the same partitions are used everywhere.

Experiment grid (12 model fits + persistence):
  * temporal:  E1-E4 x M1 x {logistic_regression, hist_gb}
  * modality:  M2, M3 x E4 x {logistic_regression, hist_gb}
  * persistence baseline (no fit; predicts AF_t).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from data.sequences import Sample
from features import build_features, feature_names, sample_training_rows
from metrics import (
    bootstrap_ci_from_values,
    compute_metrics,
    f1_optimal_threshold,
    mean_event_metrics,
)
from models import fit_model, persistence_proba, predict_proba


@dataclass
class DayData:
    """One preprocessed day loaded from the processed cache."""

    features: np.ndarray  # (H, W, 18+6+K) float32
    active_fire: np.ndarray  # (H, W) uint8
    shape: tuple[int, int]


def load_processed_day(path: Path) -> DayData:
    with np.load(path) as z:
        return DayData(
            features=z["features"],
            active_fire=z["active_fire"],
            shape=tuple(z["shape"]),
        )


def day_to_preprocessed(day: DayData, n_cont: int, n_ang: int) -> dict:
    """Split a cached day into the dict shape preprocess_day returns."""
    f = day.features
    return {
        "continuous": f[:, :, :n_cont],
        "angular": f[:, :, n_cont : n_cont + n_ang],
        "landcover": f[:, :, n_cont + n_ang :],
        "active_fire": day.active_fire,
    }


@dataclass
class ExperimentRunner:
    root: Path
    cfg: dict
    stats: dict
    samples: list[Sample]
    n_cont: int = 18
    n_ang: int = 6

    def __post_init__(self):
        self.processed = self.root / self.cfg["paths"]["processed_data"]
        self.landcover_classes: list[int] = self.stats["landcover_classes"]
        self.split_of = {s.event: s.split for s in self.samples}

    def day_path(self, sample: Sample, d: date) -> Path:
        return (
            self.processed
            / "days"
            / sample.event
            / f"{d.isoformat()}.npz"
        )

    def featurize_sample(
        self, sample: Sample, window_days: int, mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full-image features X (H*W, F) and target y for one sample."""
        input_dates = sample.input_dates[-window_days:]
        days = []
        for d in input_dates:
            raw = load_processed_day(self.day_path(sample, d))
            days.append(day_to_preprocessed(raw, self.n_cont, self.n_ang))
        X = build_features(days, active_fire_mode=mode)
        target = load_processed_day(self.day_path(sample, sample.target_date))
        y = target.active_fire.astype(np.uint8).ravel()
        return X, y

    def collect_training_rows(
        self, window_days: int, mode: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the bounded training matrix for one configuration."""
        samp_cfg = self.cfg["sampling"]
        rng = np.random.default_rng(seed)
        train = [s for s in self.samples if s.split == "train"]
        X_chunks: list[np.ndarray] = []
        y_chunks: list[np.ndarray] = []
        for sample in train:
            X, y = self.featurize_sample(sample, window_days, mode)
            rows = sample_training_rows(
                y,
                rng,
                samp_cfg["max_pixels_per_sample"],
                samp_cfg["negative_to_positive_ratio"],
            )
            X_chunks.append(X[rows])
            y_chunks.append(y[rows])
            del X
        X = np.concatenate(X_chunks)
        y = np.concatenate(y_chunks)
        cap = samp_cfg["max_train_rows"]
        if y.size > cap:
            pos = np.flatnonzero(y == 1)
            neg = np.flatnonzero(y == 0)
            keep_neg = max(cap - pos.size, 0)
            keep = np.concatenate(
                [pos, rng.choice(neg, size=keep_neg, replace=False)]
            )
            rng.shuffle(keep)
            X, y = X[keep], y[keep]
        return X, y

    def predict_split(
        self, model, split: str, window_days: int, mode: str,
        kind: str,
    ) -> list[dict]:
        """Predict every sample of one split; returns per-sample dicts."""
        out = []
        for sample in [s for s in self.samples if s.split == split]:
            X, y = self.featurize_sample(sample, window_days, mode)
            if kind == "persistence":
                # Last input day's active-fire map, as 0/1 scores.
                last = load_processed_day(
                    self.day_path(sample, sample.input_dates[-1])
                )
                score = persistence_proba(last.active_fire)
            else:
                score = predict_proba(model, X)
            out.append(
                {
                    "event": sample.event,
                    "target_date": sample.target_date.isoformat(),
                    "y_true": y,
                    "y_score": score,
                }
            )
            del X
        return out

    @staticmethod
    def evaluate_predictions(
        preds: list[dict], thresholds: dict[str, float]
    ) -> dict:
        """Pooled + per-event metrics at each named threshold."""
        pooled: dict[str, dict] = {}
        per_event: dict[str, list[dict]] = {}
        for name, t in thresholds.items():
            yt = np.concatenate([p["y_true"] for p in preds])
            ys = np.concatenate([p["y_score"] for p in preds])
            pooled[name] = compute_metrics(yt, ys, threshold=t)
            evs = []
            for p in preds:
                evs.append(
                    {
                        "event": p["event"],
                        "target_date": p["target_date"],
                        **compute_metrics(p["y_true"], p["y_score"], threshold=t),
                    }
                )
            per_event[name] = evs
        return {"pooled": pooled, "per_event": per_event}

    def run_configuration(
        self, name: str, window_days: int, mode: str, kind: str,
        keep_artifacts: bool = False,
    ) -> dict:
        """Run one (window, modality, model) configuration end to end.

        With ``keep_artifacts``, the fitted model and test predictions
        are kept in the result under ``_model`` / ``_test_preds`` for
        figure generation (not JSON-serialisable; excluded from the
        summary by ``summarise``).
        """
        t0 = time.time()
        seed = self.cfg["random_seed"]
        result: dict = {
            "config": name,
            "window_days": window_days,
            "active_fire_mode": mode,
            "model": kind,
            "n_features": len(
                feature_names(window_days, self.landcover_classes)
            ),
        }
        model = None
        if kind != "persistence":
            X, y = self.collect_training_rows(window_days, mode, seed)
            result["n_train_rows"] = int(y.size)
            result["n_train_pos"] = int((y == 1).sum())
            # Resolve the model config by its "type" field (real
            # config keys are e.g. "hist_gb" with type
            # "hist_gradient_boosting"); fall back to matching the
            # config key itself (older/test fixtures without "type").
            model_cfg = None
            for key, v in self.cfg["models"].items():
                if isinstance(v, dict) and v.get("type", key) == kind:
                    model_cfg = dict(v)
                    break
            if model_cfg is None:
                raise ValueError(f"no model config for kind={kind!r}")
            model_cfg["n_jobs"] = self.cfg.get("n_threads", 1)
            t_fit = time.time()
            model = fit_model(kind, X, y, model_cfg, seed)
            result["fit_seconds"] = round(time.time() - t_fit, 1)
            del X, y

        # Validation predictions -> F1-optimal threshold (validation only).
        # If there are no validation samples, fall back to the fixed
        # threshold alone (defensive; the real config has validation).
        val_preds = self.predict_split(model, "validation", window_days, mode, kind)
        if val_preds:
            yt_v = np.concatenate([p["y_true"] for p in val_preds])
            ys_v = np.concatenate([p["y_score"] for p in val_preds])
            best_t, best_f1 = f1_optimal_threshold(yt_v, ys_v)
            thresholds = {"fixed_0.5": 0.5, "f1_optimal_val": best_t}
        else:
            best_t, best_f1 = None, None
            thresholds = {"fixed_0.5": 0.5}
        result["f1_optimal_threshold"] = best_t
        result["validation_f1_at_optimal"] = best_f1
        result["validation"] = self.evaluate_predictions(val_preds, thresholds)

        # Held-out test predictions.
        test_preds = self.predict_split(model, "test", window_days, mode, kind)
        result["test"] = self.evaluate_predictions(test_preds, thresholds)
        if keep_artifacts:
            result["_model"] = model
            result["_test_preds"] = test_preds
            result["_window_days"] = window_days
            result["_mode"] = mode
        for name_t in thresholds:
            result["test"][f"mean_event_{name_t}"] = mean_event_metrics(
                result["test"]["per_event"][name_t]
            )
        # Event-level bootstrap CI for AP, computed ONCE (AP does not
        # depend on the operating threshold). The CI is over the
        # per-event AP values (one AP per test event, pooling that
        # event's days) -- resampling pixel arrays would re-run AP on
        # ~8.5M pixels 1000x and stall the run. With only 3 test
        # events the interval is indicative only.
        from sklearn.metrics import average_precision_score

        def _ap(yt, ys):
            if (yt == 1).sum() == 0 or len(np.unique(ys)) <= 1:
                return float("nan")
            return float(average_precision_score(yt, ys))

        per_event_ap = []
        for ev in sorted({p["event"] for p in test_preds}):
            yt = np.concatenate(
                [p["y_true"] for p in test_preds if p["event"] == ev]
            )
            ys = np.concatenate(
                [p["y_score"] for p in test_preds if p["event"] == ev]
            )
            per_event_ap.append({"event": ev, "average_precision": _ap(yt, ys)})
        result["test"]["per_event_AP"] = per_event_ap
        lo, hi = bootstrap_ci_from_values(
            [e["average_precision"] for e in per_event_ap],
            n_boot=self.cfg["evaluation"]["bootstrap_samples"],
            seed=seed,
        )
        for name_t in thresholds:
            result["test"][f"ap_bootstrap_ci95_{name_t}"] = [lo, hi]
        result["runtime_seconds"] = round(time.time() - t0, 1)
        return result

    def run_all(self, on_result=None) -> list[dict]:
        """Run the full experiment grid.

        If ``on_result`` is given it is called with each result dict
        as soon as that configuration finishes, so a caller can save
        partial results incrementally (the grid is long; a killed run
        should not lose finished configurations).
        """
        results: list[dict] = []

        def _emit(r: dict) -> None:
            results.append(r)
            if on_result is not None:
                on_result(r)

        # Persistence baseline (window-independent).
        _emit(self.run_configuration("persistence", 5, "all", "persistence"))
        # Temporal ablation E1-E4 (M1), both learned models.
        # Keep artifacts for E4/HistGB (used for figure generation).
        for exp, spec in self.cfg["experiments"]["temporal"].items():
            for kind in ("logistic_regression", "hist_gradient_boosting"):
                keep = spec["history_days"] == 5 and kind == "hist_gradient_boosting"
                _emit(
                    self.run_configuration(
                        f"{exp}_{kind}", spec["history_days"], "all", kind,
                        keep_artifacts=keep,
                    )
                )
        # Modality ablation M2, M3 at E4 (M1/E4 already run above).
        for exp, spec in self.cfg["experiments"]["modality"].items():
            if spec["active_fire"] == "all":
                continue
            for kind in ("logistic_regression", "hist_gradient_boosting"):
                _emit(
                    self.run_configuration(
                        f"E4_{exp}_{kind}", 5, spec["active_fire"], kind
                    )
                )
        return results


def summarise(results: list[dict]) -> dict:
    """Compact summary table: test pooled metrics at the 0.5 threshold."""
    rows = []
    for r in results:
        pooled = r["test"]["pooled"]["fixed_0.5"]
        rows.append(
            {
                "config": r["config"],
                "model": r["model"],
                "window_days": r["window_days"],
                "active_fire_mode": r["active_fire_mode"],
                "n_features": r["n_features"],
                "test_AP": pooled["average_precision"],
                "test_precision": pooled["precision"],
                "test_recall": pooled["recall"],
                "test_F1": pooled["f1"],
                "test_IoU": pooled["iou"],
                "test_prevalence": pooled["prevalence"],
                "test_n_pos": pooled["n_pos"],
                "runtime_seconds": r["runtime_seconds"],
            }
        )
    return {"results": rows}
