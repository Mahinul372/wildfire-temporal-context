"""End-to-end pipeline smoke test on synthetic data.

Builds two tiny fake fire events (processed-day .npz files + stats +
samples), runs one learned configuration (E1 logistic regression) and
the persistence baseline through ExperimentRunner, and checks the
result structure. No real data is used.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.sequences import Sample
from experiments import ExperimentRunner


def _make_fake_processed(root: Path, rng: np.random.Generator) -> None:
    """Write fake processed days for 2 events x 12 days.

    Event A (train): fire grows from a corner. Event B (test): fire
    grows from the opposite corner. 8x8 images, 18+6+2 = 26 features.
    """
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    stats = {
        "medians": {str(i): 0.0 for i in range(23)},
        "means": {str(i): 0.0 for i in range(23)},
        "stds": {str(i): 1.0 for i in range(23)},
        "landcover_classes": [8, 10],
    }
    import json

    (proc / "preprocess_stats.json").write_text(json.dumps(stats))
    events = {"2020/fire_a": "train", "2019/fire_c": "validation",
              "2021/fire_b": "test"}
    samples = []
    for event, split in events.items():
        start = {"2020": date(2020, 8, 1), "2019": date(2019, 6, 1),
                 "2021": date(2021, 7, 1)}[event[:4]]
        for i in range(12):
            d = start + timedelta(days=i)
            feats = rng.normal(0, 1, size=(8, 8, 26)).astype(np.float32)
            af = np.zeros((8, 8), dtype=np.uint8)
            # Fire: a growing square from a different corner per event.
            k = min(i, 4)
            if "2020" in event:
                af[: k + 1, : k + 1] = 1
            elif "2019" in event:
                af[: k + 1, -(k + 1):] = 1
            else:
                af[-(k + 1):, -(k + 1):] = 1
            # Make the fire slightly predictable from feature 0.
            feats[:, :, 0] = af * 2.0 + rng.normal(0, 0.1, size=(8, 8))
            out = proc / "days" / event / f"{d.isoformat()}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, features=feats, active_fire=af,
                     shape=np.array([8, 8]))
        # Samples: targets d[5..11] (7 per event).
        for j in range(5, 12):
            target = start + timedelta(days=j)
            inputs = tuple(start + timedelta(days=j - 5 + k)
                           for k in range(5))
            samples.append(
                Sample(event=event, split=split, target_date=target,
                       input_dates=inputs,
                       input_paths=tuple(
                           proc / "days" / event / f"{dd.isoformat()}.npz"
                           for dd in inputs),
                       target_path=proc / "days" / event
                       / f"{target.isoformat()}.npz")
            )
    import json as _json

    (proc / "samples.json").write_text(
        _json.dumps([{"event": s.event, "split": s.split,
                      "target_date": s.target_date.isoformat(),
                      "input_dates": [d.isoformat() for d in s.input_dates]}
                     for s in samples]))
    return samples


@pytest.fixture()
def runner(tmp_path):
    rng = np.random.default_rng(0)
    samples = _make_fake_processed(tmp_path, rng)
    cfg = {
        "random_seed": 42,
        "n_threads": 1,
        "paths": {"processed_data": "data/processed"},
        "experiments": {
            "temporal": {"E1": {"history_days": 1}},
            "modality": {},
        },
        # Model config mirrors the real config/experiment.yaml shape:
        # keys are short names, each entry carries a "type" field.
        # (A previous bug: run_configuration looked the config up by
        # the type name, which KeyError'd on "hist_gradient_boosting"
        # because the real key is "hist_gb".)
        "models": {
            "logistic_regression": {"type": "logistic_regression",
                                    "C": 1.0, "class_weight": "balanced",
                                    "max_iter": 200},
            "hist_gb": {"type": "hist_gradient_boosting",
                        "max_iter": 50, "learning_rate": 0.1,
                        "max_leaf_nodes": 15},
        },
        "sampling": {"max_pixels_per_sample": 500,
                     "negative_to_positive_ratio": 5,
                     "max_train_rows": 5000},
        "evaluation": {"bootstrap_samples": 100},
    }
    import json

    stats = json.loads(
        (tmp_path / "data" / "processed" / "preprocess_stats.json").read_text())
    return ExperimentRunner(tmp_path, cfg, stats, samples)


def test_pipeline_runs_and_returns_metrics(runner):
    res = runner.run_configuration("E1_logistic_regression", 1, "all",
                                   "logistic_regression")
    assert res["config"] == "E1_logistic_regression"
    assert res["n_train_rows"] > 0
    pooled = res["test"]["pooled"]["fixed_0.5"]
    assert 0.0 <= pooled["average_precision"] <= 1.0
    assert pooled["n_total"] > 0
    assert pooled["n_pos"] > 0  # the fake test event has fire
    # Per-event metrics exist for the test event.
    assert len(res["test"]["per_event"]["fixed_0.5"]) == 7
    # F1-optimal threshold came from validation (no validation samples
    # in this fixture -> validation predictions are empty; guard).
    assert "f1_optimal_threshold" in res


def test_persistence_baseline_runs(runner):
    res = runner.run_configuration("persistence", 1, "all", "persistence")
    pooled = res["test"]["pooled"]["fixed_0.5"]
    # Persistence on a growing fire: some overlap -> recall > 0.
    assert pooled["recall"] > 0
    assert pooled["n_pos"] > 0


def test_histgb_runs_and_resolves_config_by_type(runner):
    # Regression test: run_configuration must resolve the HistGB
    # config by its "type" field, not by key name (the real config
    # key is "hist_gb"; looking up "hist_gradient_boosting" raised
    # KeyError and killed the first real-data grid at config 3/13).
    res = runner.run_configuration("E1_hist_gradient_boosting", 1, "all",
                                   "hist_gradient_boosting")
    pooled = res["test"]["pooled"]["fixed_0.5"]
    assert 0.0 <= pooled["average_precision"] <= 1.0
    assert pooled["n_total"] > 0
    assert res["n_train_rows"] > 0


def test_make_figures_runs_with_artifacts(runner, tmp_path):
    # Regression test: make_figures used runner.load_processed_day,
    # which does not exist (it is a module-level function), so figure
    # generation crashed at the end of the first complete grid run.
    # Fabricate a minimal result set with an E4 HistGB entry carrying
    # _test_preds, and check the charts + example maps are written.
    from run_experiments import make_figures

    def _fake(config, model, ap=0.4):
        pooled = {"average_precision": ap, "precision": 0.3,
                  "recall": 0.7, "f1": 0.42, "iou": 0.27,
                  "n_total": 100, "n_pos": 10, "prevalence": 0.1}
        return {"config": config, "model": model, "window_days": 5,
                "active_fire_mode": "all", "n_features": 10,
                "runtime_seconds": 1.0,
                "test": {"pooled": {"fixed_0.5": pooled},
                         "per_event": {"fixed_0.5": []}},
                "f1_optimal_threshold": 0.5}

    results = []
    for exp in ("E1", "E2", "E3", "E4"):
        results.append(_fake(f"{exp}_logistic_regression",
                             "logistic_regression"))
        results.append(_fake(f"{exp}_hist_gradient_boosting",
                             "hist_gradient_boosting"))
    for exp in ("M2", "M3"):
        results.append(_fake(f"E4_{exp}_logistic_regression",
                             "logistic_regression"))
        results.append(_fake(f"E4_{exp}_hist_gradient_boosting",
                             "hist_gradient_boosting"))
    # Attach fake test predictions for the E4 HistGB entry, using the
    # fixture's real test samples (2021/fire_b, 8x8).
    e4 = next(r for r in results
              if r["config"] == "E4_hist_gradient_boosting")
    preds = []
    for s in runner.samples:
        if s.split != "test":
            continue
        preds.append({"event": s.event,
                      "target_date": s.target_date.isoformat(),
                      "y_true": np.zeros(64, dtype=np.uint8),
                      "y_score": np.random.default_rng(0).random(64)})
    preds[0]["y_true"][:5] = 1
    e4["_test_preds"] = preds
    e4["_window_days"] = 5
    e4["_mode"] = "all"

    figs = tmp_path / "figs"
    make_figures(runner, results, figs)
    assert (figs / "temporal_ap.svg").exists()
    assert (figs / "modality_ap.svg").exists()
    assert list(figs.glob("example_*_inputs.png"))
    assert list(figs.glob("example_*_maps.png"))
