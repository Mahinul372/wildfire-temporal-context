"""Evaluation metrics for next-day active-fire prediction.

Active-fire pixels are extremely rare (~0.17% of pixels in the full
dataset), so accuracy is never used. Primary metric: Average Precision.
All functions handle the degenerate cases explicitly (no positives in
the evaluated set -> AP is undefined and reported as NaN, never as 0).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict:
    """Pooled pixel-level metrics at one operating threshold."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n_pos = int((y_true == 1).sum())
    n_total = int(y_true.size)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    ap = (
        float(average_precision_score(y_true, y_score))
        if n_pos > 0 and len(np.unique(y_score)) > 1
        else float("nan")
    )
    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "prevalence": _safe_div(n_pos, n_total),
        "average_precision": ap,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
    }


def f1_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Threshold maximising F1. Fit on VALIDATION data only.

    Returns (best_threshold, best_f1). Ties go to the higher threshold.
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1 or (f1 == best_f1 and t > best_t):
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def mean_event_metrics(per_event: list[dict]) -> dict:
    """Mean of per-event metrics, skipping events where a metric is NaN
    (e.g. AP undefined when an event has no positive test pixels)."""
    keys = ["average_precision", "precision", "recall", "f1", "iou"]
    out: dict = {"n_events": len(per_event)}
    for k in keys:
        vals = [e[k] for e in per_event if np.isfinite(e[k])]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
        out[f"n_events_{k}"] = len(vals)
    return out


def bootstrap_ci_event_level(
    per_event_arrays: list[tuple[np.ndarray, np.ndarray]],
    metric_fn,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI for a pooled metric, resampling EVENTS (not pixels).

    ``per_event_arrays`` is a list of (y_true, y_score) per event.
    With very few events the interval is indicative only -- callers
    must say so in the report.
    """
    rng = np.random.default_rng(seed)
    n = len(per_event_arrays)
    if n < 2:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = np.concatenate([per_event_arrays[i][0] for i in idx])
        ys = np.concatenate([per_event_arrays[i][1] for i in idx])
        try:
            stats.append(metric_fn(yt, ys))
        except Exception:
            continue
    if not stats:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return (lo, hi)


def bootstrap_ci_from_values(
    values: list[float],
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI for the MEAN of per-event metric values.

    Resamples the (few) per-event values themselves -- O(n_boot * n),
    not O(n_boot * n_pixels). Use this when the metric has already
    been computed once per event; ``bootstrap_ci_event_level`` above
    re-concatenates full pixel arrays on every resample, which is
    prohibitively slow on millions of pixels (it stalled the first
    real-data smoke test). With very few events the interval is
    indicative only -- callers must say so in the report.
    """
    vals = [float(v) for v in values if np.isfinite(v)]
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = arr.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return (lo, hi)
