"""Tests for metrics (metrics.py) against hand-computed examples."""

import numpy as np
import pytest

from metrics import (
    bootstrap_ci_event_level,
    compute_metrics,
    f1_optimal_threshold,
    mean_event_metrics,
)


def test_perfect_prediction():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    m = compute_metrics(y, s, threshold=0.5)
    assert m["average_precision"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["iou"] == pytest.approx(1.0)
    assert m["prevalence"] == pytest.approx(0.5)
    assert m["n_pos"] == 3


def test_hand_computed_counts():
    # Scores: predict 1 where s >= 0.5 -> [0,1,1,0]; truth [0,1,0,1].
    # tp=1 (idx1), fp=1 (idx2), fn=1 (idx3), tn=1 (idx0).
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.6, 0.4])
    m = compute_metrics(y, s, threshold=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["iou"] == pytest.approx(1 / 3)


def test_no_positives_ap_is_nan_not_zero():
    y = np.zeros(10, dtype=int)
    s = np.linspace(0, 1, 10)
    m = compute_metrics(y, s, threshold=0.5)
    assert np.isnan(m["average_precision"])
    # 5 predicted positives, all false -> precision 0; recall 0 by
    # definition (no true positives exist).
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_f1_optimal_threshold_uses_validation_only():
    # Best F1 on this validation set is at threshold 0.5 (perfect split).
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t, f1 = f1_optimal_threshold(y, s)
    assert f1 == pytest.approx(1.0)
    assert 0.3 < t <= 0.7


def test_mean_event_metrics_skips_nan():
    per = [
        {"average_precision": 0.8, "precision": 0.7, "recall": 0.6,
         "f1": 0.65, "iou": 0.5},
        {"average_precision": float("nan"), "precision": 0.0,
         "recall": 0.0, "f1": 0.0, "iou": 0.0},
    ]
    m = mean_event_metrics(per)
    assert m["mean_average_precision"] == pytest.approx(0.8)
    assert m["n_events_average_precision"] == 1
    assert m["mean_precision"] == pytest.approx(0.35)


def test_bootstrap_ci_covers_and_is_bounded():
    rng = np.random.default_rng(0)
    evs = []
    for _ in range(5):
        y = (rng.random(200) < 0.1).astype(int)
        s = rng.random(200) * 0.5 + y * 0.4
        evs.append((y, s))
    from sklearn.metrics import average_precision_score

    lo, hi = bootstrap_ci_event_level(evs, average_precision_score,
                                      n_boot=200, seed=1)
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_from_values_covers_mean_and_is_fast():
    from metrics import bootstrap_ci_from_values

    vals = [0.20, 0.35, 0.50]
    lo, hi = bootstrap_ci_from_values(vals, n_boot=1000, seed=42)
    assert 0.0 <= lo <= float(np.mean(vals)) <= hi <= 1.0
    # Deterministic for a fixed seed.
    lo2, hi2 = bootstrap_ci_from_values(vals, n_boot=1000, seed=42)
    assert (lo, hi) == (lo2, hi2)
    # Fewer than 2 finite values -> NaN interval, not an exception.
    lo1, hi1 = bootstrap_ci_from_values([0.4])
    assert np.isnan(lo1) and np.isnan(hi1)
    lo0, hi0 = bootstrap_ci_from_values([])
    assert np.isnan(lo0) and np.isnan(hi0)
