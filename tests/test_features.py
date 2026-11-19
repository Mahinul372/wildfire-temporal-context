"""Tests for feature construction (features.py) on synthetic arrays."""

import numpy as np

from features import (
    build_features,
    feature_names,
    neighbourhood_mean,
    per_day_feature_count,
    sample_training_rows,
)


def _fake_day(h=6, w=5, n_lc=3, fire=None, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "continuous": rng.normal(0, 1, size=(h, w, 18)).astype(np.float32),
        "angular": rng.uniform(-1, 1, size=(h, w, 6)).astype(np.float32),
        "landcover": np.eye(n_lc, dtype=np.float32)[
            rng.integers(0, n_lc, size=(h, w))
        ],
        "active_fire": (
            np.zeros((h, w), dtype=np.uint8)
            if fire is None
            else fire.astype(np.uint8)
        ),
    }


def test_feature_count_and_names():
    assert per_day_feature_count(3) == 18 + 6 + 3 + 2
    names = feature_names(5, [8, 10, 12])
    assert len(names) == 5 * (18 + 6 + 3 + 2)
    assert names[0] == "t-4_cont_viirs_m11"
    assert names[-1] == "t_af_nbr"
    assert names[-2] == "t_af"
    # Names are unique.
    assert len(set(names)) == len(names)


def test_build_features_shape_and_window():
    days = [_fake_day(seed=i) for i in range(5)]
    X = build_features(days, active_fire_mode="all")
    assert X.shape == (6 * 5, 5 * (18 + 6 + 3 + 2))
    X1 = build_features(days[-1:], active_fire_mode="all")
    assert X1.shape == (6 * 5, 18 + 6 + 3 + 2)
    # E1 columns equal the last block of E5 (same day, same features).
    assert np.allclose(X[:, -X1.shape[1]:], X1)


def test_modality_zeroing_keeps_dimensions():
    fire = np.zeros((6, 5), dtype=np.uint8)
    fire[2, 2] = 1
    days = [_fake_day(fire=fire, seed=i) for i in range(3)]
    x_all = build_features(days, "all")
    x_cur = build_features(days, "current")
    x_none = build_features(days, "none")
    assert x_all.shape == x_cur.shape == x_none.shape
    f = per_day_feature_count(3)
    af_bin = [f * d + (18 + 6 + 3) for d in range(3)]      # [27, 56, 85]
    af_nbr = [f * d + (18 + 6 + 3) + 1 for d in range(3)]  # [28, 57, 86]
    # M3: every AF column is zero.
    assert np.all(x_none[:, af_bin + af_nbr] == 0)
    # M2: days 0-1 AF zeroed, last day's AF kept.
    assert np.all(x_cur[:, af_bin[:2] + af_nbr[:2]] == 0)
    assert np.any(x_cur[:, af_bin[2]] != 0)
    assert np.any(x_cur[:, af_nbr[2]] != 0)
    # M1: all AF columns match the raw fire map.
    assert np.all(x_all[:, af_bin[2]] == fire.ravel())
    assert np.all(x_all[:, af_bin[0]] == fire.ravel())


def test_neighbourhood_mean():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    m = neighbourhood_mean(mask)
    assert m[2, 2] == 1 / 9
    assert m[2, 1] == 1 / 9
    assert m[0, 0] == 0.0
    full = np.ones((4, 4), dtype=np.uint8)
    m2 = neighbourhood_mean(full)
    assert m2[1, 1] == 1.0
    assert m2[0, 0] == 4 / 9  # corner sees 4 of 9 cells


def test_sampling_keeps_all_positives():
    rng = np.random.default_rng(0)
    y = np.zeros(1000, dtype=np.uint8)
    y[[10, 20, 30]] = 1
    rows = sample_training_rows(y, rng, max_pixels=100, neg_pos_ratio=10)
    assert set([10, 20, 30]).issubset(set(rows.tolist()))
    assert len(rows) <= 100
    # 3 positives -> at most 30 negatives.
    assert len(rows) == 33


def test_sampling_no_positives_returns_bounded_negatives():
    rng = np.random.default_rng(0)
    y = np.zeros(500, dtype=np.uint8)
    rows = sample_training_rows(y, rng, max_pixels=50, neg_pos_ratio=10)
    assert len(rows) == 50
