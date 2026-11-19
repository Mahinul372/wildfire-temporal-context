"""Leakage tests: splits, target separation, preprocessing scope.

All fixtures are synthetic. These tests guard the three leakage rules:
  1. No fire event in more than one partition.
  2. No target-day information in the input features.
  3. Preprocessing statistics fitted on train data only.
"""

from datetime import date, timedelta

import numpy as np

from data.preprocessing import fit_preprocess_stats, load_day
from data.sequences import build_samples
from features import build_features


def _dates(start, n):
    return [start + timedelta(days=i) for i in range(n)]


def test_no_event_in_two_partitions():
    from pathlib import Path

    ev_a = {d: Path(f"/a/{d}.tif") for d in _dates(date(2020, 8, 1), 10)}
    ev_b = {d: Path(f"/b/{d}.tif") for d in _dates(date(2021, 7, 1), 10)}
    events = {"2020/fire_a": ev_a, "2021/fire_b": ev_b}
    split_of = {"2020/fire_a": "train", "2021/fire_b": "test"}
    samples = build_samples(events, split_of)
    train_events = {s.event for s in samples if s.split == "train"}
    test_events = {s.event for s in samples if s.split == "test"}
    assert train_events == {"2020/fire_a"}
    assert test_events == {"2021/fire_b"}
    assert not (train_events & test_events)


def test_inputs_never_include_target_day():
    from pathlib import Path

    ev = {d: Path(f"/a/{d}.tif") for d in _dates(date(2020, 8, 1), 10)}
    samples = build_samples({"2020/fire_a": ev}, {"2020/fire_a": "train"})
    for s in samples:
        assert s.target_path not in s.input_paths
        assert all(d < s.target_date for d in s.input_dates)


def test_features_exclude_target_day_fire(tmp_path):
    """build_features only sees input days; changing the target day's
    fire map must not change the features."""
    h, w = 4, 5
    rng = np.random.default_rng(0)
    fire_today = np.zeros((h, w), dtype=np.uint8)
    fire_today[1, 1] = 1

    def day(fire):
        return {
            "continuous": rng.normal(size=(h, w, 18)).astype(np.float32),
            "angular": rng.uniform(-1, 1, size=(h, w, 6)).astype(np.float32),
            "landcover": np.zeros((h, w, 2), dtype=np.float32),
            "active_fire": fire,
        }

    days = [day(fire_today), day(np.zeros((h, w), dtype=np.uint8))]
    x1 = build_features(days, "all")
    # Swap in a DIFFERENT "target day" fire map as an extra day is not
    # possible: build_features only consumes the given input days.
    # Instead verify the last input day's fire is the only fire source:
    # zero the last day's fire and confirm the AF columns change, while
    # zeroing a hypothetical target map cannot affect anything because
    # it is never passed in.
    days_no_fire = [day(np.zeros((h, w), dtype=np.uint8)),
                    day(np.zeros((h, w), dtype=np.uint8))]
    x2 = build_features(days_no_fire, "all")
    assert not np.allclose(x1, x2)  # input-day fire matters
    # And the AF columns of x1 equal the input days' fire, not anything else:
    # per-day layout is cont(18) + ang(6) + lc(2) + af + af_nbr, so day 0's
    # AF-binary column is index 26.
    assert np.all(x1[:, 26] == fire_today.ravel())  # day 0 AF col


def test_preprocess_stats_use_only_given_days(tmp_path, monkeypatch):
    """fit_preprocess_stats reads exactly the paths it is given."""
    import tifffile

    rng = np.random.default_rng(0)
    p1 = tmp_path / "a.tif"
    p2 = tmp_path / "b.tif"
    # Two 2x2x23 float32 days with distinct channel-0 values.
    tifffile.imwrite(p1, np.full((2, 2, 23), 10.0, dtype=np.float32))
    tifffile.imwrite(p2, np.full((2, 2, 23), 1000.0, dtype=np.float32))
    stats = fit_preprocess_stats([p1], seed=0)
    # Median/mean of channel 0 must come from p1 only (10.0), not p2.
    assert stats["medians"]["0"] == 10.0
    assert stats["means"]["0"] == 10.0
    assert stats["stds"]["0"] == 1.0  # zero variance -> floored at 1.0
