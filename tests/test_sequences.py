"""Tests for sequence construction (data/sequences.py).

Uses synthetic event inventories only -- no real data.
"""

from datetime import date, timedelta
from pathlib import Path

from data.sequences import build_samples, inventory_events


def _event_dict(dates, event="2020/fire_1"):
    return {event: {d: Path(f"/fake/{d.isoformat()}.tif") for d in dates}}


def _dates(start, n):
    return [start + timedelta(days=i) for i in range(n)]


def test_consecutive_run_builds_expected_samples():
    # 10 consecutive days -> targets d[5..9] (need 5 inputs + target).
    dates = _dates(date(2020, 8, 1), 10)
    samples = build_samples(_event_dict(dates), {"2020/fire_1": "train"})
    assert len(samples) == 5
    assert samples[0].target_date == date(2020, 8, 6)
    assert samples[0].input_dates == tuple(_dates(date(2020, 8, 1), 5))
    assert samples[-1].target_date == date(2020, 8, 10)
    assert samples[-1].input_dates == tuple(_dates(date(2020, 8, 5), 5))


def test_gap_breaks_eligibility():
    # A missing day must break the run: no sample may span the gap.
    dates = _dates(date(2020, 8, 1), 5) + _dates(date(2020, 8, 7), 5)
    samples = build_samples(_event_dict(dates), {"2020/fire_1": "train"})
    # No 6-day consecutive run exists (5 days, gap, 5 days), so no
    # sample is eligible: target 8-11 lacks input 8-6, target 8-12
    # is not itself a downloaded day.
    assert len(samples) == 0


def test_inputs_always_before_target():
    dates = _dates(date(2021, 7, 1), 12)
    samples = build_samples(_event_dict(dates), {"2020/fire_1": "test"})
    for s in samples:
        assert all(d < s.target_date for d in s.input_dates)
        assert s.input_dates[-1] == s.target_date - timedelta(days=1)
        assert len(s.input_dates) == 5


def test_same_population_for_all_windows():
    # Eligibility is defined once (5-day requirement); E1-E4 slice the
    # same samples, so the sample list must be identical regardless of
    # the window a caller later selects.
    dates = _dates(date(2019, 6, 1), 8)
    samples = build_samples(_event_dict(dates), {"2020/fire_1": "validation"})
    targets = [s.target_date for s in samples]
    assert targets == [date(2019, 6, 6), date(2019, 6, 7), date(2019, 6, 8)]


def test_inventory_parses_event_dirs(tmp_path):
    base = tmp_path / "raw"
    (base / "2020" / "fire_1").mkdir(parents=True)
    (base / "2020" / "fire_1" / "2020-08-01.tif").write_bytes(b"x")
    (base / "2020" / "fire_1" / "2020-08-02.tif").write_bytes(b"x")
    (base / "2021" / "fire_2").mkdir(parents=True)
    (base / "2021" / "fire_2" / "2021-07-01.tif").write_bytes(b"x")
    inv = inventory_events(base)
    assert set(inv) == {"2020/fire_1", "2021/fire_2"}
    assert len(inv["2020/fire_1"]) == 2
