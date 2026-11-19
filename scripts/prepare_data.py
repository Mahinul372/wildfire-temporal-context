#!/usr/bin/env python3
"""Prepare data for experiments (run once after download).

Steps:
  1. Fit preprocessing statistics on TRAIN events only -> stats JSON.
  2. Preprocess every downloaded day -> data/processed/days/<event>/<date>.npz
     (features float32 + active-fire uint8). One file at a time.
  3. Build the eligible sample list (5-day history requirement) -> samples.json.
  4. Inspect the downloaded data -> inventory.json.

Usage (from the repository root):
    .venv/bin/python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
import numpy as np  # noqa: E402

from data.inspection import inspect_all  # noqa: E402
from data.preprocessing import (  # noqa: E402
    fit_preprocess_stats,
    load_day,
    load_stats,
    preprocess_day,
    save_stats,
)
from data.sequences import build_samples, inventory_events, split_counts  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_data"]
    proc_dir = ROOT / cfg["paths"]["processed_data"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    events = inventory_events(raw_dir)
    print(f"inventoried {len(events)} events")
    split_of = {}
    for split, names in cfg["events"].items():
        for e in names:
            split_of[e] = split
    missing = [e for e in split_of if e not in events]
    if missing:
        print(f"ERROR: configured events not downloaded: {missing}")
        return 1

    # 1. Fit stats on train days only.
    train_days = [
        p for e, by_date in events.items() if split_of[e] == "train"
        for p in by_date.values()
    ]
    print(f"fitting preprocessing stats on {len(train_days)} train days ...")
    stats = fit_preprocess_stats(train_days, seed=cfg["random_seed"])
    stats_path = proc_dir / "preprocess_stats.json"
    save_stats(stats, stats_path)
    print(f"  land-cover classes: {stats['landcover_classes']}")
    print(f"  stats -> {stats_path}")

    # 2. Preprocess every day (streaming, one file at a time).
    days_dir = proc_dir / "days"
    n_done = 0
    for event, by_date in sorted(events.items()):
        for d, path in sorted(by_date.items()):
            out = days_dir / event / f"{d.isoformat()}.npz"
            if out.exists():
                n_done += 1
                continue
            day = load_day(path)
            prep = preprocess_day(day, stats)
            feats = np.concatenate(
                [prep["continuous"], prep["angular"], prep["landcover"]],
                axis=2,
            ).astype(np.float32)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                out,
                features=feats,
                active_fire=prep["active_fire"],
                shape=np.array(day.shape[:2]),
            )
            n_done += 1
            del day, prep, feats
        print(f"  preprocessed {event}: {len(by_date)} days")
    print(f"preprocessed {n_done} days -> {days_dir}")

    # 3. Build samples.
    samples = build_samples(events, split_of)
    samples_json = [
        {
            "event": s.event,
            "split": s.split,
            "target_date": s.target_date.isoformat(),
            "input_dates": [d.isoformat() for d in s.input_dates],
        }
        for s in samples
    ]
    (proc_dir / "samples.json").write_text(json.dumps(samples_json, indent=2))
    print(f"built {len(samples)} samples: {split_counts(samples)}")

    # 4. Inspection / inventory.
    inventory = inspect_all(raw_dir, proc_dir / "inventory.json")
    print(
        f"inventory: {inventory['n_events']} events, {inventory['n_days']} days, "
        f"{inventory['n_fire_days']} days with fire"
    )
    print(f"prepare_data done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
