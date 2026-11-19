#!/usr/bin/env python3
"""Inspect the downloaded subset and print a summary.

Usage (from the repository root):
    .venv/bin/python scripts/inspect_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from data.inspection import inspect_all  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    raw_dir = ROOT / cfg["paths"]["raw_data"]
    proc_dir = ROOT / cfg["paths"]["processed_data"]
    inv = inspect_all(raw_dir, proc_dir / "inventory.json")
    print(f"events: {inv['n_events']}, days: {inv['n_days']}, "
          f"fire days: {inv['n_fire_days']}")
    for event, e in sorted(inv["events"].items()):
        print(
            f"  {event}: {e['n_days']} days "
            f"{e['first_date']}..{e['last_date']} "
            f"shapes={e['shapes']} fire_days={e['n_fire_days']} "
            f"fire_pixels={e['total_fire_pixels']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
