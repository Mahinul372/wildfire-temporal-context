"""Inspection of downloaded data: integrity, inventory and fire statistics.

Loads one GeoTIFF at a time (bounded memory) and reports, per event:
day count, date range, image shape, per-day active-fire pixel counts,
and channel NaN rates on a bounded pixel subsample. Results are written
to ``data/processed/inventory.json`` for the report and for audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.preprocessing import active_fire_mask, load_day
from data.sequences import inventory_events


def inspect_event(
    by_date: dict, seed: int = 42, max_pixels_per_day: int = 50_000
) -> dict:
    """Inspect one event's downloaded days (streaming, one file at a time)."""
    rng = np.random.default_rng(seed)
    dates = sorted(by_date)
    shapes: set[tuple[int, int]] = set()
    fire_pixels: dict[str, int] = {}
    total_pixels = 0
    nan_counts = np.zeros(23, dtype=np.int64)
    nan_total = 0
    for d in dates:
        day = load_day(by_date[d])
        h, w, _ = day.shape
        shapes.add((h, w))
        total_pixels += h * w
        fire_pixels[d.isoformat()] = int(active_fire_mask(day).sum())
        n = h * w
        idx = np.arange(n)
        if n > max_pixels_per_day:
            idx = rng.choice(idx, size=max_pixels_per_day, replace=False)
        flat = day.reshape(n, 23)[idx]
        nan_counts += np.isnan(flat).sum(axis=0)
        nan_total += flat.shape[0]
        del day
    return {
        "n_days": len(dates),
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "shapes": sorted(f"{h}x{w}" for h, w in shapes),
        "total_pixels": total_pixels,
        "fire_pixels_per_day": fire_pixels,
        "n_fire_days": sum(1 for v in fire_pixels.values() if v > 0),
        "total_fire_pixels": sum(fire_pixels.values()),
        "nan_rate_per_channel": {
            str(i): float(nan_counts[i] / max(nan_total, 1)) for i in range(23)
        },
    }


def inspect_all(raw_dir: Path, out_path: Path) -> dict:
    """Inspect every downloaded event and write the inventory JSON."""
    events = inventory_events(raw_dir)
    inventory: dict = {"events": {}}
    for event, by_date in sorted(events.items()):
        inventory["events"][event] = inspect_event(by_date)
    inventory["n_events"] = len(events)
    inventory["n_days"] = sum(
        e["n_days"] for e in inventory["events"].values()
    )
    inventory["n_fire_days"] = sum(
        e["n_fire_days"] for e in inventory["events"].values()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(inventory, indent=2))
    return inventory
