#!/usr/bin/env python3
"""Download the configured WildfireSpreadTS subset.

Reads the event list from config/experiment.yaml and extracts only those
fire events' GeoTIFFs from the remote 48.4 GB ZIP via HTTP Range
requests (see src/data/acquisition.py). The full archive is never
downloaded. Idempotent: already-downloaded files with the expected size
are skipped, so a killed run can simply be re-executed.

Usage (from the repository root):
    .venv/bin/python scripts/download_subset.py [--workers N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from data.acquisition import (  # noqa: E402
    ZIP_URL,
    download_events,
    fetch_central_directory,
    list_fire_events,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=ROOT / "config" / "experiment.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    wanted: list[str] = []
    for split in ("train", "validation", "test"):
        wanted.extend(cfg["events"][split])
    print(f"events requested: {len(wanted)}")
    for e in wanted:
        print(f"  {e}")

    t0 = time.time()
    print("fetching ZIP central directory ...")
    members = fetch_central_directory()
    by_name = {m.name: m for m in members}
    events = list_fire_events(members)
    print(f"archive has {len(members)} members, {len(events)} fire events")

    selected = {e: events[e] for e in wanted if e in events}
    missing = [e for e in wanted if e not in events]
    if missing:
        print(f"WARNING: events not found in archive: {missing}")
    n_files = sum(len(v) for v in selected.values())
    n_bytes = sum(by_name[n].compress_size for v in selected.values() for n in v)
    print(f"selected {len(selected)} events, {n_files} files, "
          f"~{n_bytes / 1e9:.2f} GB to transfer")

    out_dir = ROOT / cfg["paths"]["raw_data"]
    results = download_events(selected, by_name, out_dir, workers=args.workers)
    write_manifest(results, out_dir / "manifest.json")
    n_ok = sum(1 for r in results if r["status"] in ("downloaded", "skipped"))
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print(f"done in {time.time() - t0:.0f}s: {n_ok} ok, {n_fail} failed")
    if n_fail:
        for r in results:
            if r["status"] == "failed":
                print("  FAILED:", r["name"], r.get("error"))
        return 1
    print(f"manifest: {out_dir / 'manifest.json'}")
    print(f"source: {ZIP_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
