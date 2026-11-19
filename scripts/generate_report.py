#!/usr/bin/env python3
"""Generate reports/research_report.md from the verified artefacts.

Usage (from the repository root):
    .venv/bin/python scripts/generate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from reporting import build_report  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    proc_dir = ROOT / cfg["paths"]["processed_data"]
    inventory = json.loads((proc_dir / "inventory.json").read_text())
    samples = json.loads((proc_dir / "samples.json").read_text())
    summary_path = ROOT / cfg["paths"]["summary_json"]
    summary = None
    results_detail = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    partial_path = ROOT / "results" / "partial_results.jsonl"
    if partial_path.exists():
        results_detail = [json.loads(line) for line in
                          partial_path.read_text().splitlines() if line.strip()]
    report = build_report(ROOT, cfg, inventory, samples, summary,
                          results_detail)
    out = ROOT / cfg["paths"]["report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"wrote {out} ({len(report)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
