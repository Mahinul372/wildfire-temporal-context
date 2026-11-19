#!/usr/bin/env bash
# One-command full run: tests -> data prep -> experiments -> report.
# Run from the repository root. Requires the subset download first:
#   .venv/bin/python scripts/download_subset.py
set -euo pipefail
cd "$(dirname "$0")"

echo "== tests =="
.venv/bin/python -m pytest tests/ -q

echo "== prepare data =="
.venv/bin/python scripts/prepare_data.py

echo "== experiments =="
.venv/bin/python scripts/run_experiments.py

echo "== report =="
.venv/bin/python scripts/generate_report.py

echo "Done. See reports/research_report.md and results/experiment_summary.json"
