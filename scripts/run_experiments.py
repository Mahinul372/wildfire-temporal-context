#!/usr/bin/env python3
"""Run the full experiment grid and save results + figures.

Reads data/processed/ (stats, samples, cached days) written by
scripts/prepare_data.py. Writes:
  * results/tables/<config>_test_metrics.csv (pooled + per-event)
  * results/experiment_summary.json
  * results/figures/temporal_ap.svg, modality_ap.svg
  * results/figures/example_<event>_<date>.png (input strip + maps)

Usage (from the repository root):
    .venv/bin/python scripts/run_experiments.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
import numpy as np  # noqa: E402

from data.sequences import Sample  # noqa: E402
from data.preprocessing import load_stats  # noqa: E402
from experiments import ExperimentRunner, load_processed_day, summarise  # noqa: E402
from visualization import comparison_bars, fire_map_panel, input_sequence_strip  # noqa: E402


def load_samples(proc_dir: Path) -> list[Sample]:
    raw = json.loads((proc_dir / "samples.json").read_text())
    samples = []
    for s in raw:
        target = date.fromisoformat(s["target_date"])
        inputs = tuple(date.fromisoformat(d) for d in s["input_dates"])
        samples.append(
            Sample(
                event=s["event"],
                split=s["split"],
                target_date=target,
                input_dates=inputs,
                input_paths=tuple(
                    proc_dir / "days" / s["event"] / f"{d.isoformat()}.npz"
                    for d in inputs
                ),
                target_path=proc_dir / "days" / s["event"]
                / f"{target.isoformat()}.npz",
            )
        )
    return samples


def write_metrics_csv(result: dict, tables_dir: Path) -> None:
    """Pooled + per-event test metrics at both thresholds."""
    path = tables_dir / f"{result['config']}_test_metrics.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "event_or_pooled", "threshold", "average_precision",
                    "precision", "recall", "f1", "iou", "n_total", "n_pos"])
        for tname, pooled in result["test"]["pooled"].items():
            w.writerow(["pooled", "all", tname,
                        pooled["average_precision"], pooled["precision"],
                        pooled["recall"], pooled["f1"], pooled["iou"],
                        pooled["n_total"], pooled["n_pos"]])
        for tname, evs in result["test"]["per_event"].items():
            for e in evs:
                w.writerow(["per_event", f"{e['event']}@{e['target_date']}",
                            tname, e["average_precision"], e["precision"],
                            e["recall"], e["f1"], e["iou"],
                            e["n_total"], e["n_pos"]])
    print(f"  wrote {path}")


def make_figures(runner: ExperimentRunner, results: list[dict],
                 figs_dir: Path) -> None:
    """Comparison charts + example prediction maps."""
    summary = summarise(results)["results"]
    temporal = [r for r in summary if r["config"].startswith(("E1_", "E2_", "E3_", "E4_"))
                and "_M" not in r["config"] and r["model"] != "persistence"]
    # temporal rows are E1..E4 x 2 models; keep config order E1..E4
    order = ["E1", "E2", "E3", "E4"]
    temporal.sort(key=lambda r: (order.index(r["config"].split("_")[0]),
                                 r["model"]))
    comparison_bars(
        temporal, "test_AP",
        "Temporal history ablation: test Average Precision (pooled, threshold 0.5)",
        figs_dir / "temporal_ap.svg",
    )
    modality = [r for r in summary
                if r["config"].startswith("E4_M") or r["config"] in
                ("E4_logistic_regression", "E4_hist_gradient_boosting")]
    modality.sort(key=lambda r: (r["active_fire_mode"], r["model"]))
    comparison_bars(
        modality, "test_AP",
        "Active-fire modality ablation at 5-day history: test Average Precision",
        figs_dir / "modality_ap.svg",
    )
    # Example maps from the E4 HistGB run (artifacts kept).
    best = next((r for r in results if r["config"] == "E4_hist_gradient_boosting"),
                None)
    if best is None or "_test_preds" not in best:
        print("  (no E4 HistGB artifacts; skipping example maps)")
        return
    thresh = best["f1_optimal_threshold"]
    preds = best["_test_preds"]
    # Pick up to 3 test samples with the most target fire pixels.
    preds_sorted = sorted(preds, key=lambda p: int((p["y_true"] == 1).sum()),
                          reverse=True)[:3]
    for p in preds_sorted:
        event = p["event"]
        target = date.fromisoformat(p["target_date"])
        sample = next(s for s in runner.samples
                      if s.event == event and s.target_date == target)
        h, w = load_processed_day(
            runner.day_path(sample, target)).shape
        af_maps = []
        labels = []
        for d in sample.input_dates:
            day = load_processed_day(runner.day_path(sample, d))
            af_maps.append(day.active_fire)
            labels.append(d.isoformat())
        input_sequence_strip(
            af_maps, labels,
            f"{event} input active-fire maps (target {target.isoformat()})",
            figs_dir / f"example_{event.replace('/', '_')}_{target.isoformat()}_inputs.png",
        )
        last_af = af_maps[-1]
        fire_map_panel(
            last_af,
            p["y_true"].reshape(h, w),
            p["y_score"].reshape(h, w),
            thresh,
            f"{event} target {target.isoformat()} "
            f"(E4 HistGB, F1-optimal threshold {thresh:.2f})",
            figs_dir / f"example_{event.replace('/', '_')}_{target.isoformat()}_maps.png",
        )
    print(f"  wrote figures -> {figs_dir}")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    proc_dir = ROOT / cfg["paths"]["processed_data"]
    stats = load_stats(proc_dir / "preprocess_stats.json")
    samples = load_samples(proc_dir)
    print(f"loaded {len(samples)} samples")

    runner = ExperimentRunner(ROOT, cfg, stats, samples)
    t0 = time.time()
    tables_dir = ROOT / cfg["paths"]["results_tables"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    partial_path = ROOT / "results" / "partial_results.jsonl"
    # Incremental writes: each finished configuration is saved
    # immediately (CSV + JSONL), so a killed run loses nothing done.
    if partial_path.exists():
        partial_path.unlink()
    done = [0]

    def _save_partial(result: dict) -> None:
        write_metrics_csv(result, tables_dir)
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(partial_path, "a") as f:
            f.write(json.dumps(clean, default=str) + "\n")
        done[0] += 1
        print(f"[{done[0]}/13] {result['config']}: "
              f"test AP={result['test']['pooled']['fixed_0.5']['average_precision']:.3f} "
              f"in {result['runtime_seconds']:.0f}s "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    results = runner.run_all(on_result=_save_partial)
    print(f"grid done in {time.time() - t0:.0f}s")

    # Summary JSON: strip non-serialisable artifacts first.
    clean = []
    for r in results:
        clean.append({k: v for k, v in r.items() if not k.startswith("_")})
    summary = summarise(clean)
    summary["config"] = {
        "events": cfg["events"],
        "temporal": cfg["experiments"]["temporal"],
        "modality": cfg["experiments"]["modality"],
        "models": cfg["models"],
        "sampling": cfg["sampling"],
        "random_seed": cfg["random_seed"],
    }
    summary_path = ROOT / cfg["paths"]["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {summary_path}")

    figs_dir = ROOT / cfg["paths"]["results_figures"]
    make_figures(runner, clean, figs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
