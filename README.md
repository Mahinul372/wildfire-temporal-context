# Investigating the Importance of Temporal Context in Multimodal Wildfire Spread Forecasting

**Summary.** Does adding historical satellite observations improve next-day wildfire forecasting beyond what the current fire map predicts? On 9 real WildfireSpreadTS fire events (394 days, 349 samples), tested with lightweight CPU-only baselines — persistence, logistic regression, histogram gradient boosting — the answer is mostly no: the current fire map is nearly sufficient, earlier fire maps add almost nothing, and without any fire map the models collapse (AP 0.04–0.06). Longer history does not clearly help. Persistence alone has the best hard-map F1 (0.535).

**CPU-only:** no GPU, no large models. Full experiment grid ≈ 63 min, peak memory ≈ 1 GB. This is an independent ablation on a small subset — **not** a reproduction of CMPF, VideoMAEv2, or the original WildfireSpreadTS benchmark.

## Findings

- **No fire map, no forecast (M3):** with all active-fire inputs removed, AP falls to 0.042–0.058. Weather, fuel, vegetation and forecasts alone cannot locate next-day fire at pixel level.
- **History beyond today adds almost nothing (M2 vs M1):** keeping only the current-day fire map matches keeping all history — gradient boosting AP 0.428 vs 0.453, logistic regression 0.423 vs 0.375 (better).
- **Longer context does not clearly help (E1–E4):** logistic regression degrades as history grows (AP 0.440 → 0.375); gradient boosting is flat (0.425 → 0.453, within noise for 3 test events).
- **Persistence is a strong baseline:** predicting tomorrow's fire map as today's gives the best hard-map F1 (0.535) and IoU (0.365). Learned models win on ranking (AP) and match it on F1 only after threshold tuning.

## Results

Test set: 128 samples from 3 fire events (2021); target prevalence 0.381%. Pooled pixel-level metrics at a fixed 0.5 threshold (per-event tables in `results/tables/`, full summary in `results/experiment_summary.json`).

| Config | Model | AP | Prec | Rec | F1 | IoU |
|---|---|---|---|---|---|---|
| Persistence | — | 0.288 | 0.533 | 0.537 | 0.535 | 0.365 |
| E1 (t) | Logistic regression | 0.440 | 0.125 | 0.779 | 0.215 | 0.120 |
| E1 (t) | Hist. grad. boosting | 0.425 | 0.386 | 0.730 | 0.505 | 0.338 |
| E2 (t−1, t) | Logistic regression | 0.423 | 0.086 | 0.817 | 0.156 | 0.085 |
| E2 (t−1, t) | Hist. grad. boosting | 0.441 | 0.350 | 0.757 | 0.479 | 0.315 |
| E3 (t−2..t) | Logistic regression | 0.401 | 0.074 | 0.819 | 0.136 | 0.073 |
| E3 (t−2..t) | Hist. grad. boosting | 0.459 | 0.340 | 0.765 | 0.471 | 0.308 |
| E4 (t−4..t) | Logistic regression | 0.375 | 0.055 | 0.833 | 0.103 | 0.054 |
| E4 (t−4..t) | Hist. grad. boosting | 0.453 | 0.331 | 0.772 | 0.463 | 0.301 |
| E4, M2 (current AF only) | Logistic regression | 0.423 | 0.044 | 0.795 | 0.084 | 0.044 |
| E4, M2 (current AF only) | Hist. grad. boosting | 0.428 | 0.394 | 0.716 | 0.508 | 0.341 |
| E4, M3 (no AF) | Logistic regression | 0.042 | 0.015 | 0.551 | 0.029 | 0.015 |
| E4, M3 (no AF) | Hist. grad. boosting | 0.058 | 0.188 | 0.063 | 0.095 | 0.050 |

## Data

WildfireSpreadTS (Gerard et al., NeurIPS 2023), western USA, 2018–2021: 607 fires, 13,607 daily images, 23 channels at 375 m, CC BY 4.0 ([Zenodo 8006177](https://doi.org/10.5281/zenodo.8006177)). The full archive (48.36 GB) was **not** downloaded — the 9 events (394 GeoTIFFs, 1.30 GB) were extracted remotely via HTTP Range requests. Events were selected before any modelling, by longest consecutive daily run and year. Splits by event: train 2018+2020 (4 events), validation 2019 (2), test 2021 (3) — fold 2 of the benchmark's 12-fold year cross-validation. Raw data lives in `data/` (gitignored, never committed).

## Method

Same eligible samples, splits, preprocessing and seed (42) for every config; only the temporal window and modality change. Temporal context: E1 = current day only, E2 = t−1..t, E3 = t−2..t, E4 = t−4..t. Modality: M1 = all fire history, M2 = current-day fire only, M3 = no fire inputs (target label never removed). Models: persistence, class-weighted logistic regression, histogram gradient boosting. Primary metric: Average Precision; also precision, recall, F1, IoU. 26 unit tests, all passing.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q            # unit tests
.venv/bin/python scripts/download_subset.py --workers 4   # acquire the 9 events
.venv/bin/python scripts/prepare_data.py        # preprocess, build samples
.venv/bin/python scripts/run_experiments.py     # full E×M grid + persistence
.venv/bin/python scripts/generate_report.py     # reports/research_report.md
```

All settings in `config/experiment.yaml`. Full report: `reports/research_report.md`; figures in `results/figures/` (SVG committed, PNG local-only).

## Limitations

- Only 9 events / 3 test events — small differences (AP ±0.02–0.03) are not distinguishable from event-to-event variation.
- Pixel-level models (one pixel's features + 3×3 fire neighbourhood), not spatial image models; western USA only.
- No augmentation or hyperparameter search (fixed, documented settings); VIIRS detections are a noisy target.

## License and citation

GPLv3 (see `LICENSE`). If you use the WildfireSpreadTS data, cite:

```bibtex
@inproceedings{gerard2023wildfirespreadts,
  title={WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction},
  author={Sebastian Gerard and Yu Zhao and Josephine Sullivan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=RgdGkPRQ03}
}
```
