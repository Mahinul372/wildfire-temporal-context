# Investigating the Importance of Temporal Context in Multimodal Wildfire Spread Forecasting

An independent, controlled temporal-context and input-modality ablation study on
next-day wildfire active-fire forecasting, using lightweight CPU baselines on a
small subset of the WildfireSpreadTS dataset.

**Research question:** does adding historical satellite observations improve
next-day wildfire active-fire forecasting beyond what the current fire map and
environmental conditions predict?

This is **not** a reproduction of CMPF, VideoMAEv2, or the original
WildfireSpreadTS benchmark results. It is a small, CPU-only investigation that
asks a narrower question — how much temporal context helps, and how much of the
signal comes from the active-fire history itself — with simple, transparent
baselines (persistence, logistic regression, histogram gradient boosting) on
nine real fire events.

## Status

- [x] Dataset inspected (Zenodo record 8006177; 14,218 archive entries; 607 fires)
- [x] 9 fire events acquired by remote ZIP extraction (394 daily GeoTIFFs, 1.30 GB)
- [x] Preprocessing, sequence construction, features, models, metrics, tests
      implemented (26 unit tests, all passing)
- [x] Full experiment grid run on real data (13 configurations, 349 samples)
- [x] Report and figures generated (`reports/research_report.md`,
      `results/figures/`)

## Dataset

WildfireSpreadTS (Gerard et al., NeurIPS 2023 Datasets & Benchmarks), western
USA, 2018–2021, 607 fires, 13,607 daily images, 23 channels at 375 m
resolution, CC BY 4.0, Zenodo DOI
[10.5281/zenodo.8006177](https://doi.org/10.5281/zenodo.8006177).
Each image has 23 channels: VIIRS reflectance (M11, I2, I1), NDVI, EVI2,
precipitation, wind speed/direction, min/max temperature, energy release
component, specific humidity, slope, aspect, elevation, Palmer drought
severity index, land-cover class, forecast precipitation, forecast wind
speed/direction, forecast temperature, forecast specific humidity, and
active fire. The target is the binarised next-day active-fire mask.

### Acquisition

The full archive is 48.36 GB, so it was **not** downloaded. Individual
fire-event GeoTIFFs were extracted remotely from the Zenodo ZIP using HTTP
Range requests and `remotezip` (the archive supports `Accept-Ranges: bytes`;
the ZIP64 central directory was fetched and parsed first). Nine events were
selected **before any modelling, by longest consecutive daily run and year**
— never by prediction performance:

| Split | Event | Days |
|---|---|---|
| Train (2018) | `2018/fire_21889943` | 46 |
| Train (2018) | `2018/fire_21890524` | 52 |
| Train (2020) | `2020/fire_24332764` | 46 |
| Train (2020) | `2020/fire_24333000` | 45 |
| Validation (2019) | `2019/fire_23159792` | 30 |
| Validation (2019) | `2019/fire_22938749` | 32 |
| Test (2021) | `2021/fire_25295808` | 49 |
| Test (2021) | `2021/fire_25411093` | 46 |
| Test (2021) | `2021/fire_25295855` | 48 |

Total: **9 events, 394 days, 1.30 GB**. This mirrors the original
benchmark's year-based split (train 2018+2020, validation 2019, test 2021 —
fold 2 of its 12-fold year cross-validation). Raw data lives in `data/`
(gitignored, never committed).

## Relation to the original implementation

The original loader (`SebastianGer/WildfireSpreadTS`,
`src/dataloader/FireSpreadDataset.py`) was read in full. This project follows
its semantics, with the following deliberate, documented differences:

- **Angular features (Feb 2026 issue).** The original applies **sin only** to
  the degree channels (indices 7 wind direction, 13 aspect, 19 forecast wind
  direction). The repository README (Feb 2026) notes this loses information
  and that sin+cos should be used. This project uses **sin and cos** for all
  three angular channels — the fix the issue requests.
- **Active-fire values.** Raw active-fire values are VIIRS detection times
  (hhmm); the original divides by 100 to get the hour and binarises with
  `> 0`. This project binarises `> 0` directly, which is equivalent for the
  binary mask. The original additionally keeps the standardised detection-hour
  channel alongside the binary mask; this project uses the binary mask plus a
  3×3 neighbourhood mean instead, which keeps the M1/M2/M3 ablation clean.
- **Sequence construction.** The original takes `n+1` consecutive images by
  file index, without checking calendar consecutiveness. This project requires
  consecutive calendar days; on the selected events (fully consecutive runs)
  the two are identical.
- **Dataset-class bug (commit `ab3c8f3`, 2024-07-25).** The original's
  global-index → fire lookup could resolve to the wrong fire when a fire name
  repeated across years. This project builds samples directly from per-event
  inventories, so that failure mode cannot occur.
- **Normalisation statistics.** The original uses precomputed fixed
  mean/std tables per year combination. This project fits medians/means/stds
  on the training events only (no test statistics anywhere).
- **Land cover.** The original one-hot encodes over a fixed 17-class matrix
  with class value − 1 indexing. This project one-hots over the classes
  observed in the training events (unseen classes map to all zeros).
- **No augmentation, no cropping.** The original trains with random square
  crops and 90° rotations/flips. This project uses full frames (images are
  ~300×250 px; pixel-level models, no GPU) and no augmentation.

## Experiments

All configurations are evaluated on the **same eligible samples** (a target
day is eligible only if the target day and the five preceding days are all
present as consecutive days), so metrics are comparable across configurations.

**Temporal context** (same target day t+1 for all):

| Config | Input window |
|---|---|
| E1 | current day only (t) |
| E2 | t−1, t |
| E3 | t−2, t−1, t |
| E4 | t−4 … t |

**Input modality** (target label never removed):

| Config | Active-fire inputs |
|---|---|
| M1 | all historical active-fire retained |
| M2 | earlier active-fire removed, current-day retained |
| M3 | all active-fire inputs removed (weather, forecasts, fuel, vegetation, topography, land cover only) |

**Models:** persistence baseline (predict t+1 from the t active-fire map),
logistic regression (class-weighted), histogram gradient boosting. Same model
family, splits, preprocessing, and seed (42) across configs; only the temporal
window and modality change.

**Splits** are by wildfire event (never by pixel or shuffled day); no event
appears in more than one partition; no future data enters the inputs; no test
statistics enter preprocessing; thresholds are tuned on validation only.

**Metrics:** Average Precision (primary), precision, recall, F1, IoU at a
fixed 0.5 threshold, plus an F1-optimal threshold taken from validation only.
Positive prevalence, per-event metrics, pooled vs mean event-level metrics,
and event-level bootstrap confidence intervals (1,000 resamples) are
reported. Accuracy is never used (fire pixels are ~0.17% of all pixels).

## Results

Test set: 128 samples from 3 fire events in 2021; target prevalence
0.381%. Pooled pixel-level metrics at the fixed 0.5 threshold (full
per-event tables in `results/tables/`, summary in
`results/experiment_summary.json`).

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

**What the numbers say (3 test events — treat differences as indicative,
not precise):**
- **Without any active-fire inputs (M3), the models fail** (AP 0.042–0.058):
  weather, fuel, vegetation and forecasts alone cannot locate next-day
  fire at pixel level. The predictive signal is overwhelmingly in the
  fire history itself.
- **Earlier fire maps add almost nothing beyond the current day's map:**
  M2 ≈ M1 for gradient boosting (AP 0.428 vs 0.453) and *better* for
  logistic regression (0.423 vs 0.375).
- **Longer temporal context does not clearly help these baselines:**
  logistic regression degrades with history (AP 0.440 → 0.375);
  gradient boosting is flat (0.425 → 0.453, within noise).
- **Persistence is a strong baseline** on hard-map metrics (F1 0.535,
  IoU 0.365 at 0.5 — the best F1 in the table). Learned models win on
  ranking (AP) and match it on F1 only after validation-threshold tuning
  (F1 up to 0.510 at the F1-optimal threshold).
- Per-event test AP varies widely (e.g. E4 HistGB: 0.441 / 0.278 / 0.479
  across the three test events; the quiet fire is always worst), and
  event-level bootstrap CIs are wide — see the report for details.
- Total grid runtime ≈ 63 min of compute across the runs (HistGB is the
  slow part: 293 s at E1 → 723 s at E4); peak memory ≈ 1 GB.

## Reproducibility

CPU-only. Tested on a Linux VM with ~1.5 shared CPUs and ~8 GB RAM.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q            # unit tests (synthetic fixtures)
.venv/bin/python scripts/download_subset.py --workers 4   # acquire the 9 events
.venv/bin/python scripts/prepare_data.py        # preprocess, build samples
.venv/bin/python scripts/run_experiments.py     # full E×M grid + persistence
.venv/bin/python scripts/generate_report.py     # reports/research_report.md
```

Or everything after download with `./run_all.sh`. All settings live in
`config/experiment.yaml`. The full report is in `reports/research_report.md`;
tables in `results/tables/`, figures in `results/figures/` (SVG committed,
PNG local-only).

## Limitations

- **Only 9 fire events / 3 test events.** Event-level bootstrap intervals are
  indicative at best; do not over-interpret small differences.
- **Pixel-level models, not image models.** Logistic regression and gradient
  boosting see one pixel's feature vector (plus a 3×3 active-fire
  neighbourhood), not spatial context the way a U-Net or video model would.
- **Small, geographically narrow subset** (western USA, 2018–2021); results
  do not generalise to other regions or fire regimes without further work.
- **No data augmentation, no hyperparameter search** (fixed, documented
  settings); the question is the *ablation*, not squeezing the best score.
- Active-fire detections are VIIRS satellite overpasses, not ground truth;
  cloudy or missed overpasses appear as absent fire.

## Motivation

This project was built in connection with an application to the Arizona State
University School of Geographical Sciences and Urban Planning (Geography
MA/PhD, Fall 2027), and in particular Prof. Wenwen Li's GeoAI work on
adapting video foundation models for spatiotemporal wildfire forecasting
(CMPF; Li, Hsu & Wang, IEEE TGRS 2026). It is an independent student
investigation that takes the temporal-context question from that line of work
and tests it with simple, transparent baselines — it does not reproduce or
claim to reproduce CMPF, VideoMAEv2, or any published benchmark number.

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
