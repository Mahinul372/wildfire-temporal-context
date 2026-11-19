# Investigating the Importance of Temporal Context in Multimodal Wildfire Spread Forecasting

## Abstract

This study asks whether adding historical satellite observations improves next-day wildfire active-fire forecasting beyond what the current fire map and environmental conditions predict. Using a subset of the WildfireSpreadTS dataset (Gerard et al., 2023), we run a controlled ablation study with lightweight CPU-only baselines: a persistence baseline, logistic regression and histogram-based gradient boosting. Four temporal histories (1, 2, 3 and 5 days) and three active-fire input modalities (all history, current day only, none) are compared on identical samples with event-level splits. This is an independent investigation of temporal context; it does not reproduce CMPF, VideoMAEv2 or the original benchmark results (Li et al., 2026).

## 1. Introduction and research motivation

Wildfire spread forecasting from satellite data is a spatiotemporal prediction problem: given daily multimodal observations up to day t, predict the active-fire map on day t+1. Recent work (Li, Hsu and Wang, 2026) adapts large video foundation models to this task with cross-modal progressive fine-tuning (CMPF), showing that longer temporal context helps (e.g. VideoMAEv2 average precision 0.370 at 1 day vs 0.377 at 5 days on 2021 WildfireSpreadTS data). Those models need GPUs and large-scale pretraining. The open question addressed here is how much of that temporal-context effect is already visible with cheap, CPU-only baselines -- and how much of it is simply the persistence of the fire itself. Answering it matters for practitioners without GPU access and for understanding what historical observations actually add.

## 2. Related research

* Gerard, Zhao and Sullivan (2023) introduced WildfireSpreadTS (NeurIPS Datasets and Benchmarks): 607 fires, 13,607 daily 23-channel images at 375 m resolution, western USA, 2018-2021, with next-day binarised active fire as the benchmark target.
* Huot et al. (2022) introduced the Next Day Wildfire Spread dataset and CNN baselines for the same task at 1 km resolution.
* Li, Hsu and Wang (2026) adapt video foundation models (VideoMAEv2, MViTv2, ViT) with CMPF, reaching AP 0.407 at 5 days on 2021 data. Our study uses the same dataset and target definition but lightweight CPU baselines, and asks a narrower question: the marginal value of temporal context and of historical active-fire inputs.
* Original benchmark baselines (Gerard et al., 2023, mean AP across the 12 year-permutation folds): persistence 0.193, logistic regression 0.279-0.288, U-Net (ResNet-18) 0.341 (1 day) to 0.344 (5 days), UTAE 0.372 (best reported). These are full-dataset numbers over 607 fires, not directly comparable to our 9-event subset, but they set the scale: persistence alone is a strong baseline, and extra context and features historically bought only modest gains.

## 3. Research questions and hypotheses

1. Does five-day context improve next-day forecasting over current-day observations? (Hypothesis: yes, modestly.)
2. Does additional historical context improve consistently, or plateau/deteriorate with longer histories?
3. How much does the model depend on historical active-fire observations (M1 vs M2 vs M3)?
4. Can environmental variables predict without earlier fire maps (M3)?
5. How much apparent performance is explained by persistence?

## 4. Dataset and preprocessing

Source: WildfireSpreadTS (Zenodo record 8006177, DOI 10.5281/zenodo.8006177, CC BY 4.0). The full archive is 48.4 GB; only the configured subset was extracted via HTTP Range requests and remote ZIP access (see `src/data/acquisition.py`). Subset: 9 fire events (4 train, 2 validation, 3 test), 394 daily images, of which 340 days contain at least one active-fire pixel. Events were chosen by sequence completeness (longest consecutive run of daily observations) before any modelling.

Each day is a 23-channel GeoTIFF at 375 m (VIIRS reflectance bands M11/I2/I1, NDVI, EVI2, precipitation, wind speed and direction, min/max temperature, energy release component, specific humidity, slope, aspect, elevation, Palmer drought severity index, land-cover class, and 5 forecast variables, plus active fire). Preprocessing: active-fire NaN (no detection) replaced by 0 and binarised; other channels' NaNs filled with train medians; continuous channels standardised with train means/stds; wind direction, aspect and forecast wind direction encoded as (sin, cos) to preserve circularity; land cover one-hot encoded. All fitted statistics use train events only.

Relation to the original loader (SebastianGer/WildfireSpreadTS): the original applies sin only to the three degree channels, which its README (Feb 2026) notes loses information -- sin and cos are used here instead, as the issue requests. The original's 2024 dataset-class bug (commit ab3c8f3), where a global index could resolve to the wrong fire, cannot occur here because samples are built directly from per-event inventories. The original builds sequences from consecutive file indices; consecutive calendar days are required here (identical on the selected fully-consecutive events). Forecast wind-direction values in one inspected file ranged about -88 to 86 degrees, suggesting a -180..180 convention rather than 0..360; the sin/cos encoding is correct under either convention.

Samples: one sample per (event, target date) with 5 consecutive input days and a consecutive target day. Eligible samples: 349 total (169 train, 52 validation, 128 test). Splits are by fire event; no event appears in more than one partition. This follows the original benchmark's year-based split (train 2018+2020, validation 2019, test 2021).

## 5. Experimental methodology

Temporal ablation (E1-E4): input windows of 1, 2, 3 and 5 days, same target day t+1, same sample population, same models, splits, preprocessing and seeds. Modality ablation (M1-M3) at the 5-day window: M1 retains all historical active fire, M2 keeps only the current day's, M3 removes all active-fire inputs (environmental channels retained). The persistence baseline predicts the t+1 map as the day-t active-fire map.

## 6. Models and evaluation metrics

Models: persistence; logistic regression (class_weight=balanced, C=1.0); histogram-based gradient boosting (200 trees, learning rate 0.1, 31 leaves). Pixel-level features: per-day standardised channels, sin/cos angles, one-hot land cover, active-fire binary and its 3x3 neighbourhood mean. Training rows are bounded (all positives kept, negatives subsampled 10:1 per positive, 20k pixels per sample, 200k rows total). Primary metric: Average Precision; also precision, recall, F1 and IoU at a fixed 0.5 threshold and at an F1-optimal threshold selected on validation data only. Accuracy is not reported (0.17% positive prevalence in the full dataset). Pooled pixel-level and mean event-level metrics are reported separately, with event-level bootstrap confidence intervals for AP (indicative with 3 test events).

## 7. Results

Test-set pooled metrics at the fixed 0.5 threshold (validation results and per-event metrics are in `results/tables/`):

| Config | Model | Window | AF mode | AP | Prec | Rec | F1 | IoU |
|---|---|---|---|---|---|---|---|---|
| persistence | persistence | 5 | all | 0.288 | 0.533 | 0.537 | 0.535 | 0.365 |
| E1_logistic_regression | logistic_regression | 1 | all | 0.440 | 0.125 | 0.779 | 0.215 | 0.120 |
| E1_hist_gradient_boosting | hist_gradient_boosting | 1 | all | 0.425 | 0.386 | 0.730 | 0.505 | 0.338 |
| E2_logistic_regression | logistic_regression | 2 | all | 0.423 | 0.086 | 0.817 | 0.156 | 0.085 |
| E2_hist_gradient_boosting | hist_gradient_boosting | 2 | all | 0.441 | 0.350 | 0.757 | 0.479 | 0.315 |
| E3_logistic_regression | logistic_regression | 3 | all | 0.401 | 0.074 | 0.819 | 0.136 | 0.073 |
| E3_hist_gradient_boosting | hist_gradient_boosting | 3 | all | 0.459 | 0.340 | 0.765 | 0.471 | 0.308 |
| E4_logistic_regression | logistic_regression | 5 | all | 0.375 | 0.055 | 0.833 | 0.103 | 0.054 |
| E4_hist_gradient_boosting | hist_gradient_boosting | 5 | all | 0.453 | 0.331 | 0.772 | 0.463 | 0.301 |
| E4_M2_logistic_regression | logistic_regression | 5 | current | 0.423 | 0.044 | 0.795 | 0.084 | 0.044 |
| E4_M2_hist_gradient_boosting | hist_gradient_boosting | 5 | current | 0.428 | 0.394 | 0.716 | 0.508 | 0.341 |
| E4_M3_logistic_regression | logistic_regression | 5 | none | 0.042 | 0.015 | 0.551 | 0.029 | 0.015 |
| E4_M3_hist_gradient_boosting | hist_gradient_boosting | 5 | none | 0.058 | 0.188 | 0.063 | 0.095 | 0.050 |

Comparison charts: `results/figures/temporal_ap.svg` and `results/figures/modality_ap.svg`. Example prediction maps: `results/figures/`.

## 8. Discussion

The headline result is the modality ablation, not the temporal one. Removing all active-fire inputs (M3) collapses average precision from 0.453 (E4 HistGB, M1) to 0.058, and from 0.375 to 0.042 for logistic regression. At pixel level, weather, fuel, vegetation, topography and forecasts alone cannot locate tomorrow's fire: without knowing where the fire is today, the models have almost nothing to anchor on. This answers research question 4 plainly -- no, environmental variables alone do not predict next-day fire maps at 375 m with these baselines.

By contrast, earlier fire maps add almost nothing beyond the current day's map (question 3). At the 5-day window, keeping only the current day's active fire (M2) instead of all five days (M1) changes AP by -0.025 for gradient boosting and +0.048 for logistic regression -- the linear model actually improves when the noisier historical fire features are removed. The fire's present location dominates; its history is largely redundant information for these pixel-level models.

Temporal context (questions 1-2) shows no clear gain either. From 1 to 5 days of history, AP changes by -0.065 for logistic regression (it degrades as features multiply while training rows stay capped at 200k) and by +0.028 for gradient boosting, whose best score is 0.459 (E3_hist_gradient_boosting). That is the same direction as Li et al. (2026), who report a small T=1 to T=5 gain (VideoMAEv2 AP 0.370 to 0.377), but here the change is within the noise of three test events and is dwarfed by the M3 effect. A longer history does not rescue a pixel-level baseline the way it helps a video foundation model -- which is itself informative: the temporal-context benefit in the literature likely comes from spatiotemporal representation learning, not from merely stacking more daily feature maps.

Persistence (question 5) explains a large share of apparent performance. It scores AP 0.288 -- below the learned models -- but at the fixed 0.5 threshold its F1 (0.535) and IoU are the best in the table. The learned models only match it on hard-map metrics after threshold tuning on validation data. Logistic regression is the starkest case: at 0.5 its precision collapses as history grows (E1 0.125 to E4 0.055) while recall stays near 0.8 -- the balanced class weight pushes scores up, and the F1-optimal threshold (0.85-0.95) is doing the real work. Any comparison that reports only AP, or only F1 at 0.5, would mislead here; both are needed.

Uncertainty is dominated by between-event variation. For E4 gradient boosting, per-event test AP is fire_25295808 0.441, fire_25295855 0.278, fire_25411093 0.479 -- the quiet fire is always the worst -- and the event-level bootstrap 95% interval is [0.278, 0.479]. With three test events, a difference of 0.02-0.03 AP between configurations is not distinguishable from event-to-event variation. Validation AP is systematically far below test AP (e.g. E1 logistic regression 0.094 validation vs 0.440 test) because validation prevalence (0.115%) is about a third of test prevalence (0.381%) and one validation event is very quiet; thresholds tuned on validation are therefore high and transfer imperfectly. These are properties of the subset, not of the method, and they bound what can be claimed.

In short: for cheap pixel-level baselines, the current fire map is nearly sufficient, history beyond it is nearly redundant, and environmental variables without any fire map are nearly useless. That is a useful negative result for anyone deciding where to spend compute -- it points effort toward spatial/spatiotemporal modelling of spread from the current perimeter, rather than toward longer feature stacks.


## 9. Limitations

* Small subset: only a handful of independent fire events, so geographical generalisation is limited and uncertainty intervals are wide.
* Single region (western USA) and single target (next-day active fire at 375 m).
* Pixel-level models ignore spatial context beyond a 3x3 active-fire neighbourhood; no spatial modelling of spread dynamics.
* VIIRS active-fire detections contain false alarms and miss small or short-lived fires; the target is a noisy proxy for true fire extent.
* Human firefighting is not represented in the data.
* Validation and test prevalences differ threefold (0.115% vs 0.381%), and only two validation events exist, so F1-optimal thresholds tuned on validation transfer imperfectly to test; absolute AP values are not comparable across the two partitions.

## 10. Conclusion

Answering the five research questions on this 9-event subset: (1) five-day context does not clearly improve on current-day observations for these baselines (gradient boosting AP 0.425 at E1 vs 0.453 at E4); (2) longer histories plateau for gradient boosting and deteriorate for logistic regression; (3) the models depend heavily on active-fire inputs, but almost entirely on the current day's map; (4) environmental variables alone cannot predict next-day fire maps (M3 AP 0.058); (5) much of the apparent performance is persistence -- the no-model baseline already reaches F1 0.535 at 0.5. The contribution is a reproducible, CPU-only ablation that separates these effects on identical samples, with honest uncertainty bounds. The full pipeline (acquisition, preprocessing, experiments, evaluation) is in this repository and can be re-run with one command.

## 11. References

* Gerard, S., Zhao, Y., Sullivan, J. (2023). WildfireSpreadTS: A Dataset of Multi-Modal Time Series for Wildfire Spread Prediction. NeurIPS 2023 Datasets and Benchmarks. https://doi.org/10.5281/zenodo.8006177
* Li, W., Hsu, C.-Y., Wang, S. (2026). Adapting Video Foundation Models for Spatiotemporal Wildfire Forecasting via Cross-Modal Progressive Fine-Tuning. IEEE Transactions on Geoscience and Remote Sensing, 64, 3000218. https://doi.org/10.1109/TGRS.2026.3652453
* Huot, F. et al. (2022). Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data. IEEE TGRS, 60, 1-13.
* Artés, T. et al. (2019). A global wildfire dataset for the analysis of fire regimes and fire behaviour. Scientific Data, 6.

## Reproducibility

Environment: CPU-only, Python 3.12, scikit-learn, numpy, scipy, pandas, matplotlib, tifffile, h5py, PyYAML, remotezip (see `requirements.txt`). Data: download the configured subset with `.venv/bin/python scripts/download_subset.py`, then `.venv/bin/python scripts/prepare_data.py`, then `.venv/bin/python scripts/run_experiments.py`. Random seed: 42 throughout. Raw data is never committed.
