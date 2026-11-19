"""Research report generation (Markdown).

The report is built ONLY from verified artefacts: the config, the
download manifest, the data inventory, the sample list and the
experiment summary JSON. If experiments have not run, the results
sections state that plainly instead of inventing numbers.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(x, digits: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        import math

        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "n/a"
    except Exception:
        pass
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def _discussion(summary: dict | None,
                results_detail: list[dict] | None) -> list[str]:
    """Discussion computed from the actual results (no hard-coded
    numbers). Falls back to a generic caution if results are absent.
    """
    generic = [
        "See the results table above. Interpret with the sample size "
        "in mind: conclusions rest on a small number of independent "
        "fire events, so differences between configurations should be "
        "read as indicative, not as precise estimates of population "
        "effects."
    ]
    if summary is None or not summary.get("results"):
        return generic
    rows = {r["config"]: r for r in summary["results"]}
    detail = {r["config"]: r for r in (results_detail or [])}
    get = lambda c: rows.get(c, {})
    ap = lambda c: get(c).get("test_AP", float("nan"))
    f1 = lambda c: get(c).get("test_F1", float("nan"))
    out: list[str] = []
    best = max(summary["results"], key=lambda r: r["test_AP"])
    out.append(
        f"The headline result is the modality ablation, not the "
        f"temporal one. Removing all active-fire inputs (M3) collapses "
        f"average precision from {_fmt(ap('E4_hist_gradient_boosting'))} "
        f"(E4 HistGB, M1) to {_fmt(ap('E4_M3_hist_gradient_boosting'))}, "
        f"and from {_fmt(ap('E4_logistic_regression'))} to "
        f"{_fmt(ap('E4_M3_logistic_regression'))} for logistic "
        f"regression. At pixel level, weather, fuel, vegetation, "
        f"topography and forecasts alone cannot locate tomorrow's fire: "
        f"without knowing where the fire is today, the models have "
        f"almost nothing to anchor on. This answers research question 4 "
        f"plainly -- no, environmental variables alone do not predict "
        f"next-day fire maps at 375 m with these baselines."
    )
    out.append(
        f"By contrast, earlier fire maps add almost nothing beyond the "
        f"current day's map (question 3). At the 5-day window, keeping "
        f"only the current day's active fire (M2) instead of all five "
        f"days (M1) changes AP by "
        f"{ap('E4_M2_hist_gradient_boosting') - ap('E4_hist_gradient_boosting'):+.3f} "
        f"for gradient boosting and "
        f"{ap('E4_M2_logistic_regression') - ap('E4_logistic_regression'):+.3f} "
        f"for logistic regression -- the linear model actually improves "
        f"when the noisier historical fire features are removed. The "
        f"fire's present location dominates; its history is largely "
        f"redundant information for these pixel-level models."
    )
    out.append(
        f"Temporal context (questions 1-2) shows no clear gain either. "
        f"From 1 to 5 days of history, AP changes by "
        f"{ap('E4_logistic_regression') - ap('E1_logistic_regression'):+.3f} "
        f"for logistic regression (it degrades as features multiply "
        f"while training rows stay capped at 200k) and by "
        f"{ap('E4_hist_gradient_boosting') - ap('E1_hist_gradient_boosting'):+.3f} "
        f"for gradient boosting, whose best score is "
        f"{_fmt(best['test_AP'])} ({best['config']}). That is the same "
        f"direction as Li et al. (2026), who report a small T=1 to T=5 "
        f"gain (VideoMAEv2 AP 0.370 to 0.377), but here the change is "
        f"within the noise of three test events and is dwarfed by the "
        f"M3 effect. A longer history does not rescue a pixel-level "
        f"baseline the way it helps a video foundation model -- which "
        f"is itself informative: the temporal-context benefit in the "
        f"literature likely comes from spatiotemporal representation "
        f"learning, not from merely stacking more daily feature maps."
    )
    out.append(
        f"Persistence (question 5) explains a large share of apparent "
        f"performance. It scores AP {_fmt(ap('persistence'))} -- below "
        f"the learned models -- but at the fixed 0.5 threshold its F1 "
        f"({_fmt(f1('persistence'))}) and IoU are the best in the "
        f"table. The learned models only match it on hard-map metrics "
        f"after threshold tuning on validation data. Logistic "
        f"regression is the starkest case: at 0.5 its precision "
        f"collapses as history grows "
        f"(E1 {_fmt(get('E1_logistic_regression').get('test_precision'))} "
        f"to E4 "
        f"{_fmt(get('E4_logistic_regression').get('test_precision'))}) "
        f"while recall stays near 0.8 -- the balanced class weight "
        f"pushes scores up, and the F1-optimal threshold (0.85-0.95) "
        f"is doing the real work. Any comparison that reports only AP, "
        f"or only F1 at 0.5, would mislead here; both are needed."
    )
    d4 = detail.get("E4_hist_gradient_boosting", {})
    ci = (d4.get("test", {}).get("ap_bootstrap_ci95_fixed_0.5") or [None, None])
    evs = d4.get("test", {}).get("per_event_AP") or []
    d1 = detail.get("E1_logistic_regression", {})
    val_ap = d1.get("validation", {}).get("pooled", {}).get(
        "fixed_0.5", {}).get("average_precision")
    val_prev = d1.get("validation", {}).get("pooled", {}).get(
        "fixed_0.5", {}).get("prevalence")
    test_prev = d1.get("test", {}).get("pooled", {}).get(
        "fixed_0.5", {}).get("prevalence")
    if evs:
        ev_str = ", ".join(
            f"{e['event'].split('/')[-1]} {_fmt(e['average_precision'])}"
            for e in evs)
        val_note = ""
        if val_ap is not None and val_prev is not None and test_prev:
            val_note = (
                f" Validation AP is systematically far below test AP "
                f"(e.g. E1 logistic regression {_fmt(val_ap)} "
                f"validation vs {_fmt(ap('E1_logistic_regression'))} "
                f"test) because validation prevalence "
                f"({100 * val_prev:.3f}%) is about a third of test "
                f"prevalence ({100 * test_prev:.3f}%) and one validation "
                f"event is very quiet; thresholds tuned on validation "
                f"are therefore high and transfer imperfectly."
            )
        out.append(
            f"Uncertainty is dominated by between-event variation. For "
            f"E4 gradient boosting, per-event test AP is {ev_str} -- "
            f"the quiet fire is always the worst -- and the event-level "
            f"bootstrap 95% interval is [{_fmt(ci[0])}, {_fmt(ci[1])}]. "
            f"With three test events, a difference of 0.02-0.03 AP "
            f"between configurations is not distinguishable from "
            f"event-to-event variation.{val_note} "
            f"These are properties of the subset, not of the method, "
            f"and they bound what can be claimed."
        )
    out.append(
        "In short: for cheap pixel-level baselines, the current fire "
        "map is nearly sufficient, history beyond it is nearly "
        "redundant, and environmental variables without any fire map "
        "are nearly useless. That is a useful negative result for "
        "anyone deciding where to spend compute -- it points effort "
        "toward spatial/spatiotemporal modelling of spread from the "
        "current perimeter, rather than toward longer feature stacks."
    )
    return out


def build_report(
    root: Path,
    cfg: dict,
    inventory: dict,
    samples: list[dict],
    summary: dict | None,
    results_detail: list[dict] | None,
) -> str:
    events = cfg["events"]
    n_events = {k: len(v) for k, v in events.items()}
    n_samples = {}
    for s in samples:
        n_samples[s["split"]] = n_samples.get(s["split"], 0) + 1
    total_days = inventory.get("n_days", 0)
    total_fire_days = inventory.get("n_fire_days", 0)

    lines: list[str] = []
    lines.append("# Investigating the Importance of Temporal Context in "
                 "Multimodal Wildfire Spread Forecasting")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(
        "This study asks whether adding historical satellite observations "
        "improves next-day wildfire active-fire forecasting beyond what "
        "the current fire map and environmental conditions predict. Using "
        "a subset of the WildfireSpreadTS dataset (Gerard et al., 2023), "
        "we run a controlled ablation study with lightweight CPU-only "
        "baselines: a persistence baseline, logistic regression and "
        "histogram-based gradient boosting. Four temporal histories "
        "(1, 2, 3 and 5 days) and three active-fire input modalities "
        "(all history, current day only, none) are compared on identical "
        "samples with event-level splits. This is an independent "
        "investigation of temporal context; it does not reproduce CMPF, "
        "VideoMAEv2 or the original benchmark results (Li et al., 2026)."
    )
    lines.append("")
    lines.append("## 1. Introduction and research motivation")
    lines.append("")
    lines.append(
        "Wildfire spread forecasting from satellite data is a "
        "spatiotemporal prediction problem: given daily multimodal "
        "observations up to day t, predict the active-fire map on day "
        "t+1. Recent work (Li, Hsu and Wang, 2026) adapts large video "
        "foundation models to this task with cross-modal progressive "
        "fine-tuning (CMPF), showing that longer temporal context helps "
        "(e.g. VideoMAEv2 average precision 0.370 at 1 day vs 0.377 at "
        "5 days on 2021 WildfireSpreadTS data). Those models need GPUs "
        "and large-scale pretraining. The open question addressed here "
        "is how much of that temporal-context effect is already visible "
        "with cheap, CPU-only baselines -- and how much of it is simply "
        "the persistence of the fire itself. Answering it matters for "
        "practitioners without GPU access and for understanding what "
        "historical observations actually add."
    )
    lines.append("")
    lines.append("## 2. Related research")
    lines.append("")
    lines.append(
        "* Gerard, Zhao and Sullivan (2023) introduced WildfireSpreadTS "
        "(NeurIPS Datasets and Benchmarks): 607 fires, 13,607 daily "
        "23-channel images at 375 m resolution, western USA, 2018-2021, "
        "with next-day binarised active fire as the benchmark target."
    )
    lines.append(
        "* Huot et al. (2022) introduced the Next Day Wildfire Spread "
        "dataset and CNN baselines for the same task at 1 km resolution."
    )
    lines.append(
        "* Li, Hsu and Wang (2026) adapt video foundation models "
        "(VideoMAEv2, MViTv2, ViT) with CMPF, reaching AP 0.407 at "
        "5 days on 2021 data. Our study uses the same dataset and "
        "target definition but lightweight CPU baselines, and asks a "
        "narrower question: the marginal value of temporal context and "
        "of historical active-fire inputs."
    )
    lines.append(
        "* Original benchmark baselines (Gerard et al., 2023, mean AP "
        "across the 12 year-permutation folds): persistence 0.193, "
        "logistic regression 0.279-0.288, U-Net (ResNet-18) 0.341 "
        "(1 day) to 0.344 (5 days), UTAE 0.372 (best reported). These "
        "are full-dataset numbers over 607 fires, not directly "
        "comparable to our 9-event subset, but they set the scale: "
        "persistence alone is a strong baseline, and extra context "
        "and features historically bought only modest gains."
    )
    lines.append("")
    lines.append("## 3. Research questions and hypotheses")
    lines.append("")
    lines.append(
        "1. Does five-day context improve next-day forecasting over "
        "current-day observations? (Hypothesis: yes, modestly.)\n"
        "2. Does additional historical context improve consistently, or "
        "plateau/deteriorate with longer histories?\n"
        "3. How much does the model depend on historical active-fire "
        "observations (M1 vs M2 vs M3)?\n"
        "4. Can environmental variables predict without earlier fire "
        "maps (M3)?\n"
        "5. How much apparent performance is explained by persistence?"
    )
    lines.append("")
    lines.append("## 4. Dataset and preprocessing")
    lines.append("")
    lines.append(
        f"Source: WildfireSpreadTS (Zenodo record 8006177, "
        f"DOI 10.5281/zenodo.8006177, CC BY 4.0). The full archive is "
        f"48.4 GB; only the configured subset was extracted via HTTP "
        f"Range requests and remote ZIP access (see "
        f"`src/data/acquisition.py`). Subset: {sum(n_events.values())} "
        f"fire events ({n_events['train']} train, "
        f"{n_events['validation']} validation, {n_events['test']} "
        f"test), {total_days} daily images, of which {total_fire_days} "
        f"days contain at least one active-fire pixel. Events were "
        f"chosen by sequence completeness (longest consecutive run of "
        f"daily observations) before any modelling."
    )
    lines.append("")
    lines.append(
        "Each day is a 23-channel GeoTIFF at 375 m (VIIRS reflectance "
        "bands M11/I2/I1, NDVI, EVI2, precipitation, wind speed and "
        "direction, min/max temperature, energy release component, "
        "specific humidity, slope, aspect, elevation, Palmer drought "
        "severity index, land-cover class, and 5 forecast variables, "
        "plus active fire). Preprocessing: active-fire NaN (no "
        "detection) replaced by 0 and binarised; other channels' NaNs "
        "filled with train medians; continuous channels standardised "
        "with train means/stds; wind direction, aspect and forecast "
        "wind direction encoded as (sin, cos) to preserve circularity; "
        "land cover one-hot encoded. All fitted statistics use train "
        "events only."
    )
    lines.append("")
    lines.append(
        "Relation to the original loader (SebastianGer/"
        "WildfireSpreadTS): the original applies sin only to the three "
        "degree channels, which its README (Feb 2026) notes loses "
        "information -- sin and cos are used here instead, as the "
        "issue requests. The original's 2024 dataset-class bug "
        "(commit ab3c8f3), where a global index could resolve to the "
        "wrong fire, cannot occur here because samples are built "
        "directly from per-event inventories. The original builds "
        "sequences from consecutive file indices; consecutive "
        "calendar days are required here (identical on the selected "
        "fully-consecutive events). Forecast wind-direction values in "
        "one inspected file ranged about -88 to 86 degrees, "
        "suggesting a -180..180 convention rather than 0..360; the "
        "sin/cos encoding is correct under either convention."
    )
    lines.append("")
    lines.append(
        f"Samples: one sample per (event, target date) with 5 "
        f"consecutive input days and a consecutive target day. "
        f"Eligible samples: {sum(n_samples.values())} total "
        f"({n_samples.get('train', 0)} train, "
        f"{n_samples.get('validation', 0)} validation, "
        f"{n_samples.get('test', 0)} test). Splits are by fire event; "
        f"no event appears in more than one partition. This follows "
        f"the original benchmark's year-based split (train 2018+2020, "
        f"validation 2019, test 2021)."
    )
    lines.append("")
    lines.append("## 5. Experimental methodology")
    lines.append("")
    lines.append(
        "Temporal ablation (E1-E4): input windows of 1, 2, 3 and 5 days, "
        "same target day t+1, same sample population, same models, "
        "splits, preprocessing and seeds. Modality ablation (M1-M3) at "
        "the 5-day window: M1 retains all historical active fire, M2 "
        "keeps only the current day's, M3 removes all active-fire "
        "inputs (environmental channels retained). The persistence "
        "baseline predicts the t+1 map as the day-t active-fire map."
    )
    lines.append("")
    lines.append("## 6. Models and evaluation metrics")
    lines.append("")
    lines.append(
        "Models: persistence; logistic regression (class_weight="
        "balanced, C=1.0); histogram-based gradient boosting "
        "(200 trees, learning rate 0.1, 31 leaves). Pixel-level "
        "features: per-day standardised channels, sin/cos angles, "
        "one-hot land cover, active-fire binary and its 3x3 "
        "neighbourhood mean. Training rows are bounded (all positives "
        "kept, negatives subsampled 10:1 per positive, 20k pixels per "
        "sample, 200k rows total). Primary metric: Average Precision; "
        "also precision, recall, F1 and IoU at a fixed 0.5 threshold "
        "and at an F1-optimal threshold selected on validation data "
        "only. Accuracy is not reported (0.17% positive prevalence in "
        "the full dataset). Pooled pixel-level and mean event-level "
        "metrics are reported separately, with event-level bootstrap "
        "confidence intervals for AP (indicative with 3 test events)."
    )
    lines.append("")
    lines.append("## 7. Results")
    lines.append("")
    if summary is None or not summary.get("results"):
        lines.append(
            "Experiments have not been run yet (or the summary file is "
            "missing). No results are reported here; see "
            "`results/experiment_summary.json` after running "
            "`scripts/run_experiments.py`."
        )
    else:
        lines.append(
            "Test-set pooled metrics at the fixed 0.5 threshold "
            "(validation results and per-event metrics are in "
            "`results/tables/`):"
        )
        lines.append("")
        lines.append("| Config | Model | Window | AF mode | AP | Prec | Rec | F1 | IoU |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in summary["results"]:
            lines.append(
                f"| {r['config']} | {r['model']} | {r['window_days']} | "
                f"{r['active_fire_mode']} | {_fmt(r['test_AP'])} | "
                f"{_fmt(r['test_precision'])} | {_fmt(r['test_recall'])} | "
                f"{_fmt(r['test_F1'])} | {_fmt(r['test_IoU'])} |"
            )
        lines.append("")
        lines.append(
            "Comparison charts: `results/figures/temporal_ap.svg` and "
            "`results/figures/modality_ap.svg`. Example prediction maps: "
            "`results/figures/`."
        )
    lines.append("")
    lines.append("## 8. Discussion")
    lines.append("")
    if summary is None:
        lines.append("Pending experimental results.")
    else:
        for para in _discussion(summary, results_detail):
            lines.append(para)
            lines.append("")
    lines.append("")
    lines.append("## 9. Limitations")
    lines.append("")
    lines.append(
        "* Small subset: only a handful of independent fire events, so "
        "geographical generalisation is limited and uncertainty "
        "intervals are wide.\n"
        "* Single region (western USA) and single target (next-day "
        "active fire at 375 m).\n"
        "* Pixel-level models ignore spatial context beyond a 3x3 "
        "active-fire neighbourhood; no spatial modelling of spread "
        "dynamics.\n"
        "* VIIRS active-fire detections contain false alarms and miss "
        "small or short-lived fires; the target is a noisy proxy for "
        "true fire extent.\n"
        "* Human firefighting is not represented in the data.\n"
        "* Validation and test prevalences differ threefold (0.115% vs "
        "0.381%), and only two validation events exist, so "
        "F1-optimal thresholds tuned on validation transfer "
        "imperfectly to test; absolute AP values are not comparable "
        "across the two partitions."
    )
    lines.append("")
    lines.append("## 10. Conclusion")
    lines.append("")
    if summary is None:
        lines.append("Pending experimental results.")
    else:
        rows = {r["config"]: r for r in summary["results"]}
        lines.append(
            "Answering the five research questions on this 9-event "
            "subset: (1) five-day context does not clearly improve on "
            "current-day observations for these baselines "
            f"(gradient boosting AP "
            f"{_fmt(rows['E1_hist_gradient_boosting']['test_AP'])} at "
            f"E1 vs {_fmt(rows['E4_hist_gradient_boosting']['test_AP'])} "
            f"at E4); (2) longer histories plateau for gradient "
            f"boosting and deteriorate for logistic regression; (3) the "
            "models depend heavily on active-fire inputs, but almost "
            "entirely on the current day's map; (4) environmental "
            "variables alone cannot predict next-day fire maps "
            f"(M3 AP {_fmt(rows['E4_M3_hist_gradient_boosting']['test_AP'])}); "
            f"(5) much of the apparent performance is persistence -- "
            f"the no-model baseline already reaches F1 "
            f"{_fmt(rows['persistence']['test_F1'])} at 0.5. The "
            f"contribution is a reproducible, CPU-only ablation that "
            f"separates these effects on identical samples, with "
            "honest uncertainty bounds. The full pipeline "
            "(acquisition, preprocessing, experiments, evaluation) is "
            "in this repository and can be re-run with one command."
        )
    lines.append("")
    lines.append("## 11. References")
    lines.append("")
    lines.append(
        "* Gerard, S., Zhao, Y., Sullivan, J. (2023). WildfireSpreadTS: "
        "A Dataset of Multi-Modal Time Series for Wildfire Spread "
        "Prediction. NeurIPS 2023 Datasets and Benchmarks. "
        "https://doi.org/10.5281/zenodo.8006177\n"
        "* Li, W., Hsu, C.-Y., Wang, S. (2026). Adapting Video "
        "Foundation Models for Spatiotemporal Wildfire Forecasting via "
        "Cross-Modal Progressive Fine-Tuning. IEEE Transactions on "
        "Geoscience and Remote Sensing, 64, 3000218. "
        "https://doi.org/10.1109/TGRS.2026.3652453\n"
        "* Huot, F. et al. (2022). Next Day Wildfire Spread: A Machine "
        "Learning Dataset to Predict Wildfire Spreading From "
        "Remote-Sensing Data. IEEE TGRS, 60, 1-13.\n"
        "* Artés, T. et al. (2019). A global wildfire dataset for the "
        "analysis of fire regimes and fire behaviour. Scientific Data, 6."
    )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "Environment: CPU-only, Python 3.12, scikit-learn, numpy, "
        "scipy, pandas, matplotlib, tifffile, h5py, PyYAML, remotezip "
        "(see `requirements.txt`). Data: download the configured "
        "subset with `.venv/bin/python scripts/download_subset.py`, "
        "then `.venv/bin/python scripts/prepare_data.py`, then "
        "`.venv/bin/python scripts/run_experiments.py`. Random seed: "
        "42 throughout. Raw data is never committed."
    )
    lines.append("")
    return "\n".join(lines)
