"""Preprocessing for WildfireSpreadTS daily GeoTIFFs.

Verified facts (documentation + inspection of a real file,
``2018/fire_21890058/2018-07-24.tif``):
  * Each file is one day: shape (H, W, 23), float32, 375 m pixels,
    UTM projection. H and W vary per fire event.
  * Channel order is the dataset's own order (see config/experiment.yaml).
  * Active fire (channel 22) is NaN wherever no fire was detected and a
    value (1.0) where fire was detected. The dataset documentation
    prescribes replacing missing active-fire values with zeros.
  * Other channels have occasional NaNs (clouds / sensor gaps).
  * Wind direction, aspect and forecast wind direction are angular
    (degrees). They are encoded as (sin, cos) so the encoding is
    invariant to the 0-360 vs -180-180 convention and preserves
    circularity. This deliberately avoids reproducing any
    degrees-as-linear preprocessing mistake.
  * Land cover (channel 16) is categorical (MODIS IGBP classes) and is
    one-hot encoded, never treated as a continuous number.

All fitted statistics (per-channel medians for NaN filling, means/stds
for standardisation, observed land-cover classes) are computed on
TRAIN events only and stored as JSON, so no validation/test
information leaks into preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

# Channel indices (0-based) in the dataset's own order.
N_CHANNELS = 23
TARGET_CHANNEL = 22
ANGULAR_CHANNELS = (7, 19, 13)  # wind dir, fcst wind dir, aspect
LANDCOVER_CHANNEL = 16
# Continuous channels that are standardised (all except angular,
# landcover and the target).
CONTINUOUS_CHANNELS = (
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 17, 18, 20, 21,
)


def load_day(path: str | Path) -> np.ndarray:
    """Load one daily GeoTIFF as a float32 (H, W, 23) array."""
    arr = tifffile.imread(str(path))
    if arr.ndim != 3 or arr.shape[2] != N_CHANNELS:
        raise ValueError(f"{path}: expected (H, W, 23), got {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def active_fire_mask(day: np.ndarray) -> np.ndarray:
    """Binarised active-fire map for one day.

    NaN (no detection) -> 0; any positive value -> 1. Pixels with
    non-finite, non-NaN values are treated as 0.
    """
    af = day[:, :, TARGET_CHANNEL]
    with np.errstate(invalid="ignore"):
        mask = np.where(np.isfinite(af) & (af > 0), 1, 0).astype(np.uint8)
    return mask


def fit_preprocess_stats(
    day_paths: list[Path],
    seed: int = 42,
    max_pixels_per_day: int = 50_000,
) -> dict:
    """Fit NaN-fill medians and standardisation stats on TRAIN days only.

    Returns a JSON-serialisable dict with per-channel median/mean/std
    (continuous channels) and the sorted observed land-cover classes.
    """
    rng = np.random.default_rng(seed)
    # Medians are needed for continuous channels AND angular channels
    # (NaN angles are filled with the train median angle before sin/cos).
    median_channels = tuple(sorted(set(CONTINUOUS_CHANNELS) | set(ANGULAR_CHANNELS)))
    medians = {c: [] for c in median_channels}
    sums = {c: 0.0 for c in CONTINUOUS_CHANNELS}
    sqsums = {c: 0.0 for c in CONTINUOUS_CHANNELS}
    counts = {c: 0 for c in CONTINUOUS_CHANNELS}
    landcover_classes: set[int] = set()
    for path in day_paths:
        day = load_day(path)
        h, w, _ = day.shape
        n = h * w
        idx = np.arange(n)
        if n > max_pixels_per_day:
            idx = rng.choice(idx, size=max_pixels_per_day, replace=False)
        flat = day.reshape(n, N_CHANNELS)[idx]
        lc = flat[:, LANDCOVER_CHANNEL]
        lc = lc[np.isfinite(lc)]
        landcover_classes.update(int(v) for v in np.unique(lc))
        for c in median_channels:
            v = flat[:, c]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            medians[c].append(float(np.median(v)))
            if c in CONTINUOUS_CHANNELS:
                sums[c] += float(v.sum())
                sqsums[c] += float((v.astype(np.float64) ** 2).sum())
                counts[c] += int(v.size)
    stats: dict = {"medians": {}, "means": {}, "stds": {}}
    for c in median_channels:
        med = float(np.median(medians[c])) if medians[c] else 0.0
        stats["medians"][str(c)] = med
    for c in CONTINUOUS_CHANNELS:
        if counts[c] > 0:
            mean = sums[c] / counts[c]
            var = max(sqsums[c] / counts[c] - mean * mean, 0.0)
            std = float(np.sqrt(var)) or 1.0
        else:
            mean, std = 0.0, 1.0
        stats["means"][str(c)] = float(mean)
        stats["stds"][str(c)] = float(std)
    stats["landcover_classes"] = sorted(landcover_classes)
    return stats


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def preprocess_day(day: np.ndarray, stats: dict) -> dict[str, np.ndarray]:
    """Transform one raw day into model-ready per-channel arrays.

    Returns a dict with:
      * ``continuous``: (H, W, 18) standardised continuous channels,
        NaNs filled with the train median.
      * ``angular``: (H, W, 6) sin/cos pairs for the 3 angular channels,
        NaNs filled with the train median angle first.
      * ``landcover``: (H, W, K) one-hot over the train-observed classes.
      * ``active_fire``: (H, W) binarised active-fire map (0/1).
    The target day's active fire is NEVER included in features; the
    caller is responsible for only passing input days here.
    """
    h, w, _ = day.shape
    medians = stats["medians"]
    means = stats["means"]
    stds = stats["stds"]
    classes = stats["landcover_classes"]

    cont = np.empty((h, w, len(CONTINUOUS_CHANNELS)), dtype=np.float32)
    for j, c in enumerate(CONTINUOUS_CHANNELS):
        v = day[:, :, c].astype(np.float64)
        med = float(medians[str(c)])
        v = np.where(np.isfinite(v), v, med)
        cont[:, :, j] = ((v - float(means[str(c)])) / float(stds[str(c)])).astype(
            np.float32
        )

    ang = np.empty((h, w, 2 * len(ANGULAR_CHANNELS)), dtype=np.float32)
    for j, c in enumerate(ANGULAR_CHANNELS):
        v = day[:, :, c].astype(np.float64)
        med = float(medians.get(str(c), 0.0))
        v = np.where(np.isfinite(v), v, med)
        rad = np.deg2rad(v)
        ang[:, :, 2 * j] = np.sin(rad).astype(np.float32)
        ang[:, :, 2 * j + 1] = np.cos(rad).astype(np.float32)

    lc = day[:, :, LANDCOVER_CHANNEL]
    onehot = np.zeros((h, w, len(classes)), dtype=np.float32)
    for j, cls in enumerate(classes):
        onehot[:, :, j] = (lc == cls).astype(np.float32)

    return {
        "continuous": cont,
        "angular": ang,
        "landcover": onehot,
        "active_fire": active_fire_mask(day),
    }
