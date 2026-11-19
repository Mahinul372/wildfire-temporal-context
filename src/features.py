"""Pixel-level feature construction for the temporal/modality ablations.

One sample = one (event, target date) with an input window of 1-5 days
ending on day t. For every input day the same per-day feature block is
built:

  * 18 standardised continuous channels,
  * 6 sin/cos values for the 3 angular channels,
  * K one-hot land-cover features (K = train-observed classes),
  * 2 active-fire features: the binarised fire map and its 3x3
    neighbourhood mean (fire fraction around the pixel).

The blocks are concatenated oldest-day-first, so E1 has one block and
E4 has five. The target day's active fire is never a feature.

Modality ablation (M1-M3) is implemented by ZEROING the active-fire
features of the relevant days, never by dropping columns, so every
configuration has identical feature dimensionality and the models stay
directly comparable:

  * M1 ``all``:     all days' active-fire features retained.
  * M2 ``current``: only the last input day (t) keeps active fire.
  * M3 ``none``:    all active-fire features zeroed.
"""

from __future__ import annotations

import numpy as np

from data.preprocessing import ANGULAR_CHANNELS, CONTINUOUS_CHANNELS

# Channel names for readable feature names (dataset order).
CONTINUOUS_NAMES = [
    "viirs_m11", "viirs_i2", "viirs_i1", "ndvi", "evi2",
    "total_precipitation", "wind_speed", "min_temperature",
    "max_temperature", "energy_release", "specific_humidity", "slope",
    "elevation", "pdsi", "fcst_precipitation", "fcst_wind_speed",
    "fcst_temperature", "fcst_humidity",
]
ANGULAR_NAMES = ["wind_direction", "fcst_wind_direction", "aspect"]

ACTIVE_FIRE_FEATURES = 2  # binary map + 3x3 neighbourhood mean


def per_day_feature_count(n_landcover: int) -> int:
    return (
        len(CONTINUOUS_NAMES)
        + 2 * len(ANGULAR_NAMES)
        + n_landcover
        + ACTIVE_FIRE_FEATURES
    )


def feature_names(window_days: int, landcover_classes: list[int]) -> list[str]:
    """Deterministic feature names, oldest input day first.

    Day labels are relative to the last input day t: ``t-4`` .. ``t``.
    """
    names: list[str] = []
    for i in range(window_days):
        offset = window_days - 1 - i
        day_label = "t" if offset == 0 else f"t-{offset}"
        names += [f"{day_label}_cont_{n}" for n in CONTINUOUS_NAMES]
        for a in ANGULAR_NAMES:
            names += [f"{day_label}_ang_{a}_sin", f"{day_label}_ang_{a}_cos"]
        names += [f"{day_label}_lc_{c}" for c in landcover_classes]
        names += [f"{day_label}_af", f"{day_label}_af_nbr"]
    return names


def neighbourhood_mean(mask: np.ndarray) -> np.ndarray:
    """Mean of a binary map over the 3x3 neighbourhood (zero padding)."""
    padded = np.pad(mask.astype(np.float32), 1, mode="constant")
    total = np.zeros_like(padded[1:-1, 1:-1])
    for di in range(3):
        for dj in range(3):
            total += padded[di : di + mask.shape[0], dj : dj + mask.shape[1]]
    return (total / 9.0).astype(np.float32)


def build_features(
    preprocessed_days: list[dict],
    active_fire_mode: str = "all",
) -> np.ndarray:
    """Stack per-day feature blocks into a (H*W, F) float32 matrix.

    ``preprocessed_days`` are the dicts returned by
    ``preprocessing.preprocess_day`` for the input window, oldest first.
    ``active_fire_mode`` is one of ``all`` / ``current`` / ``none``.
    """
    if active_fire_mode not in ("all", "current", "none"):
        raise ValueError(f"unknown active_fire_mode: {active_fire_mode}")
    h, w, _ = preprocessed_days[0]["continuous"].shape
    n_days = len(preprocessed_days)
    blocks = []
    for i, day in enumerate(preprocessed_days):
        keep_af = (
            active_fire_mode == "all"
            or (active_fire_mode == "current" and i == n_days - 1)
        )
        af = day["active_fire"].astype(np.float32)
        af_nbr = neighbourhood_mean(day["active_fire"])
        if not keep_af:
            af = np.zeros_like(af)
            af_nbr = np.zeros_like(af_nbr)
        block = np.concatenate(
            [
                day["continuous"],
                day["angular"],
                day["landcover"],
                af[:, :, None],
                af_nbr[:, :, None],
            ],
            axis=2,
        )
        blocks.append(block)
    stacked = np.concatenate(blocks, axis=2)
    return stacked.reshape(h * w, stacked.shape[2]).astype(np.float32)


def sample_training_rows(
    y: np.ndarray,
    rng: np.random.Generator,
    max_pixels: int,
    neg_pos_ratio: int,
) -> np.ndarray:
    """Bounded training-row selection for one sample.

    Keeps ALL positive pixels and subsamples negatives to at most
    ``neg_pos_ratio`` per positive, capped at ``max_pixels`` rows total.
    Returns the selected row indices (deterministic given ``rng``).
    """
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if pos.size == 0:
        # No fire in the target: keep a bounded random negative subset.
        k = min(neg.size, max_pixels)
        return rng.choice(neg, size=k, replace=False) if k else neg
    n_neg = min(neg.size, neg_pos_ratio * pos.size)
    chosen_neg = rng.choice(neg, size=n_neg, replace=False) if n_neg else neg[:0]
    rows = np.concatenate([pos, chosen_neg])
    if rows.size > max_pixels:
        # Keep all positives; cap negatives to fit the budget.
        keep_neg = max_pixels - pos.size
        rows = np.concatenate(
            [pos, rng.choice(chosen_neg, size=max(keep_neg, 0), replace=False)]
            if keep_neg > 0
            else [pos]
        )
    rng.shuffle(rows)
    return rows
