"""Visualisations from genuine data only (matplotlib, Agg backend).

Charts are saved as PNG (local) and SVG (for the repository README --
the GitHub connector is text-only, so PNGs cannot be pushed).
Maps are saved as PNG only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path, svg: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def comparison_bars(
    rows: list[dict],
    value_key: str,
    title: str,
    path: Path,
    ylabel: str = "Average Precision (test, pooled)",
) -> None:
    """Bar chart of one metric across configurations."""
    labels = [r["config"] for r in rows]
    vals = [r[value_key] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, vals, color="#2c7fb8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(0.5, max([v for v in vals if np.isfinite(v)] + [0]) * 1.15))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}" if np.isfinite(v) else "n/a",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, path, svg=True)


def fire_map_panel(
    input_fire_t: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
    title: str,
    path: Path,
) -> None:
    """One event-day: current fire, next-day truth, prediction, FP, FN."""
    pred = (predicted >= threshold).astype(np.uint8)
    fp = ((pred == 1) & (target == 0)).astype(np.uint8)
    fn = ((pred == 0) & (target == 1)).astype(np.uint8)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    panels = [
        ("Fire day t", input_fire_t),
        ("Truth day t+1", target),
        ("Predicted t+1", pred),
        ("False positives", fp),
        ("False negatives", fn),
    ]
    for ax, (label, img) in zip(axes, panels):
        ax.imshow(img, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path, svg=False)


def input_sequence_strip(
    fire_maps: list[np.ndarray], dates: list[str], title: str, path: Path
) -> None:
    """Active-fire maps for the input window + target, one row."""
    n = len(fire_maps)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, img, d in zip(axes, fire_maps, dates):
        ax.imshow(img, cmap="Reds", vmin=0, vmax=1)
        ax.set_title(d, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path, svg=False)
