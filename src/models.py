"""CPU-only models: persistence, logistic regression, HistGradientBoosting.

All models expose ``fit`` / ``predict_proba`` semantics through the
helpers below. Persistence needs no fitting: the next-day prediction
is the current-day active-fire map.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def fit_model(kind: str, X: np.ndarray, y: np.ndarray, cfg: dict, seed: int):
    """Fit a model of the given kind. ``cfg`` is the model's config dict."""
    if kind == "logistic_regression":
        # Note: sklearn >= 1.8 removed the effect of n_jobs for
        # LogisticRegression; it is intentionally not passed.
        model = LogisticRegression(
            C=cfg.get("C", 1.0),
            class_weight=cfg.get("class_weight", "balanced"),
            max_iter=cfg.get("max_iter", 1000),
            random_state=seed,
        )
    elif kind == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=cfg.get("max_iter", 200),
            learning_rate=cfg.get("learning_rate", 0.1),
            max_leaf_nodes=cfg.get("max_leaf_nodes", 31),
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")
    model.fit(X, y)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Positive-class probability for every row of X."""
    return model.predict_proba(X)[:, 1].astype(np.float64)


def persistence_proba(active_fire_today: np.ndarray) -> np.ndarray:
    """Persistence baseline: P(fire tomorrow) = fire today (0/1)."""
    return active_fire_today.astype(np.float64).ravel()
