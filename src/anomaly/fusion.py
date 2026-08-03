"""Weighted voting fusion for Week 5 anomaly detection."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def per_node_quantile_threshold(
    scores: Array,
    *,
    quantile: float = 0.995,
    min_threshold: float = 1e-6,
) -> Array:
    """Calibrate one threshold per grid cell from validation scores."""

    arr = np.asarray(scores, dtype=np.float32)
    thr = np.quantile(arr, quantile, axis=0).astype(np.float32)
    return np.maximum(thr, min_threshold)


def scalar_quantile_threshold(
    scores: Array,
    *,
    quantile: float = 0.995,
    min_threshold: float = 1e-6,
) -> float:
    """Calibrate one global threshold from validation scores."""

    return float(max(np.quantile(np.asarray(scores, dtype=np.float32), quantile), min_threshold))


def confidence_from_threshold(scores: Array, threshold: Array | float) -> Array:
    """Map raw scores to [0,1] confidence with 0.5 at the threshold.

    This makes heterogeneous methods comparable for weighted voting while still
    preserving intensity above threshold.
    """

    arr = np.asarray(scores, dtype=np.float32)
    thr = np.asarray(threshold, dtype=np.float32)
    return np.clip(arr / (2.0 * thr + 1e-8), 0.0, 1.0).astype(np.float32)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    return {k: float(v) / total for k, v in weights.items()}


def fuse_weighted_vote(
    *,
    stat_conf: Array,
    recon_conf: Array,
    pred_conf: Array,
    weights: dict[str, float] | None = None,
    decision_threshold: float = 0.5,
) -> dict[str, Array]:
    """Fuse the three Week 5 method families using weighted voting."""

    if weights is None:
        weights = {"statistical": 0.35, "reconstruction": 0.30, "prediction": 0.35}
    w = normalize_weights(weights)
    fused = (
        w.get("statistical", 0.0) * np.asarray(stat_conf, dtype=np.float32)
        + w.get("reconstruction", 0.0) * np.asarray(recon_conf, dtype=np.float32)
        + w.get("prediction", 0.0) * np.asarray(pred_conf, dtype=np.float32)
    ).astype(np.float32)
    return {
        "score": fused,
        "mask": fused >= float(decision_threshold),
    }
