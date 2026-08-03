"""Statistical anomaly detection for TaxiBJ/TaxiNYC flow grids.

The detectors here implement the Week 5 statistical baseline:

- per-grid, per-time-slot 3-sigma rule;
- per-grid, per-time-slot IQR rule;
- connected-component clustering for spatially coherent alerts;
- weak labels derived from high-flow statistical anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components


Array = np.ndarray


@dataclass(frozen=True)
class SlotStats:
    """Per-slot normal range statistics.

    All arrays use shape ``(steps_per_day, N)`` where ``N = H * W``.
    """

    steps_per_day: int
    mean: Array
    std: Array
    q1: Array
    q3: Array
    iqr: Array
    lower: Array
    upper: Array
    min_std: float
    iqr_factor: float


@dataclass(frozen=True)
class StatisticalResult:
    """Node-level statistical anomaly scores and masks."""

    z: Array
    sigma_score: Array
    iqr_score: Array
    score: Array
    sigma_mask: Array
    iqr_mask: Array
    mask: Array
    high_mask: Array
    low_mask: Array


def flatten_grid(flow: Array) -> Array:
    """Return flow as ``(T, N)`` float32."""

    arr = np.asarray(flow)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Expected flow with shape (T,H,W) or (T,N), got {arr.shape}")


def slot_indices_from_timestamps(
    timestamps: Array,
    *,
    steps_per_day: int = 48,
    tz_offset: int = 0,
) -> Array:
    """Compute local time-slot indices from UTC timestamps.

    TaxiBJ uses 30-minute steps, i.e. ``steps_per_day=48``. The function also
    works for other regular daily resolutions as long as each timestamp lands
    exactly on a slot boundary.
    """

    ts = pd.DatetimeIndex(timestamps.astype("datetime64[s]"))
    local = ts + pd.Timedelta(hours=tz_offset)
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    slot_minutes = 24 * 60 / steps_per_day
    slot = np.rint(minutes / slot_minutes).astype(np.int64) % steps_per_day
    return slot


def fit_slot_stats(
    flow: Array,
    timestamps: Array | None = None,
    *,
    steps_per_day: int = 48,
    tz_offset: int = 0,
    min_std: float = 1.0,
    iqr_factor: float = 1.5,
) -> SlotStats:
    """Fit per-grid, per-time-slot statistics on training flow only."""

    flat = flatten_grid(flow)
    if timestamps is None:
        slots = np.arange(len(flat), dtype=np.int64) % steps_per_day
    else:
        slots = slot_indices_from_timestamps(
            timestamps, steps_per_day=steps_per_day, tz_offset=tz_offset
        )
        if len(slots) != len(flat):
            raise ValueError("timestamps and flow must have the same length")

    n = flat.shape[1]
    mean = np.zeros((steps_per_day, n), dtype=np.float32)
    std = np.zeros_like(mean)
    q1 = np.zeros_like(mean)
    q3 = np.zeros_like(mean)

    for slot in range(steps_per_day):
        chunk = flat[slots == slot]
        if len(chunk) == 0:
            raise ValueError(f"No training samples found for slot {slot}")
        mean[slot] = chunk.mean(axis=0)
        std[slot] = np.maximum(chunk.std(axis=0), min_std)
        q1[slot] = np.quantile(chunk, 0.25, axis=0)
        q3[slot] = np.quantile(chunk, 0.75, axis=0)

    iqr = np.maximum(q3 - q1, min_std)
    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr
    return SlotStats(
        steps_per_day=steps_per_day,
        mean=mean,
        std=std,
        q1=q1,
        q3=q3,
        iqr=iqr,
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        min_std=float(min_std),
        iqr_factor=float(iqr_factor),
    )


def score_statistical(
    flow: Array,
    slots: Array,
    stats: SlotStats,
    *,
    sigma_z: float = 3.0,
) -> StatisticalResult:
    """Score flow observations using 3-sigma and IQR rules.

    ``score`` is normalized so that ``score >= 1`` means at least one of the
    statistical rules fired for that grid cell.
    """

    flat = flatten_grid(flow)
    slots = np.asarray(slots, dtype=np.int64)
    if len(flat) != len(slots):
        raise ValueError("flow and slots must have the same length")

    mu = stats.mean[slots]
    sig = stats.std[slots]
    z = (flat - mu) / sig
    sigma_score = np.abs(z) / max(float(sigma_z), 1e-6)

    lower = stats.lower[slots]
    upper = stats.upper[slots]
    iqr = stats.iqr[slots]
    high_iqr = np.maximum((flat - upper) / iqr, 0.0)
    low_iqr = np.maximum((lower - flat) / iqr, 0.0)
    iqr_score = np.maximum(high_iqr, low_iqr) / max(stats.iqr_factor, 1e-6)

    score = np.maximum(sigma_score, iqr_score).astype(np.float32)
    sigma_mask = sigma_score >= 1.0
    iqr_mask = iqr_score >= 1.0
    high_mask = (z >= sigma_z) | (flat > upper)
    low_mask = (z <= -sigma_z) | (flat < lower)

    return StatisticalResult(
        z=z.astype(np.float32),
        sigma_score=sigma_score.astype(np.float32),
        iqr_score=iqr_score.astype(np.float32),
        score=score,
        sigma_mask=sigma_mask,
        iqr_mask=iqr_mask,
        mask=sigma_mask | iqr_mask,
        high_mask=high_mask,
        low_mask=low_mask,
    )


def generate_weak_labels(
    inflow: Array,
    slots: Array,
    stats: SlotStats,
    *,
    outflow: Array | None = None,
    outflow_stats: SlotStats | None = None,
    level: str = "A",
    label_mode: str = "z_only",
    direction: str = "high",
    z_thresh: float = 3.0,
    min_high_cells: int = 12,
    outflow_z_thresh: float = 2.5,
    min_outflow_cells: int = 6,
) -> Array:
    """Generate Level-A/Level-B weak labels from the Week 5 spec.

    Labels are timestep-level binary flags. By default this follows the
    original Week 5 weak-label rule: Level-A uses high inflow ``z`` scores only.
    Set ``direction`` to ``"low"`` or ``"bidirectional"`` for evaluation-only
    variants that measure low-flow or two-sided anomaly detection.

    Set ``label_mode='z_or_iqr'`` only for exploratory labels that also treat
    IQR-bound violations as positives.

    Level-B additionally requires synchronized high outflow anomalies. Pass
    ``outflow_stats`` fitted on training outflow to avoid using inflow
    dispersion as a proxy for outflow.
    """

    if direction not in {"high", "low", "bidirectional"}:
        raise ValueError("direction must be 'high', 'low', or 'bidirectional'")

    def _direction_mask(result: StatisticalResult, threshold: float) -> Array:
        if label_mode == "z_only":
            if direction == "high":
                return result.z > threshold
            if direction == "low":
                return result.z < -threshold
            return np.abs(result.z) > threshold
        if label_mode == "z_or_iqr":
            if direction == "high":
                return result.high_mask
            if direction == "low":
                return result.low_mask
            return result.high_mask | result.low_mask
        raise ValueError("label_mode must be 'z_only' or 'z_or_iqr'")

    in_result = score_statistical(inflow, slots, stats, sigma_z=z_thresh)
    if label_mode in {"z_only", "z_or_iqr"}:
        high_mask = _direction_mask(in_result, z_thresh)
    else:
        raise ValueError("label_mode must be 'z_only' or 'z_or_iqr'")
    high_count = high_mask.sum(axis=1)
    labels = high_count >= min_high_cells

    if level.upper() == "B":
        if outflow is None:
            return labels.astype(np.int8)
        stats_out = outflow_stats or stats
        out_result = score_statistical(outflow, slots, stats_out, sigma_z=outflow_z_thresh)
        out_high_mask = _direction_mask(out_result, outflow_z_thresh)
        out_count = out_high_mask.sum(axis=1)
        labels = labels & (out_count >= min_outflow_cells)
    elif level.upper() != "A":
        raise ValueError("level must be 'A' or 'B'")

    return labels.astype(np.int8)


def extract_clusters(
    mask_flat: Array,
    score_flat: Array | None = None,
    *,
    grid_shape: tuple[int, int] = (32, 32),
    min_cells: int = 2,
) -> list[dict[str, float | int]]:
    """Extract 8-neighborhood connected components from one timestep mask."""

    mask = np.asarray(mask_flat, dtype=bool).reshape(grid_shape)
    scores = None
    if score_flat is not None:
        scores = np.asarray(score_flat, dtype=np.float32).reshape(grid_shape)

    labeled, n_comp = connected_components(mask, structure=np.ones((3, 3), dtype=int))
    clusters: list[dict[str, float | int]] = []
    for comp_id in range(1, n_comp + 1):
        comp = labeled == comp_id
        cell_count = int(comp.sum())
        if cell_count < min_cells:
            continue
        rows, cols = np.where(comp)
        height = int(rows.max() - rows.min() + 1)
        width = int(cols.max() - cols.min() + 1)
        aspect_ratio = float(max(width / max(height, 1), height / max(width, 1)))
        item: dict[str, float | int] = {
            "cell_count": cell_count,
            "centroid_row": float(rows.mean()),
            "centroid_col": float(cols.mean()),
            "bbox_row_min": int(rows.min()),
            "bbox_row_max": int(rows.max()),
            "bbox_col_min": int(cols.min()),
            "bbox_col_max": int(cols.max()),
            "bbox_height": height,
            "bbox_width": width,
            "aspect_ratio": aspect_ratio,
        }
        if scores is not None:
            item["max_score"] = float(scores[comp].max())
            item["mean_score"] = float(scores[comp].mean())
        clusters.append(item)
    clusters.sort(key=lambda c: int(c["cell_count"]), reverse=True)
    return clusters


def mask_iou(mask_a: Array, mask_b: Array) -> float:
    """Intersection-over-union between two flattened spatial masks."""

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def mask_centroid(mask_flat: Array, *, grid_shape: tuple[int, int]) -> tuple[float, float] | None:
    """Centroid of active cells in ``(row, col)`` grid coordinates."""

    mask = np.asarray(mask_flat, dtype=bool).reshape(grid_shape)
    if not mask.any():
        return None
    rows, cols = np.where(mask)
    return float(rows.mean()), float(cols.mean())


def centroid_distance(
    mask_a: Array,
    mask_b: Array,
    *,
    grid_shape: tuple[int, int] = (32, 32),
) -> float:
    """Euclidean distance between mask centroids in grid-cell units."""

    ca = mask_centroid(mask_a, grid_shape=grid_shape)
    cb = mask_centroid(mask_b, grid_shape=grid_shape)
    if ca is None or cb is None:
        return float("inf")
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))


def spatially_continuous(
    prev_mask: Array,
    cur_mask: Array,
    *,
    grid_shape: tuple[int, int] = (32, 32),
    iou_threshold: float | None = None,
    centroid_distance_threshold: float | None = None,
) -> tuple[bool, float, float]:
    """Return whether two consecutive alert masks are spatially related."""

    if iou_threshold is None and centroid_distance_threshold is None:
        return True, mask_iou(prev_mask, cur_mask), centroid_distance(
            prev_mask, cur_mask, grid_shape=grid_shape
        )
    iou = mask_iou(prev_mask, cur_mask)
    distance = centroid_distance(prev_mask, cur_mask, grid_shape=grid_shape)
    ok_iou = iou_threshold is not None and iou >= float(iou_threshold)
    ok_distance = (
        centroid_distance_threshold is not None
        and distance <= float(centroid_distance_threshold)
    )
    return bool(ok_iou or ok_distance), iou, distance


def classify_event(
    union_mask: Array,
    *,
    duration_steps: int,
    grid_shape: tuple[int, int] = (32, 32),
    road_density_mean: float | None = None,
    in_out_imbalance: float | None = None,
    trend_slope: float | None = None,
) -> str:
    """Rule-based anomaly event classification.

    The labels follow the project PDF:
    ``mass_gathering``, ``traffic_jam``, and ``sudden_incident``.
    """

    clusters = extract_clusters(union_mask, union_mask.astype(np.float32), grid_shape=grid_shape)
    if not clusters:
        return "sudden_incident"

    largest = clusters[0]
    main_cells = int(largest["cell_count"])
    aspect_ratio = float(largest["aspect_ratio"])
    cluster_count = len(clusters)
    road_dense = road_density_mean is not None and float(road_density_mean) >= 0.0
    balanced_flow = in_out_imbalance is None or float(in_out_imbalance) <= 0.35
    slow_trend = trend_slope is None or abs(float(trend_slope)) <= 80.0

    if (
        aspect_ratio >= 2.5
        and main_cells >= 6
        and duration_steps >= 2
        and (road_dense or slow_trend)
    ):
        return "traffic_jam"
    if main_cells >= 15 and duration_steps >= 2 and balanced_flow:
        return "mass_gathering"
    if cluster_count >= 3 or duration_steps <= 2:
        return "sudden_incident"
    return "mass_gathering" if main_cells >= 12 else "sudden_incident"


def severity_level(max_score: float, max_cells: int, duration_steps: int) -> int:
    """Map fused alert intensity to L1-L4 severity."""

    if max_cells >= 25 and max_score >= 0.9 and duration_steps >= 2:
        return 4
    if max_cells >= 12 and max_score >= 0.75:
        return 3
    if max_cells >= 12 or duration_steps >= 4:
        return 2
    return 1


def build_events(
    timestamps: Array,
    masks: Array,
    scores: Array,
    *,
    grid_shape: tuple[int, int] = (32, 32),
    min_cells: int = 6,
    spatial_iou_threshold: float | None = None,
    centroid_distance_threshold: float | None = None,
    inflow: Array | None = None,
    outflow: Array | None = None,
    z: Array | None = None,
    statistical_score: Array | None = None,
    reconstruction_score: Array | None = None,
    reconstruction_error: Array | None = None,
    prediction_score: Array | None = None,
    prediction_residual: Array | None = None,
    road_density: Array | None = None,
) -> list[dict[str, object]]:
    """Group anomalous timesteps into classified events.

    By default events are contiguous in time. Passing a spatial IoU threshold,
    a centroid-distance threshold, or both additionally prevents unrelated
    locations in adjacent timesteps from being merged into one event.
    Optional score/value arrays are used only for event-level explainability.
    """

    timestamps = np.asarray(timestamps)
    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)
    active = masks.sum(axis=1) >= min_cells

    def _optional_matrix(arr: Array | None) -> Array | None:
        if arr is None:
            return None
        matrix = np.asarray(arr, dtype=np.float32)
        if matrix.shape != masks.shape:
            raise ValueError(f"Optional event array has shape {matrix.shape}, expected {masks.shape}")
        return matrix

    inflow_m = _optional_matrix(inflow)
    outflow_m = _optional_matrix(outflow)
    z_m = _optional_matrix(z)
    stat_m = _optional_matrix(statistical_score)
    recon_m = _optional_matrix(reconstruction_score)
    recon_error_m = _optional_matrix(reconstruction_error)
    pred_m = _optional_matrix(prediction_score)
    pred_resid_m = _optional_matrix(prediction_residual)
    road = None if road_density is None else np.asarray(road_density, dtype=np.float32).reshape(-1)
    if road is not None and road.shape[0] != masks.shape[1]:
        raise ValueError(f"road_density has length {road.shape[0]}, expected {masks.shape[1]}")

    def _active_values(matrix: Array | None, idx: Array) -> Array:
        if matrix is None:
            return np.asarray([], dtype=np.float32)
        return matrix[idx][masks[idx]]

    def _nanmax(matrix: Array | None, idx: Array) -> float:
        values = _active_values(matrix, idx)
        return float(np.nanmax(values)) if values.size else np.nan

    def _nanmean(matrix: Array | None, idx: Array) -> float:
        values = _active_values(matrix, idx)
        return float(np.nanmean(values)) if values.size else np.nan

    def _method_maxes(idx: Array) -> dict[str, float]:
        return {
            "statistical": _nanmax(stat_m, idx),
            "reconstruction": _nanmax(recon_m, idx),
            "prediction": _nanmax(pred_m, idx),
        }

    def _dominant_method(idx: Array) -> str:
        method_maxes = _method_maxes(idx)
        finite = {
            key: value for key, value in method_maxes.items()
            if np.isfinite(value)
        }
        if not finite:
            return "unknown"
        return max(finite.items(), key=lambda item: item[1])[0]

    def _top_cells(idx: Array, union_mask: Array, *, k: int = 5) -> str:
        node_score = scores[idx].max(axis=0)
        candidates = np.flatnonzero(union_mask)
        if candidates.size == 0:
            return ""
        ranked = candidates[np.argsort(node_score[candidates])[::-1][:k]]
        pieces: list[str] = []
        h, w = grid_shape
        for node in ranked:
            row = int(node // w)
            col = int(node % w)
            parts = [
                f"r{row}",
                f"c{col}",
                f"score={node_score[node]:.3f}",
            ]
            if inflow_m is not None:
                parts.append(f"flow={float(inflow_m[idx, node].max()):.1f}")
            if z_m is not None:
                z_abs = np.abs(z_m[idx, node])
                parts.append(f"z={float(z_abs.max()):.2f}")
            pieces.append("|".join(parts))
        return "; ".join(pieces)

    def _trend_slope(idx: Array) -> float:
        if inflow_m is None or len(idx) < 2:
            return np.nan
        means = []
        for j in idx:
            current_mask = masks[j]
            if current_mask.any():
                means.append(float(inflow_m[j, current_mask].mean()))
        if len(means) < 2:
            return np.nan
        x = np.arange(len(means), dtype=np.float32)
        return float(np.polyfit(x, np.asarray(means, dtype=np.float32), deg=1)[0])

    def _finalize_event(
        event_id: int,
        start: int,
        end: int,
        link_ious: list[float],
        link_distances: list[float],
    ) -> dict[str, object]:
        idx = np.arange(start, end + 1)
        union_mask = masks[idx].any(axis=0)
        total_cells = int(union_mask.sum())
        max_cells = int(masks[idx].sum(axis=1).max())
        max_score = float(scores[idx].max())
        mean_inflow = _nanmean(inflow_m, idx)
        mean_outflow = _nanmean(outflow_m, idx)
        flow_sum = max(abs(mean_inflow) + abs(mean_outflow), 1e-6)
        in_out_imbalance = (
            float(abs(mean_inflow - mean_outflow) / flow_sum)
            if np.isfinite(mean_inflow) and np.isfinite(mean_outflow)
            else np.nan
        )
        trend_slope = _trend_slope(idx)
        road_density_mean = (
            float(np.nanmean(road[union_mask]))
            if road is not None and union_mask.any()
            else np.nan
        )
        event_type = classify_event(
            union_mask,
            duration_steps=len(idx),
            grid_shape=grid_shape,
            road_density_mean=road_density_mean if np.isfinite(road_density_mean) else None,
            in_out_imbalance=in_out_imbalance if np.isfinite(in_out_imbalance) else None,
            trend_slope=trend_slope if np.isfinite(trend_slope) else None,
        )
        clusters = extract_clusters(
            union_mask, scores[idx].max(axis=0), grid_shape=grid_shape, min_cells=2
        )
        z_values = _active_values(z_m, idx)
        method_maxes = _method_maxes(idx)
        return {
            "event_id": event_id,
            "start_index": int(start),
            "end_index": int(end),
            "start_time": str(timestamps[start]),
            "end_time": str(timestamps[end]),
            "duration_steps": int(len(idx)),
            "duration_hours": float(len(idx) * 0.5),
            "event_type": event_type,
            "severity": severity_level(max_score, max_cells, len(idx)),
            "max_score": max_score,
            "max_cells": max_cells,
            "union_cells": total_cells,
            "cluster_count": int(len(clusters)),
            "main_centroid_row": float(clusters[0]["centroid_row"]) if clusters else np.nan,
            "main_centroid_col": float(clusters[0]["centroid_col"]) if clusters else np.nan,
            "main_aspect_ratio": float(clusters[0]["aspect_ratio"]) if clusters else np.nan,
            "road_density_mean": road_density_mean,
            "in_out_imbalance": in_out_imbalance,
            "duration_trend_slope": trend_slope,
            "mean_inflow": mean_inflow,
            "max_inflow": _nanmax(inflow_m, idx),
            "mean_outflow": mean_outflow,
            "max_outflow": _nanmax(outflow_m, idx),
            "max_z": float(np.nanmax(np.abs(z_values))) if z_values.size else np.nan,
            "max_signed_z": float(np.nanmax(z_values)) if z_values.size else np.nan,
            "statistical_score": method_maxes["statistical"],
            "reconstruction_score": method_maxes["reconstruction"],
            "prediction_score": method_maxes["prediction"],
            "statistical_score_mean": _nanmean(stat_m, idx),
            "reconstruction_score_mean": _nanmean(recon_m, idx),
            "prediction_score_mean": _nanmean(pred_m, idx),
            "dominant_method": _dominant_method(idx),
            "max_reconstruction_error": _nanmax(recon_error_m, idx),
            "max_prediction_residual": _nanmax(pred_resid_m, idx),
            "top_cells": _top_cells(idx, union_mask),
            "mean_spatial_iou_to_prev": (
                float(np.mean(link_ious)) if link_ious else np.nan
            ),
            "mean_centroid_distance_to_prev": (
                float(np.mean(link_distances)) if link_distances else np.nan
            ),
        }

    events: list[dict[str, object]] = []
    event_id = 0
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start = i
        link_ious: list[float] = []
        link_distances: list[float] = []
        while i + 1 < len(active) and active[i + 1]:
            linked, iou, distance = spatially_continuous(
                masks[i],
                masks[i + 1],
                grid_shape=grid_shape,
                iou_threshold=spatial_iou_threshold,
                centroid_distance_threshold=centroid_distance_threshold,
            )
            if not linked:
                break
            link_ious.append(iou)
            link_distances.append(distance)
            i += 1
        end = i
        event_id += 1
        events.append(_finalize_event(event_id, start, end, link_ious, link_distances))
        i += 1
    return events


def time_flags_from_node_mask(mask: Array, *, min_cells: int = 6) -> Array:
    """Convert node-level anomaly masks to timestep-level alert flags."""

    return (np.asarray(mask, dtype=bool).sum(axis=1) >= min_cells).astype(np.int8)


def binary_metrics(pred: Array, target: Array) -> dict[str, float | int]:
    """Precision/recall/F1 for binary timestep labels."""

    p = np.asarray(pred, dtype=bool)
    y = np.asarray(target, dtype=bool)
    tp = int(np.logical_and(p, y).sum())
    fp = int(np.logical_and(p, ~y).sum())
    fn = int(np.logical_and(~p, y).sum())
    tn = int(np.logical_and(~p, ~y).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "positive_rate": float(p.mean()),
    }


def summarize_event_types(events: Iterable[dict[str, object]]) -> dict[str, int]:
    """Count events by project-level type label."""

    counts = {"mass_gathering": 0, "traffic_jam": 0, "sudden_incident": 0}
    for event in events:
        key = str(event.get("event_type", "sudden_incident"))
        counts[key] = counts.get(key, 0) + 1
    return counts
