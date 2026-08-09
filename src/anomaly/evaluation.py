"""Evaluation helpers for Week 5 anomaly detection.

This module separates two evaluation concerns that are easy to conflate:

1. directional timestep metrics: high-flow, low-flow, and bidirectional labels;
2. event-level metrics: interval hits, detection delay, and duration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.anomaly.statistical import (
    Array,
    binary_metrics,
    centroid_distance,
    mask_iou,
    spatially_continuous,
    time_flags_from_node_mask,
)


@dataclass(frozen=True)
class TimeEvent:
    """A contiguous anomalous timestep interval."""

    event_id: int
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    duration_steps: int
    duration_hours: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "event_id": self.event_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_steps": self.duration_steps,
            "duration_hours": self.duration_hours,
        }


@dataclass(frozen=True)
class NodeEvent:
    """A contiguous event interval with its union spatial footprint."""

    event_id: int
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    duration_steps: int
    duration_hours: float
    union_mask: Array


def directional_node_mask(mask: Array, z: Array, direction: str) -> Array:
    """Split a bidirectional node mask by observed historical-z direction."""

    base = np.asarray(mask, dtype=bool)
    z = np.asarray(z, dtype=np.float32)
    if direction == "high":
        return base & (z > 0)
    if direction == "low":
        return base & (z < 0)
    if direction == "bidirectional":
        return base
    raise ValueError("direction must be 'high', 'low', or 'bidirectional'")


def events_from_time_flags(
    timestamps: Array,
    flags: Array,
) -> list[TimeEvent]:
    """Convert a binary timestep series into contiguous events."""

    timestamps = np.asarray(timestamps)
    active = np.asarray(flags, dtype=bool)
    events: list[TimeEvent] = []
    event_id = 0
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(active) and active[i + 1]:
            i += 1
        end = i
        event_id += 1
        duration = end - start + 1
        events.append(TimeEvent(
            event_id=event_id,
            start_index=int(start),
            end_index=int(end),
            start_time=str(timestamps[start]),
            end_time=str(timestamps[end]),
            duration_steps=int(duration),
            duration_hours=float(duration * 0.5),
        ))
        i += 1
    return events


def node_events_from_masks(
    timestamps: Array,
    masks: Array,
    *,
    min_cells: int = 6,
    grid_shape: tuple[int, int] = (32, 32),
    spatial_iou_threshold: float | None = None,
    centroid_distance_threshold: float | None = None,
) -> list[NodeEvent]:
    """Convert node-level masks into time-contiguous or spatiotemporal events."""

    timestamps = np.asarray(timestamps)
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2:
        raise ValueError(f"masks must have shape (T,N), got {masks.shape}")
    if len(timestamps) != masks.shape[0]:
        raise ValueError("timestamps and masks must have the same length")

    active = masks.sum(axis=1) >= min_cells
    events: list[NodeEvent] = []
    event_id = 0
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(active) and active[i + 1]:
            linked, _, _ = spatially_continuous(
                masks[i],
                masks[i + 1],
                grid_shape=grid_shape,
                iou_threshold=spatial_iou_threshold,
                centroid_distance_threshold=centroid_distance_threshold,
            )
            if not linked:
                break
            i += 1
        end = i
        event_id += 1
        duration = end - start + 1
        events.append(NodeEvent(
            event_id=event_id,
            start_index=int(start),
            end_index=int(end),
            start_time=str(timestamps[start]),
            end_time=str(timestamps[end]),
            duration_steps=int(duration),
            duration_hours=float(duration * 0.5),
            union_mask=masks[start:end + 1].any(axis=0),
        ))
        i += 1
    return events


def _intervals_overlap(
    pred: TimeEvent | NodeEvent,
    true: TimeEvent | NodeEvent,
    *,
    tolerance_steps: int = 0,
) -> bool:
    return (
        pred.start_index <= true.end_index + tolerance_steps
        and pred.end_index >= true.start_index - tolerance_steps
    )


def _spatially_matches(
    pred: NodeEvent,
    true: NodeEvent,
    *,
    grid_shape: tuple[int, int],
    spatial_iou_threshold: float | None = None,
    centroid_distance_threshold: float | None = None,
) -> tuple[bool, float, float]:
    iou = mask_iou(pred.union_mask, true.union_mask)
    distance = centroid_distance(pred.union_mask, true.union_mask, grid_shape=grid_shape)
    if spatial_iou_threshold is None and centroid_distance_threshold is None:
        return True, iou, distance
    ok_iou = spatial_iou_threshold is not None and iou >= float(spatial_iou_threshold)
    ok_distance = (
        centroid_distance_threshold is not None
        and distance <= float(centroid_distance_threshold)
    )
    return bool(ok_iou or ok_distance), iou, distance


def event_level_metrics(
    pred_flags: Array,
    true_flags: Array,
    timestamps: Array,
    *,
    tolerance_steps: int = 1,
) -> dict[str, float | int]:
    """Event-level precision/recall/F1 using interval overlap matching.

    A true event is counted as hit when at least one unmatched predicted event
    overlaps its time interval after applying ``tolerance_steps`` on both sides.
    Greedy matching is sufficient here because TaxiBJ weak-label events are
    short, ordered intervals and we only need a stable baseline metric.
    """

    pred_events = events_from_time_flags(timestamps, pred_flags)
    true_events = events_from_time_flags(timestamps, true_flags)
    matched_pred: set[int] = set()
    matched_true = 0
    delays: list[int] = []
    duration_errors: list[int] = []

    for true_event in true_events:
        candidates = [
            pred
            for pred in pred_events
            if pred.event_id not in matched_pred
            and _intervals_overlap(pred, true_event, tolerance_steps=tolerance_steps)
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda pred: (
                max(pred.start_index - true_event.start_index, 0),
                abs(pred.start_index - true_event.start_index),
            )
        )
        pred = candidates[0]
        matched_pred.add(pred.event_id)
        matched_true += 1
        delays.append(max(pred.start_index - true_event.start_index, 0))
        duration_errors.append(abs(pred.duration_steps - true_event.duration_steps))

    precision = len(matched_pred) / max(len(pred_events), 1)
    recall = matched_true / max(len(true_events), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "event_match_mode": "temporal",
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_hit_rate": float(recall),
        "matched_true_events": int(matched_true),
        "matched_pred_events": int(len(matched_pred)),
        "true_event_count": int(len(true_events)),
        "pred_event_count": int(len(pred_events)),
        "mean_detection_delay_steps": float(np.mean(delays)) if delays else np.nan,
        "median_detection_delay_steps": float(np.median(delays)) if delays else np.nan,
        "mean_detection_delay_hours": float(np.mean(delays) * 0.5) if delays else np.nan,
        "mean_abs_duration_error_steps": (
            float(np.mean(duration_errors)) if duration_errors else np.nan
        ),
        "mean_abs_duration_error_hours": (
            float(np.mean(duration_errors) * 0.5) if duration_errors else np.nan
        ),
    }


def event_level_metrics_from_node_masks(
    pred_mask: Array,
    true_mask: Array,
    timestamps: Array,
    *,
    min_cells: int,
    grid_shape: tuple[int, int] = (32, 32),
    tolerance_steps: int = 1,
    spatial_iou_threshold: float | None = None,
    centroid_distance_threshold: float | None = None,
) -> dict[str, float | int | str]:
    """Event-level metrics with spatiotemporal event splitting and matching."""

    pred_events = node_events_from_masks(
        timestamps,
        pred_mask,
        min_cells=min_cells,
        grid_shape=grid_shape,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
    )
    true_events = node_events_from_masks(
        timestamps,
        true_mask,
        min_cells=min_cells,
        grid_shape=grid_shape,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
    )
    matched_pred: set[int] = set()
    matched_true = 0
    delays: list[int] = []
    duration_errors: list[int] = []
    matched_ious: list[float] = []
    matched_distances: list[float] = []

    for true_event in true_events:
        candidates: list[tuple[NodeEvent, float, float]] = []
        for pred in pred_events:
            if pred.event_id in matched_pred:
                continue
            if not _intervals_overlap(pred, true_event, tolerance_steps=tolerance_steps):
                continue
            spatial_ok, iou, distance = _spatially_matches(
                pred,
                true_event,
                grid_shape=grid_shape,
                spatial_iou_threshold=spatial_iou_threshold,
                centroid_distance_threshold=centroid_distance_threshold,
            )
            if spatial_ok:
                candidates.append((pred, iou, distance))
        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                max(item[0].start_index - true_event.start_index, 0),
                abs(item[0].start_index - true_event.start_index),
                -item[1],
                item[2],
            )
        )
        pred, iou, distance = candidates[0]
        matched_pred.add(pred.event_id)
        matched_true += 1
        delays.append(max(pred.start_index - true_event.start_index, 0))
        duration_errors.append(abs(pred.duration_steps - true_event.duration_steps))
        matched_ious.append(iou)
        matched_distances.append(distance)

    precision = len(matched_pred) / max(len(pred_events), 1)
    recall = matched_true / max(len(true_events), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "event_match_mode": "spatiotemporal",
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_hit_rate": float(recall),
        "matched_true_events": int(matched_true),
        "matched_pred_events": int(len(matched_pred)),
        "true_event_count": int(len(true_events)),
        "pred_event_count": int(len(pred_events)),
        "mean_detection_delay_steps": float(np.mean(delays)) if delays else np.nan,
        "median_detection_delay_steps": float(np.median(delays)) if delays else np.nan,
        "mean_detection_delay_hours": float(np.mean(delays) * 0.5) if delays else np.nan,
        "mean_abs_duration_error_steps": (
            float(np.mean(duration_errors)) if duration_errors else np.nan
        ),
        "mean_abs_duration_error_hours": (
            float(np.mean(duration_errors) * 0.5) if duration_errors else np.nan
        ),
        "mean_matched_spatial_iou": float(np.mean(matched_ious)) if matched_ious else np.nan,
        "mean_matched_centroid_distance": (
            float(np.mean(matched_distances)) if matched_distances else np.nan
        ),
    }


def evaluate_directional_timestep_metrics(
    *,
    method_masks: dict[str, Array],
    z: Array,
    labels: dict[str, dict[str, Array]],
    min_cells: int,
) -> list[dict[str, float | int | str]]:
    """Evaluate each method under high/low/bidirectional timestep labels."""

    rows: list[dict[str, float | int | str]] = []
    for method, base_mask in method_masks.items():
        for direction, level_labels in labels.items():
            pred_mask = directional_node_mask(base_mask, z, direction)
            pred_flags = time_flags_from_node_mask(pred_mask, min_cells=min_cells)
            for level, target in level_labels.items():
                rows.append({
                    "method": method,
                    "direction": direction,
                    "label_level": level,
                    **binary_metrics(pred_flags, target),
                })
    return rows


def evaluate_directional_event_metrics(
    *,
    method_masks: dict[str, Array],
    z: Array,
    labels: dict[str, dict[str, Array]],
    timestamps: Array,
    min_cells: int,
    tolerance_steps: int = 1,
) -> list[dict[str, float | int | str]]:
    """Evaluate event-level metrics for each method and direction."""

    rows: list[dict[str, float | int | str]] = []
    for method, base_mask in method_masks.items():
        for direction, level_labels in labels.items():
            pred_mask = directional_node_mask(base_mask, z, direction)
            pred_flags = time_flags_from_node_mask(pred_mask, min_cells=min_cells)
            for level, target in level_labels.items():
                rows.append({
                    "method": method,
                    "direction": direction,
                    "label_level": level,
                    **event_level_metrics(
                        pred_flags,
                        target,
                        timestamps,
                        tolerance_steps=tolerance_steps,
                    ),
                })
    return rows


def evaluate_directional_spatiotemporal_event_metrics(
    *,
    method_masks: dict[str, Array],
    z: Array,
    label_masks: dict[str, dict[str, Array]],
    timestamps: Array,
    min_cells: int,
    grid_shape: tuple[int, int] = (32, 32),
    tolerance_steps: int = 1,
    spatial_iou_threshold: float | None = None,
    centroid_distance_threshold: float | None = None,
) -> list[dict[str, float | int | str]]:
    """Evaluate event-level metrics using node-level spatiotemporal events."""

    rows: list[dict[str, float | int | str]] = []
    for method, base_mask in method_masks.items():
        for direction, level_labels in label_masks.items():
            pred_mask = directional_node_mask(base_mask, z, direction)
            for level, target_mask in level_labels.items():
                rows.append({
                    "method": method,
                    "direction": direction,
                    "label_level": level,
                    "spatial_iou_threshold": (
                        np.nan if spatial_iou_threshold is None else float(spatial_iou_threshold)
                    ),
                    "centroid_distance_threshold": (
                        np.nan
                        if centroid_distance_threshold is None
                        else float(centroid_distance_threshold)
                    ),
                    **event_level_metrics_from_node_masks(
                        pred_mask,
                        target_mask,
                        timestamps,
                        min_cells=min_cells,
                        grid_shape=grid_shape,
                        tolerance_steps=tolerance_steps,
                        spatial_iou_threshold=spatial_iou_threshold,
                        centroid_distance_threshold=centroid_distance_threshold,
                    ),
                })
    return rows


def summarize_rows(
    rows: Iterable[dict[str, float | int | str]],
    *,
    metric: str,
) -> dict[str, dict[str, float | int | str]]:
    """Return the best row for each direction by a named metric."""

    best: dict[str, dict[str, float | int | str]] = {}
    for row in rows:
        direction = str(row["direction"])
        if direction not in best or float(row.get(metric, -1)) > float(best[direction].get(metric, -1)):
            best[direction] = dict(row)
    return best
