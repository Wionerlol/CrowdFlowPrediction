"""Evaluation helpers for Week 5 anomaly detection.

This module separates two evaluation concerns that are easy to conflate:

1. directional timestep metrics: high-flow, low-flow, and bidirectional labels;
2. event-level metrics: interval hits, detection delay, and duration error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.anomaly.statistical import Array, binary_metrics, time_flags_from_node_mask


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


def _intervals_overlap(
    pred: TimeEvent,
    true: TimeEvent,
    *,
    tolerance_steps: int = 0,
) -> bool:
    return (
        pred.start_index <= true.end_index + tolerance_steps
        and pred.end_index >= true.start_index - tolerance_steps
    )


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
