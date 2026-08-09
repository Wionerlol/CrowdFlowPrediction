"""Comprehensive Week7 evaluation for the final forecasting/anomaly system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(resolve_path(path), "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def forecast_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    *,
    null_val: float = 0.0,
    mape_eps: float = 1.0,
) -> dict[str, float | int]:
    """Compute the same masked MAE/RMSE/MAPE rules used during Week4."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {true.shape}")
    mask = np.abs(true) > float(null_val)
    if not np.any(mask):
        return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "valid_count": 0}
    error = pred[mask] - true[mask]
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE": float(
            np.mean(np.abs(error) / (np.abs(true[mask]) + float(mape_eps))) * 100.0
        ),
        "valid_count": int(mask.sum()),
    }


def kth_largest_time_score(scores: np.ndarray, *, min_cells: int) -> np.ndarray:
    """Reduce node scores to the score that controls a min-cell decision."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"scores must have shape (T,N), got {values.shape}")
    if min_cells < 1 or min_cells > values.shape[1]:
        raise ValueError(f"min_cells must be in [1,{values.shape[1]}], got {min_cells}")
    index = values.shape[1] - int(min_cells)
    return np.partition(values, index, axis=1)[:, index]


def classification_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    score: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Binary metrics including class-balanced and ranking measurements."""
    prediction = np.asarray(pred, dtype=bool).reshape(-1)
    truth = np.asarray(target, dtype=bool).reshape(-1)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and target lengths differ")
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    tn = int(np.logical_and(~prediction, ~truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    result: dict[str, float | int] = {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(len(truth), 1)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "matthews_corrcoef": float(matthews_corrcoef(truth, prediction)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "target_positive_rate": float(truth.mean()) if len(truth) else 0.0,
        "predicted_positive_rate": float(prediction.mean()) if len(prediction) else 0.0,
    }
    if score is None or np.unique(truth).size < 2:
        result.update({"roc_auc": math.nan, "average_precision": math.nan})
    else:
        ranking = np.asarray(score, dtype=np.float64).reshape(-1)
        if ranking.shape != truth.shape:
            raise ValueError("score and target lengths differ")
        result.update(
            {
                "roc_auc": float(roc_auc_score(truth, ranking)),
                "average_precision": float(average_precision_score(truth, ranking)),
            }
        )
    return result


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(latencies_ms), dtype=np.float64)
    if values.size == 0:
        raise ValueError("at least one latency measurement is required")
    mean_ms = float(values.mean())
    return {
        "request_count": int(values.size),
        "mean_ms": mean_ms,
        "std_ms": float(values.std(ddof=0)),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "sequential_requests_per_second": float(1000.0 / max(mean_ms, 1e-12)),
    }


@dataclass(frozen=True)
class _NodeEvent:
    start: int
    end: int
    union_mask: np.ndarray


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _centroid(mask: np.ndarray, grid_shape: tuple[int, int]) -> tuple[float, float]:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return (math.inf, math.inf)
    rows, cols = np.unravel_index(positions, grid_shape)
    return float(rows.mean()), float(cols.mean())


def _centroid_distance(
    left: np.ndarray,
    right: np.ndarray,
    grid_shape: tuple[int, int],
) -> float:
    left_center = _centroid(left, grid_shape)
    right_center = _centroid(right, grid_shape)
    return float(math.dist(left_center, right_center))


def _spatial_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    spatial_iou_threshold: float | None,
    centroid_distance_threshold: float | None,
) -> tuple[bool, float, float]:
    iou = _mask_iou(left, right)
    distance = _centroid_distance(left, right, grid_shape)
    if spatial_iou_threshold is None and centroid_distance_threshold is None:
        return True, iou, distance
    matches_iou = spatial_iou_threshold is not None and iou >= spatial_iou_threshold
    matches_distance = (
        centroid_distance_threshold is not None
        and distance <= centroid_distance_threshold
    )
    return bool(matches_iou or matches_distance), iou, distance


def _node_events(
    masks: np.ndarray,
    timestamps: np.ndarray,
    *,
    min_cells: int,
    grid_shape: tuple[int, int],
    spatial_iou_threshold: float | None,
    centroid_distance_threshold: float | None,
    expected_step_minutes: int,
) -> list[_NodeEvent]:
    masks = np.asarray(masks, dtype=bool)
    timestamps = np.asarray(timestamps).astype("datetime64[ns]")
    if masks.ndim != 2 or masks.shape[1] != math.prod(grid_shape):
        raise ValueError(f"masks {masks.shape} do not match grid_shape={grid_shape}")
    if len(timestamps) != len(masks):
        raise ValueError("timestamps and masks must have the same number of steps")
    active = masks.sum(axis=1) >= int(min_cells)
    expected_delta = np.timedelta64(int(expected_step_minutes), "m")
    events: list[_NodeEvent] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while (
            index + 1 < len(active)
            and active[index + 1]
            and timestamps[index + 1] - timestamps[index] == expected_delta
        ):
            linked, _, _ = _spatial_match(
                masks[index],
                masks[index + 1],
                grid_shape=grid_shape,
                spatial_iou_threshold=spatial_iou_threshold,
                centroid_distance_threshold=centroid_distance_threshold,
            )
            if not linked:
                break
            index += 1
        events.append(
            _NodeEvent(
                start=start,
                end=index,
                union_mask=masks[start : index + 1].any(axis=0),
            )
        )
        index += 1
    return events


def _maximum_cardinality_matches(
    candidates: list[list[tuple[int, _NodeEvent, float, float]]],
) -> dict[int, tuple[int, _NodeEvent, float, float]]:
    """Return a deterministic maximum-cardinality bipartite event matching."""
    pred_to_true: dict[int, int] = {}
    chosen: dict[int, tuple[int, _NodeEvent, float, float]] = {}

    def augment(true_index: int, visited_pred: set[int]) -> bool:
        for candidate in candidates[true_index]:
            pred_index = candidate[0]
            if pred_index in visited_pred:
                continue
            visited_pred.add(pred_index)
            previous_true = pred_to_true.get(pred_index)
            if previous_true is None or augment(previous_true, visited_pred):
                pred_to_true[pred_index] = true_index
                chosen[true_index] = candidate
                return True
        return False

    for true_index in range(len(candidates)):
        augment(true_index, set())
    return chosen


def spatiotemporal_event_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    timestamps: np.ndarray,
    *,
    min_cells: int,
    grid_shape: tuple[int, int],
    tolerance_steps: int,
    spatial_iou_threshold: float | None,
    centroid_distance_threshold: float | None,
    expected_step_minutes: int = 30,
) -> dict[str, float | int | str]:
    """Maximum-cardinality event matching using time and spatial proximity."""
    timestamps = np.asarray(timestamps).astype("datetime64[ns]")
    if len(timestamps) != len(pred_mask) or len(timestamps) != len(true_mask):
        raise ValueError("timestamps and masks must have the same number of steps")
    predicted = _node_events(
        pred_mask,
        timestamps,
        min_cells=min_cells,
        grid_shape=grid_shape,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
        expected_step_minutes=expected_step_minutes,
    )
    reference = _node_events(
        true_mask,
        timestamps,
        min_cells=min_cells,
        grid_shape=grid_shape,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
        expected_step_minutes=expected_step_minutes,
    )
    tolerance = np.timedelta64(int(tolerance_steps * expected_step_minutes), "m")
    all_candidates: list[list[tuple[int, _NodeEvent, float, float]]] = []
    for true_event in reference:
        candidates: list[tuple[int, _NodeEvent, float, float]] = []
        for pred_index, pred_event in enumerate(predicted):
            overlaps = (
                timestamps[pred_event.start] <= timestamps[true_event.end] + tolerance
                and timestamps[pred_event.end] >= timestamps[true_event.start] - tolerance
            )
            if not overlaps:
                continue
            spatial_ok, iou, distance = _spatial_match(
                pred_event.union_mask,
                true_event.union_mask,
                grid_shape=grid_shape,
                spatial_iou_threshold=spatial_iou_threshold,
                centroid_distance_threshold=centroid_distance_threshold,
            )
            if spatial_ok:
                candidates.append((pred_index, pred_event, iou, distance))
        candidates.sort(
            key=lambda item: (
                max(item[1].start - true_event.start, 0),
                abs(item[1].start - true_event.start),
                -item[2],
                item[3],
                item[0],
            )
        )
        all_candidates.append(candidates)
    matches = _maximum_cardinality_matches(all_candidates)

    delays: list[float] = []
    duration_errors: list[int] = []
    ious: list[float] = []
    distances: list[float] = []
    for true_index, (_, event, iou, distance) in matches.items():
        true_event = reference[true_index]
        delay_minutes = max(
            float(
                (timestamps[event.start] - timestamps[true_event.start])
                / np.timedelta64(1, "m")
            ),
            0.0,
        )
        delays.append(delay_minutes / expected_step_minutes)
        duration_errors.append(abs((event.end - event.start) - (true_event.end - true_event.start)))
        ious.append(iou)
        distances.append(distance)
    matched = len(matches)
    precision = matched / max(len(predicted), 1)
    recall = matched / max(len(reference), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "event_match_mode": "spatiotemporal_maximum_cardinality",
        "expected_step_minutes": int(expected_step_minutes),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_hit_rate": float(recall),
        "matched_true_events": matched,
        "matched_pred_events": matched,
        "true_event_count": len(reference),
        "pred_event_count": len(predicted),
        "mean_detection_delay_steps": float(np.mean(delays)) if delays else math.nan,
        "mean_detection_delay_hours": float(np.mean(delays) * 0.5) if delays else math.nan,
        "mean_abs_duration_error_steps": (
            float(np.mean(duration_errors)) if duration_errors else math.nan
        ),
        "mean_matched_spatial_iou": float(np.mean(ious)) if ious else math.nan,
        "mean_matched_centroid_distance": (
            float(np.mean(distances)) if distances else math.nan
        ),
    }


def evaluate_prediction(
    ensemble_result_path: str | Path,
    scores_path: str | Path,
    *,
    consistency_tolerance: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ensemble_result_path = resolve_path(ensemble_result_path)
    scores_path = resolve_path(scores_path)
    ensemble_result = json.loads(ensemble_result_path.read_text(encoding="utf-8"))
    ensemble = ensemble_result["ensemble"]
    with np.load(scores_path, allow_pickle=False) as scores:
        true = scores["true_flow"]
        pred = scores["pred_flow"]
        timestamps = scores["timestamps"]
    if len(true) != int(ensemble_result["windows"]):
        raise ValueError("prediction artifacts use different test-window counts")
    recomputed_h1 = forecast_metrics(pred, true)
    recorded_h1 = ensemble["metrics"]["h1"]
    differences = {
        name: abs(float(recomputed_h1[name]) - float(recorded_h1[name]))
        for name in ["MAE", "RMSE", "MAPE"]
    }
    if max(differences.values()) > consistency_tolerance:
        raise ValueError(f"h1 metric consistency check failed: {differences}")

    horizon_rows: list[dict[str, Any]] = []
    for label, values in ensemble["metrics"].items():
        steps = int(label[1:]) if label.startswith("h") else 48
        horizon_rows.append(
            {
                "scope": label,
                "cumulative_steps": steps,
                "forecast_hours": steps * 0.5,
                "MAE": float(values["MAE"]),
                "RMSE": float(values["RMSE"]),
                "MAPE": float(values["MAPE"]),
                "test_windows": len(true),
                "metric_semantics": "cumulative_first_steps" if label != "avg" else "all_48_steps",
            }
        )
    horizon_frame = pd.DataFrame(horizon_rows)

    node_rows = []
    for node_id in range(true.shape[1]):
        metrics = forecast_metrics(pred[:, node_id], true[:, node_id])
        node_rows.append({"grid_id": node_id, **metrics})
    node_frame = pd.DataFrame(node_rows)

    timestamp_index = pd.DatetimeIndex(timestamps.astype("datetime64[ns]"))
    slots = timestamp_index.hour * 2 + timestamp_index.minute // 30
    slot_rows = []
    for slot in range(48):
        selected = np.asarray(slots == slot)
        metrics = forecast_metrics(pred[selected], true[selected])
        slot_rows.append(
            {
                "slot": slot,
                "local_time": f"{slot // 2:02d}:{(slot % 2) * 30:02d}",
                "windows": int(selected.sum()),
                **metrics,
            }
        )
    slot_frame = pd.DataFrame(slot_rows)
    residual = np.abs(pred.astype(np.float64) - true.astype(np.float64))
    summary = {
        "tag": ensemble["tag"],
        "members": ensemble_result["members"],
        "test_windows": len(true),
        "nodes": int(true.shape[1]),
        "h1_recomputed": recomputed_h1,
        "h1_consistency_absolute_difference": differences,
        "residual_quantiles": {
            f"p{quantile}": float(np.percentile(residual, quantile))
            for quantile in [50, 90, 95, 99]
        },
        "residual_quantile_semantics": "unmasked_all_cells_including_zero_targets",
        "worst_node_by_mae": int(node_frame.loc[node_frame["MAE"].idxmax(), "grid_id"]),
        "worst_slot_by_mae": int(slot_frame.loc[slot_frame["MAE"].idxmax(), "slot"]),
    }
    return horizon_frame, node_frame, slot_frame, summary


def evaluate_anomaly(
    scores_path: str | Path,
    *,
    min_cells: int,
    z_threshold: float,
    grid_shape: tuple[int, int],
    tolerance_steps: int,
    spatial_iou_threshold: float | None,
    centroid_distance_threshold: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    with np.load(resolve_path(scores_path), allow_pickle=False) as scores:
        arrays = {key: scores[key] for key in scores.files}
    methods = {
        "statistical": (arrays["stat_mask"], arrays["stat_score"]),
        "reconstruction": (arrays["recon_mask"], arrays["recon_score"]),
        "prediction": (arrays["prediction_mask"], arrays["prediction_confidence"]),
        "fusion": (arrays["fusion_mask"], arrays["fusion_score"]),
    }
    labels = {
        "level_A": arrays["weak_label_A"].astype(bool),
        "level_B": arrays["weak_label_B"].astype(bool),
    }
    timestep_rows = []
    for method, (node_mask, node_score) in methods.items():
        pred_flags = np.asarray(node_mask, dtype=bool).sum(axis=1) >= min_cells
        time_score = kth_largest_time_score(node_score, min_cells=min_cells)
        for level, target in labels.items():
            timestep_rows.append(
                {
                    "method": method,
                    "label_reference": level,
                    "label_type": "weak_label",
                    "test_windows": len(target),
                    **classification_metrics(pred_flags, target, time_score),
                }
            )
    timestep_frame = pd.DataFrame(timestep_rows)

    z = np.asarray(arrays["stat_z"], dtype=np.float32)
    fusion_mask = np.asarray(arrays["fusion_mask"], dtype=bool)
    fusion_high_mask = fusion_mask & (z > 0)
    event_rows = []
    for level, flags in labels.items():
        reference_mask = (z > z_threshold) & flags[:, None]
        event_rows.append(
            {
                "method": "fusion",
                "prediction_direction": "bidirectional_final_system",
                "reference_direction": "high_inflow_only",
                "label_reference": level,
                "label_type": "weak_label_spatial_footprint_from_inflow_z",
                "tolerance_steps": tolerance_steps,
                "spatial_iou_threshold": spatial_iou_threshold,
                "centroid_distance_threshold": centroid_distance_threshold,
                **spatiotemporal_event_metrics(
                    fusion_mask,
                    reference_mask,
                    arrays["timestamps"],
                    min_cells=min_cells,
                    grid_shape=grid_shape,
                    tolerance_steps=tolerance_steps,
                    spatial_iou_threshold=spatial_iou_threshold,
                    centroid_distance_threshold=centroid_distance_threshold,
                ),
            }
        )
    event_frame = pd.DataFrame(event_rows)
    full_fusion_events = _node_events(
        fusion_mask,
        arrays["timestamps"],
        min_cells=min_cells,
        grid_shape=grid_shape,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
        expected_step_minutes=30,
    )
    high_fusion_events = _node_events(
        fusion_high_mask,
        arrays["timestamps"],
        min_cells=min_cells,
        grid_shape=grid_shape,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
        expected_step_minutes=30,
    )
    summary = {
        "test_windows": int(len(arrays["timestamps"])),
        "min_event_cells": int(min_cells),
        "weak_label_warning": (
            "TaxiBJ has no authoritative anomaly labels; all precision/recall/accuracy "
            "values are agreement with project weak-label references, not ground-truth accuracy."
        ),
        "weak_label_circularity_warning": (
            "Weak labels are derived from the inflow statistical rule and the final "
            "fusion also includes the statistical detector (configured weight 0.70). "
            "These metrics therefore measure rule agreement, not independent anomaly quality."
        ),
        "event_direction_scope": {
            "prediction": "bidirectional final fusion mask",
            "reference": "high-inflow weak-label z-score footprint",
            "full_fusion_event_count": len(full_fusion_events),
            "high_direction_fusion_event_count": len(high_fusion_events),
        },
        "fusion_positive_rate": float(
            (fusion_mask.sum(axis=1) >= min_cells).mean()
        ),
        "weak_label_positive_rates": {
            level: float(flags.mean()) for level, flags in labels.items()
        },
    }
    return timestep_frame, event_frame, summary


def pipeline_stage_timings(summary_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = resolve_path(summary_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(value.get("stages", []))
    if frame.empty:
        raise ValueError(f"pipeline summary has no stages: {path}")
    durations = pd.to_numeric(frame["duration_seconds"], errors="raise")
    total_seconds = float(durations.sum())
    actual_windows = int(value.get("actual_window_count", 0))
    inference_rows = frame.loc[frame["name"] == "model_inference", "duration_seconds"]
    inference_seconds = float(inference_rows.iloc[0]) if len(inference_rows) else math.nan
    return frame, {
        "stage_duration_sum_seconds": total_seconds,
        "slowest_stage": str(frame.loc[durations.idxmax(), "name"]),
        "slowest_stage_seconds": float(durations.max()),
        "actual_window_count": actual_windows,
        "pipeline_windows_per_second": (
            float(actual_windows / total_seconds) if total_seconds > 0 else math.nan
        ),
        "model_inference_ms_per_window": (
            float(inference_seconds * 1000 / actual_windows)
            if actual_windows > 0 and math.isfinite(inference_seconds)
            else math.nan
        ),
        "device": value.get("device"),
        "time_range": value.get("time_range"),
    }


def _available_port(host: str, preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, preferred))
        except OSError:
            probe.bind((host, 0))
        return int(probe.getsockname()[1])


def _tree_rss_bytes(pid: int) -> int:
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            return int(
                process.memory_info().rss
                + sum(child.memory_info().rss for child in process.children(recursive=True))
            )
        except (psutil.Error, OSError, ProcessLookupError):
            return 0
    else:
        status = Path(f"/proc/{pid}/status")
        if not status.exists():
            return 0
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        return 0


class _MemorySampler:
    def __init__(self, pid: int, interval_seconds: float = 0.01):
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _tree_rss_bytes(self.pid))
            self.samples += 1
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak_bytes = max(self.peak_bytes, _tree_rss_bytes(self.pid))


def benchmark_api(
    cfg: dict[str, Any],
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    host = str(cfg.get("host", "127.0.0.1"))
    port = _available_port(host, int(cfg.get("port", 18080)))
    base_url = f"http://{host}:{port}"
    log_path = output_dir / "api_server.log"
    command = [
        sys.executable,
        "scripts/run_week6_api.py",
        "--config",
        str(cfg.get("web_config", "configs/week6_web.yaml")),
        "--host",
        host,
        "--port",
        str(port),
    ]
    started = time.perf_counter()
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    sampler = _MemorySampler(process.pid)
    sampler.start()
    session = requests.Session()
    startup_timeout = float(cfg.get("startup_timeout_seconds", 30))
    try:
        deadline = time.monotonic() + startup_timeout
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"API process exited with code {process.returncode}")
            try:
                response = session.get(f"{base_url}/api/v1/status", timeout=2)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"API did not become ready in {startup_timeout}s")
            time.sleep(0.1)
        startup_seconds = time.perf_counter() - started
        idle_rss = _tree_rss_bytes(process.pid)
        repeats_default = int(cfg.get("repeats", 10))
        warmups_default = int(cfg.get("warmups", 2))
        rows = []
        for endpoint in cfg.get("endpoints", []):
            method = str(endpoint.get("method", "GET")).upper()
            url = f"{base_url}{endpoint['path']}"
            request_kwargs = {
                "params": endpoint.get("params"),
                "json": endpoint.get("json"),
                "timeout": float(endpoint.get("timeout_seconds", 30)),
            }
            for _ in range(int(endpoint.get("warmups", warmups_default))):
                warmup = session.request(method, url, **request_kwargs)
                warmup.raise_for_status()
            latencies = []
            response_bytes = []
            compressed_bytes = []
            for _ in range(int(endpoint.get("repeats", repeats_default))):
                request_started = time.perf_counter_ns()
                response = session.request(method, url, **request_kwargs)
                elapsed_ms = (time.perf_counter_ns() - request_started) / 1_000_000
                response.raise_for_status()
                latencies.append(elapsed_ms)
                response_bytes.append(len(response.content))
                content_length = response.headers.get("content-length")
                compressed_bytes.append(int(content_length) if content_length else len(response.content))
            rows.append(
                {
                    "endpoint": endpoint["name"],
                    "method": method,
                    "path": endpoint["path"],
                    "response_bytes_mean": float(np.mean(response_bytes)),
                    "wire_bytes_mean": float(np.mean(compressed_bytes)),
                    **summarize_latencies(latencies),
                }
            )
        frame = pd.DataFrame(rows)
        resources = {
            "server_command": command,
            "host": host,
            "port": port,
            "startup_seconds": float(startup_seconds),
            "idle_rss_mib": float(idle_rss / 1024**2),
            "peak_rss_mib": float(sampler.peak_bytes / 1024**2),
            "memory_samples": sampler.samples,
            "benchmark_mode": "single_client_sequential_warm_server",
            "warmups_per_endpoint": warmups_default,
            "measured_requests_per_endpoint": repeats_default,
            "wire_bytes_semantics": (
                "HTTP response body bytes from Content-Length when present; headers excluded"
            ),
            "server_log": str(log_path.relative_to(ROOT)),
        }
        return frame, resources
    except Exception as exc:
        log_handle.flush()
        detail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"API benchmark failed: {exc}\n{detail}") from exc
    finally:
        sampler.stop()
        session.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log_handle.close()


def system_environment() -> dict[str, Any]:
    memory_total = None
    try:
        import psutil

        memory_total = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    gpu = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, memory_mib, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            gpu = {"name": name, "memory_total_mib": float(memory_mib), "driver": driver}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "memory_total_gib": float(memory_total / 1024**3) if memory_total else None,
        "gpu": gpu,
        "git_commit": commit,
        "git_dirty": dirty,
        "dependencies": {
            package: metadata.version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "fastapi",
                "uvicorn",
                "psutil",
                "requests",
                "PyYAML",
            ]
        },
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.columns:
        selected[column] = selected[column].map(
            lambda value: "—"
            if pd.isna(value)
            else f"{value:.4f}"
            if isinstance(value, (float, np.floating))
            else str(value)
        )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in selected.astype(str).values.tolist()]
    return "\n".join([header, separator, *rows])


def build_report(
    summary: dict[str, Any],
    prediction_horizons: pd.DataFrame,
    anomaly_timestep: pd.DataFrame,
    anomaly_events: pd.DataFrame,
    api_benchmark: pd.DataFrame,
) -> str:
    prediction = _markdown_table(
        prediction_horizons,
        ["scope", "forecast_hours", "MAE", "RMSE", "MAPE", "test_windows"],
    )
    fusion_rows = anomaly_timestep.loc[anomaly_timestep["method"] == "fusion"]
    anomaly = _markdown_table(
        fusion_rows,
        [
            "label_reference",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ],
    )
    events = _markdown_table(
        anomaly_events,
        [
            "label_reference",
            "event_precision",
            "event_recall",
            "event_f1",
            "true_event_count",
            "pred_event_count",
            "mean_detection_delay_hours",
        ],
    )
    api = (
        _markdown_table(
            api_benchmark,
            [
                "endpoint",
                "mean_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "sequential_requests_per_second",
                "response_bytes_mean",
            ],
        )
        if not api_benchmark.empty
        else "API benchmark was skipped."
    )
    api_resources = summary.get("api", {}).get("resources", {})
    pipeline = summary["pipeline"]
    startup = api_resources.get("startup_seconds")
    idle_rss = api_resources.get("idle_rss_mib")
    peak_rss = api_resources.get("peak_rss_mib")
    startup_text = f"{startup:.4f}" if isinstance(startup, (int, float)) else "—"
    idle_text = f"{idle_rss:.2f}" if isinstance(idle_rss, (int, float)) else "—"
    peak_text = f"{peak_rss:.2f}" if isinstance(peak_rss, (int, float)) else "—"
    measured_requests = api_resources.get("measured_requests_per_endpoint", "—")
    direction_scope = summary["anomaly"]["event_direction_scope"]
    return f"""# Week7 第一阶段：最终系统全面性能评估

生成时间：{summary['generated_at']}

## 1. 评估对象与口径

- 预测模型：AGFormer gated+linear 四随机种子均值 ensemble；
- 测试窗口：{summary['prediction']['test_windows']} 个 gap-aware TaxiBJ 测试窗口；
- 预测指标沿用 Week4 的非零目标 mask 和 MAPE `eps=1.0` 口径；
- 异常检测参考为 TaxiBJ weak label，不是真实人工标签；
- 最终融合输出是双向异常，而本表参考空间足迹仅由高流量 inflow z-score 构造；
- API 基准为单客户端、顺序请求、warm server，不代表多用户极限吞吐。

## 2. 人流预测精度

{prediction}

h1 指标已从 Week5 保存的 `true_flow/pred_flow` 重新计算，并与 Week4 ensemble 结果一致。`h3/h6/h12` 表示前 3/6/12 个预测步的累计指标，`avg` 为全部 48 步。

## 3. 异常检测 timestep 指标

{anomaly}

Accuracy 容易受异常样本稀少影响，主要同时观察 balanced accuracy、precision、recall、F1 和 average precision。
此外，弱标签来自统计规则，而融合权重中统计检测占 0.70，因此这些数值是规则一致性，不是独立真值准确率。

## 4. 时空事件级指标

{events}

事件匹配同时要求时间区间重叠，以及空间 IoU 或质心距离满足配置阈值。
表中预测侧为完整双向融合输出（{direction_scope['full_fusion_event_count']} 个事件），参考侧为 high-inflow 弱标签足迹；仅保留高流量方向时融合输出为 {direction_scope['high_direction_fusion_event_count']} 个事件。事件构建会在非 30 分钟连续的时间缺口处断开，匹配采用一对一最大基数匹配。

## 5. API 响应性能与内存

{api}

- API 冷启动：{startup_text} 秒；
- API 空闲 RSS：{idle_text} MiB；
- API 基准期间峰值 RSS：{peak_text} MiB。
- 每个端点正式请求数：{measured_requests}（另有 warm-up）；p95/p99 仅代表本机顺序请求；
- `response_bytes_mean` 是解压后的响应 body，wire bytes 不含 HTTP headers。

## 6. Realtime pipeline 性能

- 阶段耗时合计：{pipeline['stage_duration_sum_seconds']:.4f} 秒；
- 窗口数：{pipeline['actual_window_count']}；
- pipeline 端到端等效吞吐：{pipeline['pipeline_windows_per_second']:.4f} windows/s；
- AGFormer ensemble 推理：{pipeline['model_inference_ms_per_window']:.4f} ms/window；
- 最慢阶段：{pipeline['slowest_stage']}（{pipeline['slowest_stage_seconds']:.4f} 秒）；
- 设备：{pipeline['device']}。

该耗时来自最近一次保存的 realtime pipeline summary，不是本次重新执行。端到端等效吞吐包含独立的 48 步 forecast 生成、产物加载和异常检测，不能解释为纯模型吞吐；`model_inference_ms_per_window` 仅对应保存记录中的模型推理阶段。

## 7. 评估边界

1. TaxiBJ 没有权威异常真标签，异常“准确率”只能解释为与弱标签规则的一致程度；
2. API 基准运行在本机 loopback，不包含公网、反向代理和浏览器渲染延迟；
3. pipeline 内存/GPU 峰值未在旧 summary 中保存，本阶段报告的是 API 服务 RSS；
4. 当前测试用于定位 Week7 后续超参数优化和系统优化方向，不应把 weak-label accuracy 当成真实业务准确率。
"""


def run_evaluation(
    cfg: dict[str, Any],
    *,
    config_path: str | Path,
    skip_api_benchmark: bool = False,
) -> dict[str, Any]:
    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_cfg = cfg["prediction"]
    horizon_frame, node_frame, slot_frame, prediction_summary = evaluate_prediction(
        prediction_cfg["ensemble_result"],
        prediction_cfg["scores"],
        consistency_tolerance=float(prediction_cfg.get("consistency_tolerance", 1e-4)),
    )
    anomaly_cfg = cfg["anomaly"]
    timestep_frame, event_frame, anomaly_summary = evaluate_anomaly(
        anomaly_cfg["scores"],
        min_cells=int(anomaly_cfg["min_event_cells"]),
        z_threshold=float(anomaly_cfg.get("z_threshold", 3.0)),
        grid_shape=tuple(int(item) for item in anomaly_cfg.get("grid_shape", [32, 32])),
        tolerance_steps=int(anomaly_cfg.get("event_tolerance_steps", 1)),
        spatial_iou_threshold=anomaly_cfg.get("spatial_iou_threshold"),
        centroid_distance_threshold=anomaly_cfg.get("centroid_distance_threshold"),
    )
    pipeline_frame, pipeline_summary = pipeline_stage_timings(cfg["pipeline_summary"])
    api_frame = pd.DataFrame()
    api_resources: dict[str, Any] = {"skipped": True}
    if not skip_api_benchmark:
        api_frame, api_resources = benchmark_api(cfg["api_benchmark"], output_dir=output_dir)

    horizon_frame.to_csv(output_dir / "prediction_metrics_by_horizon.csv", index=False)
    node_frame.to_csv(output_dir / "prediction_h1_by_node.csv", index=False)
    slot_frame.to_csv(output_dir / "prediction_h1_by_time_slot.csv", index=False)
    timestep_frame.to_csv(output_dir / "anomaly_timestep_metrics.csv", index=False)
    event_frame.to_csv(output_dir / "anomaly_event_metrics.csv", index=False)
    pipeline_frame.to_csv(output_dir / "pipeline_stage_timings.csv", index=False)
    api_frame.to_csv(output_dir / "api_benchmark.csv", index=False)
    Path(output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    web_config_path = resolve_path(cfg["api_benchmark"]["web_config"])
    web_cfg = yaml.safe_load(web_config_path.read_text(encoding="utf-8"))
    input_paths = {
        "prediction_ensemble_result": prediction_cfg["ensemble_result"],
        "anomaly_scores": anomaly_cfg["scores"],
        "pipeline_summary": cfg["pipeline_summary"],
        "web_config": cfg["api_benchmark"]["web_config"],
        **{
            f"api_artifact_{name}": path
            for name, path in web_cfg.get("artifacts", {}).items()
        },
    }
    summary = {
        "schema_version": 1,
        "tag": cfg["tag"],
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config_path": str(config_path),
        "objective": cfg.get("objective"),
        "hypothesis": cfg.get("hypothesis"),
        "data_rules": cfg.get("data_rules", {}),
        "environment": system_environment(),
        "prediction": prediction_summary,
        "anomaly": anomaly_summary,
        "pipeline": pipeline_summary,
        "api": {"resources": api_resources, "endpoint_count": int(len(api_frame))},
        "inputs": {
            name: {
                "path": str(path),
                "size_bytes": resolve_path(path).stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in input_paths.items()
        },
        "implementation": {
            str(path.relative_to(ROOT)): {
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in [
                ROOT / "src/evaluation/week7.py",
                ROOT / "scripts/evaluate_week7_system.py",
                ROOT / "scripts/run_week6_api.py",
                ROOT / "src/api/main.py",
                ROOT / "src/api/repository.py",
                ROOT / "src/api/schemas.py",
                resolve_path(config_path),
            ]
        },
        "outputs": {
            "prediction_metrics_by_horizon": "prediction_metrics_by_horizon.csv",
            "prediction_h1_by_node": "prediction_h1_by_node.csv",
            "prediction_h1_by_time_slot": "prediction_h1_by_time_slot.csv",
            "anomaly_timestep_metrics": "anomaly_timestep_metrics.csv",
            "anomaly_event_metrics": "anomaly_event_metrics.csv",
            "pipeline_stage_timings": "pipeline_stage_timings.csv",
            "api_benchmark": "api_benchmark.csv",
            "report": "week7_system_evaluation_report.md",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = build_report(
        summary, horizon_frame, timestep_frame, event_frame, api_frame
    )
    (output_dir / "week7_system_evaluation_report.md").write_text(report, encoding="utf-8")
    return summary
