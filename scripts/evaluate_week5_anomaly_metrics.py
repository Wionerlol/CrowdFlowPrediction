"""Directional and event-level evaluation for Week 5 anomaly detection.

This is a post-hoc evaluator. It does not retrain reconstruction models or
rerun the Week 4 predictor. It reads saved Week5 scores and adds:

- high / low / bidirectional timestep metrics;
- event-level precision / recall / F1 / hit rate / delay metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly.evaluation import (
    directional_node_mask,
    evaluate_directional_event_metrics,
    evaluate_directional_spatiotemporal_event_metrics,
    evaluate_directional_timestep_metrics,
    summarize_rows,
)
from src.anomaly.prediction_error import load_city_data
from src.anomaly.statistical import (
    build_events,
    fit_slot_stats,
    generate_weak_labels,
    score_statistical,
    slot_indices_from_timestamps,
    summarize_event_types,
    time_flags_from_node_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="outputs/week5_anomaly/scores.npz")
    parser.add_argument("--config", default="outputs/week5_anomaly/config_used.yaml")
    parser.add_argument("--output-dir", default="outputs/week5_anomaly/evaluation")
    parser.add_argument("--min-cells", type=int, default=None)
    parser.add_argument("--event-tolerance-steps", type=int, default=1)
    return parser.parse_args()


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_channel(data: np.ndarray, channel: int) -> np.ndarray:
    return data[:, channel].reshape(len(data), -1).astype(np.float32, copy=False)


def split_lengths(splits: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, int, int]:
    return len(splits[0]), len(splits[1]), len(splits[2])


def build_directional_labels(
    *,
    inflow: np.ndarray,
    outflow: np.ndarray,
    slots: np.ndarray,
    inflow_stats,
    outflow_stats,
    stat_cfg: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    labels: dict[str, dict[str, np.ndarray]] = {}
    for direction in ["high", "low", "bidirectional"]:
        labels[direction] = {
            "level_A": generate_weak_labels(
                inflow,
                slots,
                inflow_stats,
                level="A",
                direction=direction,
                z_thresh=float(stat_cfg.get("sigma_z", 3.0)),
                min_high_cells=int(stat_cfg.get("weak_label_min_cells", 12)),
            ),
            "level_B": generate_weak_labels(
                inflow,
                slots,
                inflow_stats,
                outflow=outflow,
                outflow_stats=outflow_stats,
                level="B",
                direction=direction,
                z_thresh=float(stat_cfg.get("sigma_z", 3.0)),
                min_high_cells=int(stat_cfg.get("weak_label_min_cells", 12)),
                outflow_z_thresh=float(stat_cfg.get("level_b_outflow_z", 3.0)),
                min_outflow_cells=int(stat_cfg.get("level_b_min_outflow_cells", 25)),
            ),
        }
    return labels


def label_node_mask_from_z(z: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    if direction == "high":
        return z > threshold
    if direction == "low":
        return z < -threshold
    if direction == "bidirectional":
        return np.abs(z) > threshold
    raise ValueError("direction must be 'high', 'low', or 'bidirectional'")


def build_directional_label_node_masks(
    *,
    z: np.ndarray,
    labels: dict[str, dict[str, np.ndarray]],
    stat_cfg: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Build node-level weak-label footprints while preserving timestep labels.

    Level-A/Level-B weak labels are timestep labels. For event-level spatial
    matching we use the inflow z-score footprint on those weak-positive
    timesteps, so the event count remains consistent with the weak-label rule
    while adding a spatial footprint for matching.
    """

    out: dict[str, dict[str, np.ndarray]] = {}
    threshold = float(stat_cfg.get("sigma_z", 3.0))
    for direction, level_labels in labels.items():
        base = label_node_mask_from_z(z, direction, threshold)
        out[direction] = {
            level: base & np.asarray(flags, dtype=bool)[:, None]
            for level, flags in level_labels.items()
        }
    return out


def event_spatial_thresholds(cfg: dict[str, Any]) -> tuple[float | None, float | None]:
    events_cfg = cfg.get("events", {})
    use_spatial_continuity = bool(events_cfg.get("use_spatial_continuity", False))
    spatial_iou_threshold = (
        float(events_cfg["spatial_iou_threshold"])
        if use_spatial_continuity and events_cfg.get("spatial_iou_threshold") is not None
        else None
    )
    centroid_distance_threshold = (
        float(events_cfg["centroid_distance_threshold"])
        if use_spatial_continuity and events_cfg.get("centroid_distance_threshold") is not None
        else None
    )
    return spatial_iou_threshold, centroid_distance_threshold


def load_road_density(city: str, expected_nodes: int) -> np.ndarray | None:
    path = ROOT / "data/processed" / f"node_features_{city}.npy"
    if not path.exists():
        return None
    features = np.load(path)
    if features.ndim != 2 or features.shape[0] != expected_nodes or features.shape[1] <= 6:
        return None
    return features[:, 6].astype(np.float32, copy=False)


def optional_score(scores: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in scores.files:
        return None
    return scores[key].astype(np.float32)


def write_events(
    *,
    output_dir: Path,
    timestamps: np.ndarray,
    label_masks: dict[str, dict[str, np.ndarray]],
    fusion_mask: np.ndarray,
    fusion_score: np.ndarray,
    z: np.ndarray,
    grid_shape: tuple[int, int],
    min_cells: int,
    spatial_iou_threshold: float | None,
    centroid_distance_threshold: float | None,
    inflow: np.ndarray | None = None,
    outflow: np.ndarray | None = None,
    statistical_score: np.ndarray | None = None,
    reconstruction_score: np.ndarray | None = None,
    reconstruction_error: np.ndarray | None = None,
    prediction_score: np.ndarray | None = None,
    prediction_residual: np.ndarray | None = None,
    road_density: np.ndarray | None = None,
) -> dict[str, Any]:
    event_paths: dict[str, Any] = {"fusion": {}, "weak_labels": {}}
    weak_score = np.abs(z).astype(np.float32, copy=False)
    for direction in ["high", "low", "bidirectional"]:
        direction_mask = directional_node_mask(fusion_mask, z, direction)
        fusion_events = build_events(
            timestamps,
            direction_mask,
            fusion_score,
            grid_shape=grid_shape,
            min_cells=min_cells,
            spatial_iou_threshold=spatial_iou_threshold,
            centroid_distance_threshold=centroid_distance_threshold,
            inflow=inflow,
            outflow=outflow,
            z=z,
            statistical_score=statistical_score,
            reconstruction_score=reconstruction_score,
            reconstruction_error=reconstruction_error,
            prediction_score=prediction_score,
            prediction_residual=prediction_residual,
            road_density=road_density,
        )
        fusion_path = output_dir / f"events_fusion_{direction}.csv"
        pd.DataFrame(fusion_events).to_csv(fusion_path, index=False)
        event_paths["fusion"][direction] = {
            "count": len(fusion_events),
            "by_type": summarize_event_types(fusion_events),
            "path": str(fusion_path),
        }

        event_paths["weak_labels"][direction] = {}
        for level, weak_mask in label_masks[direction].items():
            weak_events = build_events(
                timestamps,
                weak_mask,
                weak_score,
                grid_shape=grid_shape,
                min_cells=min_cells,
                spatial_iou_threshold=spatial_iou_threshold,
                centroid_distance_threshold=centroid_distance_threshold,
                inflow=inflow,
                outflow=outflow,
                z=z,
                road_density=road_density,
            )
            weak_path = output_dir / f"events_weak_{level}_{direction}.csv"
            pd.DataFrame(weak_events).to_csv(weak_path, index=False)
            event_paths["weak_labels"][direction][level] = {
                "count": len(weak_events),
                "path": str(weak_path),
            }
    return event_paths


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = np.load(args.scores)
    city = cfg.get("city", "bj")
    data, timestamps_all, splits = load_city_data(city)
    train_len, _, _ = split_lengths(splits)
    target_idx = scores["target_indices"].astype(np.int64)
    timestamps = scores["timestamps"]
    if len(target_idx) != len(timestamps):
        raise ValueError("target_indices and timestamps length mismatch")

    stat_cfg = cfg["statistical"]
    steps_per_day = int(cfg.get("steps_per_day", 48))
    tz_offset = int(cfg.get("tz_offset", 8 if city == "bj" else 0))
    min_cells = int(args.min_cells or cfg["fusion"].get("min_event_cells", 6))
    grid_shape = tuple(int(x) for x in cfg.get("grid_shape", [32, 32]))

    inflow_all = flatten_channel(data, 0)
    outflow_all = flatten_channel(data, 1)
    train_ts = timestamps_all[:train_len]
    inflow_stats = fit_slot_stats(
        data[:train_len, 0],
        train_ts,
        steps_per_day=steps_per_day,
        tz_offset=tz_offset,
        min_std=float(stat_cfg.get("min_std", 1.0)),
        iqr_factor=float(stat_cfg.get("iqr_factor", 1.5)),
    )
    outflow_stats = fit_slot_stats(
        data[:train_len, 1],
        train_ts,
        steps_per_day=steps_per_day,
        tz_offset=tz_offset,
        min_std=float(stat_cfg.get("min_std", 1.0)),
        iqr_factor=float(stat_cfg.get("iqr_factor", 1.5)),
    )
    slots = slot_indices_from_timestamps(
        timestamps_all[target_idx],
        steps_per_day=steps_per_day,
        tz_offset=tz_offset,
    )
    inflow = inflow_all[target_idx]
    outflow = outflow_all[target_idx]
    stat_result = score_statistical(
        inflow,
        slots,
        inflow_stats,
        sigma_z=float(stat_cfg.get("sigma_z", 3.0)),
    )
    labels = build_directional_labels(
        inflow=inflow,
        outflow=outflow,
        slots=slots,
        inflow_stats=inflow_stats,
        outflow_stats=outflow_stats,
        stat_cfg=stat_cfg,
    )
    label_masks = build_directional_label_node_masks(
        z=stat_result.z,
        labels=labels,
        stat_cfg=stat_cfg,
    )
    spatial_iou_threshold, centroid_distance_threshold = event_spatial_thresholds(cfg)

    method_masks = {
        "statistical": stat_result.mask,
        "reconstruction": scores["recon_mask"].astype(bool),
        "prediction": scores["prediction_mask"].astype(bool),
        "fusion": scores["fusion_mask"].astype(bool),
    }
    timestep_rows = evaluate_directional_timestep_metrics(
        method_masks=method_masks,
        z=stat_result.z,
        labels=labels,
        min_cells=min_cells,
    )
    temporal_event_rows = evaluate_directional_event_metrics(
        method_masks=method_masks,
        z=stat_result.z,
        labels=labels,
        timestamps=timestamps,
        min_cells=min_cells,
        tolerance_steps=int(args.event_tolerance_steps),
    )
    event_rows = evaluate_directional_spatiotemporal_event_metrics(
        method_masks=method_masks,
        z=stat_result.z,
        label_masks=label_masks,
        timestamps=timestamps,
        min_cells=min_cells,
        grid_shape=grid_shape,
        tolerance_steps=int(args.event_tolerance_steps),
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
    )

    timestep_path = output_dir / "directional_timestep_metrics.csv"
    event_path = output_dir / "event_level_metrics.csv"
    temporal_event_path = output_dir / "temporal_event_level_metrics.csv"
    pd.DataFrame(timestep_rows).to_csv(timestep_path, index=False)
    pd.DataFrame(event_rows).to_csv(event_path, index=False)
    pd.DataFrame(temporal_event_rows).to_csv(temporal_event_path, index=False)

    event_paths = write_events(
        output_dir=output_dir,
        timestamps=timestamps,
        label_masks=label_masks,
        fusion_mask=method_masks["fusion"],
        fusion_score=scores["fusion_score"].astype(np.float32),
        z=stat_result.z,
        grid_shape=grid_shape,
        min_cells=min_cells,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
        inflow=optional_score(scores, "true_flow"),
        outflow=optional_score(scores, "true_outflow"),
        statistical_score=optional_score(scores, "stat_score"),
        reconstruction_score=optional_score(scores, "recon_score"),
        reconstruction_error=optional_score(scores, "reconstruction_error"),
        prediction_score=optional_score(scores, "prediction_confidence"),
        prediction_residual=optional_score(scores, "prediction_residual"),
        road_density=load_road_density(city, method_masks["fusion"].shape[1]),
    )

    label_rates = {
        direction: {
            level: float(flags.mean())
            for level, flags in level_labels.items()
        }
        for direction, level_labels in labels.items()
    }
    method_direction_rates = {}
    for method, mask in method_masks.items():
        method_direction_rates[method] = {}
        for direction in ["high", "low", "bidirectional"]:
            pred_mask = directional_node_mask(mask, stat_result.z, direction)
            method_direction_rates[method][direction] = float(
                time_flags_from_node_mask(pred_mask, min_cells=min_cells).mean()
            )

    summary = {
        "scores": args.scores,
        "config": args.config,
        "num_windows": int(len(timestamps)),
        "min_cells": min_cells,
        "event_tolerance_steps": int(args.event_tolerance_steps),
        "event_match_mode": "spatiotemporal",
        "event_building": {
            "use_spatial_continuity": spatial_iou_threshold is not None
            or centroid_distance_threshold is not None,
            "spatial_iou_threshold": spatial_iou_threshold,
            "centroid_distance_threshold": centroid_distance_threshold,
            "temporal_reference_artifact": str(temporal_event_path),
        },
        "direction_assignment": (
            "Method detections are split into high/low by the observed inflow "
            "slot-normalized z-score sign; bidirectional uses the original mask."
        ),
        "label_positive_rates": label_rates,
        "method_direction_positive_rates": method_direction_rates,
        "best_timestep_rows_by_direction": summarize_rows(timestep_rows, metric="f1"),
        "best_event_rows_by_direction": summarize_rows(event_rows, metric="event_f1"),
        "best_temporal_event_rows_by_direction": summarize_rows(
            temporal_event_rows,
            metric="event_f1",
        ),
        "fusion_rows": {
            "timestep": [
                row for row in timestep_rows
                if row["method"] == "fusion"
            ],
            "event_level": [
                row for row in event_rows
                if row["method"] == "fusion"
            ],
            "temporal_event_level": [
                row for row in temporal_event_rows
                if row["method"] == "fusion"
            ],
        },
        "event_paths": event_paths,
        "artifacts": {
            "directional_timestep_metrics": str(timestep_path),
            "event_level_metrics": str(event_path),
            "temporal_event_level_metrics": str(temporal_event_path),
        },
    }
    summary_path = output_dir / "directional_event_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps({
        "label_positive_rates": label_rates,
        "fusion_timestep": summary["fusion_rows"]["timestep"],
        "fusion_event_level": summary["fusion_rows"]["event_level"],
        "artifacts": summary["artifacts"],
        "summary": str(summary_path),
    }, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
