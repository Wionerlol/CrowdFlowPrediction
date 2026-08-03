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
    evaluate_directional_timestep_metrics,
    events_from_time_flags,
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


def write_events(
    *,
    output_dir: Path,
    timestamps: np.ndarray,
    labels: dict[str, dict[str, np.ndarray]],
    fusion_mask: np.ndarray,
    fusion_score: np.ndarray,
    z: np.ndarray,
    grid_shape: tuple[int, int],
    min_cells: int,
) -> dict[str, Any]:
    event_paths: dict[str, Any] = {"fusion": {}, "weak_labels": {}}
    for direction in ["high", "low", "bidirectional"]:
        direction_mask = directional_node_mask(fusion_mask, z, direction)
        fusion_events = build_events(
            timestamps,
            direction_mask,
            fusion_score,
            grid_shape=grid_shape,
            min_cells=min_cells,
        )
        fusion_path = output_dir / f"events_fusion_{direction}.csv"
        pd.DataFrame(fusion_events).to_csv(fusion_path, index=False)
        event_paths["fusion"][direction] = {
            "count": len(fusion_events),
            "by_type": summarize_event_types(fusion_events),
            "path": str(fusion_path),
        }

        event_paths["weak_labels"][direction] = {}
        for level, flags in labels[direction].items():
            weak_events = [event.as_dict() for event in events_from_time_flags(timestamps, flags)]
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
    event_rows = evaluate_directional_event_metrics(
        method_masks=method_masks,
        z=stat_result.z,
        labels=labels,
        timestamps=timestamps,
        min_cells=min_cells,
        tolerance_steps=int(args.event_tolerance_steps),
    )

    timestep_path = output_dir / "directional_timestep_metrics.csv"
    event_path = output_dir / "event_level_metrics.csv"
    pd.DataFrame(timestep_rows).to_csv(timestep_path, index=False)
    pd.DataFrame(event_rows).to_csv(event_path, index=False)

    event_paths = write_events(
        output_dir=output_dir,
        timestamps=timestamps,
        labels=labels,
        fusion_mask=method_masks["fusion"],
        fusion_score=scores["fusion_score"].astype(np.float32),
        z=stat_result.z,
        grid_shape=grid_shape,
        min_cells=min_cells,
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
        "direction_assignment": (
            "Method detections are split into high/low by the observed inflow "
            "slot-normalized z-score sign; bidirectional uses the original mask."
        ),
        "label_positive_rates": label_rates,
        "method_direction_positive_rates": method_direction_rates,
        "best_timestep_rows_by_direction": summarize_rows(timestep_rows, metric="f1"),
        "best_event_rows_by_direction": summarize_rows(event_rows, metric="event_f1"),
        "fusion_rows": {
            "timestep": [
                row for row in timestep_rows
                if row["method"] == "fusion"
            ],
            "event_level": [
                row for row in event_rows
                if row["method"] == "fusion"
            ],
        },
        "event_paths": event_paths,
        "artifacts": {
            "directional_timestep_metrics": str(timestep_path),
            "event_level_metrics": str(event_path),
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
