"""Tune Week 5 weighted-fusion thresholds and weights from saved scores.

This script is intentionally post-hoc: it does not retrain VAE/Transformer or
rerun AGFormer. It reads ``outputs/week5_anomaly/scores.npz`` and searches
fusion weights/thresholds against the generated weak labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly.statistical import build_events, summarize_event_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="outputs/week5_anomaly/scores.npz")
    parser.add_argument("--output-dir", default="outputs/week5_anomaly/tuned_fusion")
    parser.add_argument("--config", default="outputs/week5_anomaly/config_used.yaml")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-cells", type=int, default=6)
    parser.add_argument("--grid-shape", default="32,32")
    return parser.parse_args()


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
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


def weight_grid(step: float) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    out = []
    for stat_i in range(n + 1):
        for recon_i in range(n + 1 - stat_i):
            pred_i = n - stat_i - recon_i
            out.append((stat_i / n, recon_i / n, pred_i / n))
    return out


def get_kth_time_score(score: np.ndarray, min_cells: int) -> np.ndarray:
    """Node thresholding equivalent: time alert iff kth score >= threshold."""

    if min_cells < 1:
        raise ValueError("min_cells must be >= 1")
    if min_cells > score.shape[1]:
        raise ValueError("min_cells cannot exceed number of nodes")
    return np.partition(score, -min_cells, axis=1)[:, -min_cells]


def row_from_metrics(
    *,
    label_name: str,
    metrics: dict[str, float | int],
    weights: tuple[float, float, float],
    threshold: float,
) -> dict[str, float | int | str]:
    ws, wr, wp = weights
    return {
        "label": label_name,
        "w_statistical": ws,
        "w_reconstruction": wr,
        "w_prediction": wp,
        "decision_threshold": threshold,
        **metrics,
    }


def choose_variants(rows: pd.DataFrame) -> dict[str, dict]:
    level_a = rows[rows["label"] == "level_A"].copy()
    best = level_a.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]

    balanced_pool = level_a[level_a["precision"] >= 0.60]
    if len(balanced_pool):
        balanced = balanced_pool.sort_values(["f1", "recall"], ascending=False).iloc[0]
    else:
        balanced = best

    precision_pool = level_a[level_a["precision"] >= 0.70]
    if len(precision_pool):
        high_precision = precision_pool.sort_values(["f1", "recall"], ascending=False).iloc[0]
    else:
        high_precision = balanced

    return {
        "best_f1_unconstrained": best.to_dict(),
        "balanced_precision_ge_0_60": balanced.to_dict(),
        "high_precision_ge_0_70": high_precision.to_dict(),
    }


def evaluate_variant(
    name: str,
    variant: dict,
    *,
    stat: np.ndarray,
    recon: np.ndarray,
    pred: np.ndarray,
    labels: dict[str, np.ndarray],
    timestamps: np.ndarray,
    min_cells: int,
    grid_shape: tuple[int, int],
    output_dir: Path,
    spatial_iou_threshold: float | None = None,
    centroid_distance_threshold: float | None = None,
) -> dict:
    weights = (
        float(variant["w_statistical"]),
        float(variant["w_reconstruction"]),
        float(variant["w_prediction"]),
    )
    threshold = float(variant["decision_threshold"])
    score = weights[0] * stat + weights[1] * recon + weights[2] * pred
    mask = score >= threshold
    time_flag = mask.sum(axis=1) >= min_cells
    events = build_events(
        timestamps,
        mask,
        score,
        grid_shape=grid_shape,
        min_cells=min_cells,
        spatial_iou_threshold=spatial_iou_threshold,
        centroid_distance_threshold=centroid_distance_threshold,
    )
    events_path = output_dir / f"events_{name}.csv"
    pd.DataFrame(events).to_csv(events_path, index=False)
    scores_path = output_dir / f"fusion_{name}.npz"
    np.savez_compressed(
        scores_path,
        timestamps=timestamps,
        fusion_score=score.astype(np.float32),
        fusion_mask=mask,
        fusion_time_flag=time_flag,
    )
    test_metrics = {
        name: binary_metrics(time_flag, target)
        for name, target in labels.items()
    }
    return {
        "weights": {
            "statistical": weights[0],
            "reconstruction": weights[1],
            "prediction": weights[2],
        },
        "decision_threshold": threshold,
        "test_metrics": test_metrics,
        "events": {
            "count": len(events),
            "by_type": summarize_event_types(events),
            "path": str(events_path),
        },
        "scores_path": str(scores_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_shape = tuple(int(x) for x in args.grid_shape.split(","))
    cfg = {}
    if Path(args.config).exists():
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
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

    data = np.load(args.scores)
    required = [
        "val_stat_score", "val_recon_score", "val_prediction_confidence",
        "val_weak_label_A", "val_weak_label_B",
        "stat_score", "recon_score", "prediction_confidence",
        "weak_label_A", "weak_label_B",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError(
            "scores.npz is missing validation/test arrays required for leakage-free "
            f"fusion tuning: {missing}. Re-run scripts/run_week5_anomaly_detection.py."
        )

    stat_val = np.clip(data["val_stat_score"] / 2.0, 0.0, 1.0).astype(np.float32)
    recon_val = data["val_recon_score"].astype(np.float32)
    pred_val = data["val_prediction_confidence"].astype(np.float32)
    stat_test = np.clip(data["stat_score"] / 2.0, 0.0, 1.0).astype(np.float32)
    recon_test = data["recon_score"].astype(np.float32)
    pred_test = data["prediction_confidence"].astype(np.float32)
    labels_val = {
        "level_A": data["val_weak_label_A"].astype(bool),
        "level_B": data["val_weak_label_B"].astype(bool),
    }
    labels_test = {
        "level_A": data["weak_label_A"].astype(bool),
        "level_B": data["weak_label_B"].astype(bool),
    }
    thresholds = np.round(
        np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step),
        6,
    )

    rows = []
    for weights in weight_grid(args.weight_step):
        fused = weights[0] * stat_val + weights[1] * recon_val + weights[2] * pred_val
        kth = get_kth_time_score(fused, args.min_cells)
        for threshold in thresholds:
            pred_time = kth >= threshold
            for label_name, target in labels_val.items():
                rows.append(row_from_metrics(
                    label_name=label_name,
                    metrics=binary_metrics(pred_time, target),
                    weights=weights,
                    threshold=float(threshold),
                ))

    df = pd.DataFrame(rows)
    grid_path = output_dir / "fusion_grid.csv"
    df.sort_values(["label", "f1", "precision"], ascending=[True, False, False]).to_csv(
        grid_path, index=False
    )

    variants = choose_variants(df)
    variant_details = {}
    for name, variant in variants.items():
        selection_metrics = {
            key: variant[key]
            for key in ["precision", "recall", "f1", "tp", "fp", "fn", "tn", "positive_rate"]
            if key in variant
        }
        variant_details[name] = {
            "label": variant["label"],
            "selection_metrics_val": selection_metrics,
            **evaluate_variant(
                name,
                variant,
                stat=stat_test,
                recon=recon_test,
                pred=pred_test,
                labels=labels_test,
                timestamps=data["timestamps"],
                min_cells=args.min_cells,
                grid_shape=grid_shape,
                output_dir=output_dir,
                spatial_iou_threshold=spatial_iou_threshold,
                centroid_distance_threshold=centroid_distance_threshold,
            ),
        }

    baseline = {
        "weights": {"statistical": 0.35, "reconstruction": 0.30, "prediction": 0.35},
        "decision_threshold": 0.50,
    }
    summary = {
        "scores": args.scores,
        "grid": {
            "weight_step": args.weight_step,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
            "min_cells": args.min_cells,
            "grid_csv": str(grid_path),
        },
        "event_building": {
            "use_spatial_continuity": use_spatial_continuity,
            "spatial_iou_threshold": spatial_iou_threshold,
            "centroid_distance_threshold": centroid_distance_threshold,
        },
        "baseline": baseline,
        "variants": variant_details,
        "caveat": (
            "Fusion parameters are selected on validation weak labels and then "
            "reported on test weak labels. The weak labels are still generated "
            "from statistical anomaly rules, so unconstrained F1 may overweight "
            "the statistical method."
        ),
    }
    summary_path = output_dir / "fusion_tuning_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["variants"], ensure_ascii=False, indent=2))
    print(f"wrote {summary_path}")
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
