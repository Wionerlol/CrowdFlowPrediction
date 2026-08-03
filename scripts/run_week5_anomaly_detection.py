"""Run Week 5 anomaly detection baselines and weighted fusion.

Implemented method families:

1. statistical detection: per-grid 3-sigma + IQR;
2. reconstruction-error detection: VAE + Transformer autoencoder;
3. prediction-error detection: Week 4 AGFormer+linear ensemble;
4. weighted voting fusion and rule-based event classification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anomaly.fusion import (
    confidence_from_threshold,
    fuse_weighted_vote,
    per_node_quantile_threshold,
)
from src.anomaly.prediction_error import load_city_data, predict_ensemble_horizon
from src.anomaly.reconstruction import (
    NodeStandardScaler,
    load_transformer_checkpoint,
    load_vae_checkpoint,
    normal_sequence_starts,
    read_checkpoint_metadata,
    save_checkpoint,
    score_transformer_targets,
    score_vae,
    train_transformer_autoencoder,
    train_vae,
)
from src.anomaly.statistical import (
    binary_metrics,
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
    parser.add_argument("--config", default="configs/week5_anomaly.yaml")
    parser.add_argument("--device", default=None, help="cuda/cpu; default auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="1 epoch + small window subset")
    parser.add_argument("--max-windows", type=int, default=None, help="limit val/test windows")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_runtime_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.force_train:
        cfg.setdefault("reconstruction", {})["force_train"] = True
    if args.smoke:
        cfg["output_dir"] = str(Path(cfg["output_dir"]).with_name("week5_anomaly_smoke"))
        cfg["reconstruction"]["checkpoint_dir"] = str(Path(cfg["output_dir"]) / "checkpoints")
        cfg["reconstruction"]["force_train"] = True
        cfg["reconstruction"]["vae"]["epochs"] = 1
        cfg["reconstruction"]["vae"]["patience"] = 1
        cfg["reconstruction"]["transformer"]["epochs"] = 1
        cfg["reconstruction"]["transformer"]["patience"] = 1
        cfg["prediction"]["ensemble_tags"] = cfg["prediction"]["ensemble_tags"][:1]
        cfg["_max_train_batches"] = 2
        cfg["_max_val_batches"] = 1
        cfg["_max_windows"] = args.max_windows or 64
    elif args.max_windows:
        cfg["_max_windows"] = args.max_windows
    return cfg


def flatten_channel(data: np.ndarray, channel: int) -> np.ndarray:
    return data[:, channel].reshape(len(data), -1).astype(np.float32, copy=False)


def split_lengths(splits: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[int, int, int]:
    return len(splits[0]), len(splits[1]), len(splits[2])


def stable_hash(obj: Any) -> str:
    """Stable short hash for JSON-serializable metadata."""

    payload = json.dumps(
        obj,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def array_hash(arr: np.ndarray) -> str:
    """Content hash that includes dtype and shape."""

    x = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(x.shape).encode("utf-8"))
    h.update(str(x.dtype).encode("utf-8"))
    h.update(x.tobytes())
    return h.hexdigest()[:16]


def checkpoint_matches(path: Path, expected: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    """Return whether an existing reconstruction checkpoint is safe to reuse."""

    try:
        metadata = read_checkpoint_metadata(path)
    except Exception as exc:  # noqa: BLE001 - checkpoint may be from older torch/code.
        return False, {}, [f"metadata_read_failed={exc}"]
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected={expected_value!r} actual={actual_value!r}")
    return not mismatches, metadata, mismatches


def load_road_density(city: str, expected_nodes: int) -> np.ndarray | None:
    """Load road-density static feature when available.

    ``node_features`` column 6 is road_density(log1p) per build_graphs.py.
    It is already standardized for model input, so zero is a reasonable
    above/below-average split for rule-based classification.
    """

    path = ROOT / "data/processed" / f"node_features_{city}.npy"
    if not path.exists():
        print(f"[events] road density unavailable: {path}")
        return None
    features = np.load(path)
    if features.ndim != 2 or features.shape[0] != expected_nodes or features.shape[1] <= 6:
        print(f"[events] ignoring unexpected node_features shape: {features.shape}")
        return None
    return features[:, 6].astype(np.float32, copy=False)


def weak_label_rules_for_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    stat_cfg = cfg["statistical"]
    return {
        "level_A": {
            "source": "inflow",
            "direction": "high",
            "label_mode": "z_only",
            "z_thresh": float(stat_cfg.get("sigma_z", 3.0)),
            "min_high_cells": int(stat_cfg.get("weak_label_min_cells", 12)),
        },
        "level_B": {
            "source": "inflow_and_outflow",
            "direction": "high",
            "label_mode": "z_only",
            "inflow_z_thresh": float(stat_cfg.get("sigma_z", 3.0)),
            "min_high_cells": int(stat_cfg.get("weak_label_min_cells", 12)),
            "outflow_z_thresh": float(stat_cfg.get("level_b_outflow_z", 2.5)),
            "min_outflow_cells": int(stat_cfg.get("level_b_min_outflow_cells", 6)),
        },
    }


def fit_and_score_statistical(
    cfg: dict[str, Any],
    data: np.ndarray,
    timestamps: np.ndarray,
    train_len: int,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    stat_cfg = cfg["statistical"]
    steps = int(cfg["steps_per_day"])
    tz = int(cfg["tz_offset"])

    train_in = data[:train_len, 0]
    train_out = data[:train_len, 1]
    train_ts = timestamps[:train_len]
    stats = fit_slot_stats(
        train_in,
        train_ts,
        steps_per_day=steps,
        tz_offset=tz,
        min_std=float(stat_cfg.get("min_std", 1.0)),
        iqr_factor=float(stat_cfg.get("iqr_factor", 1.5)),
    )
    outflow_stats = fit_slot_stats(
        train_out,
        train_ts,
        steps_per_day=steps,
        tz_offset=tz,
        min_std=float(stat_cfg.get("min_std", 1.0)),
        iqr_factor=float(stat_cfg.get("iqr_factor", 1.5)),
    )

    in_flat = flatten_channel(data, 0)
    out_flat = flatten_channel(data, 1)
    val_slots = slot_indices_from_timestamps(timestamps[val_idx], steps_per_day=steps, tz_offset=tz)
    test_slots = slot_indices_from_timestamps(timestamps[test_idx], steps_per_day=steps, tz_offset=tz)

    val_result = score_statistical(
        in_flat[val_idx], val_slots, stats, sigma_z=float(stat_cfg.get("sigma_z", 3.0))
    )
    test_result = score_statistical(
        in_flat[test_idx], test_slots, stats, sigma_z=float(stat_cfg.get("sigma_z", 3.0))
    )
    min_cells = int(stat_cfg.get("weak_label_min_cells", 12))
    labels = {
        "val_A": generate_weak_labels(
            in_flat[val_idx], val_slots, stats,
            level="A", z_thresh=float(stat_cfg.get("sigma_z", 3.0)),
            min_high_cells=min_cells,
        ),
        "test_A": generate_weak_labels(
            in_flat[test_idx], test_slots, stats,
            level="A", z_thresh=float(stat_cfg.get("sigma_z", 3.0)),
            min_high_cells=min_cells,
        ),
        "val_B": generate_weak_labels(
            in_flat[val_idx], val_slots, stats,
            outflow=out_flat[val_idx],
            outflow_stats=outflow_stats,
            level="B", z_thresh=float(stat_cfg.get("sigma_z", 3.0)),
            min_high_cells=min_cells,
            outflow_z_thresh=float(stat_cfg.get("level_b_outflow_z", 2.5)),
            min_outflow_cells=int(stat_cfg.get("level_b_min_outflow_cells", 6)),
        ),
        "test_B": generate_weak_labels(
            in_flat[test_idx], test_slots, stats,
            outflow=out_flat[test_idx],
            outflow_stats=outflow_stats,
            level="B", z_thresh=float(stat_cfg.get("sigma_z", 3.0)),
            min_high_cells=min_cells,
            outflow_z_thresh=float(stat_cfg.get("level_b_outflow_z", 2.5)),
            min_outflow_cells=int(stat_cfg.get("level_b_min_outflow_cells", 6)),
        ),
    }
    return {
        "stats": stats,
        "outflow_stats": outflow_stats,
        "val": val_result,
        "test": test_result,
        "labels": labels,
    }


def train_or_load_reconstruction(
    cfg: dict[str, Any],
    data: np.ndarray,
    train_len: int,
    val_len: int,
    train_weak_labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    rec_cfg = cfg["reconstruction"]
    ckpt_dir = Path(rec_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    inference_only = bool(rec_cfg.get("inference_only", False))

    full_in = flatten_channel(data, 0)
    train_in = full_in[:train_len]
    val_in = full_in[train_len:train_len + val_len]
    scaler = NodeStandardScaler.fit(train_in, min_std=1.0)
    train_norm_full = scaler.transform(train_in)
    val_norm = scaler.transform(val_in)

    if rec_cfg.get("train_on_weak_normal", True):
        normal = train_weak_labels == 0
        train_norm_vae = train_norm_full[normal]
    else:
        normal = np.ones(len(train_norm_full), dtype=bool)
        train_norm_vae = train_norm_full
    if len(train_norm_vae) == 0:
        raise RuntimeError("No weak-normal timesteps available for VAE training")

    force = bool(rec_cfg.get("force_train", False))
    max_train_batches = cfg.get("_max_train_batches")
    max_val_batches = cfg.get("_max_val_batches")
    input_dim = train_norm_full.shape[1]
    label_rules = weak_label_rules_for_summary(cfg)
    common_meta = {
        "city": cfg.get("city", "bj"),
        "data_path": cfg.get("data_path"),
        "data_shape": list(data.shape),
        "split_info": {
            "train_len": int(train_len),
            "val_len": int(val_len),
            "test_len": int(len(data) - train_len - val_len),
        },
        "input_dim": int(input_dim),
        "scaler_mean_hash": array_hash(scaler.mean),
        "scaler_std_hash": array_hash(scaler.std),
        "train_in_hash": array_hash(train_in),
        "label_rule_params": label_rules["level_B"],
        "label_rule_hash": stable_hash(label_rules["level_B"]),
        "train_weak_label_length": int(len(train_weak_labels)),
        "train_weak_label_positive_count": int(train_weak_labels.sum()),
        "train_on_weak_normal": bool(rec_cfg.get("train_on_weak_normal", True)),
    }

    vae_path = ckpt_dir / "vae.pt"
    vae_cfg = rec_cfg["vae"]
    vae_expected = {
        **common_meta,
        **vae_cfg,
        "model_type": "vae",
        "weak_normal_vector_count": int(len(train_norm_vae)),
        "model_config_hash": stable_hash({"vae": vae_cfg}),
    }
    ok = False
    loaded_meta: dict[str, Any] = {}
    mismatches: list[str] = []
    if vae_path.exists() and not force:
        ok, loaded_meta, mismatches = checkpoint_matches(vae_path, vae_expected)
    if ok:
        vae = load_vae_checkpoint(vae_path, input_dim=input_dim, device=device)
        vae_meta = {**loaded_meta, "loaded_from": str(vae_path)}
    else:
        if inference_only:
            raise RuntimeError(
                "VAE checkpoint is unavailable or incompatible while "
                f"reconstruction.inference_only=true: {vae_path}; "
                f"mismatches={mismatches[:5]}"
            )
        if vae_path.exists() and not force:
            print(
                "[reconstruction] VAE checkpoint metadata mismatch; retraining. "
                f"First mismatches: {mismatches[:3]}"
            )
        print("[reconstruction] training VAE...")
        vae, vae_train_meta = train_vae(
            train_norm_vae,
            val_norm,
            hidden_dim=int(vae_cfg["hidden_dim"]),
            latent_dim=int(vae_cfg["latent_dim"]),
            dropout=float(vae_cfg["dropout"]),
            lr=float(vae_cfg["lr"]),
            weight_decay=float(vae_cfg["weight_decay"]),
            batch_size=int(vae_cfg["batch_size"]),
            epochs=int(vae_cfg["epochs"]),
            patience=int(vae_cfg["patience"]),
            beta=float(vae_cfg["beta"]),
            device=device,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )
        vae_meta = {**vae_expected, **vae_train_meta, "trained": True}
        save_checkpoint(
            vae_path,
            vae,
            metadata=vae_meta,
        )

    tf_path = ckpt_dir / "transformer_ae.pt"
    tf_cfg = rec_cfg["transformer"]
    seq_len = int(tf_cfg["seq_len"])
    train_valid_starts = None
    if rec_cfg.get("train_on_weak_normal", True):
        train_valid_starts = normal_sequence_starts(train_weak_labels, seq_len)
        if len(train_valid_starts) == 0:
            raise RuntimeError(
                "No weak-normal Transformer AE windows available: "
                f"seq_len={seq_len}, train weak positives={int(train_weak_labels.sum())}"
            )
    transformer_sequence_count = (
        int(len(train_valid_starts))
        if train_valid_starts is not None
        else int(len(train_norm_full) - seq_len + 1)
    )
    tf_expected = {
        **common_meta,
        **tf_cfg,
        "model_type": "transformer_autoencoder",
        "weak_normal_window_count": transformer_sequence_count,
        "train_sequence_count": transformer_sequence_count,
        "model_config_hash": stable_hash({"transformer": tf_cfg}),
    }
    ok = False
    loaded_meta = {}
    mismatches = []
    if tf_path.exists() and not force:
        ok, loaded_meta, mismatches = checkpoint_matches(tf_path, tf_expected)
    if ok:
        transformer = load_transformer_checkpoint(tf_path, input_dim=input_dim, device=device)
        tf_meta = {**loaded_meta, "loaded_from": str(tf_path)}
    else:
        if inference_only:
            raise RuntimeError(
                "Transformer AE checkpoint is unavailable or incompatible while "
                f"reconstruction.inference_only=true: {tf_path}; "
                f"mismatches={mismatches[:5]}"
            )
        if tf_path.exists() and not force:
            print(
                "[reconstruction] Transformer AE checkpoint metadata mismatch; retraining. "
                f"First mismatches: {mismatches[:3]}"
            )
        print("[reconstruction] training Transformer autoencoder...")
        transformer, tf_train_meta = train_transformer_autoencoder(
            train_norm_full,
            val_norm,
            train_valid_starts=train_valid_starts,
            seq_len=int(tf_cfg["seq_len"]),
            d_model=int(tf_cfg["d_model"]),
            n_heads=int(tf_cfg["n_heads"]),
            num_layers=int(tf_cfg["num_layers"]),
            dim_feedforward=int(tf_cfg["dim_feedforward"]),
            dropout=float(tf_cfg["dropout"]),
            lr=float(tf_cfg["lr"]),
            weight_decay=float(tf_cfg["weight_decay"]),
            batch_size=int(tf_cfg["batch_size"]),
            epochs=int(tf_cfg["epochs"]),
            patience=int(tf_cfg["patience"]),
            device=device,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )
        tf_meta = {**tf_expected, **tf_train_meta, "trained": True}
        save_checkpoint(
            tf_path,
            transformer,
            metadata=tf_meta,
        )

    return {
        "scaler": scaler,
        "vae": vae,
        "transformer": transformer,
        "vae_meta": vae_meta,
        "transformer_meta": tf_meta,
        "vae_path": vae_path,
        "transformer_path": tf_path,
    }


def score_reconstruction(
    cfg: dict[str, Any],
    models: dict[str, Any],
    data: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    rec_cfg = cfg["reconstruction"]
    scaler: NodeStandardScaler = models["scaler"]
    full_in = flatten_channel(data, 0)
    full_norm = scaler.transform(full_in)
    val_norm = full_norm[val_idx]
    test_norm = full_norm[test_idx]
    batch = int(rec_cfg["vae"].get("batch_size", 256))

    print("[reconstruction] scoring VAE...")
    vae_val = score_vae(models["vae"], val_norm, batch_size=batch, device=device)
    vae_test = score_vae(models["vae"], test_norm, batch_size=batch, device=device)

    seq_len = int(rec_cfg["transformer"]["seq_len"])
    tf_batch = int(rec_cfg["transformer"].get("batch_size", 64))
    print("[reconstruction] scoring Transformer autoencoder...")
    tf_val = score_transformer_targets(
        models["transformer"], full_norm, val_idx,
        seq_len=seq_len, batch_size=tf_batch, device=device,
    )
    tf_test = score_transformer_targets(
        models["transformer"], full_norm, test_idx,
        seq_len=seq_len, batch_size=tf_batch, device=device,
    )

    q = float(rec_cfg.get("threshold_quantile", 0.995))
    vae_thr = per_node_quantile_threshold(vae_val, quantile=q)
    tf_thr = per_node_quantile_threshold(tf_val, quantile=q)
    vae_conf = confidence_from_threshold(vae_test, vae_thr)
    tf_conf = confidence_from_threshold(tf_test, tf_thr)
    vae_val_conf = confidence_from_threshold(vae_val, vae_thr)
    tf_val_conf = confidence_from_threshold(tf_val, tf_thr)
    recon_val_conf = ((vae_val_conf + tf_val_conf) / 2.0).astype(np.float32)
    recon_conf = ((vae_conf + tf_conf) / 2.0).astype(np.float32)
    return {
        "vae_val": vae_val,
        "vae_test": vae_test,
        "transformer_val": tf_val,
        "transformer_test": tf_test,
        "vae_threshold": vae_thr,
        "transformer_threshold": tf_thr,
        "val_confidence": recon_val_conf,
        "confidence": recon_conf,
        "mask": recon_conf >= 0.5,
    }


def score_prediction(
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    pred_cfg = cfg["prediction"]
    tags = list(pred_cfg["ensemble_tags"])
    horizon = int(cfg.get("horizon", 1))
    max_windows = cfg.get("_max_windows")
    print("[prediction] scoring validation windows...")
    val = predict_ensemble_horizon(
        tags,
        split="val",
        horizon=horizon,
        batch_size=int(pred_cfg.get("batch_size", 8)),
        num_workers=int(pred_cfg.get("num_workers", 2)),
        device=device,
        max_windows=max_windows,
    )
    print("[prediction] scoring test windows...")
    test = predict_ensemble_horizon(
        tags,
        split="test",
        horizon=horizon,
        batch_size=int(pred_cfg.get("batch_size", 8)),
        num_workers=int(pred_cfg.get("num_workers", 2)),
        device=device,
        max_windows=max_windows,
    )
    q = float(pred_cfg.get("threshold_quantile", 0.995))
    threshold = per_node_quantile_threshold(val["residual"], quantile=q)
    confidence = confidence_from_threshold(test["residual"], threshold)
    val_confidence = confidence_from_threshold(val["residual"], threshold)
    return val, test, threshold, val_confidence, confidence


def evaluate_methods(
    masks: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    *,
    min_cells: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for method, mask in masks.items():
        pred_time = time_flags_from_node_mask(mask, min_cells=min_cells)
        out[method] = {
            "level_A": binary_metrics(pred_time, labels["test_A"]),
            "level_B": binary_metrics(pred_time, labels["test_B"]),
        }
    return out


def main() -> None:
    args = parse_args()
    cfg = apply_runtime_overrides(load_config(args.config), args)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    city = cfg.get("city", "bj")
    data, timestamps, splits = load_city_data(city)
    train_len, val_len, test_len = split_lengths(splits)
    print(f"data={data.shape} train={train_len} val={val_len} test={test_len}")

    pred_val, pred_test, pred_threshold, pred_val_conf, pred_conf = score_prediction(cfg, device)
    val_idx = pred_val["target_indices"]
    test_idx = pred_test["target_indices"]
    print(f"aligned windows: val={len(val_idx)} test={len(test_idx)}")

    stat = fit_and_score_statistical(cfg, data, timestamps, train_len, val_idx, test_idx)
    stat_conf = confidence_from_threshold(stat["test"].score, 1.0)

    # Training labels for optional weak-normal reconstruction training.
    train_slots = slot_indices_from_timestamps(
        timestamps[:train_len],
        steps_per_day=int(cfg["steps_per_day"]),
        tz_offset=int(cfg["tz_offset"]),
    )
    train_in = flatten_channel(data, 0)[:train_len]
    train_out = flatten_channel(data, 1)[:train_len]
    train_weak = generate_weak_labels(
        train_in,
        train_slots,
        stat["stats"],
        outflow=train_out,
        outflow_stats=stat["outflow_stats"],
        level="B",
        z_thresh=float(cfg["statistical"].get("sigma_z", 3.0)),
        min_high_cells=int(cfg["statistical"].get("weak_label_min_cells", 12)),
        outflow_z_thresh=float(cfg["statistical"].get("level_b_outflow_z", 2.5)),
        min_outflow_cells=int(cfg["statistical"].get("level_b_min_outflow_cells", 6)),
    )

    rec_models = train_or_load_reconstruction(
        cfg, data, train_len, val_len, train_weak, device
    )
    rec = score_reconstruction(cfg, rec_models, data, val_idx, test_idx, device)

    pred_mask = pred_conf >= 0.5
    fused = fuse_weighted_vote(
        stat_conf=stat_conf,
        recon_conf=rec["confidence"],
        pred_conf=pred_conf,
        weights=cfg["fusion"]["weights"],
        decision_threshold=float(cfg["fusion"].get("decision_threshold", 0.5)),
    )

    min_event_cells = int(cfg["fusion"].get("min_event_cells", 6))
    masks = {
        "statistical": stat["test"].mask,
        "reconstruction": rec["mask"],
        "prediction": pred_mask,
        "fusion": fused["mask"],
    }
    metrics = evaluate_methods(masks, stat["labels"], min_cells=min_event_cells)
    events_cfg = cfg.get("events", {})
    use_spatial_continuity = bool(events_cfg.get("use_spatial_continuity", False))
    road_density = load_road_density(city, fused["mask"].shape[1])
    outflow_test = flatten_channel(data, 1)[test_idx]
    reconstruction_error = ((rec["vae_test"] + rec["transformer_test"]) / 2.0).astype(np.float32)
    events = build_events(
        pred_test["timestamps"],
        fused["mask"],
        fused["score"],
        grid_shape=tuple(cfg["grid_shape"]),
        min_cells=min_event_cells,
        spatial_iou_threshold=(
            float(events_cfg["spatial_iou_threshold"])
            if use_spatial_continuity and events_cfg.get("spatial_iou_threshold") is not None
            else None
        ),
        centroid_distance_threshold=(
            float(events_cfg["centroid_distance_threshold"])
            if use_spatial_continuity and events_cfg.get("centroid_distance_threshold") is not None
            else None
        ),
        inflow=pred_test["true"],
        outflow=outflow_test,
        z=stat["test"].z,
        statistical_score=stat_conf,
        reconstruction_score=rec["confidence"],
        reconstruction_error=reconstruction_error,
        prediction_score=pred_conf,
        prediction_residual=pred_test["residual"],
        road_density=road_density,
    )

    events_path = output_dir / "events.csv"
    events_df = pd.DataFrame(events)
    events_df.to_csv(events_path, index=False)
    explainability_cols = [
        "event_id", "start_time", "end_time", "duration_hours", "event_type",
        "severity", "max_score", "dominant_method", "statistical_score",
        "reconstruction_score", "prediction_score", "top_cells",
        "main_centroid_row", "main_centroid_col", "union_cells", "cluster_count",
        "main_aspect_ratio", "road_density_mean", "in_out_imbalance",
        "duration_trend_slope", "mean_inflow", "max_inflow", "mean_outflow",
        "max_outflow", "max_z", "max_signed_z", "max_reconstruction_error",
        "max_prediction_residual", "mean_spatial_iou_to_prev",
        "mean_centroid_distance_to_prev",
    ]
    explainability_path = output_dir / "event_explainability.csv"
    events_df.reindex(columns=explainability_cols).to_csv(explainability_path, index=False)
    scores_path = output_dir / "scores.npz"
    np.savez_compressed(
        scores_path,
        timestamps=pred_test["timestamps"],
        target_indices=test_idx,
        true_flow=pred_test["true"],
        true_outflow=outflow_test,
        pred_flow=pred_test["pred"],
        val_timestamps=pred_val["timestamps"],
        val_target_indices=val_idx,
        val_true_flow=pred_val["true"],
        val_pred_flow=pred_val["pred"],
        stat_z=stat["test"].z,
        val_stat_score=stat["val"].score,
        val_stat_mask=stat["val"].mask,
        val_recon_score=rec["val_confidence"],
        val_recon_mask=rec["val_confidence"] >= 0.5,
        val_prediction_residual=pred_val["residual"],
        val_prediction_confidence=pred_val_conf,
        val_prediction_mask=pred_val_conf >= 0.5,
        val_weak_label_A=stat["labels"]["val_A"],
        val_weak_label_B=stat["labels"]["val_B"],
        stat_score=stat["test"].score,
        stat_mask=stat["test"].mask,
        recon_score=rec["confidence"],
        reconstruction_error=reconstruction_error,
        recon_mask=rec["mask"],
        prediction_residual=pred_test["residual"],
        prediction_confidence=pred_conf,
        prediction_mask=pred_mask,
        fusion_score=fused["score"],
        fusion_mask=fused["mask"],
        weak_label_A=stat["labels"]["test_A"],
        weak_label_B=stat["labels"]["test_B"],
        pred_threshold=pred_threshold,
        vae_threshold=rec["vae_threshold"],
        transformer_threshold=rec["transformer_threshold"],
    )

    summary = {
        "city": city,
        "device": str(device),
        "num_test_windows": int(len(test_idx)),
        "horizon": int(cfg.get("horizon", 1)),
        "fusion": {
            "weights": cfg["fusion"]["weights"],
            "decision_threshold": float(cfg["fusion"].get("decision_threshold", 0.5)),
            "min_event_cells": min_event_cells,
            "selected_on": "validation",
            "evaluated_on": "test",
        },
        "weak_label_rules": weak_label_rules_for_summary(cfg),
        "event_building": {
            "merge_by_time_continuity": True,
            "use_spatial_continuity": use_spatial_continuity,
            "spatial_iou_threshold": events_cfg.get("spatial_iou_threshold"),
            "centroid_distance_threshold": events_cfg.get("centroid_distance_threshold"),
            "classification_features": [
                "aspect_ratio",
                "road_density",
                "in_out_imbalance",
                "duration_trend",
                "cluster_count",
            ],
        },
        "method_time_positive_rates": {
            name: float(time_flags_from_node_mask(mask, min_cells=min_event_cells).mean())
            for name, mask in masks.items()
        },
        "weak_label_positive_rates": {
            "level_A": float(stat["labels"]["test_A"].mean()),
            "level_B": float(stat["labels"]["test_B"].mean()),
        },
        "metrics_vs_weak_labels": metrics,
        "events": {
            "count": int(len(events)),
            "by_type": summarize_event_types(events),
            "path": str(events_path),
            "explainability_path": str(explainability_path),
        },
        "artifacts": {
            "scores": str(scores_path),
            "config": str(output_dir / "config_used.yaml"),
            "events": str(events_path),
            "event_explainability": str(explainability_path),
            "vae_checkpoint": str(rec_models["vae_path"]),
            "transformer_checkpoint": str(rec_models["transformer_path"]),
        },
        "reconstruction": {
            "vae": rec_models["vae_meta"],
            "transformer": rec_models["transformer_meta"],
        },
        "prediction": {
            "members": cfg["prediction"]["ensemble_tags"],
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
