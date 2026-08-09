"""Week7 interpretability for the final forecasting and anomaly system.

The module deliberately separates three evidence types:

* native AGFormer temporal attention and adaptive spatial adjacency;
* TreeSHAP values for a measured-fidelity surrogate of saved ensemble outputs;
* external-context candidates for anomaly causes (weather/calendar/news input).

Attention is not claimed to be causal importance, and anomaly causes remain
candidates unless an external event source confirms them.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import yaml

from scripts.train import load_static_features
from src.anomaly.prediction_error import build_split_dataset, load_model_from_checkpoint
from src.data.flow_dataset import load_graphs, load_taxibj
from src.features.temporal import load_bj_holidays, load_bj_makeup_workdays


ROOT = Path(__file__).resolve().parents[2]

DYNAMIC_FEATURE_NAMES = [
    "inflow",
    "outflow",
    "slot_sin",
    "slot_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "is_holiday",
    "is_workday",
]

STATIC_FEATURE_NAMES = [
    "latitude_norm",
    "longitude_norm",
    "distance_to_center_norm",
    "ring_id_norm",
    "distance_to_transport_hub_norm",
    "distance_to_commercial_center_norm",
    "road_density_total",
    "road_density_highway",
    "road_density_local",
    "road_density_service",
    "road_density_active",
    "poi_food",
    "poi_shopping",
    "poi_entertainment",
    "poi_office",
    "poi_residential",
    "poi_education",
    "poi_healthcare",
    "poi_government",
    "poi_tourism",
    "poi_sports",
    "poi_religious",
    "satellite_ndvi",
    "satellite_ndbi",
    "satellite_mndwi",
    "satellite_b4_mean",
    "satellite_b4_std",
    "subway_station_density",
    "subway_entrance_density",
    "bus_stop_density",
    "subway_line_count",
    "is_transfer_hub",
    "bus_route_count",
]

STATIC_GROUPS = {
    "static_geography": (0, 4),
    "static_distance": (4, 6),
    "static_road": (6, 11),
    "static_poi": (11, 22),
    "static_satellite": (22, 27),
    "static_transit_density": (27, 30),
    "static_rail": (30, 32),
    "static_bus_routes": (32, 33),
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.generic,)):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evenly_spaced_indices(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=np.int64))


def surrogate_feature_names(t_in: int = 48) -> tuple[list[str], list[str]]:
    """Return feature names and their aggregation group for TreeSHAP."""
    names: list[str] = []
    groups: list[str] = []
    for flow_name in ["inflow", "outflow"]:
        for lag in range(t_in, 0, -1):
            names.append(f"{flow_name}_lag_{lag}")
            groups.append(f"{flow_name}_history")
    for name in DYNAMIC_FEATURE_NAMES[2:]:
        names.append(name)
        groups.append("time_calendar")
    for index, name in enumerate(STATIC_FEATURE_NAMES):
        names.append(name)
        group = next(
            group_name
            for group_name, (start, stop) in STATIC_GROUPS.items()
            if start <= index < stop
        )
        groups.append(group)
    return names, groups


def _clean_checkpoint_hashes(tags: list[str], checkpoint_root: str | Path) -> dict[str, Any]:
    root = resolve_path(checkpoint_root)
    return {
        tag: {
            "path": str((root / tag / "best.pt").relative_to(ROOT)),
            "size_bytes": (root / tag / "best.pt").stat().st_size,
            "sha256": file_sha256(root / tag / "best.pt"),
        }
        for tag in tags
    }


@torch.no_grad()
def extract_attention(
    model_cfg: dict[str, Any],
    tags: list[str],
    *,
    checkpoint_root: str | Path,
    split: str,
    mean: float,
    std: float,
    sample_windows: int,
    recent_steps: int,
    top_edges_per_node: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract native head-averaged attention and adaptive adjacency."""
    dataset, _, target_timestamps = build_split_dataset(
        model_cfg, split, mean=mean, std=std
    )
    selected = evenly_spaced_indices(len(dataset), sample_windows)
    x = torch.stack([dataset[int(index)][0] for index in selected]).to(device)
    graphs = load_graphs(
        model_cfg.get("city", "bj"),
        model_cfg.get("graphs", ["spatial"]),
        device=device,
    )
    num_layers = int(model_cfg.get("num_layers", 2))
    t_in = int(model_cfg["T_in"])
    n_nodes = int(model_cfg["N"])
    matrix_sums = [np.zeros((t_in, t_in), dtype=np.float64) for _ in range(num_layers)]
    node_last_sums = [np.zeros((n_nodes, t_in), dtype=np.float64) for _ in range(num_layers)]
    matrix_count = np.zeros(num_layers, dtype=np.int64)
    node_count = np.zeros(num_layers, dtype=np.int64)
    adaptive_sum = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for tag in tags:
        model, cfg, checkpoint_mean, checkpoint_std, _ = load_model_from_checkpoint(
            tag,
            device=device,
            checkpoint_root=resolve_path(checkpoint_root),
        )
        if cfg["T_in"] != t_in or cfg["N"] != n_nodes:
            raise RuntimeError(f"Checkpoint {tag} has incompatible model dimensions")
        if abs(checkpoint_mean - mean) > 1e-6 or abs(checkpoint_std - std) > 1e-6:
            raise RuntimeError(f"Checkpoint {tag} uses different normalization statistics")
        handles = []
        batch_size = len(selected)

        def make_hook(layer_index: int):
            def hook(_module, _inputs, outputs):
                weights = outputs[1]
                if weights is None:
                    raise RuntimeError("MultiheadAttention did not return attention weights")
                values = weights.detach().float().reshape(
                    batch_size, n_nodes, t_in, t_in
                )
                matrix_sums[layer_index] += values.sum(dim=(0, 1)).cpu().numpy()
                node_last_sums[layer_index] += values[:, :, -1, :].sum(dim=0).cpu().numpy()
                matrix_count[layer_index] += batch_size * n_nodes
                node_count[layer_index] += batch_size

            return hook

        for layer_index, block in enumerate(model.blocks):
            block.capture_attention_weights = True
            handles.append(block.attn.register_forward_hook(make_hook(layer_index)))
        try:
            model(x, graphs)
        finally:
            for handle in handles:
                handle.remove()
            for block in model.blocks:
                block.capture_attention_weights = False
        adaptive_sum += model.adaptive_adj.detach().float().cpu().numpy()
        del model

    matrices = [matrix_sums[i] / matrix_count[i] for i in range(num_layers)]
    node_last = [node_last_sums[i] / node_count[i] for i in range(num_layers)]
    adaptive = adaptive_sum / len(tags)

    matrix_rows: list[dict[str, Any]] = []
    last_rows: list[dict[str, Any]] = []
    for layer_index, matrix in enumerate(matrices):
        for query in range(t_in):
            for key in range(t_in):
                matrix_rows.append(
                    {
                        "layer": layer_index + 1,
                        "query_step": query,
                        "key_step": key,
                        "key_lag_from_input_end_steps": t_in - 1 - key,
                        "attention_weight": float(matrix[query, key]),
                    }
                )
        profile = matrix[-1]
        for key, value in enumerate(profile):
            last_rows.append(
                {
                    "layer": layer_index + 1,
                    "input_step": key,
                    "lag_steps": t_in - 1 - key,
                    "lag_hours": (t_in - 1 - key) * 0.5,
                    "attention_weight": float(value),
                }
            )

    node_profile = np.mean(np.stack(node_last, axis=0), axis=0)
    node_profile = node_profile / np.clip(node_profile.sum(axis=1, keepdims=True), 1e-12, None)
    recent_mass = node_profile[:, -int(recent_steps):].sum(axis=1)
    entropy = -(node_profile * np.log(np.clip(node_profile, 1e-12, None))).sum(axis=1)
    entropy /= math.log(t_in)
    source_influence = adaptive.mean(axis=0)
    row_entropy = -(adaptive * np.log(np.clip(adaptive, 1e-12, None))).sum(axis=1)
    row_entropy /= math.log(n_nodes)
    top_k_concentration = np.sort(adaptive, axis=1)[:, -int(top_edges_per_node):].sum(axis=1)
    node_rows = [
        {
            "grid_id": node,
            "row": node // 32,
            "col": node % 32,
            "recent_attention_mass": float(recent_mass[node]),
            "temporal_attention_entropy": float(entropy[node]),
            "adaptive_source_influence": float(source_influence[node]),
            "adaptive_self_weight": float(adaptive[node, node]),
            "adaptive_row_entropy": float(row_entropy[node]),
            "adaptive_top_k_concentration": float(top_k_concentration[node]),
        }
        for node in range(n_nodes)
    ]
    edge_rows: list[dict[str, Any]] = []
    for target in range(n_nodes):
        order = np.argsort(adaptive[target])[::-1][: int(top_edges_per_node)]
        for rank, source in enumerate(order, start=1):
            edge_rows.append(
                {
                    "target_grid_id": target,
                    "target_row": target // 32,
                    "target_col": target % 32,
                    "source_grid_id": int(source),
                    "source_row": int(source) // 32,
                    "source_col": int(source) % 32,
                    "rank": rank,
                    "adaptive_weight": float(adaptive[target, source]),
                }
            )
    summary = {
        "checkpoint_count": len(tags),
        "sample_window_indices": selected.tolist(),
        "sample_target_timestamps": [str(target_timestamps[index]) for index in selected],
        "sampling_scope": "qualitative four-date sample, not a full-test attention estimate",
        "attention_semantics": (
            "head-averaged native temporal self-attention in the nonlinear "
            "Transformer-GCN branch"
        ),
        "attention_coverage_warning": (
            "The final checkpoint uses use_linear_output=true. Temporal attention and "
            "adaptive adjacency do not explain the parallel linear residual head and "
            "therefore do not represent the complete ensemble output."
        ),
        "spatial_semantics": (
            "learned adaptive adjacency in the nonlinear branch only; predefined "
            "spatial GCN and the parallel linear residual head are separate"
        ),
        "recent_steps": int(recent_steps),
        "recent_hours": float(recent_steps * 0.5),
        "mean_recent_attention_mass": float(recent_mass.mean()),
        "mean_temporal_attention_entropy": float(entropy.mean()),
        "mean_adaptive_row_entropy": float(row_entropy.mean()),
        "mean_adaptive_top_k_concentration": float(top_k_concentration.mean()),
    }
    return (
        pd.DataFrame(matrix_rows),
        pd.DataFrame(last_rows),
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
        summary,
    )


def _time_calendar_features(timestamps: np.ndarray) -> np.ndarray:
    """Build the exact seven temporal channels used by the best checkpoint."""
    utc = pd.DatetimeIndex(timestamps.astype("datetime64[s]"))
    local = utc + pd.Timedelta(hours=8)
    slot = local.hour.to_numpy(np.float32) * 2 + local.minute.to_numpy(np.float32) / 30
    weekday = local.dayofweek.to_numpy(np.float32)
    weekend = (weekday >= 5).astype(np.float32)
    holidays = load_bj_holidays()
    holiday = np.asarray([date in holidays for date in local.date], dtype=np.float32)
    workday = 1.0 - np.maximum(weekend, holiday)
    return np.stack(
        [
            np.sin(2 * np.pi * slot / 48),
            np.cos(2 * np.pi * slot / 48),
            np.sin(2 * np.pi * weekday / 7),
            np.cos(2 * np.pi * weekday / 7),
            weekend,
            holiday,
            workday,
        ],
        axis=1,
    ).astype(np.float32)


def build_surrogate_rows(
    *,
    scores_path: str | Path,
    model_cfg: dict[str, Any],
    sample_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str]]:
    """Build leakage-safe rows that contain only information available at forecast time."""
    with np.load(resolve_path(scores_path), allow_pickle=False) as scores:
        target_indices = scores["target_indices"].astype(np.int64)
        predictions = scores["pred_flow"].astype(np.float32)
        target_timestamps = scores["timestamps"]
    data, timestamps = load_taxibj()
    if predictions.shape != (len(target_indices), int(model_cfg["N"])):
        raise ValueError("Saved predictions do not match checkpoint node/window dimensions")
    if int(model_cfg["T_in"]) != 48 or int(model_cfg["F_in"]) != 9:
        raise ValueError("Week7 surrogate feature schema currently expects T_in=48 and F_in=9")
    if np.any(target_indices < int(model_cfg["T_in"])):
        raise ValueError("Target index does not have a complete historical context")
    if not np.array_equal(timestamps[target_indices], target_timestamps.astype("datetime64[s]")):
        raise ValueError("Saved score timestamps do not align with TaxiBJ target indices")

    num_windows, num_nodes = predictions.shape
    total = num_windows * num_nodes
    count = min(int(sample_rows), total)
    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(total, size=count, replace=False)
    window_indices = flat_indices // num_nodes
    node_indices = flat_indices % num_nodes
    absolute_targets = target_indices[window_indices]

    history_times = absolute_targets[:, None] - np.arange(48, 0, -1, dtype=np.int64)[None, :]
    flow = data.reshape(len(data), 2, -1).transpose(0, 2, 1).astype(np.float32)
    inflow = flow[history_times, node_indices[:, None], 0]
    outflow = flow[history_times, node_indices[:, None], 1]
    calendar = _time_calendar_features(timestamps)[absolute_targets - 1]
    static = load_static_features(model_cfg)
    if static is None or tuple(static.shape) != (num_nodes, 33):
        raise ValueError("Expected leakage-free static feature matrix with shape (N,33)")
    static_rows = static.numpy()[node_indices]
    matrix = np.concatenate([inflow, outflow, calendar, static_rows], axis=1)
    names, groups = surrogate_feature_names(48)
    if matrix.shape[1] != len(names):
        raise RuntimeError(f"Surrogate matrix has {matrix.shape[1]} != {len(names)} columns")
    metadata_frame = pd.DataFrame(
        {
            "sample_row": np.arange(count),
            "window_index": window_indices,
            "grid_id": node_indices,
            "target_index": absolute_targets,
            "target_timestamp": target_timestamps[window_indices].astype(str),
        }
    )
    target = predictions[window_indices, node_indices]
    return matrix.astype(np.float32), target, metadata_frame, names, groups


def fit_tree_shap(
    matrix: np.ndarray,
    target: np.ndarray,
    metadata_frame: pd.DataFrame,
    feature_names: list[str],
    feature_groups: list[str],
    *,
    cfg: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit a time-held-out model-behavior surrogate and explain it with TreeSHAP."""
    window_indices = metadata_frame["window_index"].to_numpy()
    split_window = int(math.floor((window_indices.max() + 1) * 0.8))
    train_mask = window_indices < split_window
    validation_mask = ~train_mask
    if train_mask.sum() < 100 or validation_mask.sum() < 100:
        raise ValueError("Surrogate time split does not contain enough rows")
    model = ExtraTreesRegressor(
        n_estimators=int(cfg.get("n_estimators", 120)),
        max_depth=int(cfg.get("max_depth", 18)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 4)),
        max_features=float(cfg.get("max_features", 0.7)),
        n_jobs=int(cfg.get("n_jobs", 4)),
        random_state=seed,
    )
    model.fit(matrix[train_mask], target[train_mask])
    validation_pred = model.predict(matrix[validation_mask])
    fidelity = {
        "target_semantics": (
            "saved final AGFormer 4-seed ensemble h1 prediction, including the "
            "parallel linear residual head"
        ),
        "surrogate_model": "ExtraTreesRegressor",
        "split_semantics": "first 80% test-window indices fit; final 20% held out",
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_r2": float(r2_score(target[validation_mask], validation_pred)),
        "validation_mae_flow_units": float(
            mean_absolute_error(target[validation_mask], validation_pred)
        ),
    }
    validation_rows = metadata_frame.loc[validation_mask].copy().reset_index(drop=True)
    validation_rows["ensemble_prediction"] = target[validation_mask]
    validation_rows["surrogate_prediction"] = validation_pred
    validation_rows["surrogate_abs_error"] = np.abs(
        validation_rows["surrogate_prediction"] - validation_rows["ensemble_prediction"]
    )

    explain_count = min(int(cfg.get("explain_rows", 1000)), int(validation_mask.sum()))
    validation_positions = np.flatnonzero(validation_mask)
    chosen_local = evenly_spaced_indices(len(validation_positions), explain_count)
    chosen = validation_positions[chosen_local]
    explain_matrix = matrix[chosen]
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(
        explainer.shap_values(explain_matrix, check_additivity=False), dtype=np.float32
    )
    if shap_values.shape != explain_matrix.shape:
        raise RuntimeError(
            f"Unexpected TreeSHAP shape {shap_values.shape}, expected {explain_matrix.shape}"
        )
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    explained_prediction = model.predict(explain_matrix)
    additive_prediction = base_value + shap_values.sum(axis=1)
    additivity_error = np.abs(additive_prediction - explained_prediction)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    feature_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_group": feature_groups,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "importance_rank": pd.Series(mean_abs).rank(method="first", ascending=False).astype(int),
        }
    ).sort_values("importance_rank")
    group_frame = (
        feature_frame.groupby("feature_group", as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "sum"),
            mean_signed_shap=("mean_signed_shap", "sum"),
            feature_count=("feature", "count"),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    group_frame["importance_share"] = group_frame["mean_abs_shap"] / max(
        group_frame["mean_abs_shap"].sum(), 1e-12
    )
    explanation_rows = metadata_frame.iloc[chosen].copy().reset_index(drop=True)
    explanation_rows["ensemble_prediction"] = target[chosen]
    explanation_rows["surrogate_prediction"] = explained_prediction
    explanation_rows["shap_additive_prediction"] = additive_prediction
    explanation_rows["shap_additivity_abs_error"] = additivity_error

    np.savez_compressed(
        output_dir / "shap_values.npz",
        values=shap_values,
        feature_values=explain_matrix,
        feature_names=np.asarray(feature_names),
        sample_rows=explanation_rows["sample_row"].to_numpy(),
        base_value=np.asarray(base_value, dtype=np.float32),
    )
    joblib.dump(model, output_dir / "shap_surrogate.joblib", compress=3)
    fidelity.update(
        {
            "explain_rows": explain_count,
            "feature_count": len(feature_names),
            "feature_representation": {
                "flow": "all 48 historical steps for inflow and outflow",
                "calendar": "seven channels at the final context timestep only",
                "static": "33 leakage-free node features",
            },
            "shap_method": "TreeSHAP on held-out surrogate rows",
            "shap_base_value": base_value,
            "mean_additivity_abs_error": float(additivity_error.mean()),
            "max_additivity_abs_error": float(additivity_error.max()),
            "interpretation_boundary": (
                "SHAP explains the surrogate approximation of the ensemble, not a direct "
                "gradient through the four AGFormer checkpoints. Interpret only alongside fidelity."
            ),
        }
    )
    return feature_frame, group_frame, validation_rows, fidelity


def _weather_label(code: float | int | None) -> str:
    if code is None or not math.isfinite(float(code)):
        return "unknown"
    value = int(round(float(code)))
    if value in {0}:
        return "clear"
    if value in {1, 2, 3}:
        return "cloudy"
    if value in {45, 48}:
        return "fog"
    if 51 <= value <= 67 or 80 <= value <= 82:
        return "rain"
    if 71 <= value <= 77 or 85 <= value <= 86:
        return "snow"
    if 95 <= value <= 99:
        return "thunderstorm"
    return f"code_{value}"


def _load_external_events(path_value: str | Path | None, timezone_name: str) -> pd.DataFrame:
    columns = ["start_time", "end_time", "title", "category", "source", "url"]
    if not path_value:
        return pd.DataFrame(columns=columns)
    path = resolve_path(path_value)
    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"External event file {path} is missing columns: {missing}")
    for column in ["start_time", "end_time"]:
        parsed = pd.to_datetime(frame[column])
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize(timezone_name)
        frame[column] = parsed.dt.tz_convert("UTC")
    return frame


def _match_external_events(
    external: pd.DataFrame,
    *,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    tolerance: pd.Timedelta,
    event_row: float,
    event_col: float,
) -> pd.DataFrame:
    """Match time first, then apply optional grid-centroid spatial constraints."""
    if external.empty:
        return external.copy()
    matches = external.loc[
        (external["start_time"] <= end_utc + tolerance)
        & (external["end_time"] >= start_utc - tolerance)
    ].copy()
    scopes = []
    keep = []
    spatial_columns = {"center_row", "center_col", "radius_cells"}
    has_spatial_schema = spatial_columns.issubset(matches.columns)
    for _, match in matches.iterrows():
        spatial_values = [match.get(column) for column in spatial_columns]
        has_spatial_values = has_spatial_schema and all(pd.notna(value) for value in spatial_values)
        if has_spatial_values and math.isfinite(event_row) and math.isfinite(event_col):
            distance = math.dist(
                (event_row, event_col),
                (float(match["center_row"]), float(match["center_col"])),
            )
            keep.append(distance <= float(match["radius_cells"]))
            scopes.append("spatiotemporal")
        else:
            keep.append(True)
            scopes.append("citywide_time_cooccurrence")
    matches = matches.loc[np.asarray(keep, dtype=bool)].copy()
    matches["_match_scope"] = np.asarray(scopes, dtype=object)[np.asarray(keep, dtype=bool)]
    return matches


def _event_weather_rows(
    weather: pd.DataFrame,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> pd.DataFrame:
    selected = weather.loc[(weather["utc"] >= start_utc) & (weather["utc"] <= end_utc)]
    if not selected.empty:
        return selected
    distances = (weather["utc"] - start_utc).abs()
    nearest = int(distances.argmin())
    if distances.iloc[nearest] <= pd.Timedelta(minutes=30):
        return weather.iloc[[nearest]]
    return weather.iloc[0:0]


def candidate_cause(
    *,
    event: dict[str, Any],
    weather_rows: pd.DataFrame,
    local_start: pd.Timestamp,
    holidays: set,
    makeup_workdays: set,
    weather_thresholds: dict[str, float],
    external_matches: pd.DataFrame,
) -> dict[str, Any]:
    """Rank auditable cause candidates without claiming causal proof."""
    signals: list[dict[str, Any]] = []
    precipitation = float(weather_rows["precipitation"].max()) if len(weather_rows) else math.nan
    rain = float(weather_rows["rain_1h"].max()) if len(weather_rows) else math.nan
    snow = float(weather_rows["snow_1h"].max()) if len(weather_rows) else math.nan
    wind = float(weather_rows["wind_speed"].max()) if len(weather_rows) else math.nan
    temperature = float(weather_rows["temperature"].mean()) if len(weather_rows) else math.nan
    weather_code = (
        float(weather_rows["weather_code"].mode().iloc[0])
        if len(weather_rows) and weather_rows["weather_code"].notna().any()
        else math.nan
    )
    if (math.isfinite(precipitation) and precipitation >= 1.0) or (
        math.isfinite(rain) and rain >= 1.0
    ):
        signals.append(
            {
                "cause": "heavy_precipitation",
                "score": 0.85,
                "evidence": f"precipitation_max={precipitation:.2f}, rain_1h_max={rain:.2f}",
                "evidence_type": "weather",
            }
        )
    if math.isfinite(snow) and snow > 0:
        signals.append(
            {
                "cause": "snow",
                "score": 0.90,
                "evidence": f"snow_1h_max={snow:.2f}",
                "evidence_type": "weather",
            }
        )
    if math.isfinite(wind) and wind >= weather_thresholds["wind_p95"]:
        signals.append(
            {
                "cause": "strong_wind",
                "score": 0.65,
                "evidence": f"wind_max={wind:.2f} >= historical_p95",
                "evidence_type": "weather",
            }
        )
    if math.isfinite(temperature) and (
        temperature <= weather_thresholds["temperature_p05"]
        or temperature >= weather_thresholds["temperature_p95"]
    ):
        signals.append(
            {
                "cause": "extreme_temperature",
                "score": 0.55,
                "evidence": f"temperature_mean={temperature:.2f} outside historical_p05_p95",
                "evidence_type": "weather",
            }
        )

    local_date = local_start.date()
    is_holiday = local_date in holidays
    is_makeup_workday = local_date in makeup_workdays
    is_weekend = local_start.dayofweek >= 5
    if is_holiday:
        signals.append(
            {
                "cause": "official_holiday",
                "score": 0.80,
                "evidence": f"local_date={local_date} is in BJ_Holiday.txt",
                "evidence_type": "calendar",
            }
        )
    elif is_makeup_workday:
        signals.append(
            {
                "cause": "makeup_workday",
                "score": 0.65,
                "evidence": f"local_date={local_date} is an official makeup workday",
                "evidence_type": "calendar",
            }
        )
    elif is_weekend:
        signals.append(
            {
                "cause": "weekend_pattern",
                "score": 0.35,
                "evidence": f"local_date={local_date} is weekend",
                "evidence_type": "calendar",
            }
        )

    hour = local_start.hour + local_start.minute / 60
    road_density = float(event.get("road_density_mean", math.nan))
    if (
        (7 <= hour <= 10 or 16.5 <= hour <= 20)
        and math.isfinite(road_density)
        and road_density >= 0.5
    ):
        signals.append(
            {
                "cause": "commute_high_road_context",
                "score": 0.40,
                "evidence": f"local_hour={hour:.1f}, road_density_mean={road_density:.3f}",
                "evidence_type": "spatiotemporal_context",
            }
        )
    event_type = str(event.get("event_type", "unknown"))
    for _, match in external_matches.iterrows():
        scope = str(match.get("_match_scope", "citywide_time_cooccurrence"))
        signals.append(
            {
                "cause": (
                    "external_reported_event"
                    if scope == "spatiotemporal"
                    else "external_citywide_time_cooccurrence"
                ),
                "score": 0.95 if scope == "spatiotemporal" else 0.60,
                "evidence": f"{match['title']} ({match['source']}), scope={scope}",
                "evidence_type": (
                    "external_event_spatiotemporal"
                    if scope == "spatiotemporal"
                    else "external_event_time_only"
                ),
            }
        )
    signals.sort(key=lambda item: (-float(item["score"]), item["cause"]))
    if signals:
        primary = signals[0]
    else:
        primary = {
            "cause": "no_external_context_match",
            "score": 0.0,
            "evidence": "no configured external or context threshold matched",
            "evidence_type": "none",
        }
        signals.append(primary)
    weather_name = _weather_label(weather_code)
    external_status = "matched" if len(external_matches) else "not_matched"
    return {
        "primary_context_candidate": primary["cause"],
        "evidence_priority": float(primary["score"]),
        "primary_evidence_type": primary["evidence_type"],
        "context_candidates_json": json.dumps(signals, ensure_ascii=False),
        "context_status": "candidate_evidence_not_causal_proof",
        "detector_morphology_label": event_type,
        "local_start_time": local_start.isoformat(),
        "is_holiday": bool(is_holiday),
        "is_weekend": bool(is_weekend),
        "is_makeup_workday": bool(is_makeup_workday),
        "weather_rows": int(len(weather_rows)),
        "weather_summary": weather_name,
        "temperature_mean_c": temperature,
        "wind_speed_max": wind,
        "precipitation_max": precipitation,
        "rain_1h_max": rain,
        "snow_1h_max": snow,
        "external_event_status": external_status,
        "external_event_titles": "; ".join(external_matches["title"].astype(str))
        if len(external_matches)
        else "",
        "external_event_sources": "; ".join(external_matches["source"].astype(str))
        if len(external_matches)
        else "",
        "external_event_urls": "; ".join(external_matches["url"].astype(str))
        if len(external_matches)
        else "",
        "external_match_scopes": "; ".join(
            external_matches.get("_match_scope", pd.Series(dtype=str)).astype(str)
        )
        if len(external_matches)
        else "",
    }


def analyze_anomaly_causes(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    events = pd.read_csv(resolve_path(cfg["events"]))
    weather = pd.read_csv(resolve_path(cfg["weather"]))
    weather["utc"] = pd.to_datetime(weather["utc"], utc=True)
    timezone_name = str(cfg.get("timezone", "Asia/Shanghai"))
    external = _load_external_events(cfg.get("external_events"), timezone_name)
    holidays = load_bj_holidays(resolve_path(cfg["holidays"]))
    makeup_workdays = load_bj_makeup_workdays()
    thresholds = {
        "temperature_p05": float(weather["temperature"].quantile(0.05)),
        "temperature_p95": float(weather["temperature"].quantile(0.95)),
        "wind_p95": float(weather["wind_speed"].quantile(0.95)),
    }
    tolerance = pd.Timedelta(hours=float(cfg.get("external_event_tolerance_hours", 3)))
    rows: list[dict[str, Any]] = []
    for event in events.to_dict(orient="records"):
        start_utc = pd.Timestamp(event["start_time"], tz="UTC")
        end_utc = pd.Timestamp(event["end_time"], tz="UTC")
        local_start = start_utc.tz_convert(timezone_name)
        weather_rows = _event_weather_rows(weather, start_utc, end_utc)
        matches = _match_external_events(
            external,
            start_utc=start_utc,
            end_utc=end_utc,
            tolerance=tolerance,
            event_row=float(event.get("main_centroid_row", math.nan)),
            event_col=float(event.get("main_centroid_col", math.nan)),
        )
        cause = candidate_cause(
            event=event,
            weather_rows=weather_rows,
            local_start=local_start,
            holidays=holidays,
            makeup_workdays=makeup_workdays,
            weather_thresholds=thresholds,
            external_matches=matches,
        )
        rows.append({**event, **cause})
    cause_frame = pd.DataFrame(rows)
    counts = (
        cause_frame.groupby(
            ["primary_context_candidate", "primary_evidence_type"], as_index=False
        )
        .agg(
            event_count=("event_id", "count"),
            mean_max_score=("max_score", "mean"),
            mean_evidence_priority=("evidence_priority", "mean"),
        )
        .sort_values(["event_count", "mean_max_score"], ascending=False)
        .reset_index(drop=True)
    )
    summary = {
        "event_count": int(len(cause_frame)),
        "weather_coverage_rate": float((cause_frame["weather_rows"] > 0).mean()),
        "holiday_event_count": int(cause_frame["is_holiday"].sum()),
        "weekend_event_count": int(cause_frame["is_weekend"].sum()),
        "makeup_workday_event_count": int(cause_frame["is_makeup_workday"].sum()),
        "external_events_configured": bool(cfg.get("external_events")),
        "external_event_match_count": int(
            (cause_frame["external_event_status"] == "matched").sum()
        ),
        "weather_thresholds": thresholds,
        "cause_semantics": (
            "Ranked external/context evidence priorities, not calibrated probabilities or "
            "causal claims. Detector morphology remains a separate field. News is only "
            "used when an explicit external_events CSV is configured."
        ),
    }
    return cause_frame, counts, summary


def _save_attention_figures(
    matrix_frame: pd.DataFrame,
    node_frame: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    figure_paths: list[str] = []
    layers = sorted(matrix_frame["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(6.2 * len(layers), 5.2), squeeze=False)
    for axis, layer in zip(axes[0], layers):
        selected = matrix_frame.loc[matrix_frame["layer"] == layer]
        matrix = selected.pivot(index="query_step", columns="key_step", values="attention_weight")
        image = axis.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap="magma")
        axis.set_title(f"AGFormer layer {layer}")
        axis.set_xlabel("Key input step (0=oldest)")
        axis.set_ylabel("Query input step")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Nonlinear branch: native temporal self-attention "
        "(excludes linear residual head)"
    )
    fig.tight_layout()
    path = output_dir / "attention_temporal_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path.name)

    recent = node_frame.pivot(index="row", columns="col", values="recent_attention_mass")
    source = node_frame.pivot(index="row", columns="col", values="adaptive_source_influence")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, values, title, cmap in [
        (axes[0], recent, "Recent 3h temporal attention mass", "viridis"),
        (
            axes[1],
            source,
            "Nonlinear branch adaptive adjacency source influence",
            "plasma",
        ),
    ]:
        image = axis.imshow(values.to_numpy(), origin="upper", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("Grid column")
        axis.set_ylabel("Grid row")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Nonlinear Transformer-GCN branch (excludes linear residual head)")
    fig.tight_layout()
    path = output_dir / "attention_spatial_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path.name)
    return figure_paths


def _save_shap_figures(
    feature_frame: pd.DataFrame,
    group_frame: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    paths: list[str] = []
    top = feature_frame.nsmallest(20, "importance_rank").sort_values("mean_abs_shap")
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(top["feature"], top["mean_abs_shap"], color="#2563EB")
    axis.set_xlabel("Mean |SHAP value| (flow units)")
    axis.set_title("Top 20 TreeSHAP features for ensemble h1 behavior")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "shap_feature_importance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)

    groups = group_frame.sort_values("mean_abs_shap")
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(groups["feature_group"], groups["importance_share"] * 100, color="#059669")
    axis.set_xlabel("Share of summed mean |SHAP| (%)")
    axis.set_title("Grouped TreeSHAP importance")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "shap_group_importance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)
    return paths


def _save_anomaly_figures(
    cause_frame: pd.DataFrame,
    cause_counts: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    paths: list[str] = []
    counts = cause_counts.sort_values("event_count")
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(counts["primary_context_candidate"], counts["event_count"], color="#D97706")
    axis.set_xlabel("Event count")
    axis.set_title("Primary external/context evidence candidates")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "anomaly_cause_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)

    fig, axis = plt.subplots(figsize=(8, 5.5))
    scatter = axis.scatter(
        cause_frame["precipitation_max"].fillna(0),
        cause_frame["max_score"],
        c=cause_frame["evidence_priority"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    axis.set_xlabel("Event-period max precipitation")
    axis.set_ylabel("Fusion max score")
    axis.set_title("Weather context versus anomaly score")
    fig.colorbar(scatter, ax=axis, label="Evidence priority (rule-based)")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path = output_dir / "anomaly_weather_context.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path.name)
    return paths


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    selected = frame.loc[:, columns].head(max_rows).copy()
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
    attention_last: pd.DataFrame,
    shap_features: pd.DataFrame,
    shap_groups: pd.DataFrame,
    cause_counts: pd.DataFrame,
) -> str:
    top_lags = (
        attention_last.sort_values(["layer", "attention_weight"], ascending=[True, False])
        .groupby("layer", as_index=False, group_keys=False)
        .head(5)
    )
    lag_table = _markdown_table(
        top_lags, ["layer", "lag_steps", "lag_hours", "attention_weight"], max_rows=10
    )
    feature_table = _markdown_table(
        shap_features,
        ["importance_rank", "feature", "feature_group", "mean_abs_shap", "mean_signed_shap"],
        max_rows=15,
    )
    group_table = _markdown_table(
        shap_groups,
        ["feature_group", "feature_count", "mean_abs_shap", "importance_share"],
        max_rows=20,
    )
    cause_table = _markdown_table(
        cause_counts,
        ["primary_context_candidate", "primary_evidence_type", "event_count", "mean_max_score"],
        max_rows=20,
    )
    attention = summary["attention"]
    shap_summary = summary["shap"]
    anomaly = summary["anomaly_causes"]
    return f"""# Week7 第三阶段：模型可解释性分析

生成时间：{summary['generated_at']}

## 1. 范围与方法

- 解释对象：Week4 最终 AGFormer gated+linear 四随机种子均值 ensemble；
- 数据规则：TaxiBJ、4022 个 gap-aware 测试窗口、48 步输入、33 维无泄漏静态特征；
- 注意力部分直接提取四个 checkpoint 非线性 Transformer-GCN 分支的真实、head-averaged 时序注意力，并分析该分支的 learned adaptive adjacency；
- 最终 checkpoint 启用了并行 linear residual head，它绕过 Transformer/GCN，因此注意力图不覆盖完整最终输出；
- SHAP 部分使用时间留出的 ExtraTrees 模型行为代理解释保存的完整 ensemble h1 预测，包含 linear head 的效果；7 个日历特征只表示目标前最后一个上下文时刻，96 个流量特征保留完整 48 步；
- 异常原因部分只输出天气、日历、形态和可选外部事件的候选证据，不将相关性写成因果结论。

## 2. 时间与空间注意力

这是覆盖 4 个日期的定性抽样分析，不代表全部 4022 个测试窗口的注意力分布。抽样测试窗口：{attention['sample_window_indices']}。

![Temporal attention](attention_temporal_heatmap.png)

最后 query 最关注的输入滞后：

{lag_table}

![Spatial attention summary](attention_spatial_summary.png)

- 最近 {attention['recent_hours']:.1f} 小时平均注意力权重占比（attention mass）：{attention['mean_recent_attention_mass']:.4f}；
- 平均时序注意力归一化熵：{attention['mean_temporal_attention_entropy']:.4f}；
- adaptive adjacency 平均行熵：{attention['mean_adaptive_row_entropy']:.4f}；
- 每个目标节点 top-k adaptive 权重平均集中度：{attention['mean_adaptive_top_k_concentration']:.4f}。

这里的时间与空间图只解释非线性 Transformer-GCN 分支。AGFormer 同时使用预定义 spatial GCN，并通过并行 linear residual head 直接将历史 inflow 映射到输出；因此不能把这些图当成完整最终预测的贡献分解。Attention 也不等同于因果重要性。

## 3. TreeSHAP 特征重要性

代理模型在时间留出数据上的拟合度：R²={shap_summary['validation_r2']:.4f}，MAE={shap_summary['validation_mae_flow_units']:.4f} flow units。SHAP 加法平均绝对误差为 {shap_summary['mean_additivity_abs_error']:.6f}。

![SHAP feature importance](shap_feature_importance.png)

{feature_table}

![SHAP group importance](shap_group_importance.png)

{group_table}

这些 SHAP 值解释的是“代理模型如何近似最终 ensemble 的完整 h1 输出”，其中包含 linear residual head。只有在代理 R² 足够高时，才适合把全局排序视为完整 AGFormer 行为的近似；它不是对四个 checkpoint 的直接梯度 SHAP。

## 4. 异常外部/上下文证据分析

共分析 {anomaly['event_count']} 个融合异常事件，天气覆盖率 {anomaly['weather_coverage_rate']:.2%}；节假日事件 {anomaly['holiday_event_count']} 个，周末事件 {anomaly['weekend_event_count']} 个，调休工作日事件 {anomaly['makeup_workday_event_count']} 个。
周末与调休工作日是两个独立标记，计数可能重叠。

![Anomaly context distribution](anomaly_cause_distribution.png)

{cause_table}

![Anomaly weather context](anomaly_weather_context.png)

外部新闻/活动表已配置：{anomaly['external_events_configured']}；匹配事件数：{anomaly['external_event_match_count']}。检测器的 `event_type` 仅作为 morphology label 单独保留，不再作为外部原因。`evidence_priority` 是手工证据排序，不是校准概率。当前未配置新闻表时，结果不会声称任何新闻事件已经得到验证；未来只有带网格中心/半径的记录才是时空匹配，否则明确标记为全市时间共现。

## 5. 解释边界

1. 注意力权重只说明非线性 Transformer-GCN 分支的信息聚合方式，不覆盖并行 linear residual head，也不天然等于特征贡献或因果关系；
2. adaptive adjacency 不包含预定义 spatial GCN 或 linear residual head 的贡献；
3. TreeSHAP 解释的是经过时间留出验证的模型行为代理，应结合代理 R² 阅读；
4. TaxiBJ 没有权威异常真标签，天气、节假日和时空上下文只能提供候选证据；
5. 新闻或活动必须通过带来源和 URL 的外部事件 CSV 显式注入；没有空间字段时只算全市时间共现，系统不会自动编造因果匹配。
"""


def system_environment(device: torch.device) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
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
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "allocated_peak_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2),
        }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "shap": shap.__version__,
        "dependencies": {
            package: metadata.version(package)
            for package in [
                "numpy",
                "pandas",
                "scikit-learn",
                "joblib",
                "matplotlib",
                "shap",
                "PyYAML",
            ]
        },
        "device": str(device),
        "gpu": gpu,
        "git_commit": commit,
        "git_dirty": dirty,
    }


def run_interpretability(
    cfg: dict[str, Any],
    *,
    config_path: str | Path,
    device_override: str | None = None,
) -> dict[str, Any]:
    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 2026))
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = device_override or cfg.get("device", "cuda")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Week7 interpretability requested CUDA, but CUDA is unavailable")
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_section = cfg["model"]
    tags = list(model_section["checkpoint_tags"])
    checkpoint_root = model_section.get("checkpoint_root", "outputs/checkpoints")
    first_checkpoint = torch.load(
        resolve_path(checkpoint_root) / tags[0] / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_cfg = first_checkpoint["cfg"]
    mean = float(first_checkpoint["mean"])
    std = float(first_checkpoint["std"])

    attention_cfg = cfg["attention"]
    attention_matrix, attention_last, attention_nodes, adaptive_edges, attention_summary = (
        extract_attention(
            model_cfg,
            tags,
            checkpoint_root=checkpoint_root,
            split=model_section.get("split", "test"),
            mean=mean,
            std=std,
            sample_windows=int(attention_cfg.get("sample_windows", 4)),
            recent_steps=int(attention_cfg.get("recent_steps", 6)),
            top_edges_per_node=int(attention_cfg.get("top_edges_per_node", 5)),
            device=device,
        )
    )
    attention_matrix.to_csv(output_dir / "attention_temporal_matrix.csv", index=False)
    attention_last.to_csv(output_dir / "attention_last_query_by_lag.csv", index=False)
    attention_nodes.to_csv(output_dir / "attention_node_summary.csv", index=False)
    adaptive_edges.to_csv(output_dir / "adaptive_top_edges.csv", index=False)

    shap_cfg = cfg["shap"]
    matrix, target, surrogate_rows, feature_names, feature_groups = build_surrogate_rows(
        scores_path=model_section["scores"],
        model_cfg=model_cfg,
        sample_rows=int(shap_cfg.get("sample_rows", 30000)),
        seed=seed,
    )
    shap_features, shap_groups, surrogate_validation, shap_summary = fit_tree_shap(
        matrix,
        target,
        surrogate_rows,
        feature_names,
        feature_groups,
        cfg=shap_cfg,
        seed=seed,
        output_dir=output_dir,
    )
    shap_features.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    shap_groups.to_csv(output_dir / "shap_group_importance.csv", index=False)
    surrogate_validation.to_csv(output_dir / "shap_surrogate_validation.csv", index=False)

    cause_frame, cause_counts, cause_summary = analyze_anomaly_causes(cfg["anomaly_causes"])
    cause_frame.to_csv(output_dir / "anomaly_cause_analysis.csv", index=False)
    cause_counts.to_csv(output_dir / "anomaly_cause_summary.csv", index=False)

    figures = []
    figures.extend(_save_attention_figures(attention_matrix, attention_nodes, output_dir))
    figures.extend(_save_shap_figures(shap_features, shap_groups, output_dir))
    figures.extend(_save_anomaly_figures(cause_frame, cause_counts, output_dir))
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    input_paths: dict[str, str | Path] = {
        "scores": model_section["scores"],
        "taxibj": "data/processed/taxibj_clean.npz",
        "static_features": model_cfg.get("static_path", "data/processed/node_features_bj.npy"),
        "spatial_graph": "data/processed/graph_spatial_bj.pt",
        "anomaly_events": cfg["anomaly_causes"]["events"],
        "weather": cfg["anomaly_causes"]["weather"],
        "holidays": cfg["anomaly_causes"]["holidays"],
    }
    if cfg["anomaly_causes"].get("external_events"):
        input_paths["external_events"] = cfg["anomaly_causes"]["external_events"]
    summary = {
        "schema_version": 1,
        "tag": cfg["tag"],
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config_path": str(config_path),
        "objective": cfg.get("objective"),
        "hypothesis": cfg.get("hypothesis"),
        "model": {
            "checkpoint_tags": tags,
            "checkpoint_hashes": _clean_checkpoint_hashes(tags, checkpoint_root),
            "data_rules": model_section.get("data_rules", {}),
        },
        "attention": attention_summary,
        "shap": shap_summary,
        "anomaly_causes": cause_summary,
        "environment": system_environment(device),
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
                ROOT / "src/interpretability/week7.py",
                ROOT / "scripts/run_week7_interpretability.py",
                resolve_path(config_path),
            ]
        },
        "outputs": {
            "attention_temporal_matrix": "attention_temporal_matrix.csv",
            "attention_last_query_by_lag": "attention_last_query_by_lag.csv",
            "attention_node_summary": "attention_node_summary.csv",
            "adaptive_top_edges": "adaptive_top_edges.csv",
            "shap_feature_importance": "shap_feature_importance.csv",
            "shap_group_importance": "shap_group_importance.csv",
            "shap_surrogate_validation": "shap_surrogate_validation.csv",
            "shap_values": "shap_values.npz",
            "shap_surrogate": "shap_surrogate.joblib",
            "anomaly_cause_analysis": "anomaly_cause_analysis.csv",
            "anomaly_cause_summary": "anomaly_cause_summary.csv",
            "figures": figures,
            "report": "week7_interpretability_report.md",
            "artifact_manifest": "artifact_manifest.json",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = build_report(summary, attention_last, shap_features, shap_groups, cause_counts)
    (output_dir / "week7_interpretability_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at": summary["generated_at"],
        "semantics": "SHA256 for every formal artifact except this manifest itself",
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        },
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
