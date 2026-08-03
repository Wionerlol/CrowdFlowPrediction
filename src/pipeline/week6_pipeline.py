"""Week 6 end-to-end data/inference/anomaly/alert pipelines.

Two modes are supported:

- batch: orchestrates preprocessing/feature validation, Week 5 anomaly
  detection, and Week 6 alert generation;
- realtime: runs a small rolling micro-batch over the latest available windows
  using existing checkpoints and Week 5 calibration thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.run_week5_anomaly_detection import (
    flatten_channel,
    load_config as load_week5_config,
    load_road_density,
    set_seed,
    split_lengths,
    train_or_load_reconstruction,
)
from src.alerts.levels import alert_levels_from_config, annotate_event_alerts, summarize_alerts
from src.anomaly.fusion import confidence_from_threshold, fuse_weighted_vote
from src.anomaly.prediction_error import load_city_data, predict_ensemble_horizon
from src.anomaly.reconstruction import score_transformer_targets, score_vae
from src.anomaly.statistical import (
    build_events,
    fit_slot_stats,
    generate_weak_labels,
    score_statistical,
    slot_indices_from_timestamps,
    summarize_event_types,
)


Array = np.ndarray

EVENT_COLUMNS = [
    "event_id", "start_index", "end_index", "start_time", "end_time",
    "duration_steps", "duration_hours", "event_type", "severity", "max_score",
    "max_cells", "union_cells", "cluster_count", "main_centroid_row",
    "main_centroid_col", "main_aspect_ratio", "road_density_mean",
    "in_out_imbalance", "duration_trend_slope", "mean_inflow", "max_inflow",
    "mean_outflow", "max_outflow", "max_z", "max_signed_z",
    "statistical_score", "reconstruction_score", "prediction_score",
    "dominant_method", "max_reconstruction_error", "max_prediction_residual",
    "top_cells",
]

ALERT_COLUMNS = EVENT_COLUMNS + [
    "week5_severity", "event_type_zh", "alert_level_id", "alert_level",
    "alert_level_zh", "alert_color", "alert_ui_behavior", "alert_popup",
    "alert_sound", "alert_action_hint", "alert_message",
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class StageResult:
    name: str
    status: str
    started_at: str
    ended_at: str
    duration_seconds: float
    artifacts: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "artifacts": self.artifacts,
            "message": self.message,
        }


class Week6Pipeline:
    """End-to-end Week 6 pipeline runner."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.cfg = cfg
        self.config_path = str(config_path) if config_path is not None else None
        self.output_dir = Path(cfg.get("output_dir", "outputs/week6_pipeline"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stage_results: list[StageResult] = []

    def _record_stage(
        self,
        name: str,
        started_at: str,
        start_time: float,
        *,
        status: str,
        artifacts: dict[str, Any] | None = None,
        message: str = "",
    ) -> StageResult:
        result = StageResult(
            name=name,
            status=status,
            started_at=started_at,
            ended_at=utc_now(),
            duration_seconds=float(time.time() - start_time),
            artifacts=artifacts or {},
            message=message,
        )
        self.stage_results.append(result)
        return result

    def _validate_artifacts(self, stage_name: str, paths: list[str]) -> StageResult:
        started_at = utc_now()
        start = time.time()
        artifacts: dict[str, Any] = {}
        missing = []
        for raw_path in paths:
            path = Path(raw_path)
            if path.exists():
                artifacts[str(path)] = {"exists": True, "size_bytes": int(path.stat().st_size)}
            else:
                artifacts[str(path)] = {"exists": False}
                missing.append(str(path))
        if missing:
            return self._record_stage(
                stage_name,
                started_at,
                start,
                status="failed",
                artifacts=artifacts,
                message=f"Missing required artifacts: {missing}",
            )
        return self._record_stage(
            stage_name,
            started_at,
            start,
            status="ok",
            artifacts=artifacts,
            message="required artifacts validated",
        )

    def _run_commands(
        self,
        stage_name: str,
        commands: list[list[str]],
        *,
        log_dir: Path,
    ) -> StageResult:
        started_at = utc_now()
        start = time.time()
        log_dir.mkdir(parents=True, exist_ok=True)
        logs = []
        for i, command in enumerate(commands, start=1):
            log_path = log_dir / f"{stage_name}_{i}.log"
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            logs.append(str(log_path))
            if proc.returncode != 0:
                return self._record_stage(
                    stage_name,
                    started_at,
                    start,
                    status="failed",
                    artifacts={"logs": logs},
                    message=f"command failed with return code {proc.returncode}: {command}",
                )
        return self._record_stage(
            stage_name,
            started_at,
            start,
            status="ok",
            artifacts={"logs": logs},
            message=f"ran {len(commands)} command(s)",
        )

    def _run_pipeline_commands_or_validate(self, stage_name: str, stage_cfg: dict[str, Any]) -> StageResult:
        commands = stage_cfg.get("commands", [])
        if bool(stage_cfg.get("run", False)):
            command_result = self._run_commands(
                stage_name,
                commands,
                log_dir=self.output_dir / "logs",
            )
            if command_result.status != "ok":
                return command_result
            required = list(stage_cfg.get("required_artifacts", []))
            if required:
                validation = self._validate_artifacts(f"{stage_name}_artifacts", required)
                if validation.status != "ok":
                    return validation
            return command_result
        return self._validate_artifacts(stage_name, list(stage_cfg.get("required_artifacts", [])))

    @staticmethod
    def _require_ok(result: StageResult) -> None:
        if result.status != "ok":
            raise RuntimeError(f"Pipeline stage failed: {result.name}: {result.message}")

    def run_batch(self, *, skip_anomaly: bool = False, smoke: bool = False) -> dict[str, Any]:
        """Run or orchestrate the full offline batch pipeline."""

        self.stage_results = []
        batch_cfg = self.cfg.get("batch", {})
        self._require_ok(self._run_pipeline_commands_or_validate(
            "data_preprocessing",
            self.cfg.get("preprocessing", {}),
        ))
        self._require_ok(self._run_pipeline_commands_or_validate(
            "feature_engineering",
            self.cfg.get("feature_engineering", {}),
        ))

        anomaly_artifacts: dict[str, Any] = {}
        if not skip_anomaly and bool(batch_cfg.get("run_anomaly_detection", True)):
            started_at = utc_now()
            start = time.time()
            log_path = self.output_dir / "logs" / "batch_week5_anomaly.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-u",
                "scripts/run_week5_anomaly_detection.py",
                "--config",
                str(batch_cfg.get("week5_config", "configs/week5_anomaly.yaml")),
                "--device",
                self.device,
            ]
            if smoke:
                command.append("--smoke")
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            anomaly_artifacts = {"log": str(log_path)}
            status = "ok" if proc.returncode == 0 else "failed"
            stage = self._record_stage(
                "model_inference_and_anomaly_detection",
                started_at,
                start,
                status=status,
                artifacts=anomaly_artifacts,
                message=f"return_code={proc.returncode}",
            )
            self._require_ok(stage)
            if not smoke:
                self._require_ok(self._validate_artifacts(
                    "model_inference_and_anomaly_detection_artifacts",
                    list(batch_cfg.get("required_week5_artifacts", [])),
                ))
        else:
            self._require_ok(self._validate_artifacts(
                "model_inference_and_anomaly_detection",
                list(batch_cfg.get("required_week5_artifacts", [])),
            ))

        alerts_cfg_path = Path(batch_cfg.get("alerts_config", "configs/week6_alerts.yaml"))
        started_at = utc_now()
        start = time.time()
        log_path = self.output_dir / "logs" / "batch_week6_alerts.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "scripts/generate_week6_alerts.py",
            "--config",
            str(alerts_cfg_path),
        ]
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                command,
                cwd=Path.cwd(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        stage = self._record_stage(
            "alert_generation",
            started_at,
            start,
            status="ok" if proc.returncode == 0 else "failed",
            artifacts={"log": str(log_path)},
            message=f"return_code={proc.returncode}",
        )
        self._require_ok(stage)
        self._require_ok(self._validate_artifacts(
            "alert_generation_artifacts",
            [
                self.cfg.get("alerts_output", "outputs/week6_alerts/alerts.csv"),
                self.cfg.get("alerts_summary", "outputs/week6_alerts/alert_summary.json"),
            ],
        ))

        summary = {
            "mode": "batch",
            "config": self.config_path,
            "device": self.device,
            "started_at": self.stage_results[0].started_at if self.stage_results else None,
            "ended_at": utc_now(),
            "stages": [stage.as_dict() for stage in self.stage_results],
            "outputs": {
                "week5_summary": batch_cfg.get("week5_summary", "outputs/week5_anomaly/summary.json"),
                "week5_events": batch_cfg.get("week5_events", "outputs/week5_anomaly/events.csv"),
                "week6_alerts": self.cfg.get("alerts_output", "outputs/week6_alerts/alerts.csv"),
                "week6_alert_summary": self.cfg.get(
                    "alerts_summary", "outputs/week6_alerts/alert_summary.json"
                ),
            },
        }
        path = self.output_dir / "batch_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        return summary

    def run_realtime(self, *, window_count: int | None = None, window_start: int | None = None) -> dict[str, Any]:
        """Run a rolling micro-batch inference pipeline over latest windows."""

        self.stage_results = []
        realtime_cfg = self.cfg.get("realtime", {})
        output_dir = Path(realtime_cfg.get("output_dir", self.output_dir / "realtime"))
        output_dir.mkdir(parents=True, exist_ok=True)
        week5_cfg = load_week5_config(realtime_cfg.get("week5_config", "configs/week5_anomaly.yaml"))
        week5_cfg = dict(week5_cfg)
        week5_cfg.setdefault("reconstruction", {})["force_train"] = False
        week5_cfg["reconstruction"]["inference_only"] = True
        set_seed(int(week5_cfg.get("seed", 42)))
        device = torch.device(self.device)

        window_count = int(window_count or realtime_cfg.get("window_count", 24))
        window_start = int(window_start if window_start is not None else realtime_cfg.get("window_start", -window_count))
        if window_count <= 0:
            raise ValueError(f"window_count must be positive, got {window_count}")

        self._require_ok(self._validate_artifacts(
            "realtime_preprocessing_artifacts",
            list(realtime_cfg.get("required_artifacts", [])),
        ))

        calibration_scores = Path(realtime_cfg.get("calibration_scores", "outputs/week5_anomaly/scores.npz"))
        started_at = utc_now()
        start = time.time()
        if not calibration_scores.exists():
            self._record_stage(
                "load_calibration",
                started_at,
                start,
                status="failed",
                message=f"missing calibration scores: {calibration_scores}",
            )
            raise FileNotFoundError(calibration_scores)
        calibration = np.load(calibration_scores)
        required_calibration = ["pred_threshold", "vae_threshold", "transformer_threshold"]
        missing = [key for key in required_calibration if key not in calibration.files]
        if missing:
            self._record_stage(
                "load_calibration",
                started_at,
                start,
                status="failed",
                message=f"missing calibration arrays: {missing}",
            )
            raise RuntimeError(f"calibration scores missing arrays: {missing}")
        self._record_stage(
            "load_calibration",
            started_at,
            start,
            status="ok",
            artifacts={"scores": str(calibration_scores)},
            message="loaded Week5 thresholds",
        )

        started_at = utc_now()
        start = time.time()
        data, timestamps, splits = load_city_data(str(week5_cfg.get("city", "bj")))
        train_len, val_len, _ = split_lengths(splits)
        self._record_stage(
            "load_data",
            started_at,
            start,
            status="ok",
            artifacts={"shape": list(data.shape), "train_len": train_len, "val_len": val_len},
            message="loaded processed city data",
        )

        started_at = utc_now()
        start = time.time()
        pred_cfg = week5_cfg["prediction"]
        pred = predict_ensemble_horizon(
            list(pred_cfg["ensemble_tags"]),
            split="test",
            horizon=int(week5_cfg.get("horizon", 1)),
            batch_size=int(pred_cfg.get("batch_size", 8)),
            num_workers=int(pred_cfg.get("num_workers", 2)),
            device=device,
            max_windows=window_count,
            window_start=window_start,
        )
        test_idx = pred["target_indices"]
        self._record_stage(
            "model_inference",
            started_at,
            start,
            status="ok",
            artifacts={"windows": int(len(test_idx))},
            message="predicted selected realtime windows",
        )

        started_at = utc_now()
        start = time.time()
        stat_cfg = week5_cfg["statistical"]
        steps = int(week5_cfg["steps_per_day"])
        tz = int(week5_cfg["tz_offset"])
        train_slots = slot_indices_from_timestamps(timestamps[:train_len], steps_per_day=steps, tz_offset=tz)
        full_in = flatten_channel(data, 0)
        full_out = flatten_channel(data, 1)
        stats = fit_slot_stats(
            data[:train_len, 0],
            timestamps[:train_len],
            steps_per_day=steps,
            tz_offset=tz,
            min_std=float(stat_cfg.get("min_std", 1.0)),
            iqr_factor=float(stat_cfg.get("iqr_factor", 1.5)),
        )
        outflow_stats = fit_slot_stats(
            data[:train_len, 1],
            timestamps[:train_len],
            steps_per_day=steps,
            tz_offset=tz,
            min_std=float(stat_cfg.get("min_std", 1.0)),
            iqr_factor=float(stat_cfg.get("iqr_factor", 1.5)),
        )
        train_weak = generate_weak_labels(
            full_in[:train_len],
            train_slots,
            stats,
            outflow=full_out[:train_len],
            outflow_stats=outflow_stats,
            level="B",
            z_thresh=float(stat_cfg.get("sigma_z", 3.0)),
            min_high_cells=int(stat_cfg.get("weak_label_min_cells", 12)),
            outflow_z_thresh=float(stat_cfg.get("level_b_outflow_z", 2.5)),
            min_outflow_cells=int(stat_cfg.get("level_b_min_outflow_cells", 6)),
        )
        test_slots = slot_indices_from_timestamps(timestamps[test_idx], steps_per_day=steps, tz_offset=tz)
        stat_result = score_statistical(
            full_in[test_idx],
            test_slots,
            stats,
            sigma_z=float(stat_cfg.get("sigma_z", 3.0)),
        )
        stat_conf = confidence_from_threshold(stat_result.score, 1.0)
        self._record_stage(
            "statistical_detection",
            started_at,
            start,
            status="ok",
            artifacts={"train_weak_positive_count": int(train_weak.sum())},
            message="fitted train slot stats and scored realtime windows",
        )

        started_at = utc_now()
        start = time.time()
        rec_models = train_or_load_reconstruction(week5_cfg, data, train_len, val_len, train_weak, device)
        scaler = rec_models["scaler"]
        full_norm = scaler.transform(full_in)
        rec_cfg = week5_cfg["reconstruction"]
        vae_err = score_vae(
            rec_models["vae"],
            full_norm[test_idx],
            batch_size=int(rec_cfg["vae"].get("batch_size", 256)),
            device=device,
        )
        tf_err = score_transformer_targets(
            rec_models["transformer"],
            full_norm,
            test_idx,
            seq_len=int(rec_cfg["transformer"]["seq_len"]),
            batch_size=int(rec_cfg["transformer"].get("batch_size", 64)),
            device=device,
        )
        vae_conf = confidence_from_threshold(vae_err, calibration["vae_threshold"])
        tf_conf = confidence_from_threshold(tf_err, calibration["transformer_threshold"])
        recon_conf = ((vae_conf + tf_conf) / 2.0).astype(np.float32)
        reconstruction_error = ((vae_err + tf_err) / 2.0).astype(np.float32)
        self._record_stage(
            "reconstruction_detection",
            started_at,
            start,
            status="ok",
            artifacts={
                "vae_checkpoint": str(rec_models["vae_path"]),
                "transformer_checkpoint": str(rec_models["transformer_path"]),
            },
            message="loaded reconstruction checkpoints and scored windows",
        )

        pred_conf = confidence_from_threshold(pred["residual"], calibration["pred_threshold"])
        fused = fuse_weighted_vote(
            stat_conf=stat_conf,
            recon_conf=recon_conf,
            pred_conf=pred_conf,
            weights=week5_cfg["fusion"]["weights"],
            decision_threshold=float(week5_cfg["fusion"].get("decision_threshold", 0.5)),
        )

        started_at = utc_now()
        start = time.time()
        events_cfg = week5_cfg.get("events", {})
        use_spatial_continuity = bool(events_cfg.get("use_spatial_continuity", False))
        road_density = load_road_density(str(week5_cfg.get("city", "bj")), fused["mask"].shape[1])
        outflow = full_out[test_idx]
        events = build_events(
            pred["timestamps"],
            fused["mask"],
            fused["score"],
            grid_shape=tuple(week5_cfg["grid_shape"]),
            min_cells=int(week5_cfg["fusion"].get("min_event_cells", 6)),
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
            inflow=pred["true"],
            outflow=outflow,
            z=stat_result.z,
            statistical_score=stat_conf,
            reconstruction_score=recon_conf,
            reconstruction_error=reconstruction_error,
            prediction_score=pred_conf,
            prediction_residual=pred["residual"],
            road_density=road_density,
        )
        events_df = pd.DataFrame(events)
        if events_df.empty:
            events_df = pd.DataFrame(columns=EVENT_COLUMNS)
        events_path = output_dir / "events.csv"
        events_df.to_csv(events_path, index=False)
        self._record_stage(
            "anomaly_event_building",
            started_at,
            start,
            status="ok",
            artifacts={"events": str(events_path), "event_count": int(len(events_df))},
            message="fused scores and built realtime events",
        )

        started_at = utc_now()
        start = time.time()
        alert_cfg_path = Path(realtime_cfg.get("alerts_config", "configs/week6_alerts.yaml"))
        with open(alert_cfg_path, encoding="utf-8") as f:
            alert_cfg = yaml.safe_load(f) or {}
        if len(events_df):
            alerts_df = annotate_event_alerts(
                events_df,
                levels=alert_levels_from_config(alert_cfg.get("levels", {})),
            )
        else:
            alerts_df = pd.DataFrame(columns=ALERT_COLUMNS)
        alerts_path = output_dir / "alerts.csv"
        alerts_df.to_csv(alerts_path, index=False)
        self._record_stage(
            "alert_generation",
            started_at,
            start,
            status="ok",
            artifacts={"alerts": str(alerts_path), "alert_count": int(len(alerts_df))},
            message="mapped Week5 events to Week6 alert levels",
        )

        scores_path = output_dir / "scores.npz"
        np.savez_compressed(
            scores_path,
            timestamps=pred["timestamps"],
            target_indices=test_idx,
            true_flow=pred["true"],
            true_outflow=outflow,
            pred_flow=pred["pred"],
            prediction_residual=pred["residual"],
            stat_score=stat_result.score,
            stat_z=stat_result.z,
            reconstruction_error=reconstruction_error,
            recon_score=recon_conf,
            prediction_confidence=pred_conf,
            fusion_score=fused["score"],
            fusion_mask=fused["mask"],
        )
        time_summary = pd.DataFrame({
            "timestamp": pred["timestamps"].astype(str),
            "target_index": test_idx,
            "fusion_time_flag": fused["mask"].sum(axis=1) >= int(week5_cfg["fusion"].get("min_event_cells", 6)),
            "anomaly_cell_count": fused["mask"].sum(axis=1),
            "max_fusion_score": fused["score"].max(axis=1),
            "mean_true_flow": pred["true"].mean(axis=1),
            "max_prediction_residual": pred["residual"].max(axis=1),
        })
        time_summary_path = output_dir / "time_summary.csv"
        time_summary.to_csv(time_summary_path, index=False)

        alert_summary = summarize_alerts(alerts_df) if len(alerts_df) else {
            "total_alerts": 0,
            "by_level": {},
            "by_level_zh": {},
            "by_level_and_event_type": {},
            "popup_alerts": 0,
            "sound_alerts": 0,
        }
        summary = {
            "mode": "realtime",
            "config": self.config_path,
            "device": str(device),
            "city": week5_cfg.get("city", "bj"),
            "window_start": window_start,
            "requested_window_count": window_count,
            "actual_window_count": int(len(test_idx)),
            "time_range": {
                "start": str(pred["timestamps"][0]) if len(test_idx) else None,
                "end": str(pred["timestamps"][-1]) if len(test_idx) else None,
            },
            "fusion": week5_cfg["fusion"],
            "calibration": {
                "scores": str(calibration_scores),
                "threshold_arrays": required_calibration,
                "source": "Week5 validation-calibrated thresholds",
            },
            "checkpoints": {
                "vae": str(rec_models["vae_path"]),
                "transformer": str(rec_models["transformer_path"]),
                "vae_model_config_hash": rec_models["vae_meta"].get("model_config_hash"),
                "transformer_model_config_hash": rec_models["transformer_meta"].get("model_config_hash"),
                "vae_label_rule_hash": rec_models["vae_meta"].get("label_rule_hash"),
                "transformer_label_rule_hash": rec_models["transformer_meta"].get("label_rule_hash"),
            },
            "events": {
                "count": int(len(events_df)),
                "by_type": summarize_event_types(events) if len(events_df) else {},
                "path": str(events_path),
            },
            "alerts": {
                **alert_summary,
                "path": str(alerts_path),
            },
            "artifacts": {
                "scores": str(scores_path),
                "time_summary": str(time_summary_path),
                "events": str(events_path),
                "alerts": str(alerts_path),
            },
            "stages": [stage.as_dict() for stage in self.stage_results],
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )
        return summary
