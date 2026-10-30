"""Read-only query layer over Week 5/6 inference artifacts.

The API deliberately serves immutable snapshots produced by the pipeline.  It
does not load PyTorch models inside Web workers, which avoids duplicate GPU
allocation and makes multi-worker FastAPI deployment safe.  Re-running the
realtime pipeline atomically refreshes the files; this repository notices file
mtime changes and reloads them lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


class ArtifactNotReadyError(RuntimeError):
    """Raised when a pipeline artifact required by an endpoint is missing."""


class QueryNotFoundError(LookupError):
    """Raised when an artifact exists but the requested record does not."""


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_web_config(path: str | Path) -> dict[str, Any]:
    config_path = _resolve_path(path)
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _iso_timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.datetime64, pd.Timestamp, datetime)):
        return _iso_timestamp(value)
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


@dataclass(frozen=True)
class Region:
    grid_ids: tuple[int, ...] | None = None
    row_min: int | None = None
    row_max: int | None = None
    col_min: int | None = None
    col_max: int | None = None


class ArtifactCache:
    """Small mtime-aware cache for JSON, CSV and selected NPZ arrays."""

    def __init__(self) -> None:
        self._items: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._lock = RLock()

    @staticmethod
    def _mtime(path: Path) -> int:
        if not path.exists():
            raise ArtifactNotReadyError(
                f"Required artifact is missing: {path}. Run the Week6 pipeline first."
            )
        return path.stat().st_mtime_ns

    def json(self, path: Path) -> dict[str, Any]:
        key = ("json", path)
        mtime = self._mtime(path)
        with self._lock:
            if key in self._items and self._items[key][0] == mtime:
                return self._items[key][1]
            value = json.loads(path.read_text(encoding="utf-8"))
            self._items[key] = (mtime, value)
            return value

    def csv(self, path: Path) -> pd.DataFrame:
        key = ("csv", path)
        mtime = self._mtime(path)
        with self._lock:
            if key in self._items and self._items[key][0] == mtime:
                return self._items[key][1]
            value = pd.read_csv(path)
            self._items[key] = (mtime, value)
            return value

    def npz(self, path: Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
        selected = tuple(sorted(set(keys)))
        cache_key = ("npz", path, selected)
        mtime = self._mtime(path)
        with self._lock:
            if cache_key in self._items and self._items[cache_key][0] == mtime:
                return self._items[cache_key][1]
            with np.load(path, allow_pickle=False) as source:
                missing = [key for key in selected if key not in source.files]
                if missing:
                    raise ArtifactNotReadyError(f"Artifact {path} is missing arrays: {missing}")
                value = {key: source[key] for key in selected}
            self._items[cache_key] = (mtime, value)
            return value

    def derived(
        self,
        name: str,
        dependencies: Iterable[Path],
        factory: Callable[[], Any],
    ) -> Any:
        """Cache validated/derived views and invalidate on any source mtime."""

        paths = tuple(dependencies)
        token = tuple(self._mtime(path) for path in paths)
        key = ("derived", name, paths)
        with self._lock:
            if key in self._items and self._items[key][0] == token:
                return self._items[key][1]
            value = factory()
            self._items[key] = (token, value)
            return value


class Week6ArtifactRepository:
    """Query facade shared by FastAPI and inexpensive service tests."""

    SCORE_KEYS = ("timestamps", "true_flow", "pred_flow", "fusion_score", "fusion_mask")
    FORECAST_KEYS = ("city", "origin_time", "step_minutes", "timestamps", "pred_flow", "member_tags")

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.city = str(cfg.get("city", "bj"))
        artifacts = cfg.get("artifacts", {})
        self.paths = {name: _resolve_path(path) for name, path in artifacts.items()}
        self.cache = ArtifactCache()
        self.anomaly_cfg = cfg.get("anomaly", {})

    def _path(self, name: str) -> Path:
        if name not in self.paths:
            raise ArtifactNotReadyError(f"Artifact path '{name}' is not configured")
        return self.paths[name]

    def _grid_meta(self) -> pd.DataFrame:
        path = self._path("grid_meta")

        def build() -> pd.DataFrame:
            grid = self.cache.csv(path).copy()
            required = {"grid_id", "row", "col", "center_lon", "center_lat"}
            missing = required - set(grid.columns)
            if missing:
                raise ArtifactNotReadyError(f"grid metadata missing columns: {sorted(missing)}")
            return grid.sort_values("grid_id").reset_index(drop=True)

        return self.cache.derived("grid_meta_validated", [path], build)

    def _select_grid(self, region: Region | None) -> pd.DataFrame:
        grid = self._grid_meta()
        if region is None:
            return grid
        mask = np.ones(len(grid), dtype=bool)
        if region.grid_ids is not None:
            mask &= grid["grid_id"].isin(region.grid_ids).to_numpy()
        if region.row_min is not None:
            mask &= grid["row"].to_numpy() >= region.row_min
        if region.row_max is not None:
            mask &= grid["row"].to_numpy() <= region.row_max
        if region.col_min is not None:
            mask &= grid["col"].to_numpy() >= region.col_min
        if region.col_max is not None:
            mask &= grid["col"].to_numpy() <= region.col_max
        selected = grid.loc[mask].copy()
        if selected.empty:
            raise QueryNotFoundError("Region selection contains no grid cells")
        return selected

    def _scores(self, source: str) -> dict[str, np.ndarray]:
        key = "realtime_scores" if source == "realtime" else "historical_scores"
        if source not in {"realtime", "historical"}:
            raise ValueError("source must be realtime or historical")
        return self.cache.npz(self._path(key), self.SCORE_KEYS)

    @staticmethod
    def _timestamp_index(timestamps: np.ndarray, requested: str | None) -> int:
        if len(timestamps) == 0:
            raise QueryNotFoundError("Artifact contains no timestamps")
        if requested is None:
            return len(timestamps) - 1
        target = np.datetime64(pd.Timestamp(requested).to_datetime64(), "s")
        values = timestamps.astype("datetime64[s]")
        exact = np.flatnonzero(values == target)
        if len(exact):
            return int(exact[0])
        distances = np.abs(values.astype(np.int64) - target.astype(np.int64))
        return int(np.argmin(distances))

    def status(self) -> dict[str, Any]:
        artifacts = {}
        now = datetime.now(timezone.utc)
        for name, path in self.paths.items():
            exists = path.exists()
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) if exists else None
            artifacts[name] = {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "ready": exists,
                "size_bytes": int(path.stat().st_size) if exists else None,
                "modified_at": modified.isoformat() if modified else None,
                "age_seconds": (now - modified).total_seconds() if modified else None,
            }
        realtime_summary = None
        if self.paths.get("realtime_summary", Path("__missing__")).exists():
            realtime_summary = self.cache.json(self._path("realtime_summary"))
        return {
            "status": "ok" if artifacts and all(item["ready"] for item in artifacts.values()) else "degraded",
            "city": self.city,
            "server_time": now.isoformat(),
            "artifacts": artifacts,
            "realtime": realtime_summary,
        }

    def heatmap(
        self,
        *,
        source: str,
        timestamp: str | None,
        value_type: str,
        region: Region | None = None,
    ) -> dict[str, Any]:
        if value_type not in {"observed", "predicted"}:
            raise ValueError("value_type must be observed or predicted")
        arrays = self._scores(source)
        index = self._timestamp_index(arrays["timestamps"], timestamp)
        grid = self._select_grid(region)
        ids = grid["grid_id"].to_numpy(np.int64)
        observed = arrays["true_flow"][index, ids]
        predicted = arrays["pred_flow"][index, ids]
        score = arrays["fusion_score"][index, ids]
        mask = arrays["fusion_mask"][index, ids]
        selected_time = _iso_timestamp(arrays["timestamps"][index])
        values = observed if value_type == "observed" else predicted

        cells = grid[["grid_id", "row", "col", "center_lon", "center_lat"]].copy()
        cells["value"] = values
        cells["observed_flow"] = observed
        cells["predicted_flow"] = predicted
        cells["anomaly_score"] = score
        cells["is_anomaly"] = mask
        return {
            "city": self.city,
            "source": source,
            "timestamp": selected_time,
            "requested_timestamp": timestamp,
            "value_type": value_type,
            "cell_count": int(len(cells)),
            "summary": {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "total": float(np.sum(values)),
                "anomaly_cells": int(np.sum(mask)),
                "max_anomaly_score": float(np.max(score)),
            },
            "cells": _records(cells),
        }

    def forecast(
        self,
        *,
        start_time: str | None,
        end_time: str | None,
        horizon_steps: int,
        region: Region | None = None,
    ) -> dict[str, Any]:
        arrays = self.cache.npz(self._path("forecast_24h"), self.FORECAST_KEYS)
        timestamps = arrays["timestamps"].astype("datetime64[s]")
        time_mask = np.ones(len(timestamps), dtype=bool)
        if start_time is not None:
            time_mask &= timestamps >= np.datetime64(pd.Timestamp(start_time).to_datetime64(), "s")
        if end_time is not None:
            time_mask &= timestamps <= np.datetime64(pd.Timestamp(end_time).to_datetime64(), "s")
        positions = np.flatnonzero(time_mask)[: int(horizon_steps)]
        if len(positions) == 0:
            raise QueryNotFoundError("No forecast steps overlap the requested time range")

        grid = self._select_grid(region)
        ids = grid["grid_id"].to_numpy(np.int64)
        predictions = arrays["pred_flow"][positions][:, ids]
        frames = []
        for row_index, position in enumerate(positions):
            values = predictions[row_index]
            frames.append(
                {
                    "step": int(position + 1),
                    "timestamp": _iso_timestamp(timestamps[position]),
                    "mean": float(np.mean(values)),
                    "max": float(np.max(values)),
                    "total": float(np.sum(values)),
                    "values": values.astype(float, copy=False).tolist(),
                }
            )
        return {
            "city": str(arrays["city"].item()),
            "model": "AGFormer gated+linear 4-seed ensemble",
            "origin_time": _iso_timestamp(arrays["origin_time"].item()),
            "horizon_steps": int(len(frames)),
            "step_minutes": int(arrays["step_minutes"].item()),
            "member_tags": [str(item) for item in arrays["member_tags"].tolist()],
            "cells": _records(grid[["grid_id", "row", "col", "center_lon", "center_lat"]]),
            "frames": frames,
        }

    def _events(self, source: str) -> pd.DataFrame:
        if source not in {"realtime", "historical"}:
            raise ValueError("source must be realtime or historical")
        key = "realtime_alerts" if source == "realtime" else "historical_alerts"
        alerts_path = self._path(key)
        grid_path = self._path("grid_meta")

        def build() -> pd.DataFrame:
            frame = self.cache.csv(alerts_path).copy()
            for column in ["start_time", "end_time"]:
                if column in frame:
                    frame[column] = pd.to_datetime(frame[column], errors="coerce")
            centroid_columns = {"main_centroid_row", "main_centroid_col"}
            if len(frame) and centroid_columns.issubset(frame.columns):
                # Recompute coordinates from the authoritative grid mapping;
                # avoid merge suffixes if a future alert export already
                # carries stale center_lon/center_lat columns.
                frame = frame.drop(columns=["center_lon", "center_lat"], errors="ignore")
                main_row = pd.to_numeric(frame["main_centroid_row"], errors="coerce").round()
                main_col = pd.to_numeric(frame["main_centroid_col"], errors="coerce").round()
                main_valid = main_row.between(0, 31) & main_col.between(0, 31)

                # Spatially fragmented events may not have a single main
                # cluster centroid. Their first top_cells entry is the
                # highest-scoring detected grid and provides a deterministic,
                # explainable map anchor without changing the event itself.
                top_cell = (
                    frame.get("top_cells", pd.Series("", index=frame.index))
                    .fillna("")
                    .astype(str)
                    .str.extract(r"r(?P<row>\d+)\|c(?P<col>\d+)")
                    .apply(pd.to_numeric, errors="coerce")
                )
                top_valid = top_cell["row"].between(0, 31) & top_cell["col"].between(0, 31)
                frame["_center_row_key"] = main_row.where(main_valid, top_cell["row"]).astype("Int64")
                frame["_center_col_key"] = main_col.where(main_valid, top_cell["col"]).astype("Int64")
                frame["coordinate_source"] = np.select(
                    [main_valid, ~main_valid & top_valid],
                    ["main_centroid", "top_cell_fallback"],
                    default="unavailable",
                )
                locations = self._grid_meta()[
                    ["grid_id", "row", "col", "center_lon", "center_lat"]
                ].rename(columns={"row": "_center_row_key", "col": "_center_col_key"})
                frame = frame.merge(
                    locations,
                    on=["_center_row_key", "_center_col_key"],
                    how="left",
                    sort=False,
                    validate="many_to_one",
                ).rename(columns={"grid_id": "map_grid_id"})
                frame["map_alignment_status"] = np.where(
                    frame["center_lon"].notna() & frame["center_lat"].notna(),
                    "aligned",
                    "unavailable",
                )
                frame = frame.rename(
                    columns={
                        "_center_row_key": "map_row",
                        "_center_col_key": "map_col",
                    }
                )
            return frame

        return self.cache.derived(
            f"events_enriched_{source}", [alerts_path, grid_path], build
        )

    def query_events(
        self,
        *,
        source: str,
        start_time: str | None = None,
        end_time: str | None = None,
        event_type: str | None = None,
        alert_level: str | None = None,
        min_alert_level: int | None = None,
        min_score: float | None = None,
        region: Region | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        frame = self._events(source)
        if start_time is not None and "end_time" in frame:
            frame = frame.loc[frame["end_time"] >= pd.Timestamp(start_time)]
        if end_time is not None and "start_time" in frame:
            frame = frame.loc[frame["start_time"] <= pd.Timestamp(end_time)]
        if event_type is not None and "event_type" in frame:
            frame = frame.loc[frame["event_type"] == event_type]
        if alert_level is not None and "alert_level" in frame:
            frame = frame.loc[frame["alert_level"] == alert_level]
        if min_alert_level is not None and "alert_level_id" in frame:
            frame = frame.loc[frame["alert_level_id"] >= int(min_alert_level)]
        if min_score is not None and "max_score" in frame:
            frame = frame.loc[frame["max_score"] >= float(min_score)]
        if region is not None:
            row_column = "map_row" if "map_row" in frame else "main_centroid_row"
            col_column = "map_col" if "map_col" in frame else "main_centroid_col"
            if region.row_min is not None:
                frame = frame.loc[frame[row_column] >= region.row_min]
            if region.row_max is not None:
                frame = frame.loc[frame[row_column] <= region.row_max]
            if region.col_min is not None:
                frame = frame.loc[frame[col_column] >= region.col_min]
            if region.col_max is not None:
                frame = frame.loc[frame[col_column] <= region.col_max]
            if region.grid_ids is not None:
                selected = self._select_grid(region)
                pairs = set(zip(selected["row"].astype(int), selected["col"].astype(int)))
                rounded = zip(
                    frame[row_column].round().astype("Int64"),
                    frame[col_column].round().astype("Int64"),
                )
                frame = frame.loc[[
                    pd.notna(row) and pd.notna(col) and (int(row), int(col)) in pairs
                    for row, col in rounded
                ]]
        total = int(len(frame))
        coordinate_source_series = frame.get(
            "coordinate_source", pd.Series("unavailable", index=frame.index, dtype=str)
        )
        alignment_status_series = frame.get(
            "map_alignment_status", pd.Series("unavailable", index=frame.index, dtype=str)
        )
        coordinate_sources = coordinate_source_series.value_counts().to_dict()
        map_alignment = {
            "aligned": int((alignment_status_series == "aligned").sum()),
            "main_centroid": int(coordinate_sources.get("main_centroid", 0)),
            "top_cell_fallback": int(coordinate_sources.get("top_cell_fallback", 0)),
            "unavailable": int(coordinate_sources.get("unavailable", 0)),
        }
        if "start_time" in frame:
            frame = frame.sort_values("start_time", ascending=False)
        page = frame.iloc[int(offset) : int(offset) + int(limit)].copy()
        for column in ["start_time", "end_time"]:
            if column in page:
                page[column] = page[column].map(_iso_timestamp)
        return {
            "source": source,
            "total": total,
            "offset": int(offset),
            "limit": int(limit),
            "map_alignment": map_alignment,
            "events": _records(page),
        }

    def event_detail(self, event_id: int, *, source: str) -> dict[str, Any]:
        frame = self._events(source)
        selected = frame.loc[frame["event_id"].astype(int) == int(event_id)]
        if selected.empty:
            raise QueryNotFoundError(f"Event {event_id} was not found in {source} artifacts")
        row = selected.iloc[0].copy()
        for column in ["start_time", "end_time"]:
            if column in row:
                row[column] = _iso_timestamp(row[column])
        return {str(key): _json_scalar(value) for key, value in row.items()}

    def detect(
        self,
        *,
        source: str,
        timestamp: str | None,
        threshold: float | None,
        min_cells: int | None,
        top_k: int,
        region: Region | None = None,
    ) -> dict[str, Any]:
        arrays = self._scores(source)
        index = self._timestamp_index(arrays["timestamps"], timestamp)
        grid = self._select_grid(region)
        ids = grid["grid_id"].to_numpy(np.int64)
        scores = arrays["fusion_score"][index, ids]
        if threshold is None:
            mask = arrays["fusion_mask"][index, ids]
            threshold = float(self.anomaly_cfg.get("decision_threshold", 0.56))
        else:
            mask = scores >= float(threshold)
        required_cells = int(min_cells or self.anomaly_cfg.get("min_event_cells", 6))
        selected_time = _iso_timestamp(arrays["timestamps"][index])
        order = np.argsort(scores)[::-1][: int(top_k)]
        top = grid.iloc[order][["grid_id", "row", "col", "center_lon", "center_lat"]].copy()
        top["fusion_score"] = scores[order]
        top["is_anomaly"] = mask[order]
        matched = self.query_events(
            source=source,
            start_time=selected_time,
            end_time=selected_time,
            region=region,
            limit=100,
        )["events"]
        return {
            "city": self.city,
            "source": source,
            "execution_mode": "pipeline_artifact",
            "timestamp": selected_time,
            "requested_timestamp": timestamp,
            "decision_threshold": float(threshold),
            "min_event_cells": required_cells,
            "is_anomaly": bool(mask.sum() >= required_cells),
            "anomaly_cell_count": int(mask.sum()),
            "max_fusion_score": float(np.max(scores)),
            "pipeline_matched_event_count": int(len(matched)),
            "pipeline_matched_events": matched,
            "event_match_semantics": (
                "Events are prebuilt by the pipeline with its configured threshold and "
                "then filtered by time and requested region; custom request thresholds do "
                "not rebuild event boundaries."
            ),
            "top_cells": _records(top),
        }

    def statistics(self, **query: Any) -> dict[str, Any]:
        query = dict(query)
        query["limit"] = 1_000_000
        query["offset"] = 0
        result = self.query_events(**query)
        frame = pd.DataFrame(result["events"])
        if frame.empty:
            return {
                "source": result["source"],
                "total_events": 0,
                "by_event_type": {},
                "by_alert_level": {},
                "by_severity": {},
                "daily": [],
                "mean_duration_hours": None,
                "mean_max_score": None,
            }
        start = pd.to_datetime(frame["start_time"], errors="coerce")
        daily = start.dt.date.astype(str).value_counts().sort_index()
        return {
            "source": result["source"],
            "total_events": int(len(frame)),
            "by_event_type": {str(k): int(v) for k, v in frame["event_type"].value_counts().items()},
            "by_alert_level": {str(k): int(v) for k, v in frame["alert_level"].value_counts().items()},
            "by_severity": {str(k): int(v) for k, v in frame["week5_severity"].value_counts().sort_index().items()},
            "daily": [{"date": str(date), "count": int(count)} for date, count in daily.items()],
            "mean_duration_hours": float(frame["duration_hours"].mean()),
            "mean_max_score": float(frame["max_score"].mean()),
        }
