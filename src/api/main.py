"""FastAPI application for Week 6 prediction, anomaly and alert queries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from src.api.repository import (
    ArtifactNotReadyError,
    QueryNotFoundError,
    Region,
    Week6ArtifactRepository,
    load_web_config,
)
from src.api.schemas import AnomalyDetectionRequest, PredictionRequest


DEFAULT_CONFIG = "configs/week6_web.yaml"


def _region(
    grid_ids: str | None,
    row_min: int | None,
    row_max: int | None,
    col_min: int | None,
    col_max: int | None,
) -> Region | None:
    ids = None
    if grid_ids:
        try:
            ids = tuple(sorted({int(item.strip()) for item in grid_ids.split(",") if item.strip()}))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="grid_ids must be comma-separated integers") from exc
    if ids is None and all(item is None for item in [row_min, row_max, col_min, col_max]):
        return None
    if row_min is not None and row_max is not None and row_min > row_max:
        raise HTTPException(status_code=422, detail="row_min must be <= row_max")
    if col_min is not None and col_max is not None and col_min > col_max:
        raise HTTPException(status_code=422, detail="col_min must be <= col_max")
    return Region(ids, row_min, row_max, col_min, col_max)


def _request_region(request: PredictionRequest | AnomalyDetectionRequest) -> Region | None:
    if request.grid_ids is None and all(
        item is None for item in [request.row_min, request.row_max, request.col_min, request.col_max]
    ):
        return None
    return Region(
        tuple(request.grid_ids) if request.grid_ids is not None else None,
        request.row_min,
        request.row_max,
        request.col_min,
        request.col_max,
    )


def create_app(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    repository: Week6ArtifactRepository | None = None,
) -> FastAPI:
    cfg = load_web_config(config_path)
    repo = repository or Week6ArtifactRepository(cfg)
    app = FastAPI(
        title="城市人流预测与异常预警 API",
        description="Week6 TaxiBJ 人流热力图、24小时预测、异常事件与三级预警服务",
        version="1.0.0",
    )
    origins = cfg.get("api", {}).get("cors_origins", [])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1_000)
    app.state.repository = repo

    @app.exception_handler(ArtifactNotReadyError)
    async def artifact_not_ready(_, exc: ArtifactNotReadyError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(QueryNotFoundError)
    async def query_not_found(_, exc: QueryNotFoundError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid_query(_, exc: ValueError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": app.title, "docs": "/docs", "health": "/api/v1/status"}

    @app.get("/api/v1/status")
    def status() -> dict:
        return repo.status()

    @app.get("/api/v1/heatmap")
    def heatmap(
        source: Literal["realtime", "historical"] = "realtime",
        timestamp: str | None = None,
        value_type: Literal["observed", "predicted"] = "observed",
        grid_ids: str | None = None,
        row_min: int | None = Query(default=None, ge=0),
        row_max: int | None = Query(default=None, ge=0),
        col_min: int | None = Query(default=None, ge=0),
        col_max: int | None = Query(default=None, ge=0),
    ) -> dict:
        return repo.heatmap(
            source=source,
            timestamp=timestamp,
            value_type=value_type,
            region=_region(grid_ids, row_min, row_max, col_min, col_max),
        )

    @app.get("/api/v1/predictions")
    def predictions(
        start_time: str | None = None,
        end_time: str | None = None,
        horizon_steps: int = Query(default=48, ge=1, le=48),
        grid_ids: str | None = None,
        row_min: int | None = Query(default=None, ge=0),
        row_max: int | None = Query(default=None, ge=0),
        col_min: int | None = Query(default=None, ge=0),
        col_max: int | None = Query(default=None, ge=0),
    ) -> dict:
        return repo.forecast(
            start_time=start_time,
            end_time=end_time,
            horizon_steps=horizon_steps,
            region=_region(grid_ids, row_min, row_max, col_min, col_max),
        )

    @app.post("/api/v1/predict")
    def predict(request: PredictionRequest) -> dict:
        if request.city != repo.city:
            raise HTTPException(status_code=422, detail=f"Only city={repo.city} is configured")
        return repo.forecast(
            start_time=request.start_time,
            end_time=request.end_time,
            horizon_steps=request.horizon_steps,
            region=_request_region(request),
        )

    @app.post("/api/v1/anomaly/detect")
    def detect_anomaly(request: AnomalyDetectionRequest) -> dict:
        if request.city != repo.city:
            raise HTTPException(status_code=422, detail=f"Only city={repo.city} is configured")
        return repo.detect(
            source=request.source,
            timestamp=request.timestamp,
            threshold=request.decision_threshold,
            min_cells=request.min_event_cells,
            top_k=request.top_k,
            region=_request_region(request),
        )

    def event_query(
        source: Literal["realtime", "historical"],
        start_time: str | None,
        end_time: str | None,
        event_type: str | None,
        alert_level: str | None,
        min_alert_level: int | None,
        min_score: float | None,
        grid_ids: str | None,
        row_min: int | None,
        row_max: int | None,
        col_min: int | None,
        col_max: int | None,
        limit: int,
        offset: int,
    ) -> dict:
        return repo.query_events(
            source=source,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            alert_level=alert_level,
            min_alert_level=min_alert_level,
            min_score=min_score,
            region=_region(grid_ids, row_min, row_max, col_min, col_max),
            limit=limit,
            offset=offset,
        )

    @app.get("/api/v1/anomalies")
    def anomalies(
        source: Literal["realtime", "historical"] = "historical",
        start_time: str | None = None,
        end_time: str | None = None,
        event_type: str | None = None,
        alert_level: str | None = None,
        min_alert_level: int | None = Query(default=None, ge=1, le=3),
        min_score: float | None = Query(default=None, ge=0.0, le=1.0),
        grid_ids: str | None = None,
        row_min: int | None = Query(default=None, ge=0),
        row_max: int | None = Query(default=None, ge=0),
        col_min: int | None = Query(default=None, ge=0),
        col_max: int | None = Query(default=None, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        return event_query(source, start_time, end_time, event_type, alert_level, min_alert_level, min_score, grid_ids, row_min, row_max, col_min, col_max, limit, offset)

    @app.get("/api/v1/alerts")
    def alerts(
        source: Literal["realtime", "historical"] = "realtime",
        start_time: str | None = None,
        end_time: str | None = None,
        event_type: str | None = None,
        alert_level: str | None = None,
        min_alert_level: int | None = Query(default=None, ge=1, le=3),
        min_score: float | None = Query(default=None, ge=0.0, le=1.0),
        grid_ids: str | None = None,
        row_min: int | None = Query(default=None, ge=0),
        row_max: int | None = Query(default=None, ge=0),
        col_min: int | None = Query(default=None, ge=0),
        col_max: int | None = Query(default=None, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        return event_query(source, start_time, end_time, event_type, alert_level, min_alert_level, min_score, grid_ids, row_min, row_max, col_min, col_max, limit, offset)

    @app.get("/api/v1/anomalies/{event_id}")
    def anomaly_detail(event_id: int, source: Literal["realtime", "historical"] = "historical") -> dict:
        return repo.event_detail(event_id, source=source)

    @app.get("/api/v1/statistics")
    def statistics(
        source: Literal["realtime", "historical"] = "historical",
        start_time: str | None = None,
        end_time: str | None = None,
        event_type: str | None = None,
        alert_level: str | None = None,
        min_alert_level: int | None = Query(default=None, ge=1, le=3),
        min_score: float | None = Query(default=None, ge=0.0, le=1.0),
        row_min: int | None = Query(default=None, ge=0),
        row_max: int | None = Query(default=None, ge=0),
        col_min: int | None = Query(default=None, ge=0),
        col_max: int | None = Query(default=None, ge=0),
    ) -> dict:
        return repo.statistics(
            source=source,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            alert_level=alert_level,
            min_alert_level=min_alert_level,
            min_score=min_score,
            region=_region(None, row_min, row_max, col_min, col_max),
        )

    return app


app = create_app(os.environ.get("WEEK6_WEB_CONFIG", DEFAULT_CONFIG))
