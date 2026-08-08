"""Request schemas for the Week 6 REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


SourceName = Literal["realtime", "historical"]


class RegionSelection(BaseModel):
    """Optional grid selection shared by inference endpoints."""

    grid_ids: list[int] | None = None
    row_min: int | None = Field(default=None, ge=0)
    row_max: int | None = Field(default=None, ge=0)
    col_min: int | None = Field(default=None, ge=0)
    col_max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "RegionSelection":
        if self.row_min is not None and self.row_max is not None and self.row_min > self.row_max:
            raise ValueError("row_min must be <= row_max")
        if self.col_min is not None and self.col_max is not None and self.col_min > self.col_max:
            raise ValueError("col_min must be <= col_max")
        return self


class PredictionRequest(RegionSelection):
    city: str = "bj"
    start_time: str | None = None
    end_time: str | None = None
    horizon_steps: int = Field(default=48, ge=1, le=48)


class AnomalyDetectionRequest(RegionSelection):
    city: str = "bj"
    timestamp: str | None = None
    source: SourceName = "realtime"
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_event_cells: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=20, ge=1, le=100)
