"""Display-only geographic calibration for the TaxiBJ 32x32 grid.

The TaxiBJ archive does not carry an authoritative geographic bounding box.  The
project's original grid metadata is therefore kept as the modelling coordinate
system, while this module provides a separate, reversible translation for map
display.  Node identity, row/column indices and all model features remain
unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


GRID_SIZE = 32
SOURCE_BOUNDS = {
    "west": 116.25,
    "south": 39.75,
    "east": 116.75,
    "north": 40.25,
}
DEFAULT_ROW_OFFSET = 5
DEFAULT_COL_OFFSET = -8
ROW_OFFSET_RANGE = (4, 7)
COL_OFFSET_RANGE = (-9, -6)
CELL_LON_DEGREES = (SOURCE_BOUNDS["east"] - SOURCE_BOUNDS["west"]) / GRID_SIZE
CELL_LAT_DEGREES = (SOURCE_BOUNDS["north"] - SOURCE_BOUNDS["south"]) / GRID_SIZE


def calibrated_bounds(
    *,
    row_offset: int = DEFAULT_ROW_OFFSET,
    col_offset: int = DEFAULT_COL_OFFSET,
) -> dict[str, float]:
    """Return the translated display bounds without changing their extent."""
    west = SOURCE_BOUNDS["west"] + col_offset * CELL_LON_DEGREES
    north = SOURCE_BOUNDS["north"] - row_offset * CELL_LAT_DEGREES
    return {
        "west": west,
        "south": north - GRID_SIZE * CELL_LAT_DEGREES,
        "east": west + GRID_SIZE * CELL_LON_DEGREES,
        "north": north,
    }


def build_display_grid_frame(
    source: pd.DataFrame,
    *,
    row_offset: int = DEFAULT_ROW_OFFSET,
    col_offset: int = DEFAULT_COL_OFFSET,
) -> pd.DataFrame:
    """Copy grid metadata and replace only its derived geographic columns."""
    required = {"grid_id", "row", "col"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"grid metadata is missing columns: {sorted(missing)}")

    display = source.copy()
    rows = pd.to_numeric(display["row"], errors="raise")
    cols = pd.to_numeric(display["col"], errors="raise")
    lon_min = SOURCE_BOUNDS["west"] + (cols + col_offset) * CELL_LON_DEGREES
    lat_max = SOURCE_BOUNDS["north"] - (rows + row_offset) * CELL_LAT_DEGREES

    display["center_lon"] = lon_min + CELL_LON_DEGREES / 2
    display["center_lat"] = lat_max - CELL_LAT_DEGREES / 2
    display["lon_min"] = lon_min
    display["lon_max"] = lon_min + CELL_LON_DEGREES
    display["lat_min"] = lat_max - CELL_LAT_DEGREES
    display["lat_max"] = lat_max
    return display


def adjust_display_frame(
    frame: pd.DataFrame,
    *,
    row_offset: int,
    col_offset: int,
) -> pd.DataFrame:
    """Translate display coordinates relative to the committed default grid."""
    required = {"center_lon", "center_lat"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"map data is missing columns: {sorted(missing)}")

    adjusted = frame.copy()
    longitude_delta = (col_offset - DEFAULT_COL_OFFSET) * CELL_LON_DEGREES
    latitude_delta = -(row_offset - DEFAULT_ROW_OFFSET) * CELL_LAT_DEGREES
    adjusted["center_lon"] = pd.to_numeric(
        adjusted["center_lon"], errors="coerce"
    ) + longitude_delta
    adjusted["center_lat"] = pd.to_numeric(
        adjusted["center_lat"], errors="coerce"
    ) + latitude_delta
    return adjusted


def adjust_display_records(
    records: Sequence[dict[str, Any]],
    *,
    row_offset: int,
    col_offset: int,
) -> list[dict[str, Any]]:
    """Record-oriented wrapper used by Streamlit map renderers."""
    if not records:
        return []
    return adjust_display_frame(
        pd.DataFrame(records), row_offset=row_offset, col_offset=col_offset
    ).to_dict("records")
