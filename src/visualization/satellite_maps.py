"""Interactive Week6 maps backed by the local georeferenced Landsat image."""

from __future__ import annotations

import base64
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk


ASSET_DIR = Path(__file__).resolve().parent / "assets"
SATELLITE_IMAGE = ASSET_DIR / "beijing_landsat_20150416.webp"
SATELLITE_METADATA = ASSET_DIR / "beijing_landsat_20150416.json"
FLOW_COLOR_RANGE = [
    [11, 31, 58, 25],
    [18, 60, 105, 90],
    [29, 112, 162, 150],
    [44, 166, 164, 185],
    [253, 231, 76, 220],
    [250, 121, 33, 240],
    [215, 38, 61, 255],
]
ALERT_COLORS = {
    "general": [250, 204, 21, 220],
    "important": [249, 115, 22, 235],
    "urgent": [220, 38, 38, 245],
}


@lru_cache(maxsize=1)
def satellite_metadata() -> dict[str, Any]:
    return json.loads(SATELLITE_METADATA.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def satellite_image_uri() -> str:
    encoded = base64.b64encode(SATELLITE_IMAGE.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def satellite_caption() -> str:
    metadata = satellite_metadata()
    return (
        f"底图：NASA/USGS Landsat‑8，北京场景 {metadata['acquired_on']}，"
        f"原始空间分辨率 {metadata['native_resolution_m']} m；底图为静态遥感影像，并非实时卫星画面。"
    )


def _bounds() -> list[float]:
    bounds = satellite_metadata()["bounds"]
    return [bounds["west"], bounds["south"], bounds["east"], bounds["north"]]


def _view_state() -> pdk.ViewState:
    west, south, east, north = _bounds()
    return pdk.ViewState(
        longitude=(west + east) / 2,
        latitude=(south + north) / 2,
        zoom=9.15,
        min_zoom=8.1,
        max_zoom=14,
        pitch=0,
        bearing=0,
    )


def _bitmap_layer(opacity: float) -> pdk.Layer:
    return pdk.Layer(
        "BitmapLayer",
        data=None,
        # Explicit Image prevents pydeck from prefixing a data URI with "@@="
        # and making deck.gl parse it as a JavaScript accessor expression.
        image=pdk.types.Image(str(SATELLITE_IMAGE)),
        bounds=_bounds(),
        opacity=float(opacity),
        pickable=False,
    )


def _html(deck: pdk.Deck) -> str:
    return deck.to_html(as_string=True, notebook_display=False)


def flow_satellite_map_html(
    cells: list[dict[str, Any]],
    *,
    value_field: str = "value",
    height: int = 570,
    basemap_opacity: float = 0.72,
) -> str:
    frame = pd.DataFrame(cells)
    if frame.empty:
        return "<p>当前筛选范围没有热力图数据。</p>"
    frame[value_field] = pd.to_numeric(frame[value_field], errors="coerce").fillna(0.0)
    heatmap = pdk.Layer(
        "HeatmapLayer",
        data=frame,
        get_position="[center_lon, center_lat]",
        get_weight=value_field,
        radius_pixels=28,
        intensity=1.15,
        threshold=0.035,
        color_range=FLOW_COLOR_RANGE,
        opacity=0.86,
    )
    hover_points = pdk.Layer(
        "ScatterplotLayer",
        data=frame,
        get_position="[center_lon, center_lat]",
        get_radius=520,
        get_fill_color=[255, 255, 255, 2],
        opacity=0.02,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 90],
        stroked=False,
    )
    tooltip = {
        "html": (
            "<b>网格 #{grid_id}</b><br/>"
            "人流：{" + value_field + "}<br/>"
            "异常分数：{anomaly_score}<br/>异常：{is_anomaly}"
        ),
        "style": {"backgroundColor": "#102a43", "color": "white"},
    }
    deck = pdk.Deck(
        layers=[_bitmap_layer(basemap_opacity), heatmap, hover_points],
        initial_view_state=_view_state(),
        map_style=None,
        tooltip=tooltip,
    )
    html = _html(deck)
    return html.replace("height: 100vh", f"height: {height}px")


def event_satellite_map_html(
    events: list[dict[str, Any]],
    *,
    height: int = 520,
    basemap_opacity: float = 0.72,
) -> str:
    frame = pd.DataFrame(events)
    required = {"center_lon", "center_lat"}
    if frame.empty or not required.issubset(frame.columns):
        return "<p>当前筛选范围没有可定位的异常事件。</p>"
    frame = frame.dropna(subset=["center_lon", "center_lat"]).copy()
    if frame.empty:
        return "<p>当前筛选范围没有可定位的异常事件。</p>"
    frame["_fill_color"] = frame.get("alert_level", "general").map(
        lambda value: ALERT_COLORS.get(str(value), ALERT_COLORS["general"])
    )
    scores = pd.to_numeric(frame.get("max_score", 0.0), errors="coerce").fillna(0.0)
    frame["_radius_m"] = 550 + scores.clip(0, 1) * 1_250
    event_layer = pdk.Layer(
        "ScatterplotLayer",
        data=frame,
        get_position="[center_lon, center_lat]",
        get_radius="_radius_m",
        get_fill_color="_fill_color",
        get_line_color=[255, 255, 255, 240],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": (
            "<b>事件 #{event_id} · {alert_level_zh}</b><br/>"
            "类型：{event_type_zh}<br/>开始：{start_time}<br/>"
            "最高融合分数：{max_score}"
        ),
        "style": {"backgroundColor": "#102a43", "color": "white"},
    }
    deck = pdk.Deck(
        layers=[_bitmap_layer(basemap_opacity), event_layer],
        initial_view_state=_view_state(),
        map_style=None,
        tooltip=tooltip,
    )
    html = _html(deck)
    return html.replace("height: 100vh", f"height: {height}px")


def plotly_satellite_layer(opacity: float = 0.72) -> dict[str, Any]:
    west, south, east, north = _bounds()
    return {
        "sourcetype": "image",
        "source": satellite_image_uri(),
        "coordinates": [
            [west, north],
            [east, north],
            [east, south],
            [west, south],
        ],
        "opacity": float(opacity),
        "below": "traces",
    }
