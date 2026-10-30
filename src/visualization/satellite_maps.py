"""Interactive Week6 maps backed by the local georeferenced Landsat image."""

from __future__ import annotations

import base64
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pydeck as pdk

from src.visualization.geo_calibration import DEFAULT_COL_OFFSET, DEFAULT_ROW_OFFSET


ASSET_DIR = Path(__file__).resolve().parent / "assets"
SATELLITE_IMAGE = ASSET_DIR / "beijing_landsat_20150416.webp"
SATELLITE_METADATA = ASSET_DIR / "beijing_landsat_20150416.json"
RING_ROADS_GEOJSON = ASSET_DIR / "beijing_ring_roads.geojson"
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


@lru_cache(maxsize=1)
def ring_roads_geojson() -> dict[str, Any]:
    return json.loads(RING_ROADS_GEOJSON.read_text(encoding="utf-8"))


def satellite_caption(
    *,
    row_offset: int = DEFAULT_ROW_OFFSET,
    col_offset: int = DEFAULT_COL_OFFSET,
) -> str:
    metadata = satellite_metadata()
    vertical = f"向南 {row_offset} 格" if row_offset >= 0 else f"向北 {abs(row_offset)} 格"
    horizontal = (
        f"向东 {col_offset} 格" if col_offset >= 0 else f"向西 {abs(col_offset)} 格"
    )
    return (
        f"底图：NASA/USGS Landsat‑8，北京场景 {metadata['acquired_on']}，"
        f"原始空间分辨率 {metadata['native_resolution_m']} m；"
        f"当前展示网格相对原元数据{vertical}、{horizontal}。该校准只影响地图显示，"
        "不改变训练节点或模型输入；底图为静态遥感影像，并非实时卫星画面。"
    )


def _bounds() -> list[float]:
    bounds = satellite_metadata()["bounds"]
    return [bounds["west"], bounds["south"], bounds["east"], bounds["north"]]


def _view_state() -> pdk.ViewState:
    west, south, east, north = _bounds()
    return pdk.ViewState(
        longitude=(west + east) / 2,
        latitude=(south + north) / 2,
        zoom=9.25,
        min_zoom=8.1,
        max_zoom=15,
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


def _ring_roads_layer(opacity: float = 0.78) -> pdk.Layer:
    alpha = max(0, min(255, round(230 * opacity)))
    return pdk.Layer(
        "GeoJsonLayer",
        data=ring_roads_geojson(),
        stroked=True,
        filled=False,
        get_line_color=[255, 194, 71, alpha],
        line_width_min_pixels=2,
        line_width_max_pixels=5,
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
    show_ring_roads: bool = True,
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
        radius_pixels=32,
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
    anomaly_mask = (
        frame["is_anomaly"].fillna(False).astype(bool)
        if "is_anomaly" in frame
        else pd.Series(False, index=frame.index)
    )
    anomaly_frame = frame.loc[anomaly_mask].copy()
    anomaly_layer = pdk.Layer(
        "ScatterplotLayer",
        data=anomaly_frame,
        get_position="[center_lon, center_lat]",
        get_radius=780,
        get_fill_color=[220, 38, 38, 24],
        get_line_color=[255, 255, 255, 245],
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=False,
    )
    tooltip = {
        "html": (
            "<b>网格 #{grid_id}</b><br/>"
            "人流：{" + value_field + "}<br/>"
            "异常分数：{anomaly_score}<br/>异常：{is_anomaly}"
        ),
        "style": {"backgroundColor": "#102a43", "color": "white"},
    }
    layers = [_bitmap_layer(basemap_opacity)]
    if show_ring_roads:
        layers.append(_ring_roads_layer())
    layers.extend([heatmap, anomaly_layer, hover_points])
    deck = pdk.Deck(
        layers=layers,
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
    show_ring_roads: bool = True,
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
    selected = frame.get("is_selected", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame.loc[selected, "_radius_m"] *= 1.45
    frame["_line_color"] = selected.map(
        {True: [56, 189, 248, 255], False: [255, 255, 255, 240]}
    )
    frame["_line_width"] = selected.map({True: 5, False: 2})
    event_layer = pdk.Layer(
        "ScatterplotLayer",
        data=frame,
        get_position="[center_lon, center_lat]",
        get_radius="_radius_m",
        get_fill_color="_fill_color",
        get_line_color="_line_color",
        get_line_width="_line_width",
        line_width_units="pixels",
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
            "最高融合分数：{max_score}<br/>坐标来源：{coordinate_source}"
        ),
        "style": {"backgroundColor": "#102a43", "color": "white"},
    }
    layers = [_bitmap_layer(basemap_opacity)]
    if show_ring_roads:
        layers.append(_ring_roads_layer())
    layers.append(event_layer)
    deck = pdk.Deck(
        layers=layers,
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


def plotly_ring_roads_layer(opacity: float = 0.78) -> dict[str, Any]:
    return {
        "sourcetype": "geojson",
        "source": ring_roads_geojson(),
        "type": "line",
        "color": "#ffc247",
        "line": {"width": 2.0},
        "opacity": float(opacity),
        "below": "traces",
    }
