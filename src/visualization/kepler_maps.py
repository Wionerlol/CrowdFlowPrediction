"""Kepler.gl map builders for flow grids and anomaly event markers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from keplergl import KeplerGl


FLOW_COLORS = [
    "#0B1F3A",
    "#123C69",
    "#1D70A2",
    "#2CA6A4",
    "#9BC53D",
    "#FDE74C",
    "#FA7921",
    "#D7263D",
]


def _map_state(frame: pd.DataFrame, zoom: float = 9.3) -> dict[str, Any]:
    return {
        "latitude": float(frame["center_lat"].dropna().mean()),
        "longitude": float(frame["center_lon"].dropna().mean()),
        "zoom": zoom,
        "bearing": 0,
        "pitch": 0,
    }


def _html(map_object: KeplerGl) -> str:
    value = map_object._repr_html_()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def flow_map_html(cells: list[dict[str, Any]], *, value_field: str = "value", height: int = 570) -> str:
    frame = pd.DataFrame(cells)
    if frame.empty:
        return "<p>当前筛选范围没有热力图数据。</p>"
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [
                    {
                        "id": "flow-heatmap",
                        "type": "heatmap",
                        "config": {
                            "dataId": "flow",
                            "label": "人流强度",
                            "color": [18, 60, 105],
                            "columns": {"lat": "center_lat", "lng": "center_lon"},
                            "isVisible": True,
                            "visConfig": {
                                "opacity": 0.82,
                                "colorRange": {
                                    "name": "CrowdFlow",
                                    "type": "sequential",
                                    "category": "Custom",
                                    "colors": FLOW_COLORS,
                                },
                                "radius": 24,
                            },
                        },
                        "visualChannels": {
                            "weightField": {"name": value_field, "type": "real"},
                            "weightScale": "linear",
                        },
                    }
                ],
                "interactionConfig": {
                    "tooltip": {
                        "enabled": True,
                        "fieldsToShow": {
                            "flow": [
                                {"name": "grid_id", "format": None},
                                {"name": value_field, "format": ".2f"},
                                {"name": "anomaly_score", "format": ".3f"},
                                {"name": "is_anomaly", "format": None},
                            ]
                        },
                    }
                },
            },
            "mapState": _map_state(frame),
            "mapStyle": {"styleType": "dark", "visibleLayerGroups": {"label": True}},
        },
    }
    result = KeplerGl(height=height, config=config)
    result.add_data(data=frame, name="flow")
    return _html(result)


def event_map_html(events: list[dict[str, Any]], *, height: int = 520) -> str:
    frame = pd.DataFrame(events)
    if frame.empty or frame[["center_lon", "center_lat"]].dropna().empty:
        return "<p>当前筛选范围没有可定位的异常事件。</p>"
    frame = frame.dropna(subset=["center_lon", "center_lat"]).copy()
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [
                    {
                        "id": "event-points",
                        "type": "point",
                        "config": {
                            "dataId": "events",
                            "label": "异常事件",
                            "color": [220, 38, 38],
                            "columns": {"lat": "center_lat", "lng": "center_lon", "altitude": None},
                            "isVisible": True,
                            "visConfig": {
                                "radius": 24,
                                "fixedRadius": False,
                                "opacity": 0.88,
                                "outline": True,
                                "strokeColor": [255, 255, 255],
                                "thickness": 2,
                            },
                        },
                        "visualChannels": {
                            "sizeField": {"name": "max_score", "type": "real"},
                            "sizeScale": "sqrt",
                            "colorField": {"name": "alert_level_id", "type": "integer"},
                            "colorScale": "ordinal",
                        },
                    }
                ],
                "interactionConfig": {
                    "tooltip": {
                        "enabled": True,
                        "fieldsToShow": {
                            "events": [
                                {"name": "event_id", "format": None},
                                {"name": "event_type_zh", "format": None},
                                {"name": "alert_level_zh", "format": None},
                                {"name": "start_time", "format": None},
                                {"name": "max_score", "format": ".3f"},
                                {"name": "coordinate_source", "format": None},
                            ]
                        },
                    }
                },
            },
            "mapState": _map_state(frame, zoom=9.0),
            "mapStyle": {"styleType": "dark", "visibleLayerGroups": {"label": True}},
        },
    }
    result = KeplerGl(height=height, config=config)
    result.add_data(data=frame, name="events")
    return _html(result)
