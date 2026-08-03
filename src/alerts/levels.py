"""Three-level warning rules aligned with Week 5 anomaly events.

Week 5 keeps a research-oriented severity score in ``{1, 2, 3, 4}``.
Week 6 needs a product-facing warning mechanism with three levels:

- general: visible in the alert list, no interruption;
- important: popup/banner, no sound by default;
- urgent: popup + sound.

This module deliberately does not rewrite the Week 5 severity. It adds alert
fields that the FastAPI and Streamlit layers can consume directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


EVENT_TYPE_ZH = {
    "mass_gathering": "大型集会",
    "traffic_jam": "交通拥堵",
    "sudden_incident": "突发事件",
}

ALERT_EXTRA_COLUMNS = [
    "week5_severity",
    "event_type_zh",
    "alert_level_id",
    "alert_level",
    "alert_level_zh",
    "alert_color",
    "alert_ui_behavior",
    "alert_popup",
    "alert_sound",
    "alert_action_hint",
    "alert_message",
]


@dataclass(frozen=True)
class AlertLevel:
    """Product-facing alert level metadata."""

    id: int
    name: str
    name_zh: str
    week5_severities: tuple[int, ...]
    color: str
    ui_behavior: str
    popup: bool
    sound: bool
    action_hint: str


DEFAULT_ALERT_LEVELS: tuple[AlertLevel, ...] = (
    AlertLevel(
        id=1,
        name="general",
        name_zh="一般",
        week5_severities=(1, 2),
        color="#F59E0B",
        ui_behavior="sidebar_badge",
        popup=False,
        sound=False,
        action_hint="记录并观察事件走势；在历史异常列表中展示。",
    ),
    AlertLevel(
        id=2,
        name="important",
        name_zh="重要",
        week5_severities=(3,),
        color="#F97316",
        ui_behavior="popup_and_banner",
        popup=True,
        sound=False,
        action_hint="弹窗提示并高亮地图区域；建议运营人员核查原因。",
    ),
    AlertLevel(
        id=3,
        name="urgent",
        name_zh="紧急",
        week5_severities=(4,),
        color="#DC2626",
        ui_behavior="modal_banner_and_sound",
        popup=True,
        sound=True,
        action_hint="强制弹窗、红色横幅和声音提示；建议立即处置。",
    ),
)


def _level_lookup(levels: tuple[AlertLevel, ...]) -> dict[int, AlertLevel]:
    lookup: dict[int, AlertLevel] = {}
    for level in levels:
        for severity in level.week5_severities:
            lookup[int(severity)] = level
    return lookup


def alert_levels_from_config(levels_cfg: dict[str, Any]) -> tuple[AlertLevel, ...]:
    """Build alert-level metadata from a YAML config ``levels`` mapping."""

    if not levels_cfg:
        raise ValueError("Alert config must define levels")
    levels = []
    for name, item in levels_cfg.items():
        levels.append(AlertLevel(
            id=int(item["id"]),
            name=str(name),
            name_zh=str(item["name_zh"]),
            week5_severities=tuple(int(x) for x in item["week5_severities"]),
            color=str(item["color"]),
            ui_behavior=str(item["ui_behavior"]),
            popup=bool(item["popup"]),
            sound=bool(item["sound"]),
            action_hint=str(item["action_hint"]),
        ))
    levels.sort(key=lambda x: x.id)
    return tuple(levels)


def build_alert_message(row: pd.Series | dict[str, Any]) -> str:
    """Build a compact UI/API alert message for one event."""

    get = row.get
    name_zh = get("alert_level_zh", "一般")
    event_type = get("event_type", "unknown")
    event_type_zh = get("event_type_zh", EVENT_TYPE_ZH.get(str(event_type), str(event_type)))
    start = get("start_time", "")
    end = get("end_time", "")
    score = float(get("max_score", np.nan))
    cells = int(get("union_cells", get("max_cells", 0)) or 0)
    centroid_row = get("main_centroid_row", np.nan)
    centroid_col = get("main_centroid_col", np.nan)
    if pd.notna(centroid_row) and pd.notna(centroid_col):
        region = f"中心网格({float(centroid_row):.1f}, {float(centroid_col):.1f})"
    else:
        region = "中心区域未知"
    return (
        f"{name_zh}预警：{event_type_zh}，{start} 至 {end}，"
        f"{region}，影响约 {cells} 个网格，最高融合分数 {score:.3f}。"
    )


def annotate_event_alerts(
    events: pd.DataFrame,
    *,
    levels: tuple[AlertLevel, ...] = DEFAULT_ALERT_LEVELS,
) -> pd.DataFrame:
    """Append Week 6 alert fields to Week 5 event rows."""

    if "severity" not in events.columns:
        raise ValueError("events must contain a Week 5 'severity' column")

    lookup = _level_lookup(levels)
    fallback = min(levels, key=lambda x: x.id)
    rows = []
    for _, row in events.iterrows():
        severity = int(row["severity"])
        level = lookup.get(severity, fallback)
        item = row.to_dict()
        item.update({
            "week5_severity": severity,
            "event_type_zh": EVENT_TYPE_ZH.get(str(item.get("event_type")), str(item.get("event_type"))),
            "alert_level_id": level.id,
            "alert_level": level.name,
            "alert_level_zh": level.name_zh,
            "alert_color": level.color,
            "alert_ui_behavior": level.ui_behavior,
            "alert_popup": level.popup,
            "alert_sound": level.sound,
            "alert_action_hint": level.action_hint,
        })
        item["alert_message"] = build_alert_message(item)
        rows.append(item)

    out = pd.DataFrame(rows)
    sort_cols = [col for col in ["alert_level_id", "max_score", "start_index"] if col in out.columns]
    if sort_cols:
        ascending = [False if col in {"alert_level_id", "max_score"} else True for col in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    return out


def summarize_alerts(alerts: pd.DataFrame) -> dict[str, Any]:
    """Summarize alert distribution for reports and API status endpoints."""

    if len(alerts) == 0 or "alert_level" not in alerts.columns:
        return {
            "total_alerts": 0,
            "by_level": {},
            "by_level_zh": {},
            "by_level_and_event_type": {},
            "popup_alerts": 0,
            "sound_alerts": 0,
        }
    counts = alerts["alert_level"].value_counts().to_dict()
    counts_zh = alerts["alert_level_zh"].value_counts().to_dict()
    by_type = (
        alerts.groupby(["alert_level", "event_type"]).size().unstack(fill_value=0).to_dict()
        if {"alert_level", "event_type"}.issubset(alerts.columns)
        else {}
    )
    return {
        "total_alerts": int(len(alerts)),
        "by_level": {str(k): int(v) for k, v in counts.items()},
        "by_level_zh": {str(k): int(v) for k, v in counts_zh.items()},
        "by_level_and_event_type": {
            str(event_type): {str(level): int(count) for level, count in values.items()}
            for event_type, values in by_type.items()
        },
        "popup_alerts": int(alerts["alert_popup"].sum()) if "alert_popup" in alerts else 0,
        "sound_alerts": int(alerts["alert_sound"].sum()) if "alert_sound" in alerts else 0,
    }
