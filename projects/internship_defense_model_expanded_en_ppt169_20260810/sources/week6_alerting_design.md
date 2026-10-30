# Week6 分级预警机制设计

更新时间：2026-08-04

## 目标

Week6 第 4 项要求实现三级异常预警机制：

1. 一般；
2. 重要；
3. 紧急。

Week5 已经输出异常事件和研究用严重度字段：

```text
outputs/week5_anomaly/events.csv
severity ∈ {1, 2, 3, 4}
```

因此 Week6 不应重做异常检测，也不应直接改掉 Week5 的 severity。推荐做法是增加一层产品预警映射：

```text
Week5 event severity  →  Week6 alert level
```

这样既保留 Week5 的评估可复现性，又让 FastAPI/Streamlit 只面对清晰的三档预警。

## 与 Week5 的对齐规则

当前映射：

| Week5 severity | Week6 预警 | UI 行为 | 声音 |
|----------------|------------|---------|------|
| 1 | 一般 | 侧栏/列表标记 | 否 |
| 2 | 一般 | 侧栏/列表标记 | 否 |
| 3 | 重要 | 弹窗 + 顶部横幅 + 地图高亮 | 否 |
| 4 | 紧急 | 强制弹窗 + 红色横幅 + 地图高亮 | 是 |

选择 `1–2 → 一般、3 → 重要、4 → 紧急` 的原因：

1. Week5 severity=1/2 数量较多，若都弹窗会造成预警疲劳；
2. Week5 severity=3 代表高分数或较大空间影响，应进入人工关注；
3. Week5 severity=4 已经要求高分数、大范围、持续性，更适合触发声音和强提示；
4. 三级预警与 Week6 UI/API 更匹配，但原始 severity 仍可用于高级筛选。

## 当前 Week5 结果映射分布

基于 `outputs/week5_anomaly/events.csv` 生成：

```text
outputs/week6_alerts/alerts.csv
outputs/week6_alerts/alert_summary.json
```

当前 131 个事件的预警分布：

| Week6 预警 | 数量 | 行为 |
|------------|----:|------|
| 一般 | 110 | 仅列表/侧栏展示 |
| 重要 | 15 | 弹窗和横幅 |
| 紧急 | 6 | 弹窗、横幅和声音 |

按事件类型拆分：

| 预警 | 大型集会 | 交通拥堵 | 突发事件 |
|------|---------:|---------:|---------:|
| 一般 | 2 | 2 | 106 |
| 重要 | 7 | 4 | 4 |
| 紧急 | 5 | 0 | 1 |

该分布适合作为 Week6 初始 UI 策略：大部分事件进入历史列表，只有 21 个事件触发弹窗，其中 6 个触发声音。

## 输出字段

`outputs/week6_alerts/alerts.csv` 在 Week5 事件字段基础上新增：

| 字段 | 含义 |
|------|------|
| week5_severity | 原始 Week5 severity，保留 1–4 |
| event_type_zh | 中文事件类型 |
| alert_level_id | Week6 预警等级，1–3 |
| alert_level | 英文等级：general / important / urgent |
| alert_level_zh | 中文等级：一般 / 重要 / 紧急 |
| alert_color | UI 颜色 |
| alert_ui_behavior | UI 行为 |
| alert_popup | 是否弹窗 |
| alert_sound | 是否播放声音 |
| alert_action_hint | 处置建议 |
| alert_message | API/UI 可直接展示的预警文案 |

## API 对齐建议

FastAPI 后端建议直接返回 Week6 alert 字段，而不是让前端再计算预警等级。

推荐接口：

```text
GET /alerts
GET /alerts/{event_id}
GET /alerts?level=urgent
GET /alerts?start_time=...&end_time=...
GET /alerts?region=row_min,row_max,col_min,col_max
```

每条 alert 返回最小字段：

```json
{
  "event_id": 65,
  "start_time": "2016-02-04T20:30:00",
  "end_time": "2016-02-05T02:30:00",
  "event_type": "mass_gathering",
  "event_type_zh": "大型集会",
  "week5_severity": 4,
  "alert_level": "urgent",
  "alert_level_zh": "紧急",
  "alert_popup": true,
  "alert_sound": true,
  "max_score": 0.9547,
  "union_cells": 95,
  "main_centroid_row": 20.1,
  "main_centroid_col": 8.1,
  "alert_message": "紧急预警：大型集会..."
}
```

## Streamlit 对齐建议

Streamlit 前端可以按 `alert_level` 驱动 UI：

| alert_level | 展示方式 |
|-------------|----------|
| general | 右侧告警列表 + 地图 marker |
| important | `st.warning` / 顶部横幅 + 地图高亮 |
| urgent | `st.error` / 强弹窗 + HTML5 audio |

声音只建议给 `urgent` 开启，避免频繁提示音降低可用性。

## 当前实现文件

| 文件 | 用途 |
|------|------|
| `configs/week6_alerts.yaml` | 三档预警配置 |
| `src/alerts/levels.py` | Week5 severity 到 Week6 alert 的映射逻辑 |
| `scripts/generate_week6_alerts.py` | 从 Week5 events 生成 alerts |
| `outputs/week6_alerts/alerts.csv` | 当前事件预警表 |
| `outputs/week6_alerts/alert_summary.json` | 当前预警汇总 |

生成命令：

```bash
.venv/bin/python scripts/generate_week6_alerts.py \
  --config configs/week6_alerts.yaml
```

## Caveats

1. 当前预警等级依赖 Week5 的 `severity_level` 规则，因此 Week5 severity 阈值变化会影响 Week6 分布。
2. Week5 severity=4 当前没有 traffic_jam，说明交通拥堵类紧急预警可能偏保守；后续接入真实道路先验后应重新校准。
3. 当前预警不直接考虑人工核验结果；如果后续 50 个事件人工核验发现误报较多，应上调 important/urgent 阈值。
4. 当前基于离线 Week5 events 生成 alerts；实时模式中需要在 pipeline 每次产生新 event 后调用同一映射函数。
