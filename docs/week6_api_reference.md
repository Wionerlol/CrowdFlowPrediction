# Week6 REST API 接口文档

更新时间：2026-08-09
API 版本：1.0.0
实现位置：`src/api/main.py`

## 1. 服务概述

Week6 API 为 TaxiBJ 城市人流系统提供只读查询服务，包括：

- 实时/历史人流热力图；
- 同一预测起点的未来 24 小时人流预测；
- 三方法融合异常检测；
- 历史与实时异常事件查询；
- 一般、重要、紧急三级预警；
- 异常事件聚合统计。

API 读取 Week5/Week6 pipeline 已生成的快照文件，不在 HTTP 请求中训练模型，也不在 FastAPI worker 内加载 PyTorch checkpoint。模型推理与 API 服务彼此隔离。

默认地址：

```text
http://127.0.0.1:8000
```

接口前缀：

```text
/api/v1
```

交互式文档：

```text
Swagger UI  http://127.0.0.1:8000/docs
ReDoc       http://127.0.0.1:8000/redoc
OpenAPI     http://127.0.0.1:8000/openapi.json
```

当前实现未提供身份认证。不要把 API 无保护地暴露到公网；正式部署应绑定内网地址，或在反向代理/VPN 后增加认证和 TLS。

## 2. 通用约定

### 2.1 数据格式

- 请求和响应使用 UTF-8 JSON；
- POST 请求需设置 `Content-Type: application/json`；
- 时间建议使用 ISO 8601，如 `2016-04-01T08:30:00`；
- TaxiBJ 项目时间按北京本地时间语义解释，artifact 中的时间字符串不携带时区后缀；
- 网格编号范围为 `0–1023`，`row/col` 范围为 `0–31`。

### 2.2 数据源

| 值 | 含义 | 主要产物 |
|---|---|---|
| `realtime` | Week6 最新实时微批模拟 | `outputs/week6_pipeline/realtime/` |
| `historical` | Week5 全测试集历史结果 | `outputs/week5_anomaly/`、`outputs/week6_alerts/` |

当前“实时”是对 TaxiBJ 离线测试窗口的滚动微批模拟，不代表已连接真实消息队列或在线出租车数据源。

### 2.3 区域筛选

大多数接口支持以下两种区域选择方式：

| 参数 | GET 格式 | POST 格式 | 说明 |
|---|---|---|---|
| `grid_ids` | 逗号分隔字符串，如 `1,2,3` | 整数数组，如 `[1,2,3]` | 指定网格集合 |
| `row_min/row_max` | 非负整数 | 非负整数 | 行号闭区间 |
| `col_min/col_max` | 非负整数 | 非负整数 | 列号闭区间 |

同时提供多种条件时按交集筛选。`row_min` 不得大于 `row_max`，`col_min` 不得大于 `col_max`。

热力图和预测接口按网格筛选；事件接口按事件的 `main_centroid_row/main_centroid_col` 主质心筛选，而不是按完整异常区域多边形相交。

### 2.4 事件类别与预警等级

事件类别：

| `event_type` | 中文 |
|---|---|
| `mass_gathering` | 大型集会 |
| `traffic_jam` | 交通拥堵 |
| `sudden_incident` | 突发事件 |

预警等级：

| `alert_level_id` | `alert_level` | 中文 | 默认 UI 行为 |
|---:|---|---|---|
| 1 | `general` | 一般 | 列表/地图展示 |
| 2 | `important` | 重要 | 横幅和弹窗 |
| 3 | `urgent` | 紧急 | 红色横幅、弹窗和声音 |

## 3. 接口总览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | 服务入口信息 |
| GET | `/api/v1/status` | 健康状态与 artifact 就绪情况 |
| GET | `/api/v1/heatmap` | 查询观测或 h1 预测热力图 |
| GET | `/api/v1/predictions` | 查询未来 24 小时预测帧 |
| POST | `/api/v1/predict` | 使用 JSON 请求查询预测 |
| POST | `/api/v1/anomaly/detect` | 对已有融合分数执行时间片判定 |
| GET | `/api/v1/anomalies` | 查询异常事件 |
| GET | `/api/v1/alerts` | 查询预警事件 |
| GET | `/api/v1/anomalies/{event_id}` | 查询单个事件详情 |
| GET | `/api/v1/statistics` | 查询异常事件聚合统计 |

## 4. 服务与健康状态

### 4.1 `GET /`

返回服务名称及文档入口。

```bash
curl http://127.0.0.1:8000/
```

响应示例：

```json
{
  "service": "城市人流预测与异常预警 API",
  "docs": "/docs",
  "health": "/api/v1/status"
}
```

### 4.2 `GET /api/v1/status`

检查配置中的所有 artifact 是否存在。即使部分 artifact 缺失，该接口仍返回 HTTP 200，但 `status` 为 `degraded`；业务查询缺少必要 artifact 时返回 HTTP 503。

```bash
curl http://127.0.0.1:8000/api/v1/status
```

核心响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | `ok` 或 `degraded` |
| `city` | string | 当前配置城市，现为 `bj` |
| `server_time` | string | API 服务器 UTC 时间 |
| `artifacts` | object | 各产物路径、大小、修改时间和就绪状态 |
| `realtime` | object/null | realtime pipeline 的 `summary.json` 内容 |

## 5. 人流热力图

### `GET /api/v1/heatmap`

查询指定时间片的观测人流或 h1 预测人流。未提供 `timestamp` 时返回该数据源的最新时间片；请求时间没有精确匹配时返回距离最近的时间片，并在 `requested_timestamp` 中保留原请求值。

查询参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `source` | `realtime\|historical` | `realtime` | 数据源 |
| `timestamp` | string | 最新时间片 | ISO 8601 时间 |
| `value_type` | `observed\|predicted` | `observed` | 返回观测值或 h1 预测值 |
| `grid_ids` | string | 全部 | 逗号分隔网格 ID |
| `row_min/row_max` | integer | 全部 | 行范围 |
| `col_min/col_max` | integer | 全部 | 列范围 |

示例：查询第 10–15 行、第 12–18 列的最新预测热力图。

```bash
curl 'http://127.0.0.1:8000/api/v1/heatmap?source=realtime&value_type=predicted&row_min=10&row_max=15&col_min=12&col_max=18'
```

响应结构：

```json
{
  "city": "bj",
  "source": "realtime",
  "timestamp": "2016-04-08T16:30:00",
  "requested_timestamp": null,
  "value_type": "predicted",
  "cell_count": 42,
  "summary": {
    "mean": 118.4,
    "max": 463.2,
    "total": 4972.8,
    "anomaly_cells": 3,
    "max_anomaly_score": 0.91
  },
  "cells": [
    {
      "grid_id": 332,
      "row": 10,
      "col": 12,
      "center_lon": 116.3203125,
      "center_lat": 40.0078125,
      "value": 121.7,
      "observed_flow": 126.0,
      "predicted_flow": 121.7,
      "anomaly_score": 0.34,
      "is_anomaly": false
    }
  ]
}
```

经纬度来自独立的 Week6 展示网格，只用于地图定位；训练节点、`grid_id`、`row/col` 和模型输入未改变。

## 6. 未来 24 小时预测

### 6.1 `GET /api/v1/predictions`

返回同一个 forecast origin 产生的最多 48 个未来时间步，每步 30 分钟。它不是连续 48 个独立 h1 预测的拼接。

查询参数：

| 参数 | 类型 | 默认值 | 约束/说明 |
|---|---|---|---|
| `start_time` | string | 不限制 | 预测帧起始时间，包含边界 |
| `end_time` | string | 不限制 | 预测帧结束时间，包含边界 |
| `horizon_steps` | integer | 48 | `1–48` |
| 区域参数 | — | 全部网格 | 见 2.3 节 |

```bash
curl 'http://127.0.0.1:8000/api/v1/predictions?horizon_steps=12&grid_ids=330,331,332'
```

### 6.2 `POST /api/v1/predict`

功能与 GET 版本相同，适合较长的网格 ID 列表。

请求体：

```json
{
  "city": "bj",
  "start_time": null,
  "end_time": null,
  "horizon_steps": 48,
  "grid_ids": [330, 331, 332],
  "row_min": null,
  "row_max": null,
  "col_min": null,
  "col_max": null
}
```

仅支持 Web 配置中的城市；当前传入非 `bj` 城市会返回 HTTP 422。

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"city":"bj","horizon_steps":2,"grid_ids":[330,331,332]}'
```

两种接口的响应结构相同：

```json
{
  "city": "bj",
  "model": "AGFormer gated+linear 4-seed ensemble",
  "origin_time": "2016-04-08T16:00:00",
  "horizon_steps": 2,
  "step_minutes": 30,
  "member_tags": [
    "agformer_gate_context_linear_seed42",
    "agformer_gate_context_linear_seed7",
    "agformer_gate_context_linear_seed21",
    "agformer_gate_context_linear_seed123"
  ],
  "cells": [
    {"grid_id": 330, "row": 10, "col": 10, "center_lon": 116.2890625, "center_lat": 40.0078125}
  ],
  "frames": [
    {
      "step": 1,
      "timestamp": "2016-04-08T16:30:00",
      "mean": 115.2,
      "max": 207.4,
      "total": 345.6,
      "values": [102.1, 116.8, 126.7]
    }
  ]
}
```

`frames[*].values` 与 `cells` 顺序一一对应。响应不返回真实未来流量。

## 7. 融合异常检测

### `POST /api/v1/anomaly/detect`

对 pipeline 已生成的统计、重构和预测误差融合分数进行指定时间片判定。

请求体字段：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `city` | string | `bj` | 当前仅支持 `bj` |
| `source` | `realtime\|historical` | `realtime` | 数据源 |
| `timestamp` | string/null | 最新时间片 | 无精确匹配时取最近时间片 |
| `decision_threshold` | float/null | 配置值/产物 mask | `0–1` |
| `min_event_cells` | integer/null | 配置值 6 | 判断全局是否异常的最小网格数 |
| `top_k` | integer | 20 | `1–100` |
| 区域字段 | — | 全部网格 | 见 2.3 节 |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/anomaly/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "city":"bj",
    "source":"realtime",
    "timestamp":"2016-04-08T16:30:00",
    "decision_threshold":0.56,
    "min_event_cells":6,
    "top_k":10
  }'
```

响应结构：

```json
{
  "city": "bj",
  "source": "realtime",
  "execution_mode": "pipeline_artifact",
  "timestamp": "2016-04-08T16:30:00",
  "requested_timestamp": "2016-04-08T16:30:00",
  "decision_threshold": 0.56,
  "min_event_cells": 6,
  "is_anomaly": true,
  "anomaly_cell_count": 9,
  "max_fusion_score": 0.94,
  "pipeline_matched_event_count": 0,
  "pipeline_matched_events": [],
  "event_match_semantics": "Events are prebuilt by the pipeline ...",
  "top_cells": [
    {
      "grid_id": 527,
      "row": 16,
      "col": 15,
      "center_lon": 116.3671875,
      "center_lat": 39.9140625,
      "fusion_score": 0.94,
      "is_anomaly": true
    }
  ]
}
```

重要边界：

- 该接口不会重新运行 VAE、Transformer AE 或 AGFormer；
- 临时传入 `decision_threshold/min_event_cells` 只改变当前时间片判定；
- `pipeline_matched_events` 是 pipeline 按其固定配置预先构建的事件，再按请求时间和区域过滤；
- 临时阈值不会重建事件开始/结束时间、空间簇或事件分类。

## 8. 异常事件与预警查询

### 8.1 `GET /api/v1/anomalies`

默认查询历史异常事件。

### 8.2 `GET /api/v1/alerts`

参数和响应与 `/anomalies` 相同，默认查询实时预警。两个接口当前都返回带三级预警字段的事件记录，路径差异主要用于调用语义和默认数据源。

共同查询参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `source` | `realtime\|historical` | anomalies=`historical`；alerts=`realtime` | 数据源 |
| `start_time` | string | 不限制 | 事件结束时间不早于该值 |
| `end_time` | string | 不限制 | 事件开始时间不晚于该值 |
| `event_type` | string | 全部 | 三类事件之一 |
| `alert_level` | string | 全部 | `general/important/urgent` |
| `min_alert_level` | integer | 不限制 | `1–3`，按等级下限筛选 |
| `min_score` | float | 不限制 | `0–1` |
| 区域参数 | — | 全部 | 事件主质心范围 |
| `limit` | integer | 200 | `1–1000` |
| `offset` | integer | 0 | 分页偏移 |

```bash
curl 'http://127.0.0.1:8000/api/v1/anomalies?source=historical&event_type=mass_gathering&min_alert_level=2&limit=20'

curl 'http://127.0.0.1:8000/api/v1/alerts?source=realtime&alert_level=urgent'
```

响应：

```json
{
  "source": "historical",
  "total": 12,
  "offset": 0,
  "limit": 20,
  "events": [
    {
      "event_id": 65,
      "start_time": "2016-02-04T20:30:00",
      "end_time": "2016-02-05T02:30:00",
      "event_type": "mass_gathering",
      "event_type_zh": "大型集会",
      "alert_level_id": 3,
      "alert_level": "urgent",
      "alert_level_zh": "紧急",
      "max_score": 0.955,
      "duration_hours": 6.5,
      "union_cells": 95,
      "center_lon": 116.2578125,
      "center_lat": 39.8515625,
      "alert_popup": true,
      "alert_sound": true,
      "alert_message": "紧急预警：大型集会……"
    }
  ]
}
```

事件记录还可能包含以下解释字段：

| 字段组 | 典型字段 |
|---|---|
| 空间形态 | `cluster_count`、`main_centroid_row/col`、`main_aspect_ratio`、`road_density_mean` |
| 流量变化 | `mean_inflow`、`max_inflow`、`mean_outflow`、`in_out_imbalance`、`duration_trend_slope` |
| 方法贡献 | `statistical_score`、`reconstruction_score`、`prediction_score`、`dominant_method` |
| 极值解释 | `max_z`、`max_reconstruction_error`、`max_prediction_residual`、`top_cells` |
| 预警行为 | `alert_color`、`alert_ui_behavior`、`alert_popup`、`alert_sound`、`alert_action_hint` |

### 8.3 `GET /api/v1/anomalies/{event_id}`

返回一个事件的完整记录。

参数：

| 参数 | 位置 | 默认值 | 说明 |
|---|---|---|---|
| `event_id` | path | — | 事件 ID |
| `source` | query | `historical` | `realtime` 或 `historical` |

```bash
curl 'http://127.0.0.1:8000/api/v1/anomalies/65?source=historical'
```

同一个 `event_id` 在不同数据源中不是全局唯一，查询详情时必须使用与列表相同的 `source`。

## 9. 历史统计

### `GET /api/v1/statistics`

对筛选后的事件进行聚合。支持 `source`、时间、事件类型、预警等级、最低分数和 row/col 区域参数；不支持 `grid_ids`。

```bash
curl 'http://127.0.0.1:8000/api/v1/statistics?source=historical&min_alert_level=2'
```

响应：

```json
{
  "source": "historical",
  "total_events": 31,
  "by_event_type": {
    "mass_gathering": 18,
    "sudden_incident": 10,
    "traffic_jam": 3
  },
  "by_alert_level": {
    "important": 19,
    "urgent": 12
  },
  "by_severity": {
    "3": 19,
    "4": 12
  },
  "daily": [
    {"date": "2016-01-31", "count": 1}
  ],
  "mean_duration_hours": 2.18,
  "mean_max_score": 0.83
}
```

无匹配事件时，计数和分组为空，两个均值字段为 `null`。

## 10. 状态码与错误响应

| 状态码 | 场景 |
|---:|---|
| 200 | 查询成功；`/status` 也可能以 200 返回 `degraded` |
| 404 | 无符合条件的预测帧，或事件 ID 不存在 |
| 422 | 参数格式、范围、城市或时间字符串非法 |
| 503 | 必要 artifact 缺失，服务尚未就绪 |

错误格式：

```json
{
  "detail": "具体错误信息"
}
```

示例：

```json
{
  "detail": "row_min must be <= row_max"
}
```

## 11. Python 调用示例

```python
import requests

base_url = "http://127.0.0.1:8000/api/v1"

status = requests.get(f"{base_url}/status", timeout=10).json()
status_text = status["status"]

forecast = requests.post(
    f"{base_url}/predict",
    json={"city": "bj", "horizon_steps": 4, "grid_ids": [330, 331, 332]},
    timeout=20,
).json()

first_time = forecast["frames"][0]["timestamp"]
first_values = forecast["frames"][0]["values"]
print(status_text, first_time, first_values)
```

## 12. 接口限制

1. 当前只配置 TaxiBJ，北京 32×32 网格；
2. 预测上限为 48 个 30 分钟时间步，即未来 24 小时；
3. API 只读取已有 artifact，不接收任意原始流量数组执行在线推理；
4. 当前无认证、配额、用户隔离和持久化数据库；
5. 异常标签来自 Week5 弱标签与无监督规则，不等同于人工确认事件；
6. 请求级自定义阈值不会重建完整异常事件；
7. 实时快照刷新由外部 pipeline 调度负责。

部署和运维步骤见 [Week6 系统部署指南](week6_deployment_guide.md)。
