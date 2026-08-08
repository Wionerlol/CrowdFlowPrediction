# Week6 FastAPI + Streamlit 交互式人流监测平台

更新时间：2026-08-09

## 实现范围

本模块完成 Week6 的模型推理 API、交互式可视化和三级异常预警要求：

1. 城市人流热力图实时展示；
2. 同一预测起点的未来 24 小时（48×30 分钟）人流预测动画；
3. 异常事件地图标记、详情和三类检测分支解释；
4. 历史异常事件按时间、区域、类别、等级和分数查询；
5. 历史事件类型、预警等级和时间分布统计；
6. 一般、重要、紧急三级预警；重要/紧急弹窗，紧急预警额外播放声音。

当前 TaxiBJ 的“实时”是对离线测试集最新窗口的微批模拟，不代表已接入生产消息队列或真实在线出租车流。

## 系统架构

```text
Week6 realtime pipeline
  ├─ scores.npz                 h1 观测、预测、融合异常分数
  ├─ forecast_24h.npz           同一 origin 的 48-step ensemble 预测
  ├─ events.csv                 实时异常事件
  └─ alerts.csv                 三级预警字段
             │
             ▼
FastAPI read-only artifact service :8000
  ├─ 热力图 / 预测 / 异常检测
  ├─ 事件 / 预警 / 历史统计
  └─ mtime cache：pipeline 更新文件后自动重新加载
             │
             ▼
Streamlit :8501
  ├─ Kepler.gl 实时热力图
  ├─ Plotly 48 帧预测动画
  ├─ Kepler.gl 异常事件地图与详情
  └─ 弹窗、横幅、声音和历史统计
```

FastAPI worker 不直接加载 PyTorch checkpoint，也不会在请求线程中占用 GPU。模型推理由 Week6 pipeline 单独运行，API 读取其版本化快照。这样可以避免启动多个 API worker 时重复分配显存。

## 首次启动

### 1. 生成最新实时产物与 24 小时预测

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode realtime \
  --window-count 24 \
  --window-start -24 \
  --device cuda
```

管道会额外生成：

```text
outputs/week6_pipeline/realtime/forecast_24h.npz
```

它包含同一个 forecast origin 的 48 个未来时间步。不能使用连续 48 个独立 h1 预测代替这一产物，否则动画不具备“同一时刻预测未来 24 小时”的语义。

### 2. 启动 FastAPI

```bash
.venv/bin/python scripts/run_week6_api.py \
  --config configs/week6_web.yaml
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

### 3. 启动 Streamlit

另开一个终端：

```bash
.venv/bin/streamlit run src/visualization/week6_app.py \
  --server.address 127.0.0.1 \
  --server.port 8501
```

界面地址：

```text
http://127.0.0.1:8501
```

## REST API

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/api/v1/status` | 服务与所有 pipeline artifact 状态 |
| GET | `/api/v1/heatmap` | 指定时间、区域的观测或 h1 预测热力图 |
| GET | `/api/v1/predictions` | 查询未来 24 小时预测帧 |
| POST | `/api/v1/predict` | 按 grid IDs 或 row/col 区域查询预测 |
| POST | `/api/v1/anomaly/detect` | 对已推理时间片执行融合阈值判定 |
| GET | `/api/v1/anomalies` | 历史/实时异常事件分页查询 |
| GET | `/api/v1/anomalies/{event_id}` | 单个事件详情与解释字段 |
| GET | `/api/v1/alerts` | 按三级预警等级查询事件 |
| GET | `/api/v1/statistics` | 历史事件聚合统计 |

区域查询使用包含边界的 `row_min/row_max/col_min/col_max`，或逗号分隔的 `grid_ids`。事件区域筛选以事件主质心为准，热力图和预测区域筛选以网格本身为准。

### 示例：区域预测

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "city": "bj",
    "horizon_steps": 48,
    "row_min": 10,
    "row_max": 15,
    "col_min": 12,
    "col_max": 18
  }'
```

### 示例：紧急历史事件

```bash
curl 'http://127.0.0.1:8000/api/v1/alerts?source=historical&alert_level=urgent'
```

### 示例：指定时间异常检测

```bash
curl -X POST http://127.0.0.1:8000/api/v1/anomaly/detect \
  -H 'Content-Type: application/json' \
  -d '{
    "city": "bj",
    "source": "realtime",
    "timestamp": "2016-04-08T16:30:00",
    "top_k": 20
  }'
```

该接口消费 pipeline 已生成的三方法融合分数，`execution_mode` 返回 `pipeline_artifact`。它不会在 HTTP 请求中重新训练 VAE、Transformer AE 或 AGFormer。返回的 `pipeline_matched_events` 是 pipeline 使用默认阈值预先构建、再按请求时间和区域过滤的事件；如果请求临时覆盖 `decision_threshold/min_event_cells`，接口只重新计算当前时间片判定，不会用临时阈值重建完整事件边界。

## 三级预警 UI 行为

| Week6 等级 | Week5 severity | 页面行为 | 声音 |
|------------|----------------|----------|------|
| 一般 | 1–2 | 事件列表和地图标记 | 否 |
| 重要 | 3 | 顶部横幅、地图高亮、弹窗 | 否 |
| 紧急 | 4 | 红色脉冲横幅、强提醒弹窗 | 是 |

界面用 `event_id` 记录本次浏览器会话中已经确认的预警，避免每次自动刷新都重复弹窗。浏览器可能在用户首次与页面交互前阻止自动播放声音；界面保留了该提示，并允许用户关闭声音。

## 配置

Web/API 配置位于：

```text
configs/week6_web.yaml
```

主要配置包括 FastAPI host/port、CORS 来源、实时与历史 artifact 路径、融合阈值、最小异常网格数和 UI 默认刷新间隔。

## 验证

```bash
.venv/bin/python -m unittest -v tests.test_week6_api

.venv/bin/python -m py_compile \
  src/api/main.py \
  src/api/repository.py \
  src/visualization/week6_app.py \
  src/anomaly/prediction_error.py \
  src/pipeline/week6_pipeline.py
```

测试使用临时 2×2 网格构造 artifact，不依赖 GPU，覆盖：

- API readiness；
- 实时热力图与区域筛选；
- 同源 forecast 查询；
- 融合异常检测与事件匹配；
- 历史事件、详情、预警和统计；
- 非法区域参数。

## 当前限制

1. 实时数据仍是 TaxiBJ 离线微批模拟，需要定时运行 pipeline 才会刷新；
2. Web API 当前严格配置为 Beijing，NYC 需要独立 Web 配置和对应 Week5/Week6 artifact；
3. 异常事件和预警沿用 Week5 弱标签与规则分类局限，不代表人工确认事件；
4. 事件区域查询使用主质心，不是完整事件多边形相交查询；
5. Kepler.gl 底图瓦片需要浏览器能够访问相应地图资源。
