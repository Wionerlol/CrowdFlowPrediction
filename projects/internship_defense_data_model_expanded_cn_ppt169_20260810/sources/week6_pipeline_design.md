# Week6 端到端数据处理与推理管道设计

更新时间：2026-08-04

## 目标

Week6 第 1 项要求把以下模块封装成完整流水线：

1. 数据预处理；
2. 特征工程；
3. 模型推理；
4. 异常检测；
5. 预警输出。

同时需要支持两种运行模式：

| 模式 | 含义 | 当前实现 |
|------|------|----------|
| batch | 离线批处理，适合重跑整段 TaxiBJ 数据 | 编排 Week5 anomaly runner + Week6 alert generator |
| realtime | 实时/微批处理，适合最新 N 个窗口 | 复用 Week5 checkpoint 和校准阈值，对最新窗口做预测、异常检测和三档预警 |

核心代码：

```text
configs/week6_pipeline.yaml
src/pipeline/week6_pipeline.py
scripts/run_week6_pipeline.py
```

## Pipeline stage

### 1. 数据预处理阶段

默认不重跑原始预处理脚本，而是校验 Week2 已生成的 processed artifacts：

```text
data/processed/taxibj_clean.npz
data/processed/taxinyc_clean.npz
data/processed/weather_bj_30min.csv
data/processed/weather_nyc_30min.csv
```

原因：原始预处理是重型、低频操作，Week6 服务管道通常应复用已清洗数据。若需要重建，可在 `configs/week6_pipeline.yaml` 中设置：

```yaml
preprocessing:
  run: true
```

### 2. 特征工程阶段

默认校验以下特征/图结构产物：

```text
data/processed/node_features_bj.npy
data/processed/graph_spatial_bj.pt
data/processed/graph_flow_bj.pt
data/processed/graph_poi_bj.pt
data/processed/graph_transit_bj.pt
data/processed/grid_coords_bj.npy
data/processed/grid_meta_bj.csv
```

这些 artifact 支撑 Week4 AGFormer 推理、Week5 事件解释和 Week6 可视化。

### 3. 模型推理阶段

预测模型沿用 Week4 最优：

```text
AGFormer + gated context + linear residual head + 4-seed ensemble
```

在 batch 模式中，推理由 `scripts/run_week5_anomaly_detection.py` 执行。

在 realtime 模式中，pipeline 直接调用 `predict_ensemble_horizon(..., window_start, max_windows)`，只对最新或指定的一小段窗口做推理，避免重跑全量测试集。

为支持 Web 端“未来 24 小时预测动画”，realtime pipeline 还会调用 `predict_ensemble_sequence(...)`，从最新有效 forecast origin 一次输出完整 48 步。该序列与滚动 h1 结果分开保存，避免把 48 个不同起点的 h1 预测误当成同一起点的 24 小时预测。

### 4. 异常检测阶段

实时模式中复用 Week5 当前规则：

| 方法 | 实时行为 |
|------|----------|
| 统计规则 | 用训练集拟合 slot stats，再对当前窗口打分 |
| VAE | 加载 Week5 checkpoint，计算当前窗口重构误差 |
| Transformer AE | 加载 Week5 checkpoint，计算当前窗口序列重构误差 |
| 预测误差 | 使用 AGFormer ensemble residual |
| 融合 | 使用 Week5 固定 fusion weights 和 threshold |

重要约束：

```text
reconstruction.inference_only = true
```

实时模式不会偷偷重训 VAE/Transformer AE。如果 checkpoint 缺失或 metadata 不匹配，pipeline 会失败，而不是自动训练。

### 5. 三档预警阶段

异常事件生成后，pipeline 调用 Week6 alert mapping：

```text
Week5 severity 1–2 -> 一般
Week5 severity 3   -> 重要
Week5 severity 4   -> 紧急
```

输出可直接供 FastAPI 和 Streamlit 使用。

## Batch mode

### 完整批处理命令

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode batch \
  --device cuda
```

该命令会：

1. 校验 processed data；
2. 校验 feature artifacts；
3. 运行 Week5 anomaly detection；
4. 生成 Week6 alerts；
5. 写出 `outputs/week6_pipeline/batch_summary.json`。

### 快速批处理检查

如果 Week5 输出已经存在，可以跳过重算异常检测：

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode batch \
  --skip-anomaly \
  --device cuda
```

当前已验证该模式可正常运行。

主要输出：

```text
outputs/week6_pipeline/batch_summary.json
outputs/week6_pipeline/logs/batch_week6_alerts.log
outputs/week6_alerts/alerts.csv
outputs/week6_alerts/alert_summary.json
```

## Realtime mode

### 最新窗口微批推理

默认配置中：

```yaml
realtime:
  window_start: -24
  window_count: 24
```

含义：取最新 24 个有效 test windows 做实时微批推理。

运行命令：

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode realtime \
  --device cuda
```

也可以覆盖窗口范围：

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode realtime \
  --window-count 8 \
  --window-start -8 \
  --device cuda
```

当前已验证 `window_count=4, window_start=-4` 可正常运行，输出最新 4 个窗口的实时推理结果。

主要输出：

```text
outputs/week6_pipeline/realtime/summary.json
outputs/week6_pipeline/realtime/time_summary.csv
outputs/week6_pipeline/realtime/scores.npz
outputs/week6_pipeline/realtime/forecast_24h.npz
outputs/week6_pipeline/realtime/events.csv
outputs/week6_pipeline/realtime/alerts.csv
```

其中：

| 文件 | 含义 |
|------|------|
| `time_summary.csv` | 每个实时窗口的融合分数、异常网格数、预测残差摘要 |
| `forecast_24h.npz` | 最新共同预测起点的 48-step AGFormer ensemble 预测，供动画/API 使用 |
| `events.csv` | 当前微批内合并出的异常事件 |
| `alerts.csv` | 当前微批内的三档预警结果 |
| `summary.json` | pipeline stage 状态、窗口范围、事件/预警计数 |

## 当前验证结果

已执行：

```bash
.venv/bin/python -m py_compile \
  src/anomaly/prediction_error.py \
  scripts/run_week5_anomaly_detection.py \
  src/alerts/__init__.py \
  src/alerts/levels.py \
  scripts/generate_week6_alerts.py \
  src/pipeline/__init__.py \
  src/pipeline/week6_pipeline.py \
  scripts/run_week6_pipeline.py
```

Batch 快检：

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode batch \
  --skip-anomaly \
  --device cuda
```

Realtime 快检：

```bash
.venv/bin/python scripts/run_week6_pipeline.py \
  --config configs/week6_pipeline.yaml \
  --mode realtime \
  --window-count 4 \
  --window-start -4 \
  --device cuda
```

实时快检结果：

```text
actual_window_count = 4
time_range = 2016-04-08T15:00:00 ~ 2016-04-08T16:30:00
events = 0
alerts = 0
```

这说明最新几个窗口没有触发异常。即使无异常，`events.csv` 和 `alerts.csv` 也会保留表头，方便 API/前端稳定读取。

## 和后续 Week6 API / UI 的关系

FastAPI 不应直接调用 Week5 脚本，而应优先读取 pipeline 输出：

| API 功能 | 推荐数据源 |
|----------|------------|
| 最新推理状态 | `outputs/week6_pipeline/realtime/summary.json` |
| 未来/当前预测摘要 | `outputs/week6_pipeline/realtime/time_summary.csv` |
| 异常事件列表 | `outputs/week6_pipeline/realtime/events.csv` |
| 预警列表 | `outputs/week6_pipeline/realtime/alerts.csv` |
| 离线历史事件 | `outputs/week5_anomaly/events.csv` / `outputs/week6_alerts/alerts.csv` |

Streamlit 可视化也应消费同一批输出，避免前后端逻辑重复。

## Caveats

1. 当前实时模式是“离线数据上的微批实时模拟”，因为项目没有真实外部流数据源。
2. 实时模式复用 Week5 校准阈值，因此如果 Week5 fusion/threshold 改变，需要先重跑 Week5 或更新 `outputs/week5_anomaly/scores.npz`。
3. 实时模式不会自动重训重构模型；checkpoint 不匹配会 fail-fast。
4. 数据预处理和特征工程默认是 artifact validation；如果要从 raw data 全量重建，需要显式打开 `run: true`。
5. 当前只对 TaxiBJ 主线做了严格对齐；TaxiNYC 后续可复用 pipeline 结构，但需要单独配置 Week5 anomaly 与 checkpoint。
