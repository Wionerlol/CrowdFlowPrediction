# Week5 异常检测模块改进记录

更新时间：2026-08-09

## 当前基线

Week5 异常检测已经实现三类方法：

1. 统计规则：per-grid/per-slot 3σ + IQR；
2. 重构误差：VAE + Transformer AutoEncoder；
3. 预测误差：Week4 最优 AGFormer ensemble；
4. 融合策略：三方法置信度加权融合，并做规则型事件分类。

当前正式输出位于：

- `outputs/week5_anomaly/summary.json`
- `outputs/week5_anomaly/scores.npz`
- `outputs/week5_anomaly/events.csv`
- `outputs/week5_anomaly/event_explainability.csv`
- `outputs/week5_anomaly/audit/event_audit_sample_50.csv`
- `outputs/week5_anomaly/tuned_fusion/fusion_tuning_summary.json`

## 已确认需要改进的评估口径

### 1. high / low / bidirectional 三套异常评估

原问题：

- 检测器实际会抓双向异常：高流量、低流量、重构异常、预测残差异常；
- 但早期弱标签只评估 high-flow 异常；
- 因此低流量异常和部分模式异常会被当成 false positive，timestep F1 偏低。

落地方案：

- `high`：只评估高流量异常；
- `low`：只评估低流量异常；
- `bidirectional`：评估双向偏离异常；
- 方法输出按观测 inflow 的 slot-normalized z-score 正负方向拆分；
- 继续保留 Level-A / Level-B 两套弱标签。

实现入口：

```bash
.venv/bin/python scripts/evaluate_week5_anomaly_metrics.py \
  --scores outputs/week5_anomaly/scores.npz \
  --config outputs/week5_anomaly/config_used.yaml
```

本轮执行记录（2026-08-09）：

- 已按当前 `outputs/week5_anomaly/scores.npz` 与 `outputs/week5_anomaly/config_used.yaml` 重跑；
- 结果输出到 `outputs/week5_anomaly/evaluation/directional_timestep_metrics.csv`；
- 同时输出方向拆分后的 fusion / weak-label 事件明细，便于后续人工抽查。

### 2. 事件级评估

原问题：

- timestep F1 要求每个 30 分钟窗口逐点对齐；
- 但异常检测更关心“事件是否被命中”，而不是每个 timestep 是否完全重合；
- 对持续事件而言，只要检测器命中事件的一部分，就有预警价值。

落地方案：

- 将 timestep labels/masks 合并为连续事件区间；
- 预测事件和弱标签事件按时间重叠匹配；
- 输出 event precision / recall / F1 / hit rate；
- 同时统计 detection delay 和 duration error。

本轮执行记录（2026-08-09）：

- 已生成 `outputs/week5_anomaly/evaluation/event_level_metrics.csv`；
- 纯时间事件指标保留在 `outputs/week5_anomaly/evaluation/temporal_event_level_metrics.csv`，只作为参考；
- 汇总入口为 `outputs/week5_anomaly/evaluation/directional_event_summary.json`；
- 后续 Week5/Week6 报告中应优先同时报告 timestep 与 event-level 指标，不能只看 timestep F1。

## 后续仍可做的工程加固

已完成：

1. Transformer AE weak-normal window 训练：窗口内所有 timestep 都不是 Level-B 弱异常才进入训练集。
2. 事件合并加入空间连续性：在时间连续基础上加入 `cluster IoU >= 0.05` 或 `centroid distance <= 4.0`。
3. 分类规则增强：加入 `aspect_ratio`、`road_density`、`in_out_imbalance`、`duration_trend`、`cluster_count`，当前能识别少量 `traffic_jam`。
4. Reconstruction checkpoint 元数据校验：保存并校验 data/split/scaler/label rule/model hyperparams hash，避免误用旧 checkpoint。
5. 人工抽样核验：输出 50 个事件到 `outputs/week5_anomaly/audit/event_audit_sample_50.csv`，含 20 high-confidence、15 borderline、15 weak-label false-positive。
6. 事件解释表：输出 `outputs/week5_anomaly/event_explainability.csv`，包含三类方法贡献、dominant method、top cells、max z、max reconstruction error、max prediction residual。
7. `summary.json` 直接写入 fusion 参数、weak-label 规则和事件合并规则。

当前最终融合配置：

```yaml
weights:
  statistical: 0.70
  reconstruction: 0.20
  prediction: 0.10
decision_threshold: 0.56
min_event_cells: 6
```

当前正式结果（TaxiBJ test，4022 个窗口）：

| 口径 | Precision | Recall | F1 |
|------|----------:|-------:|---:|
| Level-A timestep | 0.4322 | 0.5688 | 0.4912 |
| Level-B timestep | 0.1921 | 0.8947 | 0.3163 |
| High-flow Level-A timestep | 0.8010 | 0.5688 | 0.6652 |
| High-flow Level-B timestep | 0.3560 | 0.8947 | 0.5094 |
| High-flow Level-A spatiotemporal event | 0.5882 | 0.5085 | 0.5455 |
| High-flow Level-B spatiotemporal event | 0.2255 | 0.7931 | 0.3511 |

事件输出：131 个事件，其中 `mass_gathering=14`、`traffic_jam=6`、`sudden_incident=111`。

仍可继续做：

1. 人工核验 50 个样本后，把人工结果写回并重新估计 weak-label 噪声。
2. 用更完整的道路先验或真实路网线段增强 `traffic_jam` 分类，而不是只依赖 node feature 的 road-density proxy。

## Reviewer caveats（2026-08-04）

`training_reviewer` 只读复审未发现 BLOCKER/HIGH。报告中需要保留以下口径说明：

1. 事件空间合并仍以相邻 timestep 的异常 mask 整体 IoU/centroid 为主；多中心事件可能被拆分或误合并。
2. `false_positive_by_weak_label` 不是人工真伪，只是 “fusion 检出但 high Level-A 弱标签未标出” 的候选，应写成 weak-label false-positive candidates。
3. Checkpoint metadata 已覆盖配置/数据形状/split/scaler/label rule/训练窗口数等关键项，但不是完整 provenance；如果预处理代码变了但数据文件内容不变，不一定自动失效。
4. high/low direction split 使用观测 inflow 的 slot-normalized z-score sign，不代表模型直接预测了异常方向。
5. 当前 `duration_hours = steps * 0.5` 适配 TaxiBJ；跨数据集时应改为配置化时间粒度。
