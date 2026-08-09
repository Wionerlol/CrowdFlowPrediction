# Week7 第一阶段：最终系统全面性能评估

生成时间：2026-08-09T12:13:19+00:00

## 1. 评估对象与口径

- 预测模型：AGFormer gated+linear 四随机种子均值 ensemble；
- 测试窗口：4022 个 gap-aware TaxiBJ 测试窗口；
- 预测指标沿用 Week4 的非零目标 mask 和 MAPE `eps=1.0` 口径；
- 异常检测参考为 TaxiBJ weak label，不是真实人工标签；
- 最终融合输出是双向异常，而本表参考空间足迹仅由高流量 inflow z-score 构造；
- API 基准为单客户端、顺序请求、warm server，不代表多用户极限吞吐。

## 2. 人流预测精度

| scope | forecast_hours | MAE | RMSE | MAPE | test_windows |
| --- | --- | --- | --- | --- | --- |
| h1 | 0.5000 | 11.3019 | 19.8621 | 23.4913 | 4022 |
| h3 | 1.5000 | 12.6473 | 22.9856 | 26.1838 | 4022 |
| h6 | 3.0000 | 13.4194 | 24.7530 | 28.2789 | 4022 |
| h12 | 6.0000 | 14.1068 | 26.2676 | 30.6591 | 4022 |
| avg | 24.0000 | 15.9030 | 30.2250 | 35.4350 | 4022 |

h1 指标已从 Week5 保存的 `true_flow/pred_flow` 重新计算，并与 Week4 ensemble 结果一致。`h3/h6/h12` 表示前 3/6/12 个预测步的累计指标，`avg` 为全部 48 步。

## 3. 异常检测 timestep 指标

| label_reference | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| level_A | 0.9212 | 0.7576 | 0.4322 | 0.5688 | 0.4912 | 0.9335 | 0.4608 |
| level_B | 0.9269 | 0.9111 | 0.1921 | 0.8947 | 0.3163 | 0.9689 | 0.4436 |

Accuracy 容易受异常样本稀少影响，主要同时观察 balanced accuracy、precision、recall、F1 和 average precision。
此外，弱标签来自统计规则，而融合权重中统计检测占 0.70，因此这些数值是规则一致性，不是独立真值准确率。

## 4. 时空事件级指标

| label_reference | event_precision | event_recall | event_f1 | true_event_count | pred_event_count | mean_detection_delay_hours |
| --- | --- | --- | --- | --- | --- | --- |
| level_A | 0.4580 | 0.5085 | 0.4819 | 118 | 131 | 0.1833 |
| level_B | 0.1756 | 0.7931 | 0.2875 | 29 | 131 | 0.0217 |

事件匹配同时要求时间区间重叠，以及空间 IoU 或质心距离满足配置阈值。
表中预测侧为完整双向融合输出（131 个事件），参考侧为 high-inflow 弱标签足迹；仅保留高流量方向时融合输出为 102 个事件。事件构建会在非 30 分钟连续的时间缺口处断开，匹配采用一对一最大基数匹配。

## 5. API 响应性能与内存

| endpoint | mean_ms | p50_ms | p95_ms | p99_ms | sequential_requests_per_second | response_bytes_mean |
| --- | --- | --- | --- | --- | --- | --- |
| status | 1.3308 | 1.3036 | 1.4688 | 1.7280 | 751.4170 | 5683.2400 |
| realtime_heatmap_1024_cells | 12.4769 | 12.3010 | 13.2446 | 14.2561 | 80.1482 | 212737.0000 |
| forecast_48_steps_1024_cells | 41.8862 | 41.7197 | 43.0060 | 44.8263 | 23.8742 | 992367.0000 |
| anomaly_detect_top20 | 4.3330 | 4.1311 | 5.3741 | 5.9778 | 230.7877 | 3253.0000 |
| historical_anomalies_200 | 19.4158 | 19.0918 | 20.7733 | 23.0003 | 51.5044 | 235642.0000 |
| historical_statistics | 16.8250 | 16.6671 | 18.0365 | 19.0920 | 59.4353 | 1876.0000 |

- API 冷启动：0.6161 秒；
- API 空闲 RSS：125.75 MiB；
- API 基准期间峰值 RSS：137.23 MiB。
- 每个端点正式请求数：100（另有 warm-up）；p95/p99 仅代表本机顺序请求；
- `response_bytes_mean` 是解压后的响应 body，wire bytes 不含 HTTP headers。

## 6. Realtime pipeline 性能

- 阶段耗时合计：6.7881 秒；
- 窗口数：24；
- pipeline 端到端等效吞吐：3.5356 windows/s；
- AGFormer ensemble 推理：107.1625 ms/window；
- 最慢阶段：model_inference（2.5719 秒）；
- 设备：cuda。

该耗时来自最近一次保存的 realtime pipeline summary，不是本次重新执行。端到端等效吞吐包含独立的 48 步 forecast 生成、产物加载和异常检测，不能解释为纯模型吞吐；`model_inference_ms_per_window` 仅对应保存记录中的模型推理阶段。

## 7. 评估边界

1. TaxiBJ 没有权威异常真标签，异常“准确率”只能解释为与弱标签规则的一致程度；
2. API 基准运行在本机 loopback，不包含公网、反向代理和浏览器渲染延迟；
3. pipeline 内存/GPU 峰值未在旧 summary 中保存，本阶段报告的是 API 服务 RSS；
4. 当前测试用于定位 Week7 后续超参数优化和系统优化方向，不应把 weak-label accuracy 当成真实业务准确率。
