# Week7 系统性能优化报告

## 结论

在同一组 TaxiBJ 最新 24 个有效窗口、同一组 checkpoint 和同一异常阈值下，优化后暖启动流水线端到端加速 **1.39×**。数值一致性与量化采用显式门槛，未通过的候选不会进入生产配置。

## 固定条件

- 推理窗口：测试集末尾 24 个连续有效窗口
- 预测模型：AGFormer gated+linear 四随机种子 ensemble
- 预测目标：h1 异常残差，并输出同一 origin 的 48 步预测
- 对照原则：相同数据、checkpoint、阈值和输出数组；只改变执行策略

## 流水线结果

| name | pipeline_seconds | model_inference_seconds | forecast_inference_seconds | statistical_seconds | peak_rss_mib | cuda_peak_allocated_mib |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_fp32 | 6.4161 | 2.0058 | 1.7411 | 1.7231 | 3881.5195 | 826.2471 |
| optimized_cold_fp32 | 6.1063 | 3.0868 | 0.0000 | 1.9276 | 2793.1953 | 703.7471 |
| optimized_warm_fp32 | 4.6133 | 3.3640 | 0.0000 | 0.0377 | 1791.1523 | 703.7471 |
| optimized_warm_fp16 | 4.5962 | 3.3825 | 0.0000 | 0.0336 | 1894.3789 | 419.6846 |

主要改动：关闭预测路径未使用的注意力矩阵物化；以实测最优 batch=8 处理 24 个窗口；h1 异常预测与 48 步 Web 预测共享一次模型加载/前向；按数据 SHA256 和规则指纹缓存训练集统计量。批大小扫描见 `/home/louis/projects/DidiRemote/outputs/week7_performance/final_system_v5/batch_sweep.csv`。

## 数值一致性

- FP32 优化路径 fusion mask 不一致率：`0.0`
- FP32 复用预测的 forecast origin/timestamps/target indices 完全一致，48×1024 预测最大绝对差：`9.1552734375e-05`
- FP16 候选 fusion mask 不一致率：`0.0`
- FP16 是否部署：`False`（failed realtime or stratified test-sample gate; keep production fp32）

补充分层测试集核验覆盖 `256` 个窗口（`262144` 个节点预测）：FP16 相对 FP32 的平均预测差为 `0.034193`，按项目口径排除真实值为 0 后的 masked MAE 变化为 `-0.000048`，融合异常掩码不一致率为 `0.0`，结论为 `True`。核验文件及 SHA256 已写入 `summary.json`。

## 模型量化评估

- 状态：`evaluated`
- 范围：`single_model_single_latest_window_cpu_dynamic_linear_int8`
- CPU INT8 加速比：`1.3048567494858418`
- 模型 state_dict 体积下降：`0.012361021765003866`
- 是否部署：`False`
- CPU 降级路径候选：`True`
- 判断：CPU fallback microbenchmark passed, but current production inference is CUDA; do not replace the GPU ensemble without an end-to-end CPU benchmark

动态 INT8 仅量化 `Linear`，图邻接乘法与注意力主体仍是 CPU 浮点运算；因此只有在实测延迟和偏差同时通过门槛时才值得部署。当前 API 继续读取原子发布的 NPZ/CSV 快照，不在多个 Web worker 中复制 GPU 模型。

## API 热路径

API 增加了基于源文件 mtime 的派生视图缓存，并将事件质心坐标映射改为向量化 merge。完整结果见 `api_optimized.csv`；历史基线来自 Week7 固定评估产物，来源记录在 `summary.json`。

## 测量边界

- 流水线 variant 各运行一次且顺序固定，适合确认数量级与回归，不应把很小的差异解释为稳定收益。
- `peak_rss_mib` 是主进程与 DataLoader 子进程 RSS 求和，共享页可能重复计算；它用于资源风险筛查，不等同于物理 PSS。
- API 是本机 loopback、单客户端顺序请求；事件查询明显受益，但热力图和完整 48×1024 JSON 仍主要受序列化/压缩限制，单次结果可能小幅波动。
- FP16 和动态 INT8 都是候选路径；只有联合 gate 通过才允许修改生产精度。本轮生产保持 FP32。

## 可复现产物

- `summary.json`：所有耗时、资源、数值偏差和部署判断
- `pipeline_variants.csv`：各执行策略对照
- `api_optimized.csv`：优化后 API 延迟
- 各 variant 目录：实际配置、日志、scores 与 forecast 快照
