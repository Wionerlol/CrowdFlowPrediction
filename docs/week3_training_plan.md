# Week 3 训练方案

**日期**：2026-06-22  
**阶段**：Week 3 — 基础时空预测模型

---

## 一、数据集与切分

### 主数据集：TaxiBJ（1024节点 / 32×32）

| 分片 | 时间段 | 步数 | 用途 |
|------|--------|------|------|
| BJ13 + BJ14 + BJ15[:75%] | 2013-07 ~ 2015-05 | ~13,865 | 训练 |
| BJ15[75%:] | 2015-05 ~ 2015-06 | ~1,399 | 验证（调参/早停） |
| BJ16 | 2015-11 ~ 2016-04 | 7,220 | 测试（不可见，唯一评分依据） |

### 跨城市验证：TaxiNYC（75节点 / 15×5）

BJ 模型训练完后直接迁移评估（zero-shot），不单独训练。

### 滑动窗口

```
T_in  = 12 步（6小时历史）
T_out = 3  步（1.5小时预测，主指标）
       同时记录 1步 / 6步 / 12步 的单独指标
```

### 归一化

训练集全局 mean/std → 统一应用于 val/test，反归一化后计算指标。

---

## 二、数据处理管道

目前 `TaxiBJDataset` 存在两个问题需修复：
1. 只能处理单一 `T_out`，评估多时域需重建
2. 没有 TaxiNYC 对应实现

新建 `src/data/flow_dataset.py`，统一两个城市：

```
FlowDataset(data, T_in, T_out, mean=None, std=None)
  → x: (N, T_in, 2)   inflow + outflow
  → y: (N, T_out)     inflow（预测目标）

load_taxibj()  → (22484, 2, 32, 32)   ← 已有
load_taxinyc() → (17520, 2, 15, 5)    ← 需新增
```

---

## 三、各模型网络设计

### 模型一：Historical Average（HA）

- 无需训练，用训练集中同一时段的历史均值作为预测
- 目的：给出不可能低于的下界，验证其他模型有无实际意义

### 模型二：ARIMA

- 每个网格独立拟合 ARIMA(2,0,1)
- 1024 格太慢，只跑非零格（约 600~700 格）+ `joblib` 并行
- 只做 1步预测（ARIMA 多步退化严重），结果仅作参考

### 模型三：LSTM

```
架构：共享权重的节点级 LSTM
输入 (B×N, T_in, 2) → LSTM(hidden=64, layers=2) → 取最后隐状态
    → Linear(64 → T_out) → 输出 (B, N, T_out)
参数量：~0.1M，训练 <1min/epoch
```

不引入空间信息，纯时序基线，与 STGCN 对比空间建模的收益。

### 模型四：STGCN + 单图（G_spatial）

- 已实现，`src/models/stgcn.py`
- 参数量：~0.6M，训练 ~3min/epoch（GPU）

### 模型五：STGCN + 方案C 四图门控融合（主模型）

```
输入 x: (B, N, T_in, 2)
  ↓ 输入投影 Linear(2 → d_model=64)

并行 4 路 ChebGCN（各2层，共享时间步）：
  h_spatial = GCN(G_spatial, x)   → (B, N, 64)
  h_poi     = GCN(G_poi,     x)   → (B, N, 64)
  h_flow    = GCN(G_flow,    x)   → (B, N, 64)
  h_transit = GCN(G_transit, x)   → (B, N, 64)

门控融合：
  gate = Softmax(Linear(64 → 4))  per-node
  h_fused = Σ gate_k * h_k        → (B, N, 64)

时序建模：
  STGCN 时序卷积块 × 2 → OutputModule → (B, N, T_out)

参数量：~1.2M，训练 ~5min/epoch（GPU）
```

---

## 四、训练基础设施

### 统一训练配置

每个模型一个 YAML 配置文件：

```
configs/
  arima.yaml
  lstm.yaml
  stgcn.yaml
  stgcn_multigraph.yaml
```

### 统一训练入口

```bash
python scripts/train.py --config configs/stgcn_multigraph.yaml
```

内部共享：数据加载 → 归一化 → 训练循环 → 早停 → 评估 → 记录

### 优化器 / 调度（统一规范）

| 项目 | 设置 |
|------|------|
| 优化器 | AdamW(lr=1e-3, weight_decay=1e-4) |
| 调度器 | ReduceLROnPlateau(patience=5, factor=0.5) |
| 损失函数 | Huber Loss（对异常值鲁棒） |
| 梯度裁剪 | max_norm=5.0 |
| 早停 | patience=15 epoch（监控 val MAE） |
| 混合精度 | torch.cuda.amp（可选，节省显存） |

---

## 五、实验记录与存储

### 目录结构

```
outputs/
├── checkpoints/
│   ├── ha/                       # 无权重，只存统计量
│   ├── arima/                    # arima_params_{grid}.pkl（抽样格子）
│   ├── lstm/best.pt
│   ├── stgcn/best.pt
│   └── stgcn_multigraph/best.pt
├── logs/                         # TensorBoard 日志
│   ├── lstm/
│   ├── stgcn/
│   └── stgcn_multigraph/
├── results/
│   └── week3_metrics.csv         # 所有模型对比汇总表
└── reports/
    └── week3_report.pdf
```

### MLflow 实验追踪

每次训练自动记录 config + metrics + artifact（best.pt）：

```
mlruns/
  experiment: week3_baselines
    run: lstm_T12_h64_...
    run: stgcn_single_...
    run: stgcn_multigraph_...
```

### 最终对比表格（week3_metrics.csv）

| 模型 | BJ 1步MAE | BJ 3步MAE | BJ 6步MAE | BJ RMSE | BJ MAPE | NYC MAE |
|------|-----------|-----------|-----------|---------|---------|---------|
| HA | | | | | | |
| ARIMA | | | | | | — |
| LSTM | | | | | | |
| STGCN-单图 | | | | | | |
| STGCN-方案C | | | | | | |

---

## 六、文件规划

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/data/flow_dataset.py` | 新建 | 统一 BJ/NYC Dataset + load_taxinyc |
| `src/models/lstm.py` | 新建 | 节点共享 LSTM |
| `src/models/multi_graph_stgcn.py` | 新建 | 4图门控融合 STGCN |
| `src/training/metrics.py` | 新建 | MAE / RMSE / MAPE，masked |
| `src/training/trainer.py` | 新建 | 统一训练循环 |
| `configs/*.yaml` | 新建 | 各模型超参配置 |
| `scripts/train.py` | 新建 | 统一训练入口 |
| `scripts/train_arima.py` | 新建 | ARIMA 独立脚本（慢，单独跑） |
| `scripts/evaluate_all.py` | 新建 | 汇总对比表 |
| `src/data/taxibj_dataset.py` | 保留/弃用 | 被 flow_dataset.py 替代 |
| `scripts/train_stgcn.py` | 保留/弃用 | 被 train.py 替代 |
