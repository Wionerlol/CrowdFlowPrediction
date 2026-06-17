# 异常检测规格文档

**版本**: v1.0  
**日期**: 2026-06-13  
**适用数据集**: TaxiBJ（主）、TaxiNYC（泛化验证）

---

## 一、异常定义框架

城市人流异常 = **在正常时空背景下，某区域的流量出现统计意义上的显著偏离，并伴有空间聚集性**。

三个判断维度：

| 维度 | 描述 |
|------|------|
| 幅度（Magnitude） | 单格流量相对于同时槽历史水平的 Z-score 超过阈值 |
| 聚集（Spatial） | 满足幅度条件的格子在空间上有邻域聚集（而非随机散点） |
| 持续（Duration） | 连续 ≥2 个时步（≥1小时）保持异常状态 |

> 单格孤立尖峰可能是传感器噪声；持续的空间聚集才代表真实事件。

---

## 二、Z-score 归一化方案

### 2.1 时槽归一化（核心方法）

原始流量分布高度右偏（均值 103.86，中位数 56，std 130.20），直接使用全局阈值会导致大量误报。  
**解决方案**：按 (格子, 时槽) 组合单独计算历史均值/标准差。

```python
# 计算时槽历史统计（训练集上计算，保存为 .npy 供推理使用）
SLOTS = 48   # 每天48个30分钟时槽

slot_mu  = np.zeros((SLOTS, 32, 32))   # shape: (48, 32, 32)
slot_sig = np.zeros((SLOTS, 32, 32))

for s in range(SLOTS):
    chunk = inflow_train[s::SLOTS]     # 该时槽的所有历史观测
    slot_mu[s]  = chunk.mean(axis=0)
    slot_sig[s] = np.maximum(chunk.std(axis=0), 1.0)  # 避免低活跃格除零

# 推理时的实时 Z-score
def compute_zscore(flow_t, slot_idx):
    """flow_t: (32,32), slot_idx: 当前时槽 0~47"""
    return (flow_t - slot_mu[slot_idx]) / slot_sig[slot_idx]
```

**实测触发率**（基于 22484 步全量数据）：

| Z 阈值 | 触发比例（格-步级） |
|--------|-----------------|
| z > 2.0 | 1.910% |
| z > 2.5 | 0.612% |
| z > 3.0 | 0.223% |
| z > 3.5 | 0.094% |
| z > 4.0 | 0.042% |

### 2.2 可选：对数变换后 Z-score

对于极右偏分布，可先做 `log1p` 变换再计算 Z-score，使分布更接近正态，减少高峰时段误报：

```python
z_log = (np.log1p(flow_t) - log_slot_mu[slot_idx]) / log_slot_sig[slot_idx]
```

> **当前阶段使用线性 Z-score**，Week 5 异常检测模块可对比两种方案。

---

## 三、严重程度分级与告警阈值

基于实测数据，以"每步中超阈值格子数"作为空间维度过滤条件：

| 级别 | 颜色 | Z阈值 | 空间条件 | 实测触发率 | 业务含义 |
|------|------|-------|---------|-----------|---------|
| L1 轻度 | 🟡 黄色 | z > 3.0 | 同步超阈值格 ≥ 6 | **11.42%** ≈ 每天~5.5次 | 局部热点，观察 |
| L2 中度 | 🟠 橙色 | z > 3.0 | 同步超阈值格 ≥ 12 | **3.78%** ≈ 每天~1.8次 | 街区级异常，关注 |
| L3 重度 | 🔴 红色 | z > 3.5 | 同步超阈值格 ≥ 12 | **0.71%** ≈ 每3天1次 | 大型聚集/拥堵事件 |
| L4 极端 | 🚨 紧急 | z > 4.0 | 同步超阈值格 ≥ 25 | **0.01%** ≈ 每年~1次 | 城市级突发事件 |

**空间条件含义**：
- ≥6 格 ≈ 约 8km² 区域（2×3 街区块）
- ≥12 格 ≈ 约 27km² 区域（3×4 街区块，可视化可感知）
- ≥25 格 ≈ 约 58km² 区域（5×5 街区块，跨区级事件）

---

## 四、空间聚集判定算法

除"同步超阈值格数"外，要求超阈值格在**空间上形成连通分量**，排除离散噪声点：

```python
from scipy.ndimage import label, uniform_filter

def detect_anomaly_clusters(flow_t, slot_idx, z_thresh=3.0, min_cells=6):
    """
    flow_t: (32,32) 当前时步流量
    返回: (level, clusters) where clusters 是异常区域列表
    """
    z = compute_zscore(flow_t, slot_idx)
    exceed_mask = z > z_thresh          # (32,32) bool

    # 连通分量标记（8邻域）
    labeled, n_comp = label(exceed_mask, structure=np.ones((3,3)))

    clusters = []
    for comp_id in range(1, n_comp + 1):
        comp_mask = (labeled == comp_id)
        cell_count = comp_mask.sum()
        if cell_count >= 2:             # 至少2格连通才算簇
            max_z = z[comp_mask].max()
            centroid = np.array(np.where(comp_mask)).mean(axis=1)
            clusters.append({
                'cell_count': int(cell_count),
                'max_z': float(max_z),
                'centroid_row': float(centroid[0]),
                'centroid_col': float(centroid[1]),
            })

    total_cells = sum(c['cell_count'] for c in clusters)
    # 按分级表确定 level
    if total_cells >= 25 and any(c['max_z'] > 4.0 for c in clusters):
        level = 4
    elif total_cells >= 12 and any(c['max_z'] > 3.5 for c in clusters):
        level = 3
    elif total_cells >= 12:
        level = 2
    elif total_cells >= 6:
        level = 1
    else:
        level = 0  # 无告警

    return level, clusters
```

---

## 五、异常类型分类

基于空间形态和时间模式，预定义三种异常类型：

| 类型 | 英文标签 | 空间特征 | 时间特征 | 典型事件 |
|------|---------|---------|---------|---------|
| 大型聚集 | `mass_gathering` | 单一高密度热点（1个大连通分量） | 持续2~6小时后骤降 | 演唱会、庙会、体育赛事 |
| 交通拥堵 | `traffic_jam` | 线性/带状分布（沿主干道） | 缓慢上升、缓慢消散 | 早晚高峰拥堵 |
| 突发事件 | `sudden_incident` | 多分散热点（多个小连通分量） | 突然出现(<30min)、快速消散 | 交通事故、突发故障 |

分类规则（基于几何特征，Week 5 实现）：
```python
def classify_anomaly(clusters):
    if len(clusters) == 1 and clusters[0]['cell_count'] > 15:
        return 'mass_gathering'
    # 线性度检验（长宽比）
    elif aspect_ratio(clusters) > 3.0:
        return 'traffic_jam'
    else:
        return 'sudden_incident'
```

---

## 六、数据统计摘要（参考基准）

基于 TaxiBJ 全量数据（22,484 步 × 1024 格）：

| 统计量 | Inflow | Outflow |
|--------|--------|---------|
| 均值 | 103.86 | 103.86 |
| 中位数 | 56.0 | 56.0 |
| 标准差 | 130.20 | 130.28 |
| p90 | 270.0 | 270.0 |
| p95 | 384.0 | 385.0 |
| p99 | 602.0 | 603.0 |
| 最大值 | 1,285 | 1,292 |
| 零值占比 | 5.3% | 5.3% |

早高峰（07:00~09:30）均值约 117，凌晨低谷（00:00~02:00）约 88，日内振幅仅 35%——时间效应弱于空间异质性（格间最大均值差达 613 倍），**时槽归一化是必要的**。
