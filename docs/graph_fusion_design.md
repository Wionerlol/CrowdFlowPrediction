# 异构图融合方案设计文档

**版本**: v1.0  
**日期**: 2026-06-13  
**状态**: 已确定，待 Week 3/4 实现

---

## 一、背景与目标

为 TaxiBJ 32×32（1024节点）和 TaxiNYC 15×5（75节点）构建图结构，捕捉城市空间中的多维关系：地理邻近性、功能区相似性、历史流量相关性、POI语义影响。

**文献依据**：调研 2022–2025 年 KDD/AAAI/IJCAI/ICLR/NeurIPS 共 10+ 篇相关工作（详见调研记录），确定两个方案并行推进：

- **方案C（主线）**：多同构图 + 门控融合，Week 3 实现，文献最充分
- **方案B（消融对照）**：真异构图，Week 4 实现，验证跨类型节点的额外收益

---

## 二、方案C（主线）：多同构图门控融合

### 2.1 图结构

节点类型：**单一**（网格节点，1024个）

构建三条独立同构图，共享节点集合，分别建模不同空间关系：

| 图名 | 边定义 | 边数量（估计） | 实现依据 |
|------|--------|--------------|---------|
| **G_spatial** 空间邻接图 | 8-邻域（Moore邻域），边权 = exp(−d²/σ²) | ~8192 条 | 地理距离衰减 |
| **G_poi** POI语义图 | 格间 POI 类别分布余弦相似度 > 阈值τ | ~5000~15000 条 | OSM POI 12维向量 |
| **G_flow** 流量相关图 | 训练集 Pearson 相关系数 > 0.6 的格对 | ~3000~8000 条 | 历史 inflow 时序 |

### 2.2 聚合方式

每图独立 2 层 GCN，门控网络学习三图权重动态加权：

```
输入节点特征 h ∈ R^{N×d}
    ↓ enc: Linear(T_in×F_in → d)
    ├── GCN×2 on G_spatial  → h_s
    ├── GCN×2 on G_poi       → h_p
    └── GCN×2 on G_flow      → h_f
    ↓ gate(h) → w ∈ R^{N×3}  (Softmax)
    fused = w[:,0]*h_s + w[:,1]*h_p + w[:,2]*h_f
    ↓ 时序建模（STGCN / STAEformer）
    输出预测
```

### 2.3 实现规格

```python
# 存储路径（Week 2 特征工程产出）
G_spatial : data/processed/graph_spatial.pt     # edge_index + edge_weight
G_poi     : data/processed/graph_poi.pt         # edge_index + edge_weight
G_flow    : data/processed/graph_flow.pt        # edge_index + edge_weight

# 框架：PyTorch Geometric
from torch_geometric.nn import GCNConv
```

**超参数**：
- POI 图相似度阈值 τ：0.5（待消融）
- 流量相关图阈值：Pearson r > 0.6
- 空间图带宽 σ：1.5格（约 2km）
- GCN 隐藏维度 d：64

### 2.4 文献性能参考

| 相似工作 | 数据集 | vs 单图 MAE |
|---------|--------|------------|
| STAHGNet (2025) | PeMSD3/4 | ↓ 1~2% |
| Cross-Attn Multi-Graph GCN (2021) | PeMS04/08 | ↓ 5~15% |
| DMGF-Net, TKDD 2023 | 多数据集 | ↓ 5~12% |

---

## 三、方案B（消融对照）：真异构图

### 3.1 图结构

节点类型：**两类**

| 节点类型 | 数量 | 特征维度 | 来源 |
|---------|------|---------|------|
| `grid` 网格节点 | 1024（32×32） | 33维静态 + 动态时序 | TaxiBJ 流量数据 |
| `poi` POI节点 | ~500（聚类后） | 12维 POI 类别嵌入 | OSM 提取后 K-means 聚类 |

边类型：**三类**

| 边类型 | 方向 | 定义 | Meta-path 含义 |
|--------|------|------|---------------|
| `(grid, adj, grid)` | 格→格 | 8-邻域空间邻接 | 地理相邻关系 |
| `(poi, inf, grid)` | POI→格 | POI 节点在格内或半径 1km 内 | POI 对格子的语义影响 |
| `(grid, rev, poi)` | 格→POI | 双向更新（保证 POI 节点也被更新） | 反向消息 |

### 3.2 聚合方式

基于 HAN（异构图注意力网络）风格，按 meta-path 分别聚合：

```
{grid: h_g, poi: h_p}
    ↓ HeteroConv Layer 1
    h_g' = Aggregate([grid→grid消息, poi→grid消息])
    h_p' = Aggregate([grid→poi消息])
    ↓ HeteroConv Layer 2（同上）
    ↓ 取 h_g'' 接时序建模
    输出预测
```

### 3.3 实现规格

```python
# 框架：PyTorch Geometric HeteroConv
from torch_geometric.nn import HeteroConv, SAGEConv

hetero_conv = HeteroConv({
    ('grid', 'adj', 'grid'): SAGEConv(d, d),
    ('poi',  'inf', 'grid'): SAGEConv((d, d), d),
    ('grid', 'rev', 'poi' ): SAGEConv((d, d), d),
}, aggr='sum')

# 存储路径
node_feat_grid : data/processed/node_features_bj.npy   # (1024, 33)
node_feat_poi  : data/processed/poi_clusters_bj.npy    # (500, 12)
hetero_edges   : data/processed/hetero_edges_bj.pt     # dict of edge_index
```

**POI 聚类方案**：对 OSM 提取的全量 POI（数万个），按 500m 网格聚合后取每格 POI 类别向量均值，再 K-means 聚类为 500 个超级 POI 节点，减少异构图规模。

### 3.4 文献性能参考

| 相似工作 | 数据集 | vs 同构基线 MAE |
|---------|--------|----------------|
| STHGFormer (2023) | 道路速度预测 | ↓ 6~8.5% |
| HASTN (2024) | 交通流量 | 优于纯路网图 |
| MOHER, AAAI 2021 | 多模态站点 | 显著优于同构 |

> **注意**：TaxiBJ 栅格场景（非路段场景）目前无直接顶会验证，方案B在本项目中属探索性工作。

---

## 四、两方案对比

| 维度 | 方案C（主线） | 方案B（消融） |
|------|-------------|------------|
| 节点类型 | 1类（grid） | 2类（grid + poi） |
| 边类型数 | 3（逻辑上），实现为独立图 | 3（结构上异构） |
| POI 引入方式 | 节点特征（33维静态向量） | 显式 POI 节点 |
| 实现框架 | PyG 标准 GCNConv | PyG HeteroConv |
| 本机训练时长（100 epoch 骨干） | ~4.7 min | ~2.8 min |
| 完整模型预估（含时序）| 15~25 min | 20~35 min |
| 显存占用 | ~24 MB（骨干）| ~24 MB（骨干） |
| 文献支撑 | 强（多篇直接对标） | 中（道路场景为主） |
| 工程复杂度 | 中 | 高 |

---

## 五、G_transit 扩展方案（可选第四图）

### 5.1 动机

现有三图（G_spatial/G_poi/G_flow）均从静态地理或历史统计角度建模。G_transit 可从**公共交通网络拓扑**角度补充新的连接关系：共享同一地铁线路的格子之间存在强流量关联，这种关联既非地理邻近也非 POI 相似，是现有三图的盲区。

### 5.2 边定义

**方案 T1：地铁线路连接**（主推）

两个格子共享同一地铁线路（即两格内均有属于同一线路的站点），则添加边：

```
edge(i, j) if ∃ line L: station(i) ∈ L AND station(j) ∈ L
edge_weight = exp(−hop_distance / σ_t)   # hop_distance = 两站之间的站数
```

OSM 数据支撑：地铁线路以 `route=subway` 的 relation 存储，包含 member 站点的有序列表。

**方案 T2：多模态站点连接**（可作为 T1 扩展）

在 T1 基础上，将共享自行车租赁站（`amenity=bicycle_rental`）作为软连接：  
两格距离 ≤ 2km 且均有 bicycle_rental → 加入弱边（权重衰减 × 0.3），反映骑行接驳可达性。

### 5.3 实现路径

```python
# 需解析 OSM relation（需 osmium way/relation handler）
class SubwayLineHandler(osmium.SimpleHandler):
    def relation(self, r):
        if r.tags.get("route") == "subway":
            # 收集 member node id（站点）及线路名
            ...
```

### 5.4 纳入消融实验

| 实验 | 配置 |
|------|------|
| E5（现方案C） | G_spatial + G_poi + G_flow，门控融合 |
| E7（+G_transit T1） | 四图门控融合 |
| E8（+G_transit T2） | 四图 + bicycle_rental 软边 |

> **当前状态**：G_transit 为 Week 4+ 选做项，先跑通 E5 再决定是否扩展。

---

## 六、实现计划

| 周次 | 任务 | 产出 |
|------|------|------|
| **Week 2** | 构建 G_spatial、G_poi、G_flow 的边集（需 OSM POI 提取） | `data/processed/graph_*.pt` |
| **Week 3** | 实现方案C图构建 + 接入 STGCN 基线，跑通训练循环 | MAE 基准数字 |
| **Week 4** | 实现方案B HeteroConv，替换 STGCN 空间模块 | 方案B vs C MAE 对比 |
| **Week 4** | 消融实验：单图 / 双图 / 三图 / 门控 vs 均值融合 | 消融表 |

---

## 七、消融实验设计（Week 4）

验证方案C中每个图的贡献，以及方案B vs C 的收益：

| 实验组 | 配置 |
|--------|------|
| E1 | 仅 G_spatial（同构单图基线） |
| E2 | G_spatial + G_poi |
| E3 | G_spatial + G_flow |
| E4 | G_spatial + G_poi + G_flow，均值融合 |
| **E5**（方案C） | G_spatial + G_poi + G_flow，**门控融合** |
| **E6**（方案B） | HeteroConv（grid + poi节点，3类边） |
| E7（G_transit T1） | 四图门控融合，加入地铁线路拓扑图 |
| E8（G_transit T2） | 四图 + bicycle_rental 软边扩展 |

评估指标：TaxiBJ 测试集（BJ16）MAE / RMSE / MAPE，预测步长 1步 / 6步 / 12步。
