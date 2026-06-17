# 特征工程报告

**项目**：城市人流异常检测  
**日期**：2026-06-17  
**阶段**：Week 2/3 — 时空语义特征工程 + 多模态图结构构建

---

## 一、特征体系总览

本项目特征分为三类，分别服务于不同的模型输入角色：

| 类别 | 内容 | 维度 | 变化维度 | 文件 |
|------|------|------|---------|------|
| **静态节点特征** | 地理位置、路网、POI、遥感、历史统计 | (N, 36) | 仅随空间变化 | `node_features_{city}.npy` |
| **动态时间特征** | 时刻/星期/月份周期编码、节假日 | (T, 12) | 仅随时间变化 | `temporal_features_{city}.csv` |
| **动态气象特征** | 温度、风速、降水等 12 维 | (T, 12) | 仅随时间变化（全城统一） | `weather_{city}_merged.csv` |

> **气象数据**为外部时间协变量，不纳入时空特征工程，在模型阶段作为全局上下文拼接至时间编码。

---

## 二、动态时间特征

### 2.1 编码方式

采用正弦余弦编码将周期性时间变量映射到连续向量空间，避免时间边界处（如 23:59 → 00:00）的数值跳跃：

```
sin_enc = sin(2π × value / period)
cos_enc = cos(2π × value / period)
```

### 2.2 特征列表

| 特征名 | 周期 | 含义 |
|--------|------|------|
| `slot_sin` / `slot_cos` | 48 | 日内第几个 30min 步（0~47） |
| `hour_sin` / `hour_cos` | 24 | 小时（0~23） |
| `weekday_sin` / `weekday_cos` | 7 | 星期（0=周一，6=周日） |
| `month_sin` / `month_cos` | 12 | 月份（1~12） |
| `doy_sin` / `doy_cos` | 366 | 年内第几天（1~366） |
| `is_weekend` | — | 二值：0=工作日，1=周末 |
| `is_holiday` | — | 二值：0=普通日，1=节假日 |

### 2.3 节假日规则

**北京**：从 `BJ_Holiday.txt` 读取（105 天，含法定节假日及调休安排），节假日占比 8.0%。

**纽约**：按美国联邦假日算法生成，含固定假日（元旦、独立日、退伍军人节、圣诞节）和浮动假日（MLK Day、总统日、阵亡将士纪念日、劳动节、哥伦布日、感恩节），节假日占比 2.7%。

### 2.4 输出

| 文件 | 规格 |
|------|------|
| `temporal_features_bj.csv` | (22484, 12)，索引为北京本地时 |
| `temporal_features_nyc.csv` | (17520, 12)，索引为纽约本地时 |

---

## 三、静态空间特征

### 3.1 地理距离特征

使用 Haversine 公式计算每个网格中心点到城市关键地标的球面距离（km）：

| 特征 | 维度 | 地标定义 |
|------|------|---------|
| `dist_center_km` | 1 | 到城市中心（北京：天安门；纽约：中城曼哈顿） |
| `dist_hub_min_km` | 1 | 到最近交通枢纽（最小值） |
| `dist_hub{1~4}_km` | 4 | 到各枢纽距离（北京：北京站/西站/南站/首都机场；纽约：Penn/Grand Central/JFK/WTC） |
| `dist_commercial_min_km` | 1 | 到最近商业中心（最小值） |
| `dist_com{1~4}_km` | 4 | 到各商业中心（北京：王府井/三里屯/中关村/西单；纽约：时代广场/华尔街/哥伦布圆环/上东区） |

共 11 维，存储于 `spatial_features_{city}.csv`。

### 3.2 路网密度特征

从 OSM PBF 文件提取道路几何，使用 geopandas 空间裁剪（`overlay intersection`）将道路按网格边界分割，投影至 UTM 坐标系计算精确长度：

| 特征 | 计算方式 |
|------|---------|
| `road_density_km_km2` | 格内道路总长度 / 格面积 (km/km²) |
| `highway_density` | motorway/trunk/primary/secondary 长度 / 格面积 |
| `local_density` | tertiary/unclassified/residential 长度 / 格面积 |
| `service_density` | service 道路长度 / 格面积 |
| `active_density` | 步道/自行车道/人行道长度 / 格面积 |

**路网统计：**

| 城市 | 提取道路数 | 路网密度均值 |
|------|----------|------------|
| 北京 | 90,380 条 | 10.0 km/km² |
| 纽约 | 114,938 条 | 34.3 km/km² |

纽约密度显著高于北京，符合曼哈顿高密度格网路网的实际情况。

### 3.3 POI 语义特征

从 OSM PBF 节点提取兴趣点，按 12 个类别分类后聚合到网格：

| 类别 | 北京数量 | 纽约数量 | 主要 OSM Tag |
|------|---------|---------|-------------|
| food | 5,326 | 10,240 | amenity=restaurant/cafe/fast_food |
| shopping | 1,920 | 3,809 | shop=supermarket/mall/clothes/... |
| entertainment | 249 | 407 | leisure=cinema, amenity=theatre/... |
| office | 1,838 | 1,811 | office=*, amenity=bank |
| residential | 104 | 28 | building=apartments/residential |
| education | 251 | 614 | amenity=school/university/college |
| healthcare | 323 | 1,199 | amenity=hospital/clinic/pharmacy |
| transport | 23,549 | 7,639 | railway=station, public_transport=* |
| government | 265 | 137 | amenity=townhall/police/courthouse |
| tourism | 1,878 | 1,833 | tourism=*, amenity=place_of_interest |
| sports | 342 | 729 | leisure=sports_centre/stadium/gym |
| religious | 39 | 334 | amenity=place_of_worship |
| **合计** | **36,084** | **28,780** | |

每类输出两列：`{cat}_count`（数量）和 `{cat}_density`（个/km²），共 24 列。

> **注意**：北京交通类 POI 占 65%（地铁/公交站密集），纽约占 27%。此比例在后续 G_poi 图构建时需排除 transport 类，防止相似度退化。

### 3.4 遥感特征（Landsat-8）

**数据**：L1TP 级别 DN 值，使用 MTL.txt 校正参数转换为 TOA（大气层顶）反射率：

```
ρ = REFLECTANCE_MULT_BAND_n × DN + REFLECTANCE_ADD_BAND_n
ρ_adj = ρ / sin(太阳高度角)
```

北京太阳高度角 54.80°（2015-04-16），纽约 53.04°（2014-04-10）。

裁剪至研究区 bbox 后重投影至 WGS84（分辨率 0.00025°≈25m），按网格聚合像素均值：

| 特征 | 公式 | 含义 | 北京均值 | 纽约均值 |
|------|------|------|---------|---------|
| `ndvi` | (B5−B4)/(B5+B4) | 植被指数 | 0.219 | 0.034 |
| `ndbi` | (B6−B5)/(B6+B5) | 建筑密度指数 | −0.022 | −0.070 |
| `mndwi` | (B3−B6)/(B3+B6) | 水体指数 | −0.225 | 0.009 |
| `b4_mean` | 红波段平均反射率 | 建筑/裸地亮度 | 0.142 | 0.115 |
| `b4_std` | 红波段标准差 | 土地利用异质性 | 0.033 | 0.035 |

**物理解读：**
- 北京 NDVI 高于纽约（春季有植被），MNDWI 负值（以陆地为主）
- 纽约 MNDWI 均值接近 0（哈德逊河/东河边界格子贡献水体信号，std=0.23 较大）
- 纽约 NDBI 低于北京（曼哈顿超高层建筑导致 B6 反射率相对 B5 偏低）

---

## 四、静态节点特征矩阵

### 4.1 特征拼接规格

将上述所有静态特征合并为每节点 36 维向量：

| 索引 | 分组 | 维度 | 归一化方式 |
|------|------|------|----------|
| [0:4] | 地理位置 | 4 | lat/lon/dist_center: min-max；ring_id: 1~5 → /5 |
| [4:6] | 到枢纽/商圈最短距离 | 2 | min-max [0,1] |
| [6:11] | 路网密度（5类） | 5 | log1p 后 z-score |
| [11:23] | POI 密度（12类） | 12 | log1p 后 z-score |
| [23:28] | 遥感指数 | 5 | z-score |
| [28:36] | 历史流量统计 | 8 | 见下表 |

**历史统计特征（[28:36]）：**

| 特征 | 计算方式 | 归一化 |
|------|---------|-------|
| hist_mean | 全时段 inflow 均值 | min-max |
| p99_ratio | 该格 p99 / 全局 p99 | 直接使用 [0,1] |
| zero_rate | inflow=0 的时步比例 | 直接使用 [0,1] |
| wd_peak_ratio | 工作日 7~8 点均值 / 工作日全天均值 | clip [0,5] |
| we_ratio | 周末均值 / 工作日均值 | clip [0,5] |
| night_ratio | 0~5 点均值 / 全天均值 | clip [0,1] |
| flow_dir | mean(inflow − outflow)，净流向 | min-max |
| flow_cv | std(inflow) / mean(inflow)，变异系数 | clip [0,5] |

### 4.2 输出

| 文件 | 规格 | NaN |
|------|------|-----|
| `node_features_bj.npy` | (1024, 36) float32 | 0 |
| `node_features_nyc.npy` | (75, 36) float32 | 0 |

---

## 五、多模态图结构（方案C 主线）

为每个城市构建三张独立同构图，共享节点集合，后续通过门控网络融合：

### 5.1 G_spatial — 空间邻接图

**构建方式**：Moore 8邻域，边权 = 高斯距离衰减

```
w(i,j) = exp(−d(i,j)² / σ²)，σ = 1.5 格
```

邻格距离：正交邻居 d=1，斜角邻居 d=√2，对应权重 0.641 和 0.411。

| 城市 | 边数 | 均值边/节点 |
|------|------|------------|
| 北京 | 7,812 | 7.6（边角节点 3 条，内部节点 8 条） |
| 纽约 | 484 | 6.5 |

### 5.2 G_poi — POI 功能相似图

**构建方式**：POI 类别密度向量余弦相似度，top-k kNN（k=10）

**关键设计决策**：北京交通类 POI 占 65%，若将其纳入相似度计算，几乎所有格对余弦相似度 > 0.99，图退化为完全图。解决方案：排除 transport 类别，使用其余 11 类计算相似度，保留 10 个最相似的邻居。

特征向量处理：`log1p(密度)` → L2 归一化 → 余弦相似度。

| 城市 | 边数 | 均值边/节点 | 相似度均值 |
|------|------|------------|----------|
| 北京 | 10,240 | 10.0 | 0.466 |
| 纽约 | 750 | 10.0 | 0.888 |

### 5.3 G_flow — 历史流量相关图

**构建方式**：训练集 inflow 时序 Pearson 相关矩阵，top-k kNN（k=15），仅保留正相关边

```
corr(i,j) = (X_i − μ_i)/σ_i · (X_j − μ_j)/σ_j / T
```

| 城市 | 边数 | 均值边/节点 | 相关系数均值 |
|------|------|------------|------------|
| 北京 | 15,255 | 14.9 | 0.857 |
| 纽约 | 1,125 | 15.0 | 0.691 |

北京流量相关性普遍高（均值 0.857），反映城市整体通勤模式的同步性；纽约相关性较低（0.691），与曼哈顿南北向交通结构差异有关。

### 5.4 图统计汇总

| 城市 | 图 | 节点数 | 边数 | 边/节点 |
|------|----|----|------|---------|
| 北京 | G_spatial | 1024 | 7,812 | 7.6 |
| 北京 | G_poi | 1024 | 10,240 | 10.0 |
| 北京 | G_flow | 1024 | 15,255 | 14.9 |
| 纽约 | G_spatial | 75 | 484 | 6.5 |
| 纽约 | G_poi | 75 | 750 | 10.0 |
| 纽约 | G_flow | 75 | 1,125 | 15.0 |

### 5.5 存储格式

每张图存储为 PyTorch `.pt` 文件，包含：

```python
{
    "edge_index" : torch.LongTensor,   # (2, E) 有向边索引
    "edge_weight": torch.FloatTensor,  # (E,)   边权重
    "num_nodes"  : int,                # 节点数
}
```

---

## 六、超参数说明

| 参数 | 值 | 说明 | 消融建议 |
|------|-----|------|---------|
| 空间图带宽 σ | 1.5 格 | 控制距离衰减速率 | 试 1.0, 2.0 |
| POI kNN k | 10 | 每节点 POI 邻居数 | 试 5, 20 |
| Flow kNN k | 15 | 每节点流量邻居数 | 试 10, 20 |
| POI 排除类别 | transport | 防止主导类别使图退化 | 也可用 log1p+L2 均衡后不排除 |

---

## 七、产出物清单

```
data/processed/
├── temporal_features_bj.csv     (22484, 12)  北京时间特征
├── temporal_features_nyc.csv    (17520, 12)  纽约时间特征
├── spatial_features_bj.csv      (1024,  16)  北京空间距离+路网
├── spatial_features_nyc.csv     (75,    16)  纽约空间距离+路网
├── poi_features_bj.csv          (1024,  24)  北京 POI（12类×count+density）
├── poi_features_nyc.csv         (75,    24)  纽约 POI
├── satellite_features_bj.csv    (1024,   5)  北京遥感指数
├── satellite_features_nyc.csv   (75,     5)  纽约遥感指数
├── node_features_bj.npy         (1024,  36)  北京节点静态特征矩阵
├── node_features_nyc.npy        (75,    36)  纽约节点静态特征矩阵
├── graph_spatial_bj.pt          edges=7812   北京空间邻接图
├── graph_poi_bj.pt              edges=10240  北京 POI 语义图
├── graph_flow_bj.pt             edges=15255  北京流量相关图
├── graph_spatial_nyc.pt         edges=484    纽约空间邻接图
├── graph_poi_nyc.pt             edges=750    纽约 POI 语义图
└── graph_flow_nyc.pt            edges=1125   纽约流量相关图
```

---

## 八、已知局限

| 局限 | 说明 | 影响 |
|------|------|------|
| 网格粒度 | 北京/纽约格子尺寸 1.3~3.2 km，远大于 500m 目标 | 细粒度空间模式无法捕捉 |
| NYC路网 | 初始下载文件损坏（30MB，EOF），补下后恢复正常 | 已解决 |
| Landsat-8 单期 | 仅用单一时相卫星影像，无法反映季节变化 | 遥感特征为春季快照 |
| G_poi transport | 北京交通类 POI 主导，排除后图结构合理但损失一维信息 | 后续可用 log1p 均衡替代排除 |
| METR-LA DST | 洛杉矶夏令时统一按 UTC-8 处理，边界处约 1h 误差 | 仅影响跨 DST 切换的约 96 个时步 |
