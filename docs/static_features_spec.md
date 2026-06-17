# 静态特征维度规格

**版本**: v1.0  
**日期**: 2026-06-13  
**目标**: 为每个网格节点定义固定的静态属性向量，作为 GNN 节点初始嵌入

---

## 一、静态特征的作用

动态特征（人流时序）表达"发生了什么"，静态特征表达"这个地方是什么"。
在 STAEformer/TESTAM 等自适应嵌入模型中，静态特征可替代或增强可学习节点 ID 嵌入，使模型在城市迁移时保持可解释性。

---

## 二、TaxiBJ 32×32 网格节点特征

每个节点 (row, col) 对应一个 1.33km×1.73km 的格子，静态特征向量维度合计 **33维**：

### 2.1 地理位置特征（4维）

| 特征 | 维度 | 取值 | 计算方式 |
|------|------|------|---------|
| 归一化纬度 | 1 | [0,1] | `lat_norm = (lat_center - 39.75) / 0.50` |
| 归一化经度 | 1 | [0,1] | `lon_norm = (lon_center - 116.25) / 0.50` |
| 到城市中心距离 | 1 | km，归一化 | 到天安门(116.3975,39.9087)的球面距离 |
| 环路区域 ID | 1 | 1~5（二环内~五环外） | 按格子中心到天安门距离分桶 |

```python
# 预计算所有格子的中心坐标
LON_MIN, LAT_MAX = 116.25, 40.25
CELL_LON, CELL_LAT = 0.5/32, 0.5/32

rows, cols = np.meshgrid(np.arange(32), np.arange(32), indexing='ij')
lat_center = LAT_MAX - (rows + 0.5) * CELL_LAT
lon_center = LON_MIN + (cols + 0.5) * CELL_LON
```

### 2.2 POI 类别密度（12维，来自 OSM）

从 OpenStreetMap 北京 PBF 提取各类 POI，聚合到 32×32 格子，归一化为密度（个/km²）后 log1p 变换：

| 维度编号 | POI 类别 | OSM key/value |
|---------|---------|--------------|
| 0 | 餐饮 | `amenity=restaurant/cafe/fast_food` |
| 1 | 购物 | `shop=*` |
| 2 | 教育 | `amenity=school/university/college` |
| 3 | 医疗 | `amenity=hospital/clinic/pharmacy` |
| 4 | 交通枢纽 | `public_transport=station`, `railway=station` |
| 5 | 公交站 | `highway=bus_stop` |
| 6 | 办公商业 | `office=*`, `building=commercial` |
| 7 | 酒店住宿 | `tourism=hotel/hostel` |
| 8 | 文体娱乐 | `leisure=*`, `amenity=cinema/theatre` |
| 9 | 公园绿地 | `leisure=park/garden` |
| 10 | 政务机构 | `amenity=government/townhall` |
| 11 | 居住区 | `building=residential/apartments` |

```python
# Week 2 实现骨架
import osmium
def extract_poi_to_grid(pbf_path, grid_bounds, grid_size=(32,32)):
    """返回 (H, W, 12) 的 POI 密度矩阵"""
    ...
```

### 2.3 路网特征（4维，来自 OSM）

| 特征 | 维度 | 描述 |
|------|------|------|
| 路网密度 | 1 | 格内道路总长度（km/km²），log1p归一化 |
| 主干道比例 | 1 | motorway/trunk/primary 占总路长比例 |
| 交叉口密度 | 1 | 格内道路交叉口数/km² |
| 是否含地铁站 | 1 | 二值：格内有地铁站=1，否则=0 |

### 2.4 卫星遥感特征（5维，来自 Landsat-8）

Landsat-8 分辨率 30m，需重采样到 32×32 格（每格约 44×57 像素平均）：

| 特征 | 维度 | 计算公式 | 含义 |
|------|------|---------|------|
| NDVI | 1 | `(B5-B4)/(B5+B4)` | 植被指数，[-1,1] |
| NDBI | 1 | `(B6-B5)/(B6+B5)` | 建筑指数，正值=城区 |
| MNDWI | 1 | `(B3-B6)/(B3+B6)` | 水体指数，正值=水域 |
| B4 均值（红波段） | 1 | 平均地表反射率 | 建筑/裸地亮度 |
| B4 标准差 | 1 | 格内像素方差 | 土地利用异质性 |

> 使用前需将 DN 值转为地表反射率：`ρ = REFLECTANCE_MULT_BAND_n × DN + REFLECTANCE_ADD_BAND_n`（参数在 `*_MTL.txt` 中）

### 2.5 时间不变特征（8维）

| 特征 | 维度 | 描述 |
|------|------|------|
| 历史均值（全局） | 1 | 训练集上该格的 inflow 全局均值，归一化 |
| 历史峰值比 | 1 | 该格 p99 / 全局 p99（反映绝对流量量级） |
| 历史零值率 | 1 | 该格 inflow=0 的时步比例 |
| 工作日峰均比 | 1 | 工作日早高峰均值 / 全天均值（活跃度） |
| 周末峰均比 | 1 | 周末均值 / 工作日均值（功能区分） |
| 夜间占比 | 1 | 0:00~6:00 流量 / 全天流量（夜生活区） |
| 流量方向偏差 | 1 | mean(inflow - outflow)，反映净流入/流出 |
| 流量波动性 | 1 | std(inflow) / mean(inflow)（变异系数） |

---

## 三、TaxiNYC 15×5 网格节点特征

网格更小（15×5=75 节点），特征设计与 TaxiBJ 对齐，差异点：

| 差异 | TaxiBJ | TaxiNYC |
|------|--------|---------|
| 格子尺寸 | 1.33×1.73 km | 约 0.55×0.55 km（曼哈顿网格更密） |
| 坐标范围 | 北京五环内 | 曼哈顿岛 (40.70~40.78N, 73.97~74.01W) |
| Landsat-8 | 北京场景 2015-04-16 | 纽约场景 2014-04-10 |
| 地铁数据 | 北京地铁 | NYC Subway GTFS（可从 MTA 开放数据获取） |

---

## 四、特征矩阵规格汇总

| 数据集 | 节点数 | 特征维度 | 存储规格 |
|--------|--------|---------|---------|
| TaxiBJ | 1024（32×32） | 33 维 | `(1024, 33)` float32，约 128 KB |
| TaxiNYC | 75（15×5） | 33 维 | `(75, 33)` float32，约 9.4 KB |

**存储路径**：`data/processed/node_features_bj.npy` / `node_features_nyc.npy`

---

## 五、特征工程执行计划

| 周次 | 任务 | 产出 |
|------|------|------|
| Week 2 | 2.1 地理位置特征（纯数学，无需解析） | `geo_features.npy` |
| Week 2 | 2.3 路网密度（osmium 解析 OSM PBF） | `road_features.npy` |
| Week 2 | 2.5 历史统计特征（从训练集计算） | `hist_features.npy` |
| Week 3 | 2.2 POI 密度（12类，OSM 解析） | `poi_features.npy` |
| Week 3 | 2.4 Landsat-8 遥感指数（rasterio 处理） | `satellite_features.npy` |
| Week 3 | 合并为最终节点特征矩阵 | `node_features_bj.npy` |

---

## 六、特征重要性预判

按对人流预测的预期贡献排序（基于文献）：

1. **历史统计特征**（8维）— 直接反映流量模式，影响最大
2. **POI 密度**（12维）— 决定区域功能，对早高峰/晚高峰幅度影响大
3. **路网特征**（4维）— 决定区域通达性，对拥堵传播建模重要
4. **地理位置**（4维）— 捕捉城区位置效应（中心 vs 边缘）
5. **遥感特征**（5维）— 土地利用辅助，对细粒度区分有补充价值
