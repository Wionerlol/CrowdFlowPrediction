# 数据预处理报告

**项目**：城市人流异常检测  
**日期**：2026-06-17  
**阶段**：Week 2 — 多源数据清洗与标准化

---

## 一、数据集概览

| 数据集 | 用途 | 原始格式 | 时间范围 | 空间范围 |
|--------|------|---------|---------|---------|
| TaxiBJ | 主训练集（北京人流） | .mat (HDF5) | 2013–2016，共 14 个子集 | 北京五环内 32×32 网格 |
| TaxiNYC | 迁移验证集（纽约人流） | .npz | 2014 全年 | 曼哈顿 15×5 网格 |
| METR-LA | 模型验证基线（交通速度） | .parquet + .pkl | 2012-03 ~ 2012-06 | 洛杉矶 207 传感器 |
| OpenWeather | 气象协变量 | .csv (30min) | 与流量数据对齐 | 城市级统一 |
| ERA5 (Open-Meteo) | 气象协变量（主源） | .h5 (本地时) | 与流量数据对齐 | 城市级统一 |
| OSM PBF | 路网 + POI 提取 | .pbf | 静态（采集时间点） | 北京 / 纽约 bbox |
| Landsat-8 | 遥感静态特征 | L1TP .TIF | 北京 2015-04-16，纽约 2014-04-10 | 场景覆盖研究区 |

---

## 二、TaxiBJ 清洗

### 2.1 原始数据结构

- 格式：`.mat` 文件（HDF5），变量 `data`（T, 2, 32, 32），类型 `float32`
- 日期字段：字符串数组 `date`，格式 `YYYYMMDDSS`（SS = 步长编号 01~48，每步 30 分钟）
- 通道：`data[:,0,:,:]` = inflow，`data[:,1,:,:]` = outflow
- 时区：本地时 UTC+8，日期字段表示北京时

### 2.2 处理步骤

**时间戳解析**

`YYYYMMDDSS` → 北京本地时 → UTC：

```
步长 SS 对应时刻 = date_part + (SS-1) × 30 分钟
UTC = 本地时 − 8小时
```

最终时间戳以 `datetime64[s]` 格式存储，无时区歧义。

**异常值检测（per-cell z-score）**

对每个网格 (row, col, channel) 计算全时序均值和标准差，标记 |z| > 5 的时步为异常。处理方式：线性插值（前后各一个正常值之间）。

**连续性验证**

检查时间戳序列间隔是否均匀（30 分钟）。14 个子集合并后验证无重复时步、无跨集跳跃。

### 2.3 输出

| 文件 | 规格 | 说明 |
|------|------|------|
| `taxibj_clean.npz` | `data`: (22484, 2, 32, 32) float32 | inflow / outflow 计数 |
| | `timestamps`: (22484,) datetime64[s] UTC | 时间戳 |

**基本统计：**

| 指标 | inflow | outflow |
|------|--------|---------|
| 最小值 | 0 | 0 |
| 最大值 | 1285 | — |
| 均值 | 103.86 | — |
| 零值率 | 5.2% | — |

---

## 三、TaxiNYC 清洗

### 3.1 原始数据结构

- 格式：`.npz`，含 `data`（17520, 2, 15, 5）和 `timestamps`（字符串数组）
- 时间步长：30 分钟；总时步 T = 365 × 48 = 17,520
- 网格：15 行 × 5 列，覆盖曼哈顿（lon [-74.0166, -73.9004]，lat [40.6996, 40.9196]）
- 时区：本地时 UTC−5

### 3.2 处理步骤

- 时间戳转 UTC：字符串解析 → 本地时 → `+ 5小时` → UTC
- 零值检查：NYC 零值率 12.0%，高于北京，因网格边缘含河道/港口区域，属正常分布
- 坐标系统一：WGS84，格子中心坐标按等距分割计算

### 3.3 输出

| 文件 | 规格 |
|------|------|
| `taxinyc_clean.npz` | `data`: (17520, 2, 15, 5) float32；`timestamps`: datetime64[s] UTC |

**基本统计：** inflow 均值 113.22，最大值 1430，零值率 12.0%

---

## 四、METR-LA 清洗

### 4.1 原始数据结构

- 格式：宽表 parquet（train/val/test 三份）
- 列名模式：`x_t-11_d0` ~ `x_t+0_d0`（输入速度 mph）、`x_t+0_d1`（日内时间编码 0~1）、`y_t+1_d0` ~ `y_t+12_d0`（预测目标）
- 索引：`node_id`（0~206）+ `t0_timestamp`（本地 PST 时间，UTC−8）

### 4.2 重建时序矩阵

原始数据为滑窗格式，非原始时序。重建策略：

```
取每条记录的 x_t+0_d0（当前时刻速度）
按 (node_id, t0_timestamp) pivot → (T, 207) 速度矩阵
本地 PST → UTC（统一 UTC−8，忽略 DST 简化处理）
```

**去重**：同 (node_id, t0_timestamp) 重复条目取均值。  
**缺失时步**：线性插值补全，前向/后向填充处理边界。  
**值域约束**：速度超出 [0, 100] mph 的值裁剪（数据中未发现此类情况）。

### 4.3 邻接矩阵

- 来源：`adj_mx.pkl`，高斯距离衰减权重（传感器间道路距离）
- 对称误差 max|A−Aᵀ| = 0.996，**非对称**——原始设计为有向图（单行道影响）
- 非零边数：1,722 条；对角线均为 1

### 4.4 输出

| 文件 | 规格 |
|------|------|
| `metrla_clean.npz` | `data`: (34249, 207) float32 速度 mph；`timestamps`: datetime64[s] UTC |
| `metrla_adj.npy` | (207, 207) float32 有向邻接矩阵 |
| `metrla_sensor.csv` | 207 传感器经纬度（WGS84） |

---

## 五、气象数据合并

### 5.1 数据来源

| 来源 | 格式 | 变量 | 时间粒度 |
|------|------|------|---------|
| ERA5 (Open-Meteo) | `.h5`，本地时 YYYYMMDDHHII | temperature(°C), wind_speed(m/s), precipitation(mm), weather_code(WMO) | 30 分钟 |
| OpenWeather | `.csv`，UTC 时间戳 | temp, dew_point, pressure, humidity, wind_deg, wind_speed, rain_1h, snow_1h, clouds_all, weather_id, weather_code | 30 分钟 |

### 5.2 合并策略

1. ERA5 本地时转 UTC（北京 −8h，纽约 +5h）
2. 以 UTC 时间戳为键做 outer join
3. 重叠变量以 ERA5 为准：`temperature`、`wind_speed`、`weather_code`
4. OW 独有变量全部保留：dew_point、pressure、humidity、wind_deg、rain_1h、snow_1h、clouds_all、weather_id
5. 边界缺口（极少量）：前向/后向填充

### 5.3 输出

| 文件 | 规格 | NaN |
|------|------|-----|
| `weather_bj_merged.csv` | (48736, 12) — 覆盖 BJ 全时段 | 0 |
| `weather_nyc_merged.csv` | (17520, 12) — 与 TaxiNYC 对齐 | 0 |

**12 维气象特征：** temperature, wind_speed, precipitation, weather_code, dew_point, pressure, humidity, wind_deg, rain_1h, snow_1h, clouds_all, weather_id

---

## 六、网格元数据生成

### 6.1 规格

| 城市 | H×W | lon 范围 | lat 范围 | 格子尺寸 | 面积/格 |
|------|-----|---------|---------|---------|---------|
| 北京 | 32×32 | [116.25, 116.75] | [39.75, 40.25] | 约 1.33km × 1.73km | 2.32 km² |
| 纽约 | 15×5 | [−74.0166, −73.9004] | [40.6996, 40.9196] | 约 2.25km × 1.47km | 3.20 km² |

网格编号：`grid_id = row × W + col`（行从北向南，列从西向东）

### 6.2 坐标精度说明

TaxiBJ 和 TaxiNYC 均为预聚合网格数据，原始 GPS 轨迹不可获取，网格边长约 1.3~3.2 km，无法满足 500m 精度要求。本项目保留原始网格，在报告中注明此限制。

### 6.3 输出

| 文件 | 字段 | 行数 |
|------|------|------|
| `grid_meta_bj.csv` | grid_id, row, col, center_lon, center_lat, lon_min, lon_max, lat_min, lat_max, area_km2 | 1024 |
| `grid_meta_nyc.csv` | 同上 | 75 |

---

## 七、OSM 数据说明

| 城市 | 文件 | 大小 | 来源 | 状态 |
|------|------|------|------|------|
| 北京 | `beijing-latest.osm.pbf` | 35 MB | Geofabrik | 完整 |
| 纽约 | `nyc-latest.osm.pbf` | 151 MB | BBBike | 完整（初次下载 30MB 文件损坏，已重新下载） |

OSM 数据无需单独清洗，直接在特征提取时按 bbox 过滤，交由 osmium 库解析。

---

## 八、坐标系与时间戳统一

| 维度 | 规范 |
|------|------|
| 坐标系 | 统一使用 WGS84（EPSG:4326）；路网长度计算时投影至 UTM（北京 EPSG:32650，纽约 EPSG:32618） |
| 时间戳 | 统一存储为 UTC，格式 `datetime64[s]`（numpy），避免对象数组序列化问题 |
| 流量单位 | 人次计数（整数），无量纲，无需单位转换 |
| 速度单位 | METR-LA 为 mph，不与流量数据混用 |

---

## 九、产出物清单

```
data/processed/
├── taxibj_clean.npz          (22484, 2, 32, 32) 北京人流
├── taxinyc_clean.npz         (17520, 2, 15,  5) 纽约人流
├── metrla_clean.npz          (34249, 207)        LA 速度
├── metrla_adj.npy            (207, 207)          LA 邻接矩阵
├── metrla_sensor.csv         207 传感器坐标
├── weather_bj_merged.csv     (48736, 12)         北京气象
├── weather_nyc_merged.csv    (17520, 12)         纽约气象
├── grid_meta_bj.csv          (1024, 10)          北京网格元数据
└── grid_meta_nyc.csv         (75,   10)          纽约网格元数据
```
