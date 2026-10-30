# week2_report

![week2_report_p1_0.png](week2_report_files/week2_report_p1_0.png)

WEEK 2 · 数据预处理与特征工程

# 城市人流异常检测系统 第二周进度报告

基于多源时空数据融合的城市级人流异常检测与预警系统 报告日期 阶段 **2026-06-17** **Week 2 / 8** 覆盖城市 数据集 **北京（BJ）** **TaxiBJ ·** **· 纽约** **TaxiNYC ·** **（NYC）** **12数据源**

## 41 4 13 3.7GB

维节点静态特征 类多模态图结构 个预处理脚本 原始+预处理数据

<!-- Page 2 -->

**一、本周完成概览** Week 2 全部四项交付物已完成：数据预处理脚本 13 个、特征工程代码库（src/features/，16个 公开符号）、 预处理数据集（41维节点特征矩阵 + 4类多模态图，共 37 个文件）、数据质量分析 报告。

### 13 4 37 12

个脚本 个模块 个文件 个来源 数据预处理 Python 特征工程可导入代码 预处理产物（100 数据质量全覆盖分析 脚本 库 MB） **1.1 交付物清单** 数据预处理 Python 脚本（scripts/ 目录，13 个脚本） ✅ 特征工程代码库（src/features/ ，4 个模块，16 个公开符号，完整可导入） ✅ 预处理数据集：北京 (1024, 41) + 纽约 (75, 41) 节点特征矩阵，NaN=0 ✅ 预处理数据集：北京 + 纽约各 4 张多模态图（.pt 格式，PyTorch Geometric 兼容） ✅ 数据质量分析报告（覆盖全部 12 个数据来源） ✅ 特征工程技术报告（outputs/reports/feature_engineering_report.md ） ✅ 量化 / TensorRT 加速预研报告 ✅ 可视化界面设计文档 ✅ **1.2 数据规模说明**

|分类|大小|说明|
|---|---|---|
|data/raw/|3.6 GB|Landsat-8 2.3GB、MTA GTFS 0.5GB、METR-LA 0.2GB、OSM 0.2GB、<br>taxi 0.4GB 等|

data/ 100 MB 全部预处理产物（npz / csv / pt / npy，共 37 个文件） processed/

|合计|~3.7<br>GB|—|
|---|---|---|

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 2

<!-- Page 3 -->

**关于课题文档估计"约 20 GB"的说明：**该估计基于原始出租车 GPS 轨迹数据（逐条行程记录）。

本项目使用官方预聚合网格版本（TaxiBJ 32×32 / TaxiNYC 15×5），已将原始轨迹聚合为 inflow/outflow 计数， 数据量远小于原始轨迹，但信息完整，不影响模型训练与评估。

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 3

<!-- Page 4 -->

**二、数据预处理脚本** **2.1 脚本清单（scripts/ 目录）**

|脚本|功能|输出|
|---|---|---|
|`build_spatial_features.py`|路网密度（5类）+ 地理距<br>离特征（Haversine）|spatial_features_{city}.csv<br>(N×16)|

OSM POI 语义特征（11类 poi_features_{city}.csv (N×22) build_poi_features.py 功能性 POI，transport 已剥离）

|build satellite features.py<br>_ _|Landsat-8 DN→反射率，<br>NDVI/NDBI/MNDWI 指数|satellite features {city}.csv<br>_ _<br>(N×5)|
|---|---|---|

地铁站/入口/公交站密度 transit_features_{city}.csv (N×6) build_transit_features.py （OSM PBF）

|build subway features.py<br>_ _|轨道线路覆盖（政府xlsx<br>+ OSM名称匹配）+<br>G transit 图<br>_|subway grid features {city}.csv<br>_ _ _<br>(N×2)<br>graph transit {city}.pt<br>_ _|
|---|---|---|

公交线路数（OSM bus_route_features_{city}.csv build_bus_features.py route=bus 关系 + MTA (N×1) GTFS 5区）

|build temporal features.py<br>_ _|时间周期编码（sin/cos）<br>+ 节假日标志（12维）|temporal features {city}.csv<br>_ _<br>(T×12)|
|---|---|---|

G_spatial / G_poi / node_features_{city}.npy (N×41) build_graphs.py G_flow 构建 + 41维节点 graph_{spatial,poi,flow}_{city}.pt 特征合并

|merge weather.py<br>_|ERA5 +<br>OpenWeatherMap 气象<br>字段合并，插值至30分钟|weather {city} merged.csv<br>_ _|
|---|---|---|

一键顺序执行全部预处理 —（调度器） preprocess_all.py 步骤

|train stgcn.py<br>_|STGCN 基线模型训练<br>（PyTorch Geometric）|stgcn best.pt<br>_|
|---|---|---|

依赖与 GPU 环境验证 终端报告 verify_env.py

|setup env.sh<br>_|Col2|.venv/|
|---|---|---|

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 4

<!-- Page 5 -->

|Col1|Conda 环境 + 依赖安装<br>（支持 CUDA<br>118/121/128 / CPU）|Col3|
|---|---|---|

**2.2 公共交通特征管道（三步串联）** 公共交通特征的提取分三个脚本串联执行，分别产出不同粒度的特征：

build_transit_features.py   ─→地铁站/入口/公交站密度transit_features (N×6) ↓ build_subway_features.py    ─→轨道线路数/换乘枢纽subway_grid_features (N×2) ─→  轨道拓扑图graph_transit.pt ↓ build_bus_features.py       ─→网格内公交线路数bus_route_features (N×1) ↓ build_graphs.py             ─→合并 41 维节点特征矩阵node_features (N×41) **北京公交数据处理** OSM 从route=bus relation 关系直接提取 platform 节点成员，获取 2,098 条关系， bbox 内覆盖 率 88.4%；剩余未覆盖站点通过政府公交站点信息.xlsx （58,697 条记录，9,416 唯一站名） 模糊 匹配（阈值 0.80）补全 890 个站点。

**纽约公交数据处理（MTA GTFS 5区）** OSM 数据对纽约 TaxiNYC bbox 覆盖率仅 41.5%，故改用 MTA GTFS 5区公交数据。 5个子目录 （Brooklyn / Bronx / Manhattan / Queens / Staten Island）各含完整 routes.txt 和独立 stop_times.txt。 按 bbox 过滤后得 3,168 个唯一站点，255 条 route_type=3 公交线路，覆盖率 100%。

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 5

<!-- Page 6 -->

**三、特征工程代码库（src/features/）** src/features/ 为可导入 Python 包，提供 4 个独立模块和 16 个公开符号， 用于在训练/评估 代码中直接调用特征函数，不依赖任何中间文件路径。

**3.1 模块结构**

|模块|文件|公开函数|功能|
|---|---|---|---|
|`base`|base.py|`assign_to_grid`<br>`haversine_km`<br>`haversine_matrix`<br>`minmax`<br>`zscore`|网格坐标分配、球面距离计算<br>（Haversine）、归一化工具|

temporal.py 时间周期编码（sin/cos，12 temporal build_temporal 维）、节假日规则（北京/纽 load_bj_holidays 约） us_federal_holidays

|graphs|graphs.py|build spatial graph<br>_ _<br>build poi graph<br>_ _<br>build flow graph<br>_ _<br>save graph<br>_<br>load graph<br>_<br>graph stats<br>_|G spatial / G poi / G flow 构<br>_ _ _<br>建；.pt 格式存储/加载；边统<br>计|
|---|---|---|---|

node_features.py 组装 (N, 41) 节点特征矩阵；8 node_features build_node_features 维历史流量统计 hist_stats **3.2 典型用法** from src.features.base import assign_to_grid, haversine_km from src.features.temporal import build_temporal, load_bj_holidays from src.features.graphs import build_spatial_graph, save_graph, load_graph, graph_stats from src.features.node_features import build_node_features, hist_stats # 加载图 g = load_graph("data/processed/graph_transit_bj.pt") print(graph_stats("data/processed/graph_transit_bj.pt")) # → nodes=1024, edges=3080, avg_deg=3.0 # 组装节点特征 feat = build_node_features( grid_csv="data/processed/grid_bj.csv", spatial_csv="data/processed/spatial_features_bj.csv", poi_csv="data/processed/poi_features_bj.csv", 城市人流异常检测系统 ·Week 2报告 · 2026-06-17 6

<!-- Page 7 -->

satellite_csv="data/processed/satellite_features_bj.csv", transit_csv="data/processed/transit_features_bj.csv", subway_feat_csv="data/processed/subway_grid_features_bj.csv", bus_feat_csv="data/processed/bus_route_features_bj.csv", flow_npz="data/raw/taxibj/taxibj.npz", tz_offset=8 ) assert feat.shape == (1024, 41) and feat.dtype == "float32" 城市人流异常检测系统 ·Week 2报告 · 2026-06-17 7

<!-- Page 8 -->

**四、41 维静态节点特征矩阵** **4.1 特征维度规格**

|索引|分组|维<br>度|数据来源|归一化|
|---|---|---|---|---|
|**[0:4]**|地理位置（纬度/经度/城市中心距离/<br>环路ID）|4|坐标计算|lat/lon/dist: min-<br>max；ring: /5|

到交通枢纽/商业中心最短距离 2 Haversine min-max [0,1] **[4:6]**

|[6:11]|路网密度（总/主干/地方/服务/慢行）|5|OSM PBF|log1p → z-score|
|---|---|---|---|---|

POI 类别密度（11类功能性POI） 11 OSM PBF log1p → z-score **[11:22]**

|[22:27]|遥感指数（NDVI / NDBI / MNDWI /<br>B4均值 / B4标准差）|5|Landsat-8|z-score|
|---|---|---|---|---|

历史流量统计（均值/峰值比/零值率/ 8 TaxiBJ / 见注① **[27:35]** 时段模式等） TaxiNYC

|[35:38]|公共交通密度（地铁站/入口/公交<br>站）|3|OSM PBF|log1p → z-score|
|---|---|---|---|---|

轨道线路数 / 换乘枢纽标志 2 政府数据 + log1p / binary **[38:40]** OSM

|[40]|公交线路数|1|MTA GTFS /<br>OSM|log1p|
|---|---|---|---|---|

注①：hist_mean: min-max；p99_ratio: [0,1]；zero_rate: [0,1]；wd_peak_ratio / we_ratio: clip [0,5]；

night_ratio: clip [0,1]；flow_dir: min-max；flow_cv: clip [0,5]。

**4.2 各分组统计详情** **路网密度（[6:11]）**

|城市|提取道路数|路网密度均值|说明|
|---|---|---|---|
|北京|90,380 条|10.0 km/km²|OSM PBF 36.1 MB，UTM 50N 投影|

纽约 114,938 条

34. 3 km/km²

OSM PBF 151 MB（BBBike），曼哈顿格网路网密度高 **POI 语义特征（[11:22]）— transport 类完全剥离**

|类别|北京数量|纽约数量|主要 OSM Tag|
|---|---|---|---|

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 8

<!-- Page 9 -->

|food|5,326|10,240|amenity=restaurant/cafe/fast food<br>_|
|---|---|---|---|

shopping 1,920 3,809 shop=supermarket/mall/clothes/...

|entertainment|249|407|leisure=cinema, amenity=theatre|
|---|---|---|---|

office 1,838 1,811 office=*, amenity=bank

|residential|104|28|building=apartments/residential|
|---|---|---|---|

education 251 614 amenity=school/university/college

|healthcare|323|1,199|amenity=hospital/clinic/pharmacy|
|---|---|---|---|

government 265 137 amenity=townhall/police/courthouse

|tourism|1,878|1,833|tourism=*, amenity=place of interest<br>_ _|
|---|---|---|---|

sports 342 729 leisure=sports_centre/stadium/gym

|religious|39|334|amenity=place of worship<br>_ _|
|---|---|---|---|

**合计** **12,536** **21,144** — transport 类已完全从 POI 特征中剥离，改由独立的公共交通密度特征（[35:38]）建模， 避免北 京 transport 类 POI（23,549 条）主导余弦相似度导致 G_poi 图退化。

**遥感特征（[22:27]）— Landsat-8**

|特征|公式|北京均值|纽约均值|
|---|---|---|---|
|ndvi|(B5−B4)/(B5+B4)|0.219|0.034|

ndbi (B6−B5)/(B6+B5) −0.022 −0.070

|mndwi|(B3−B6)/(B3+B6)|−0.225|0.009|
|---|---|---|---|

b4_mean 红波段平均反射率

0. 142
0. 115

|b4 std<br>_|红波段标准差|0.033|0.035|
|---|---|---|---|

**公共交通特征（[35:40]）— 详细统计**

|特征|北京|纽约|
|---|---|---|
|subway_station_density|369站，253/1024格（24.7%）覆<br>盖|238站，41/75格（54.7%）覆盖|

subway_entrance_density 983入口，208/1024格覆盖 1,153入口，41/75格覆盖

|bus stop density<br>_ _|10,831站，787/1024格<br>（76.9%），均值4.56/km²|3,015站，66/75格（88.0%），<br>均值12.57/km²|
|---|---|---|

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 9

<!-- Page 10 -->

num_subway_lines 208/1024格有线路，55格为换乘枢 41/75格有线路，33格为换乘枢纽 （[38]） 纽

|is transfer hub（[39]）<br>_ _|binary（西直门/国贸/东直门等）|binary（Times Sq/Grand<br>Central等）|
|---|---|---|

num_bus_routes（[40]） 765/1024（74.7%）格，均值7.6， 48/75（64.0%）格，均值7.6， 最大64 最大41 **4.3 输出矩阵**

### (1024, 41) (75, 41)

float32 · 164.1 KB float32 · 12.1 KB node_features_bj.npy · NaN=0✅ node_features_nyc.npy · NaN=0✅ 城市人流异常检测系统 ·Week 2报告 · 2026-06-17 10

<!-- Page 11 -->

**五、多模态图结构（方案C 主线）** 每城市构建 4 张同构图，共享同一节点集合，后续通过门控网络融合。 图以 PyTorch Geometric 兼容格式存储（edge_index / edge_weight / num_nodes）。

**5.1 四类图构建方法** **G_spatial — 空间邻接图** Moore 8邻域，边权 = 高斯距离衰减，带宽 σ=1.5格：

w(i,j) = exp(−d(i,j)² / σ²) 正交邻居d=1.0 → w=0.641；斜角邻居d=√2 → w=0.411 **G_poi — POI 功能相似图** 11类功能 POI 密度向量，log1p → L2归一化 → 余弦相似度，取 top-k=10 近邻。

**G_flow — 历史流量相关图** 全量 inflow 时序 Pearson 相关矩阵，取 top-k=15 正相关近邻。北京相关均值 0.857，纽约

0. 691。

**G_transit — 轨道线路拓扑图** 同一条轨道线路途经的相邻站点之间连边，边权 = exp(−站间距/σ)。 ChebConv 要求对称拉普拉 斯，图为无向图（双向有向边实现）。 北京使用政府 xlsx（18条线，342站）与 OSM 坐标匹配；

纽约使用 OSM route=subway 关系（31条线，1,601节点）。

**5.2 图统计汇总**

|城市|图|节点数|边数|边/节点|存储文件|
|---|---|---|---|---|---|
|北京|G_spatial|1,024|7,812|7.6|graph_spatial_bj.pt|

北京 G_poi 1,024 10,240

10. 0

graph_poi_bj.pt

|北京|G flow<br>_|1,024|15,255|14.9|graph flow bj.pt<br>_ _|
|---|---|---|---|---|---|

北京 G_transit 1,024 3,080

3. 0

graph_transit_bj.pt

|纽约|G spatial<br>_|75|484|6.5|graph spatial nyc.pt<br>_ _|
|---|---|---|---|---|---|

纽约 G_poi 75 750

10. 0

graph_poi_nyc.pt

|纽约|G flow<br>_|75|1,125|15.0|graph flow nyc.pt<br>_ _|
|---|---|---|---|---|---|

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 11

<!-- Page 12 -->

纽约 G_transit 75 754

10. 1

graph_transit_nyc.pt **5.3 动态时间特征（12维）** 采用正弦余弦编码将周期性时间变量映射到连续向量空间：

|特征名|周期|含义|
|---|---|---|
|slot_sin / slot_cos|48|日内第几个 30min 步（0~47）|

hour_sin / hour_cos 24 小时（0~23）

|weekday sin / weekday cos<br>_ _|7|星期（0=周一）|
|---|---|---|

month_sin / month_cos 12 月份（1~12）

|doy sin / doy cos<br>_ _|366|年内第几天|
|---|---|---|

is_weekend — 二值：0=工作日，1=周末

|is holiday<br>_|—|二值：0=普通日，1=节假日（BJ 8.0%；NYC 2.7%）|
|---|---|---|

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 12

<!-- Page 13 -->

**六、数据质量分析** **6.1 总体评估**

|数据集|完整性|可用<br>性|主要问题|
|---|---|---|---|
|TaxiBJ 人流|✅ 无缺失|高|四段不连续，非全年|

TaxiNYC 人流 ✅无缺失 高 零值比例 10.7%（边缘格正常）

|METR-LA 速度|✅ 无缺失|高|速度非人流，定位为辅助基线|
|---|---|---|---|

OpenWeatherMap 北 ⚠️部分缺 中 visibility/sea_level/wind_gust 100% 缺失 京 失

|OpenWeatherMap 纽<br>约|⚠️ 部分缺<br>失|中高|sea level/grnd level 100% 缺失<br>_ _|
|---|---|---|---|

Open-Meteo ERA5 ✅无缺失 高 字段少但全完整，作为核心气象输入

|OSM 北京 / 纽约|✅ 完整|高|纽约初次下载文件 30MB（EOF 损坏），已重新获取<br>151MB|
|---|---|---|---|

Landsat-8 北京 / 纽约 ✅完整 高 单景静态特征；云覆盖率 <0.2%

|北京政府交通数据|✅ 完整|高|2026年版，新线路在bbox外，无影响|
|---|---|---|---|

MTA GTFS（5区公交） ✅完整 高 5区分包，100% 站点覆盖 **6.2 TaxiBJ 流量统计**

|年份|时间步数|时间段|Infolw 均值|Infolw 最大|零值占比|
|---|---|---|---|---|---|
|BJ13|4,888|2013-07-01 ~ 2013-10-29|97.3|1,230|5.68%|

BJ14 4,780 2014-03-01 ~ 2014-06-27

111. 1

1,285

5. 16%

|BJ15|5,596|2015-03-01 ~ 2015-06-30|115.4|1,267|4.71%|
|---|---|---|---|---|---|

BJ16 7,220 2015-11-01 ~ 2016-04-10

94. 5

1,250

5. 46%

|合计|22,484|四段（不连续），30分钟/步，(T,2,32,32)|Col4|
|---|---|---|---|

**6.3 OSM 关键问题记录** **纽约 OSM 文件损坏问题：**初次从 Geofabrik 下载文件仅 30.7MB（完整应为 151MB）。 OSM PBF 格式中 Node 块先于 Way 块存储，截断文件导致 POI 节点可正常解析但道路几何完全缺失， 城市人流异常检测系统 ·Week 2报告 · 2026-06-17 13

<!-- Page 14 -->

最终路网密度全部为 0。排查后从 BBBike 重新下载完整文件，路网（114,938条）、POI （21,144个）、 公交站点（3,015个）均恢复正常。

**6.4 MTA GTFS 数据结构**

|子目录|覆盖区|路线数（type=3）|bbox内站点数|
|---|---|---|---|
|gtfs_bus/gtfs_b/|Brooklyn|255|648|

gtfs_bus/gtfs_bx/ Bronx 255 656

|gtfs bus/gtfs m/<br>_ _|Manhattan|255|1,813|
|---|---|---|---|

gtfs_bus/gtfs_q/ Queens 255 27

|gtfs bus/gtfs si/<br>_ _|Staten Island|255|156|
|---|---|---|---|

**合计（去重）** 全市 **255** **3,168** 注：routes.txt 在5个区GTFS中内容相同（同一套全市线路），stop_times.txt各区独立（最大 ~141MB），使用 chunksize=200,000 分块读取。

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 14

<!-- Page 15 -->

**七、预处理产物清单** **7.1 data/processed/ 完整文件列表**

|文件|规格|状态|
|---|---|---|
|temporal_features_bj.csv|(22484, 12) 北京时间特征|✅|

temporal_features_nyc.csv (17520, 12) 纽约时间特征 ✅

|spatial features bj.csv<br>_ _|(1024, 16) 北京路网+距离|✅|
|---|---|---|

spatial_features_nyc.csv (75, 16) 纽约路网+距离 ✅

|poi features bj.csv<br>_ _|(1024, 22) 北京 POI（11类×2）|✅|
|---|---|---|

poi_features_nyc.csv (75, 22) 纽约 POI ✅

|satellite features bj.csv<br>_ _|(1024, 5) 北京遥感指数|✅|
|---|---|---|

satellite_features_nyc.csv (75, 5) 纽约遥感指数 ✅

|transit features bj.csv<br>_ _|(1024, 6) 北京公共交通密度|✅|
|---|---|---|

transit_features_nyc.csv (75, 6) 纽约公共交通密度 ✅

|subway grid features bj.csv<br>_ _ _|(1024, 2) 北京轨道线路特征|✅|
|---|---|---|

subway_grid_features_nyc.csv (75, 2) 纽约轨道线路特征 ✅

|bus route features bj.csv<br>_ _ _|(1024, 1) 北京公交线路数|✅|
|---|---|---|

bus_route_features_nyc.csv (75, 1) 纽约公交线路数 ✅

|node features bj.npy<br>_ _|(1024, 41) float32, 164.1KB, NaN=0|✅|
|---|---|---|

node_features_nyc.npy (75, 41) float32, 12.1KB, NaN=0 ✅

|graph spatial bj.pt<br>_ _|edges=7,812|✅|
|---|---|---|

graph_poi_bj.pt edges=10,240 ✅

|graph flow bj.pt<br>_ _|edges=15,255|✅|
|---|---|---|

graph_transit_bj.pt edges=3,080 ✅

|graph spatial nyc.pt<br>_ _|edges=484|✅|
|---|---|---|

graph_poi_nyc.pt edges=750 ✅ 城市人流异常检测系统 ·Week 2报告 · 2026-06-17 15

<!-- Page 16 -->

|graph flow nyc.pt<br>_ _|edges=1,125|✅|
|---|---|---|

graph_transit_nyc.pt edges=754 ✅ 城市人流异常检测系统 ·Week 2报告 · 2026-06-17 16

<!-- Page 17 -->

**八、Week 2 完成情况总结** **8.1 任务完成核查**

|状<br>态|任务|涉及数据/文件|
|---|---|---|
|✅|TaxiBJ 四段时间对齐与拼接|taxibj/*.h5|

OpenWeatherMap 时间戳解析、缺失字段处理、插值30分 openweather/*.csv ✅ 钟

|✅|TaxiNYC 气象温度单位转换（°F → °C）|taxinyc/Meteorology.h5|
|---|---|---|

OSM POI 提取与网格聚合（11类功能POI，transport剥 osm/*.pbf ✅ 离）

|✅|Landsat-8 DN → 反射率转换，NDVI/NDBI/MNDWI 计算|Landsat8 */B*.TIF<br>_|
|---|---|---|

ERA5 + OpenWeatherMap 字段合并，30分钟气象特征矩 weather/*.h5 + openweather/ ✅ 阵 *.csv

|✅|METR-LA parquet 格式解析|metr la/*.parquet<br>_|
|---|---|---|

公共交通特征（地铁站/入口/公交站密度，3维） OSM PBF ✅

|✅|轨道线路特征（num subway lines, is transfer hub，2<br>_ _ _ _<br>维）|政府xlsx + OSM PBF|
|---|---|---|

公交线路特征（num_bus_routes，1维） OSM + MTA GTFS 5区 ✅

|✅|多模态图构建（G spatial / G poi / G flow / G transit）<br>_ _ _ _|各中间特征文件|
|---|---|---|

节点特征矩阵合并（N×41，NaN=0） build_graphs.py ✅ **8.2 额外完成项** src/features/ 可导入 Python 特征工程代码库（4模块，16公开符号） ✅ 量化 / TensorRT 加速预研报告（INT8 / FP16 方案对比，推理加速评估） ✅ 可视化界面设计文档（Streamlit + Kepler.gl 交互设计） ✅ 完整的 static_features_spec.md v1.4（41维特征完整规格） ✅ graph_fusion_design.md（多图融合方案C最终设计） ✅ 数据规模差异说明（3.7GB 实际 vs 20GB PDF估计） ✅ 城市人流异常检测系统 ·Week 2报告 · 2026-06-17 17

<!-- Page 18 -->

**8.3 Week 3 预告** **下周目标：基础时空预测模型** 基于本周完成的 (N,41) 节点特征矩阵和 4 类多模态图，训练并评估： ARIMA 时序基线 → LSTM 序列模型 → 标准 GCN（单图）→ STGCN（时空图卷积）。 评估指标：MAE / RMSE / MAPE，在 TaxiBJ 和 TaxiNYC 两个数据集上对比。

城市人流异常检测系统 ·Week 2报告 · 2026-06-17 18
