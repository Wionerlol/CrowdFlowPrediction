# CrowdFlowPrediction

基于多源时空数据融合的城市级人流异常检测与预警系统

## 项目简介

本项目融合卫星遥感影像、出租车轨迹、POI 分布、气象数据、公共交通网络等多源异构时空信息，采用图神经网络与 Transformer 融合技术，实现对城市不同区域未来 24 小时的人流预测，并精准识别大型集会、交通拥堵、突发事件等异常人流事件。

**核心目标**

- 构建多源时空数据融合处理管道
- 实现基于 GNN + Transformer 的高精度人流预测模型
- 开发多维度无监督异常检测算法
- 搭建交互式可视化预警平台

## 数据集

| 数据集 | 城市 | 时间范围 | 用途 |
|--------|------|---------|------|
| TaxiBJ | 北京 | 2013–2016（四段，22484步） | 人流预测主数据 |
| TaxiNYC | 纽约 | 2014 全年（17520步） | 跨城市泛化验证 |
| METR-LA | 洛杉矶 | 2012 | 交通流量基线 |
| OpenStreetMap | 北京 / 纽约 | — | 路网 + POI + 公共交通特征 |
| NASA Landsat-8 | 北京 / 纽约 | 2015-04-16 / 2014-04-10 | 遥感静态特征 |
| OpenWeatherMap / ERA5 | 北京 / 纽约 | 2013–2016 | 气象外生变量 |
| 北京交通委政府数据 | 北京 | 2026（最新） | 轨道/公交线路站点匹配 |
| MTA GTFS（5区） | 纽约 | 2024（最新） | 公交线路站点特征 |

数据不随代码入库，详见 [`data/raw/DATASETS.md`](data/raw/DATASETS.md)。

> **数据规模**：原始数据 3.6 GB，预处理产物 100 MB，合计约 3.7 GB。课题文档估计的"约 20 GB"基于原始 GPS 轨迹格式；本项目使用官方预聚合网格版本（TaxiBJ 32×32 / TaxiNYC 15×5），数据量更小但信息完整，不影响模型训练与评估。

## 技术栈

- **深度学习**：PyTorch 2.11+cu128、PyTorch Geometric
- **模型**：STGCN → STAEformer → Spacetimeformer（渐进式）
- **地理处理**：GeoPandas、Rasterio、osmium、pyproj
- **异常检测**：PyOD、VAE、AnomalyTransformer
- **系统**：FastAPI、Streamlit、Kepler.gl
- **实验管理**：MLflow、TensorBoard

## 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU（RTX 20xx 及以上）+ 对应 CUDA 驱动
- 16 GB+ 内存，50 GB+ 磁盘

### 安装

```bash
git clone https://github.com/Wionerlol/CrowdFlowPrediction.git
cd CrowdFlowPrediction

# 根据 GPU 选择 CUDA 版本
bash scripts/setup_env.sh cu128   # RTX 50xx (Blackwell) ← 推荐
bash scripts/setup_env.sh cu121   # RTX 40xx / 30xx
bash scripts/setup_env.sh cu118   # RTX 20xx / 10xx
bash scripts/setup_env.sh cpu     # 无 GPU

source .venv/bin/activate
python scripts/verify_env.py
```

详细说明见 [`docs/environment_setup_guide.md`](docs/environment_setup_guide.md)。

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 OpenWeatherMap API Key
```

## 特征工程（Week 2 完成）

每个网格节点的静态特征向量共 **41 维**：

| 索引 | 分组 | 维度 | 数据来源 |
|------|------|------|---------|
| [0:4] | 地理位置（坐标、城市中心距离、环路ID） | 4 | 坐标计算 |
| [4:6] | 到交通枢纽 / 商业中心最短距离 | 2 | Haversine |
| [6:11] | 路网密度（总/主干/地方/服务/慢行） | 5 | OSM PBF |
| [11:22] | POI 类别密度（11类功能性POI） | 11 | OSM PBF |
| [22:27] | 遥感指数（NDVI/NDBI/MNDWI/B4均值/B4标准差） | 5 | Landsat-8 |
| [27:35] | 历史流量统计（均值/峰值比/零值率/时段模式等） | 8 | TaxiBJ/NYC |
| [35:38] | 公共交通密度（地铁站/入口/公交站） | 3 | OSM PBF |
| [38:40] | 轨道线路（穿越线路数/换乘枢纽标志） | 2 | 政府数据 + OSM |
| [40] | 公交线路数 | 1 | MTA GTFS / OSM |

多模态图结构（每城市4张图，共享节点集合）：

| 图 | 构建方式 | 北京边数 | 纽约边数 |
|----|---------|---------|---------|
| G_spatial | 8邻域高斯距离衰减 | 7,812 | 484 |
| G_poi | POI余弦相似度 top-10 kNN | 10,240 | 750 |
| G_flow | Pearson流量相关 top-15 kNN | 15,255 | 1,125 |
| G_transit | 轨道线路拓扑（exp衰减） | 3,080 | 754 |

## 项目结构

```
CrowdFlowPrediction/
├── data/
│   ├── raw/                        # 原始数据（不入库，见 DATASETS.md）
│   │   ├── taxibj/ taxinyc/        # 人流数据
│   │   ├── osm/                    # OSM PBF 文件（北京36MB，纽约151MB）
│   │   ├── Landsat8_beijing/nyc/   # 卫星影像（~2.3GB）
│   │   ├── openweather/            # 气象历史数据
│   │   ├── jtw_beijing_gov/        # 北京交通委政府数据（轨道/公交xlsx）
│   │   └── mta_gtfs/               # MTA GTFS（地铁 + 5区公交）
│   └── processed/                  # 预处理产物（自动生成）
│       ├── node_features_{bj,nyc}.npy      # 节点静态特征矩阵 (N,41)
│       ├── graph_{spatial,poi,flow,transit}_{bj,nyc}.pt  # 4类图结构
│       ├── transit_features_{bj,nyc}.csv   # 公共交通密度特征
│       ├── subway_grid_features_{bj,nyc}.csv  # 轨道线路特征
│       ├── bus_route_features_{bj,nyc}.csv    # 公交线路数特征
│       └── temporal_features_{bj,nyc}.csv     # 时间周期编码特征
├── docs/
│   ├── environment_setup_guide.md
│   ├── static_features_spec.md      # 41维静态特征规格（v1.4）
│   ├── graph_fusion_design.md       # 多图融合方案设计
│   ├── anomaly_detection_spec.md    # 异常检测方案
│   ├── weak_label_rules.md          # 弱标签规则
│   └── visualization_ui_design.md   # 可视化界面设计
├── outputs/
│   ├── reports/
│   │   ├── research_report_week1.md         # Week1 技术调研
│   │   ├── data_quality_report.md           # 数据质量分析
│   │   ├── data_preprocessing_report.md     # 数据预处理报告
│   │   ├── feature_engineering_report.md    # 特征工程报告
│   │   ├── quantization_tensorrt_report.md  # 量化加速预研
│   │   └── (anomaly_detection_report.md)    # 待生成
│   └── checkpoints/
│       └── stgcn_best.pt                    # STGCN 基线检查点
├── scripts/
│   ├── setup_env.sh / verify_env.py         # 环境配置与验证
│   ├── preprocess_all.py                    # 一键预处理入口
│   ├── build_spatial_features.py            # 路网 + 距离特征
│   ├── build_poi_features.py                # POI 语义特征
│   ├── build_satellite_features.py          # Landsat-8 遥感特征
│   ├── build_transit_features.py            # 公共交通密度特征
│   ├── build_subway_features.py             # 轨道线路特征 + G_transit
│   ├── build_bus_features.py                # 公交线路特征（GTFS/OSM）
│   ├── build_temporal_features.py           # 时间周期编码
│   ├── build_graphs.py                      # 多图构建 + 节点特征合并
│   ├── merge_weather.py                     # ERA5 + OWM 气象合并
│   └── train_stgcn.py                       # STGCN 训练脚本
├── src/
│   ├── data/                        # 数据加载与气象 API 客户端
│   ├── features/                    # 特征工程模块
│   ├── models/
│   │   └── stgcn.py                 # STGCN 实现
│   ├── anomaly/                     # 异常检测模块（待开发）
│   └── visualization/               # 可视化模块（待开发）
├── study_guides/                    # 论文阅读导读（STGCN/STAEformer/Spacetimeformer）
└── .env                             # API Key（不入库）
```

## 项目进度

- [x] **Week 1**：数据集获取、环境搭建、技术调研
  - TaxiBJ/NYC/METR-LA 数据验证，OSM/Landsat-8/气象数据获取
  - STGCN / STAEformer / Spacetimeformer 论文精读
  - 多模态图融合方案（方案C：G_spatial + G_poi + G_flow + G_transit）确定

- [x] **Week 2**：数据预处理与特征工程（全部完成）
  - 41维静态节点特征矩阵构建（NaN=0）
  - 4类多模态图结构生成（北京 + 纽约）
  - 公共交通特征：OSM密度（3维）+ 政府数据轨道线路（2维）+ MTA GTFS公交线路（1维）
  - G_transit 轨道线路拓扑图（北京3080边，纽约754边）
  - 量化/TensorRT 加速预研报告、可视化界面设计文档

- [ ] **Week 3**：基础时空预测模型（ARIMA / LSTM / GCN）
- [ ] **Week 4**：高级模型（STGCN / STAEformer / Spacetimeformer）
- [ ] **Week 5**：异常检测算法集成
- [ ] **Week 6**：系统集成与可视化
- [ ] **Week 7**：模型优化与可解释性分析
- [ ] **Week 8**：项目总结与演示

## 参考论文

1. **STGCN** — Yu et al., IJCAI 2018 · [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)
2. **STAEformer** — Liu et al., CIKM 2023 · [GitHub](https://github.com/XDZhelheim/STAEformer)
3. **Spacetimeformer** — Grigsby et al., ICLR 2022 · [arXiv:2109.12218](https://arxiv.org/abs/2109.12218)
4. **Survey** — ACM Computing Surveys 2024 · [DOI:10.1145/3766546](https://doi.org/10.1145/3766546)
