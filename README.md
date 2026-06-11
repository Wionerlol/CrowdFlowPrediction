# CrowdFlowPrediction

基于多源时空数据融合的城市级人流异常检测与预警系统

## 项目简介

本项目融合卫星遥感影像、出租车轨迹、POI 分布、气象数据等多源异构时空信息，采用图神经网络与 Transformer 融合技术，实现对城市不同区域未来 24 小时的人流预测，并精准识别大型集会、交通拥堵、突发事件等异常人流事件。

**核心目标**

- 构建多源时空数据融合处理管道
- 实现基于 GNN + Transformer 的高精度人流预测模型
- 开发多维度无监督异常检测算法
- 搭建交互式可视化预警平台

## 数据集

| 数据集 | 城市 | 时间范围 | 用途 |
|--------|------|---------|------|
| TaxiBJ | 北京 | 2013–2016（四段） | 人流预测主数据 |
| TaxiNYC | 纽约 | 2014 全年 | 跨城市泛化验证 |
| METR-LA | 洛杉矶 | 2012 | 交通流量基线 |
| OpenStreetMap | 北京 / 纽约 | — | 路网 + POI 特征 |
| NASA Landsat-8 | 北京 / 纽约 | 2014–2015 单景 | 静态空间特征 |
| OpenWeatherMap | 北京 / 纽约 | 2013–2016 | 气象外生变量 |

数据不随代码入库，详见 [`data/raw/DATASETS.md`](data/raw/DATASETS.md)。

## 技术栈

- **深度学习**：PyTorch 2.11+cu128、PyTorch Geometric、DGL
- **模型**：STGCN → STAEformer → Spacetimeformer（渐进式）
- **地理处理**：GeoPandas、Rasterio、osmium、pyproj
- **异常检测**：PyOD、VAE、AnomalyTransformer
- **系统**：FastAPI、Streamlit、Folium、Kepler.gl
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

# 激活虚拟环境
source .venv/bin/activate

# 验证安装
python scripts/verify_env.py
```

详细说明见 [`docs/environment_setup_guide.md`](docs/environment_setup_guide.md)。

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 OpenWeatherMap API Key
```

## 项目结构

```
CrowdFlowPrediction/
├── data/
│   └── raw/                    # 原始数据（不入库，见 DATASETS.md）
├── docs/
│   └── environment_setup_guide.md
├── outputs/
│   └── reports/                # 调研报告、数据质量报告
├── scripts/
│   ├── setup_env.sh            # 一键环境配置
│   ├── verify_env.py           # 环境验证
│   └── explore_data.py         # 数据探索
├── src/
│   ├── data/                   # 数据加载与气象 API 客户端
│   ├── features/               # 特征工程
│   ├── models/                 # 预测模型
│   ├── anomaly/                # 异常检测
│   └── visualization/          # 可视化
├── study_guides/               # 论文阅读导读（STGCN / STAEformer / Spacetimeformer）
└── .env                        # API Key（不入库）
```

## 项目进度

- [x] Week 1：数据集获取、环境搭建、技术调研
- [ ] Week 2：数据预处理与特征工程
- [ ] Week 3：基础时空预测模型（ARIMA / LSTM / GCN）
- [ ] Week 4：高级模型（STGCN / STAEformer / Spacetimeformer）
- [ ] Week 5：异常检测算法集成
- [ ] Week 6：系统集成与可视化
- [ ] Week 7：模型优化与可解释性分析
- [ ] Week 8：项目总结与演示

## 参考论文

1. **STGCN** — Yu et al., IJCAI 2018 · [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)
2. **STAEformer** — Liu et al., CIKM 2023 · [GitHub](https://github.com/XDZhelheim/STAEformer)
3. **Spacetimeformer** — Grigsby et al., ICLR 2022 · [arXiv:2109.12218](https://arxiv.org/abs/2109.12218)
4. **Survey** — ACM Computing Surveys 2024 · [DOI:10.1145/3766546](https://doi.org/10.1145/3766546)
