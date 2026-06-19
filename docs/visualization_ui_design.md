# 可解释性可视化界面设计文档

**版本**：v1.0  
**日期**：2026-06-17  
**阶段**：Week 6/7 预研  
**技术栈**：FastAPI + Streamlit + Kepler.gl + PyTorch（SHAP / 注意力可视化）

---

## 一、系统总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户浏览器                             │
│          Streamlit Web App (localhost:8501)              │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ 热力图/预测  │  │  异常事件面板  │  │  可解释性面板  │  │
│  │ Kepler.gl   │  │  告警列表     │  │  SHAP / Attn  │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
└─────────┼────────────────┼──────────────────┼──────────┘
          │                │                  │
          └────────────────┴──────────────────┘
                           │ HTTP REST
          ┌────────────────▼──────────────────┐
          │        FastAPI 后端                 │
          │   /predict  /anomaly  /explain     │
          │   /history  /alert   /status       │
          └────────────────┬──────────────────┘
                           │
          ┌────────────────▼──────────────────┐
          │         推理引擎层                  │
          │  STAEformer (TRT FP16)            │
          │  Z-score 异常检测                  │
          │  SHAP Explainer                   │
          └────────────────┬──────────────────┘
                           │
          ┌────────────────▼──────────────────┐
          │         数据层                     │
          │  node_features.npy  graphs.pt     │
          │  taxibj_clean.npz   slot_mu.npy   │
          │  weather_merged.csv               │
          └───────────────────────────────────┘
```

---

## 二、FastAPI 后端设计

### 2.1 项目结构

```
src/
├── api/
│   ├── main.py          # FastAPI 应用入口
│   ├── routers/
│   │   ├── predict.py   # 预测接口
│   │   ├── anomaly.py   # 异常检测接口
│   │   ├── explain.py   # 可解释性接口
│   │   └── history.py   # 历史查询接口
│   ├── models/
│   │   ├── schemas.py   # Pydantic 请求/响应模型
│   │   └── inference.py # 推理引擎封装
│   └── services/
│       ├── anomaly_service.py
│       ├── shap_service.py
│       └── cache.py     # Redis 缓存
```

### 2.2 核心 API 接口定义

```python
# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="城市人流异常检测系统",
    description="TaxiBJ/TaxiNYC 人流预测与异常检测 API",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])
```

**接口一：人流预测**

```
POST /api/v1/predict
Content-Type: application/json

请求体：
{
  "city": "bj",                      // "bj" | "nyc"
  "timestamp": "2016-04-01T08:00:00", // 预测起始时刻 (UTC)
  "horizon": 12,                      // 预测步数（默认12步=6小时）
  "grid_ids": [100, 101, 102]         // 可选，不填则返回全城
}

响应：
{
  "city": "bj",
  "pred_timestamps": ["2016-04-01T08:30:00", ...],  // 12个时刻
  "predictions": {                    // grid_id → [12步预测值]
    "100": [125.3, 132.1, ...],
    "101": [89.7, 95.2, ...],
    ...
  },
  "inference_ms": 14.2,
  "model_version": "staeformer_v1"
}
```

**接口二：异常检测**

```
POST /api/v1/anomaly/detect
{
  "city": "bj",
  "timestamp": "2016-04-01T08:00:00",
  "z_threshold": 3.0                  // 可调阈值
}

响应：
{
  "level": 2,                         // 0=无 1=轻 2=中 3=重 4=极端
  "level_label": "L2 中度",
  "triggered_cells": 15,
  "clusters": [
    {
      "cell_count": 8,
      "max_z": 3.87,
      "centroid": {"row": 14, "col": 18},
      "center_lon": 116.53,
      "center_lat": 39.96,
      "bounding_box": [[116.48, 39.92], [116.58, 40.00]]
    }
  ],
  "top_cells": [
    {"grid_id": 462, "z_score": 3.87, "flow": 412, "expected": 185}
  ]
}
```

**接口三：可解释性 — SHAP 特征重要性**

```
POST /api/v1/explain/shap
{
  "city": "bj",
  "grid_id": 462,
  "timestamp": "2016-04-01T08:00:00",
  "n_samples": 100                    // SHAP 背景样本数
}

响应：
{
  "grid_id": 462,
  "base_value": 185.3,
  "prediction": 412.1,
  "shap_values": {                    // 特征名 → SHAP 贡献值
    "hist_mean": 42.3,
    "is_holiday": 38.7,
    "temperature": -12.4,
    "poi_food_density": 28.9,
    "subway_station_density": 22.1,
    "bus_stop_density": 15.6,
    "road_density": 11.2,
    ...
  },
  "feature_groups": {                 // 按组聚合（可视化用）
    "历史统计": 67.4,
    "公共交通": 37.7,
    "时间特征": 38.7,
    "POI密度": 55.3,
    "气象": -12.4,
    "路网": 11.2,
    "遥感": 4.1
  }
}
```

**接口四：注意力权重可视化**

```
GET /api/v1/explain/attention?city=bj&grid_id=462&timestamp=2016-04-01T08:00:00

响应：
{
  "grid_id": 462,
  "temporal_attention": [0.12, 0.08, 0.15, ...],  // 对过去12步的注意力权重
  "spatial_attention": {                            // 对其他格子的注意力
    "460": 0.089,
    "461": 0.134,
    "463": 0.076,
    ...                                             // top-20 重要格子
  }
}
```

**接口五：历史查询**

```
GET /api/v1/history/anomalies?city=bj&start=2016-03-01&end=2016-04-01&min_level=2

响应：
{
  "total": 47,
  "anomalies": [
    {
      "timestamp": "2016-03-15T19:00:00",
      "level": 3,
      "cell_count": 18,
      "centroid": {"lon": 116.41, "lat": 39.91},
      "duration_steps": 4,
      "weather": {"temp": 12.3, "weather_code": 800}
    },
    ...
  ]
}
```

### 2.3 推理引擎封装

```python
# src/api/models/inference.py
import numpy as np
import torch
from pathlib import Path

class InferenceEngine:
    """单例推理引擎，应用启动时加载，避免重复初始化"""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self._load_static_data()

    def _load_model(self):
        from src.models.staeformer import STAEformer
        self.model = STAEformer(...).to(self.device)
        self.model.load_state_dict(torch.load('outputs/checkpoints/best_model.pt'))
        self.model.eval()
        self.model.half()  # FP16

    def _load_static_data(self):
        self.node_features = {
            'bj':  torch.from_numpy(np.load('data/processed/node_features_bj.npy')).half().to(self.device),
            'nyc': torch.from_numpy(np.load('data/processed/node_features_nyc.npy')).half().to(self.device),
        }
        self.graphs = {
            'bj':  torch.load('data/processed/graph_spatial_bj.pt'),
            'nyc': torch.load('data/processed/graph_spatial_nyc.pt'),
        }
        self.slot_mu  = np.load('data/processed/slot_mu_bj.npy')
        self.slot_sig = np.load('data/processed/slot_sig_bj.npy')

    @torch.no_grad()
    def predict(self, city: str, flow_window: np.ndarray) -> np.ndarray:
        """
        flow_window: (T_in, N, 2) numpy float32
        返回: (T_out, N) numpy float32
        """
        x = torch.from_numpy(flow_window).half().unsqueeze(0).to(self.device)
        pred = self.model(x, self.node_features[city], self.graphs[city])
        return pred.squeeze(0).cpu().float().numpy()
```

---

## 三、Streamlit 前端界面设计

### 3.1 页面布局

```
┌──────────────────────────────────────────────────────────┐
│  🏙️ 城市人流异常检测系统        城市: [北京▼]  时间: [时间选择器]│
├────────────────────────────────────────────────┬─────────┤
│                                                │告警面板  │
│              Kepler.gl 地图                     │         │
│                                                │⚠️ L2告警│
│  [热力图] [预测] [异常] [注意力]  ← 图层切换    │12格 3.8z│
│                                                │         │
│  ▶ 播放预测动画  时刻: [──●────]               │历史事件  │
│                                                │[列表]   │
├────────────┬───────────────────────────────────┼─────────┤
│ SHAP分析   │    注意力权重                       │时序图   │
│ [瀑布图]   │  时间注意力 [折线图]                 │选中格子 │
│            │  空间注意力 [地图叠加]               │流量曲线 │
└────────────┴───────────────────────────────────┴─────────┘
```

### 3.2 主要组件代码

```python
# src/visualization/app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from streamlit_keplergl import keplergl

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="城市人流异常检测",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── 侧边栏 ──────────────────────────────
with st.sidebar:
    st.title("⚙️ 控制面板")
    city = st.selectbox("城市", ["北京 (BJ)", "纽约 (NYC)"], index=0)
    city_code = "bj" if "BJ" in city else "nyc"

    selected_date = st.date_input("日期", value=pd.Timestamp("2016-04-01"))
    selected_hour = st.slider("小时", 0, 23, 8)
    selected_ts = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)

    z_threshold = st.slider("Z-score 告警阈值", 2.0, 5.0, 3.0, 0.1)
    horizon = st.slider("预测步数", 1, 48, 12)
    layer_mode = st.radio("显示图层", ["实时热力图", "预测动画", "异常区域", "注意力叠加"])

# ─────────────────────────── 主区域 ──────────────────────────────
col_map, col_alert = st.columns([3, 1])
```

**Kepler.gl 热力图组件**

```python
def render_heatmap(city_code, flow_data, anomaly_clusters=None):
    """
    flow_data: dict {grid_id: flow_value}
    anomaly_clusters: list of cluster dicts
    """
    # 构建 GeoDataFrame 格式的图层数据
    grid_meta = pd.read_csv(f'data/processed/grid_meta_{city_code}.csv')
    grid_meta['flow'] = grid_meta['grid_id'].map(flow_data).fillna(0)
    grid_meta['is_anomaly'] = grid_meta['grid_id'].isin(
        [c['grid_id'] for c in (anomaly_clusters or [])]
    )

    # Kepler.gl 配置
    config = {
        "version": "v1",
        "config": {
            "mapStyle": {"styleType": "dark"},
            "visState": {
                "layers": [
                    {
                        "type": "heatmap",
                        "config": {
                            "dataId": "flow_data",
                            "columns": {"lat": "center_lat", "lng": "center_lon"},
                            "visConfig": {
                                "radius": 30,
                                "weightField": "flow",
                                "colorRange": {
                                    "colors": ["#0000FF", "#00FF00", "#FFFF00", "#FF0000"]
                                }
                            }
                        }
                    }
                ]
            }
        }
    }

    return keplergl(grid_meta, config=config, height=500)
```

**预测动画播放组件**

```python
def render_prediction_animation(city_code, timestamp, horizon=12):
    """生成 Plotly 动画帧，逐步展示未来 horizon 步流量"""
    resp = requests.post(f"{API_BASE}/predict", json={
        "city": city_code,
        "timestamp": timestamp.isoformat(),
        "horizon": horizon
    })
    pred_data = resp.json()

    grid_meta = pd.read_csv(f'data/processed/grid_meta_{city_code}.csv')

    # 构建动画帧
    frames = []
    for step_idx, ts in enumerate(pred_data['pred_timestamps']):
        frame_df = grid_meta.copy()
        frame_df['flow'] = frame_df['grid_id'].astype(str).map(
            {k: v[step_idx] for k, v in pred_data['predictions'].items()}
        ).fillna(0)
        frames.append(go.Frame(
            data=[go.Scatter(
                x=frame_df['center_lon'], y=frame_df['center_lat'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=frame_df['flow'],
                    colorscale='YlOrRd',
                    cmin=0, cmax=frame_df['flow'].quantile(0.95),
                    showscale=True
                ),
                text=frame_df['grid_id']
            )],
            name=ts
        ))

    fig = go.Figure(frames=frames)
    fig.update_layout(
        updatemenus=[{"type": "buttons", "buttons": [
            {"label": "▶ 播放", "method": "animate",
             "args": [None, {"frame": {"duration": 500}, "transition": {"duration": 300}}]},
            {"label": "⏸ 暂停", "method": "animate",
             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
        ]}],
        sliders=[{"steps": [
            {"args": [[f.name], {"frame": {"duration": 500}}],
             "label": f.name, "method": "animate"}
            for f in frames
        ]}]
    )
    st.plotly_chart(fig, use_container_width=True)
```

**SHAP 瀑布图组件**

```python
def render_shap_waterfall(grid_id, timestamp, city_code):
    resp = requests.post(f"{API_BASE}/explain/shap", json={
        "city": city_code,
        "grid_id": grid_id,
        "timestamp": timestamp.isoformat(),
        "n_samples": 100
    })
    shap_data = resp.json()

    # 按组聚合
    groups = shap_data['feature_groups']
    sorted_groups = sorted(groups.items(), key=lambda x: abs(x[1]), reverse=True)

    labels = [g[0] for g in sorted_groups]
    values = [g[1] for g in sorted_groups]
    colors = ['#FF4444' if v > 0 else '#4444FF' for v in values]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(labels) + ["total"],
        y=labels + ["预测值"],
        x=values + [shap_data['prediction']],
        base=shap_data['base_value'],
        connector={"line": {"color": "gray", "width": 1}},
        increasing={"marker": {"color": "#FF4444"}},
        decreasing={"marker": {"color": "#4444FF"}},
    ))

    fig.update_layout(
        title=f"格子 {grid_id} 预测贡献分解（SHAP）",
        xaxis_title="流量贡献（人次）",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    # 详情展开
    with st.expander("查看各维度详细 SHAP 值"):
        shap_df = pd.DataFrame(
            list(shap_data['shap_values'].items()),
            columns=['特征', 'SHAP值']
        ).sort_values('SHAP值', key=abs, ascending=False)
        st.dataframe(shap_df.style.background_gradient(cmap='RdBu_r', subset=['SHAP值']))
```

**时空注意力可视化组件**

```python
def render_attention(grid_id, timestamp, city_code):
    resp = requests.get(f"{API_BASE}/explain/attention", params={
        "city": city_code,
        "grid_id": grid_id,
        "timestamp": timestamp.isoformat()
    })
    attn_data = resp.json()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("时间注意力权重")
        # 过去 T_in 步的注意力
        t_labels = [f"t-{i}" for i in range(len(attn_data['temporal_attention']), 0, -1)]
        fig = go.Figure(go.Bar(
            x=t_labels,
            y=attn_data['temporal_attention'],
            marker_color='rgba(99, 110, 250, 0.8)'
        ))
        fig.update_layout(
            title="模型关注的历史时步",
            xaxis_title="历史时步",
            yaxis_title="注意力权重",
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("空间注意力权重（top-20 格子）")
        grid_meta = pd.read_csv(f'data/processed/grid_meta_{city_code}.csv')
        spatial_attn = attn_data['spatial_attention']

        attn_df = pd.DataFrame([
            {"grid_id": int(k), "attention": v}
            for k, v in sorted(spatial_attn.items(), key=lambda x: x[1], reverse=True)[:20]
        ])
        attn_df = attn_df.merge(grid_meta[['grid_id','center_lon','center_lat']], on='grid_id')

        fig = go.Figure(go.Scattermapbox(
            lat=attn_df['center_lat'],
            lon=attn_df['center_lon'],
            mode='markers',
            marker=dict(
                size=attn_df['attention'] * 300,  # 按权重缩放
                color=attn_df['attention'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="注意力")
            ),
            text=attn_df['grid_id']
        ))
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_zoom=10,
            mapbox_center={"lat": 39.9, "lon": 116.4},
            height=250,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
```

---

## 四、分级预警系统

### 4.1 预警级别与 UI 表现

| 级别 | 颜色 | 触发条件 | UI 表现 | 声音 |
|------|------|---------|---------|------|
| L1 轻度 | 🟡 黄色 | z>3.0，≥6格 | 侧栏橙色提示条 | 无 |
| L2 中度 | 🟠 橙色 | z>3.0，≥12格 | 顶部横幅 + 地图闪烁边框 | 无 |
| L3 重度 | 🔴 红色 | z>3.5，≥12格 | 弹窗 + 持续橙色横幅 | 短促提示音（1次） |
| L4 极端 | 🚨 紫色 | z>4.0，≥25格 | 强制弹窗 + 全屏红色边框 | 持续警报音 |

### 4.2 告警弹窗实现

```python
def show_alert(level, clusters, timestamp):
    LEVEL_CONFIG = {
        1: {"emoji": "🟡", "color": "#FFA500", "title": "L1 轻度异常"},
        2: {"emoji": "🟠", "color": "#FF6B00", "title": "L2 中度异常"},
        3: {"emoji": "🔴", "color": "#FF0000", "title": "L3 重度异常"},
        4: {"emoji": "🚨", "color": "#8B0000", "title": "L4 极端异常 — 请立即关注"},
    }
    cfg = LEVEL_CONFIG[level]

    # Streamlit 原生弹窗（通过 st.warning / st.error）
    msg = (
        f"**{cfg['emoji']} {cfg['title']}** | "
        f"时间: {timestamp.strftime('%Y-%m-%d %H:%M')} | "
        f"影响格子: {sum(c['cell_count'] for c in clusters)} | "
        f"最大Z值: {max(c['max_z'] for c in clusters):.2f}"
    )

    if level >= 3:
        st.error(msg)
        # 声音通知（HTML5 Audio）
        sound_html = """
        <audio autoplay>
          <source src="data:audio/wav;base64,..." type="audio/wav">
        </audio>
        """
        st.components.v1.html(sound_html, height=0)
    elif level == 2:
        st.warning(msg)
    else:
        st.info(msg)

    # 详细信息展开
    with st.expander(f"查看 {len(clusters)} 个异常区域详情"):
        for i, cluster in enumerate(clusters):
            st.markdown(
                f"**区域 {i+1}**: 中心坐标 ({cluster['center_lon']:.4f}, "
                f"{cluster['center_lat']:.4f}), "
                f"{cluster['cell_count']} 格, 最大 Z={cluster['max_z']:.2f}"
            )
```

---

## 五、SHAP 可解释性后端实现

### 5.1 SHAP 服务

```python
# src/api/services/shap_service.py
import shap
import numpy as np
import torch

class SHAPExplainer:
    """
    使用 KernelSHAP 对静态节点特征进行可解释性分析
    KernelSHAP 是模型无关的，适合 GNN+Transformer 混合架构
    """

    FEATURE_NAMES = [
        # 地理位置 [0:4]
        'lat_norm', 'lon_norm', 'dist_center', 'ring_id',
        # 距离 [4:6]
        'dist_hub', 'dist_commercial',
        # 路网 [6:11]
        'road_density', 'highway_density', 'local_density',
        'service_density', 'active_density',
        # POI 11类 [11:22]
        'poi_food', 'poi_shopping', 'poi_entertainment', 'poi_office',
        'poi_residential', 'poi_education', 'poi_healthcare',
        'poi_government', 'poi_tourism', 'poi_sports', 'poi_religious',
        # 遥感 [22:27]
        'ndvi', 'ndbi', 'mndwi', 'b4_mean', 'b4_std',
        # 历史统计 [27:35]
        'hist_mean', 'p99_ratio', 'zero_rate', 'wd_peak_ratio',
        'we_ratio', 'night_ratio', 'flow_dir', 'flow_cv',
        # 公共交通 [35:38]
        'subway_station_density', 'subway_entrance_density', 'bus_stop_density',
    ]

    FEATURE_GROUPS = {
        '地理位置': list(range(0, 4)),
        '距离特征': list(range(4, 6)),
        '路网密度': list(range(6, 11)),
        'POI密度':  list(range(11, 22)),
        '遥感指数': list(range(22, 27)),
        '历史统计': list(range(27, 35)),
        '公共交通': list(range(35, 38)),
    }

    def __init__(self, model, node_features, background_size=50):
        self.model = model
        self.node_features = node_features  # (N, 38) numpy

        # 背景分布：随机采样 N 个格子的特征作为 baseline
        bg_idx = np.random.choice(len(node_features), background_size, replace=False)
        self.background = node_features[bg_idx]

        # KernelSHAP 解释器（对节点特征）
        self.explainer = shap.KernelExplainer(
            self._predict_fn,
            self.background
        )

    def _predict_fn(self, node_feat_perturbed):
        """将扰动后的节点特征输入模型，返回目标格子的预测值"""
        # node_feat_perturbed: (n_samples, 38)
        results = []
        for feat in node_feat_perturbed:
            # 替换目标格子的特征，其余格子保持不变
            nf = self.node_features.copy()
            nf[self.target_grid_id] = feat
            with torch.no_grad():
                pred = self.model.predict_single(
                    self.flow_window,
                    torch.from_numpy(nf).half().cuda()
                )
            results.append(pred[self.target_grid_id, 0])  # 预测第1步
        return np.array(results)

    def explain(self, grid_id, flow_window, n_samples=100):
        self.target_grid_id = grid_id
        self.flow_window = flow_window

        target_feat = self.node_features[grid_id:grid_id+1]  # (1, 38)
        shap_values = self.explainer.shap_values(target_feat, nsamples=n_samples)

        # 组聚合
        group_shap = {}
        for group_name, indices in self.FEATURE_GROUPS.items():
            group_shap[group_name] = float(shap_values[0, indices].sum())

        return {
            "shap_values": dict(zip(self.FEATURE_NAMES, shap_values[0].tolist())),
            "feature_groups": group_shap,
            "base_value": float(self.explainer.expected_value),
        }
```

### 5.2 注意力权重提取

```python
# STAEformer 注意力权重提取
class AttentionExtractor:
    """在 STAEformer 的 forward pass 中 hook 注意力权重"""

    def __init__(self, model):
        self.model = model
        self.temporal_attn = []
        self.spatial_attn = []
        self._register_hooks()

    def _register_hooks(self):
        def temporal_hook(module, input, output):
            # output[1] 是注意力权重矩阵 (B, heads, T, T)
            if output[1] is not None:
                self.temporal_attn.append(
                    output[1].mean(dim=1).detach().cpu()  # 平均所有头
                )

        def spatial_hook(module, input, output):
            if output[1] is not None:
                self.spatial_attn.append(
                    output[1].mean(dim=1).detach().cpu()
                )

        # 注册到 STAEformer 的时间/空间 Attention 层
        for name, module in self.model.named_modules():
            if 'temporal_attn' in name:
                module.register_forward_hook(temporal_hook)
            elif 'spatial_attn' in name:
                module.register_forward_hook(spatial_hook)

    def get_attention_for_grid(self, grid_id, flow_window):
        self.temporal_attn.clear()
        self.spatial_attn.clear()

        with torch.no_grad():
            _ = self.model(flow_window)

        # 时间注意力：目标格子对各历史步的权重
        t_attn = self.temporal_attn[-1][0, grid_id, :].numpy()  # (T_in,)

        # 空间注意力：目标格子对其他格子的权重
        s_attn = self.spatial_attn[-1][0, grid_id, :].numpy()   # (N,)
        top20_idx = np.argsort(s_attn)[::-1][:20]

        return {
            "temporal_attention": t_attn.tolist(),
            "spatial_attention": {
                str(idx): float(s_attn[idx]) for idx in top20_idx
            }
        }
```

---

## 六、历史异常事件查询面板

```python
def render_history_panel(city_code):
    st.subheader("📊 历史异常事件")

    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("起始日期", pd.Timestamp("2016-03-01"))
    with col2:
        end_date = st.date_input("结束日期", pd.Timestamp("2016-04-10"))
    with col3:
        min_level = st.selectbox("最低级别", [1, 2, 3, 4], index=1)

    resp = requests.get(f"{API_BASE}/history/anomalies", params={
        "city": city_code,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "min_level": min_level
    })
    data = resp.json()

    # 统计图：每日异常次数
    df = pd.DataFrame(data['anomalies'])
    if not df.empty:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily = df.groupby(['date', 'level']).size().reset_index(name='count')

        fig = go.Figure()
        colors = {1: '#FFD700', 2: '#FFA500', 3: '#FF4500', 4: '#8B0000'}
        for lvl in [1, 2, 3, 4]:
            sub = daily[daily['level'] == lvl]
            if not sub.empty:
                fig.add_trace(go.Bar(
                    x=sub['date'], y=sub['count'],
                    name=f"L{lvl}", marker_color=colors[lvl]
                ))
        fig.update_layout(
            barmode='stack', title="每日异常事件统计",
            height=200, margin=dict(t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 事件列表
        st.dataframe(
            df[['timestamp', 'level', 'cell_count', 'duration_steps']].rename(columns={
                'timestamp': '时间', 'level': '级别',
                'cell_count': '影响格子', 'duration_steps': '持续步数'
            }),
            height=200
        )
```

---

## 七、启动与部署

### 7.1 本地启动

```bash
# 启动 FastAPI 后端
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 启动 Streamlit 前端
streamlit run src/visualization/app.py --server.port 8501

# 访问地址
# API 文档: http://localhost:8000/docs
# 可视化界面: http://localhost:8501
```

### 7.2 依赖安装

```bash
pip install \
  fastapi uvicorn[standard] \
  streamlit streamlit-keplergl \
  plotly shap \
  pycuda tensorrt \
  redis[hiredis]
```

### 7.3 配置文件

```yaml
# config/app.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 2

model:
  checkpoint: "outputs/checkpoints/best_model.pt"
  device: "cuda"
  precision: "fp16"

cache:
  backend: "redis"
  ttl_seconds: 300  # 预测结果缓存 5 分钟

alert:
  levels:
    L1: {z_thresh: 3.0, min_cells: 6}
    L2: {z_thresh: 3.0, min_cells: 12}
    L3: {z_thresh: 3.5, min_cells: 12}
    L4: {z_thresh: 4.0, min_cells: 25}
```

---

## 八、Week 6/7 实现优先级

| 优先级 | 组件 | 工时估计 |
|--------|------|---------|
| P0 | FastAPI 骨架 + /predict + /anomaly 接口 | 1天 |
| P0 | Streamlit 主页 + Kepler.gl 热力图 | 1天 |
| P1 | 预测动画播放（Plotly + Slider） | 0.5天 |
| P1 | 分级预警告警面板 | 0.5天 |
| P1 | SHAP 瀑布图（接口 + 前端） | 1天 |
| P2 | 注意力权重可视化（时间+空间） | 1天 |
| P2 | 历史异常查询面板 | 0.5天 |
| P3 | TensorRT FP16 推理引擎接入 | 1天 |
| P3 | 声音预警 + 弹窗强化 | 0.5天 |
