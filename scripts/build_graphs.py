"""
多模态图结构构建（方案C 主线）
输出（每个城市各三张图 + 节点特征矩阵）：

  data/processed/graph_spatial_{city}.pt   — G_spatial: 8邻域高斯距离衰减图
  data/processed/graph_poi_{city}.pt       — G_poi:     POI余弦相似度图
  data/processed/graph_flow_{city}.pt      — G_flow:    Pearson流量相关图
  data/processed/node_features_{city}.npy  — 节点静态特征 (N, F)

节点特征维度（共 37 维）：
  [0:4]   地理位置  : lat_norm, lon_norm, dist_center_norm, ring_id(1~5)
  [4:6]   到枢纽/商圈距离 : dist_hub_min_norm, dist_com_min_norm
  [6:11]  路网特征  : road_density(log1p), highway(log1p), local(log1p),
                      service(log1p), active(log1p)
  [11:22] POI密度   : 11类 (log1p)，不含 transport（已拆分为独立特征）
  [22:27] 遥感特征  : ndvi, ndbi, mndwi, b4_mean, b4_std
  [27:35] 历史统计  : hist_mean_norm, p99_ratio, zero_rate,
                      wd_peak_ratio, we_ratio, night_ratio,
                      flow_dir_norm, flow_cv
  [35:37] 公共交通  : subway_station_density, subway_entrance_density

超参数：
  σ = 1.5 格（空间图高斯带宽）
  τ = 0.5 （POI 余弦相似度阈值）
  ρ = 0.6 （流量 Pearson 相关系数阈值）
"""

import math
import numpy as np
import pandas as pd
import torch
from pathlib import Path

OUT  = Path("data/processed")
PROC = Path("data/processed")

# ── 超参数 ────────────────────────────────────────────────────────────────────
SIGMA       = 1.5   # 空间图高斯带宽（格单位）
POI_K       = 10    # POI 图每节点保留 top-k 近邻
FLOW_K      = 15    # 流量图每节点保留 top-k 近邻
RING_THR_KM = [3.5, 7.0, 11.0, 16.0]   # 二/三/四/五环边界 km

POI_CATS = [
    "food", "shopping", "entertainment", "office", "residential",
    "education", "healthcare", "government",
    "tourism", "sports", "religious",
]  # 11 类，transport 已拆分为独立公共交通特征

# ── 城市配置 ──────────────────────────────────────────────────────────────────
CITIES = {
    "bj": {
        "H": 32, "W": 32, "tz": 8,
        "lon_range": (116.25, 116.75),
        "lat_range": (39.75, 40.25),
        "flow_npz" : PROC / "taxibj_clean.npz",
        "grid_csv" : PROC / "grid_meta_bj.csv",
        "spatial_csv": PROC / "spatial_features_bj.csv",
        "poi_csv"      : PROC / "poi_features_bj.csv",
        "satellite_csv": PROC / "satellite_features_bj.csv",
        "transit_csv"  : PROC / "transit_features_bj.csv",
    },
    "nyc": {
        "H": 15, "W": 5, "tz": -5,
        "lon_range": (-74.0166, -73.9004),
        "lat_range": (40.6996, 40.9196),
        "flow_npz"  : PROC / "taxinyc_clean.npz",
        "grid_csv"  : PROC / "grid_meta_nyc.csv",
        "spatial_csv": PROC / "spatial_features_nyc.csv",
        "poi_csv"      : PROC / "poi_features_nyc.csv",
        "satellite_csv": PROC / "satellite_features_nyc.csv",
        "transit_csv"  : PROC / "transit_features_nyc.csv",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 工具：归一化
# ═══════════════════════════════════════════════════════════════════════════════

def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)

def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)

def save_graph(edge_index: np.ndarray, edge_weight: np.ndarray,
               num_nodes: int, path: Path):
    """存储为 dict（兼容 PyG Data）"""
    torch.save({
        "edge_index" : torch.from_numpy(edge_index).long(),
        "edge_weight": torch.from_numpy(edge_weight).float(),
        "num_nodes"  : num_nodes,
    }, path)
    nnz = edge_index.shape[1]
    print(f"    → {path.name}  edges={nnz}  nodes={num_nodes}")


# ═══════════════════════════════════════════════════════════════════════════════
# G_spatial：8邻域高斯距离衰减图
# ═══════════════════════════════════════════════════════════════════════════════

def build_spatial_graph(H: int, W: int, sigma: float = SIGMA):
    """
    Moore 8邻域，边权 = exp(-d² / σ²)。
    自环不加入。
    返回 edge_index (2, E), edge_weight (E,)。
    """
    rows, cols, weights = [], [], []
    for r in range(H):
        for c in range(W):
            src = r * W + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        dst = nr * W + nc
                        d   = math.sqrt(dr**2 + dc**2)
                        w   = math.exp(-(d**2) / (sigma**2))
                        rows.append(src); cols.append(dst)
                        weights.append(w)

    edge_index  = np.array([rows, cols], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    return edge_index, edge_weight


# ═══════════════════════════════════════════════════════════════════════════════
# G_poi：POI 余弦相似度 kNN 图
# ═══════════════════════════════════════════════════════════════════════════════

def build_poi_graph(poi_feat: pd.DataFrame, k: int = POI_K):
    """
    取 11 类 POI 密度列（log1p），计算余弦相似度，
    每节点保留 top-k 近邻（有向图），边权 = 相似度值。
    transport 已从 POI 中移除，无需额外排除。
    """
    density_cols = [f"{c}_density" for c in POI_CATS]
    X = poi_feat[density_cols].values.astype(np.float32)
    X = np.log1p(X)

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_n   = X / norms
    sim   = X_n @ X_n.T   # (N, N)
    np.fill_diagonal(sim, -1.0)   # 排除自环

    N   = sim.shape[0]
    k_  = min(k, N - 1)

    src_list, dst_list, w_list = [], [], []
    for i in range(N):
        top_k = np.argpartition(sim[i], -k_)[-k_:]
        for j in top_k:
            src_list.append(i); dst_list.append(j)
            w_list.append(float(sim[i, j]))

    edge_index  = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(w_list, dtype=np.float32)
    return edge_index, edge_weight


# ═══════════════════════════════════════════════════════════════════════════════
# G_flow：历史流量 Pearson 相关 kNN 图
# ═══════════════════════════════════════════════════════════════════════════════

def build_flow_graph(flow_data: np.ndarray, k: int = FLOW_K):
    """
    flow_data: (T, 2, H, W)，取 channel 0（inflow）。
    计算格间 Pearson 相关矩阵，每节点保留 top-k 正相关近邻。
    """
    T, C, H, W = flow_data.shape
    N = H * W
    X = flow_data[:, 0, :, :].reshape(T, N).astype(np.float64)

    print(f"    计算 Pearson 相关矩阵 ({N}×{N})...")
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn  = (X - mu) / std
    corr = (Xn.T @ Xn) / T   # (N, N)
    np.fill_diagonal(corr, -1.0)

    k_ = min(k, N - 1)
    src_list, dst_list, w_list = [], [], []
    for i in range(N):
        top_k = np.argpartition(corr[i], -k_)[-k_:]
        for j in top_k:
            if corr[i, j] > 0:   # 只保留正相关
                src_list.append(i); dst_list.append(j)
                w_list.append(float(corr[i, j]))

    edge_index  = np.array([src_list, dst_list], dtype=np.int64)
    edge_weight = np.array(w_list, dtype=np.float32)
    return edge_index, edge_weight


# ═══════════════════════════════════════════════════════════════════════════════
# 节点特征矩阵
# ═══════════════════════════════════════════════════════════════════════════════

def _ring_id(dist_km: np.ndarray) -> np.ndarray:
    ring = np.ones(len(dist_km), dtype=np.float32)
    for i, thr in enumerate(RING_THR_KM):
        ring[dist_km > thr] = i + 2.0
    return ring


def _hist_stats(flow_data: np.ndarray, timestamps_utc: np.ndarray,
                tz_offset: int) -> np.ndarray:
    """
    计算每格的 8 维历史统计特征。
    flow_data: (T, 2, H, W)
    返回: (N, 8)
    """
    T, C, H, W = flow_data.shape
    N = H * W
    inflow  = flow_data[:, 0, :, :].reshape(T, N).astype(np.float64)
    outflow = flow_data[:, 1, :, :].reshape(T, N).astype(np.float64)

    # 本地时间
    ts_utc   = pd.DatetimeIndex(timestamps_utc.astype("datetime64[s]"))
    ts_local = ts_utc + pd.Timedelta(hours=tz_offset)
    hour     = ts_local.hour.values
    weekday  = ts_local.dayofweek.values   # 0=Mon … 6=Sun

    is_weekday = (weekday < 5)
    is_weekend = ~is_weekday
    is_night   = (hour < 6)
    is_wd_peak = is_weekday & ((hour == 7) | (hour == 8))

    # 防零除
    wd_sum = is_weekday.sum(); we_sum = is_weekend.sum()
    pk_sum = is_wd_peak.sum()

    hist_mean = inflow.mean(axis=0)
    hist_p99  = np.percentile(inflow, 99, axis=0)
    global_p99 = hist_p99.max() + 1e-8

    zero_rate     = (inflow == 0).mean(axis=0)
    wd_mean       = inflow[is_weekday].mean(axis=0) if wd_sum else hist_mean
    wd_all_mean   = wd_mean
    wd_peak_mean  = inflow[is_wd_peak].mean(axis=0) if pk_sum else wd_mean
    we_mean       = inflow[is_weekend].mean(axis=0) if we_sum else hist_mean
    night_mean    = inflow[is_night].mean(axis=0)
    flow_dir      = (inflow - outflow).mean(axis=0)
    flow_cv       = inflow.std(axis=0) / (hist_mean + 1e-8)

    wd_peak_ratio = wd_peak_mean / (wd_all_mean + 1e-8)
    we_ratio      = we_mean      / (wd_all_mean + 1e-8)
    night_ratio   = night_mean   / (hist_mean   + 1e-8)
    p99_ratio     = hist_p99     / global_p99

    feats = np.stack([
        minmax(hist_mean),
        p99_ratio,
        zero_rate,
        np.clip(wd_peak_ratio, 0, 5),
        np.clip(we_ratio, 0, 5),
        np.clip(night_ratio, 0, 1),
        minmax(flow_dir),
        np.clip(flow_cv, 0, 5),
    ], axis=1).astype(np.float32)   # (N, 8)
    return feats


def build_node_features(cfg: dict) -> np.ndarray:
    grid = pd.read_csv(cfg["grid_csv"]).set_index("grid_id").sort_index()
    spat = pd.read_csv(cfg["spatial_csv"], index_col=0).sort_index()
    poi  = pd.read_csv(cfg["poi_csv"],     index_col=0).sort_index()
    sat  = pd.read_csv(cfg["satellite_csv"], index_col=0).sort_index()

    lon_min, lon_max = cfg["lon_range"]
    lat_min, lat_max = cfg["lat_range"]

    # ── [0:4] 地理位置 ──────────────────────────────────────────────────────
    lat_norm = minmax(grid["center_lat"].values.astype(np.float32))
    lon_norm = minmax(grid["center_lon"].values.astype(np.float32))
    dist_c   = spat["dist_center_km"].values.astype(np.float32)
    dist_c_n = minmax(dist_c)
    ring_id  = _ring_id(dist_c) / 5.0   # 归一化到 [0.2, 1.0]

    geo = np.stack([lat_norm, lon_norm, dist_c_n, ring_id], axis=1)   # (N,4)

    # ── [4:6] 距离特征 ──────────────────────────────────────────────────────
    dist_hub = minmax(spat["dist_hub_min_km"].values.astype(np.float32))
    dist_com = minmax(spat["dist_commercial_min_km"].values.astype(np.float32))
    dist_f   = np.stack([dist_hub, dist_com], axis=1)                  # (N,2)

    # ── [6:11] 路网特征 ─────────────────────────────────────────────────────
    road_cols = ["road_density_km_km2", "highway_density",
                 "local_density", "service_density", "active_density"]
    road_raw  = spat[road_cols].values.astype(np.float32)
    road_feat = np.log1p(road_raw)
    road_feat = (road_feat - road_feat.mean(axis=0)) / (road_feat.std(axis=0) + 1e-8)
                                                                        # (N,5)

    # ── [11:22] POI 密度（11类，不含 transport）────────────────────────────
    poi_cols  = [f"{c}_density" for c in POI_CATS]
    poi_raw   = poi[poi_cols].values.astype(np.float32)
    poi_feat  = np.log1p(poi_raw)
    poi_feat  = (poi_feat - poi_feat.mean(axis=0)) / (poi_feat.std(axis=0) + 1e-8)
                                                                        # (N,11)

    # ── [22:27] 遥感特征 ────────────────────────────────────────────────────
    sat_arr = sat[["ndvi", "ndbi", "mndwi", "b4_mean", "b4_std"]].values.astype(np.float32)
    sat_arr = (sat_arr - sat_arr.mean(axis=0)) / (sat_arr.std(axis=0) + 1e-8)
                                                                        # (N,5)

    # ── [27:35] 历史统计 ────────────────────────────────────────────────────
    npz = np.load(cfg["flow_npz"], allow_pickle=False)
    hist = _hist_stats(npz["data"], npz["timestamps"], cfg["tz"])      # (N,8)

    # ── [35:37] 公共交通特征 ────────────────────────────────────────────────
    transit = pd.read_csv(cfg["transit_csv"], index_col=0).sort_index()
    tr_cols = ["subway_station_density", "subway_entrance_density"]
    tr_raw  = transit[tr_cols].values.astype(np.float32)
    tr_feat = np.log1p(tr_raw)
    tr_feat = (tr_feat - tr_feat.mean(axis=0)) / (tr_feat.std(axis=0) + 1e-8)
                                                                        # (N,2)

    # ── 拼接 ────────────────────────────────────────────────────────────────
    node_feat = np.concatenate([geo, dist_f, road_feat,
                                poi_feat, sat_arr, hist, tr_feat], axis=1)  # (N,37)
    assert node_feat.shape[1] == 37, node_feat.shape
    return node_feat.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def process_city(city: str, cfg: dict):
    print(f"\n  [{city.upper()}]")
    H, W = cfg["H"], cfg["W"]
    N    = H * W

    # ── G_spatial ──────────────────────────────────────────────────────────
    print(f"  G_spatial (8邻域, σ={SIGMA})...")
    ei, ew = build_spatial_graph(H, W, SIGMA)
    save_graph(ei, ew, N, OUT / f"graph_spatial_{city}.pt")
    print(f"    密度: {ei.shape[1]/(N*8)*100:.1f}% 最大边数已填满")

    # ── G_poi ──────────────────────────────────────────────────────────────
    print(f"  G_poi (余弦相似度 top-{POI_K} kNN, 11类 POI)...")
    poi_df = pd.read_csv(cfg["poi_csv"], index_col=0).sort_index()
    ei, ew = build_poi_graph(poi_df, POI_K)
    save_graph(ei, ew, N, OUT / f"graph_poi_{city}.pt")
    if len(ew) > 0:
        print(f"    sim 均值={ew.mean():.3f}, 最小={ew.min():.3f}")

    # ── G_flow ─────────────────────────────────────────────────────────────
    print(f"  G_flow (Pearson top-{FLOW_K} kNN)...")
    npz = np.load(cfg["flow_npz"], allow_pickle=False)
    ei, ew = build_flow_graph(npz["data"], FLOW_K)
    save_graph(ei, ew, N, OUT / f"graph_flow_{city}.pt")
    if len(ew) > 0:
        print(f"    corr 均值={ew.mean():.3f}, 最小={ew.min():.3f}")

    # ── 节点特征 ────────────────────────────────────────────────────────────
    print(f"  节点特征矩阵...")
    nf = build_node_features(cfg)
    out_path = OUT / f"node_features_{city}.npy"
    np.save(out_path, nf)
    print(f"    shape={nf.shape}, NaN数={np.isnan(nf).sum()}")
    print(f"    → {out_path.name}")


def main():
    print("=" * 55)
    print("  多模态图结构构建")
    print("=" * 55)
    for city, cfg in CITIES.items():
        process_city(city, cfg)

    print("\n  汇总统计:")
    for city in CITIES:
        print(f"\n  [{city.upper()}]")
        for gtype in ["spatial", "poi", "flow"]:
            d = torch.load(OUT / f"graph_{gtype}_{city}.pt", weights_only=True)
            E = d["edge_index"].shape[1]
            N = d["num_nodes"]
            print(f"    G_{gtype}: {E} 条边  ({E/N:.1f} 边/节点均值)")
        nf = np.load(OUT / f"node_features_{city}.npy")
        print(f"    node_features: {nf.shape}  dtype={nf.dtype}")

    print("\n  完成。")


if __name__ == "__main__":
    main()
