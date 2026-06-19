"""
多模态图结构构建

提供四类图的构建函数，统一返回 PyTorch Geometric 兼容格式：
  {"edge_index": LongTensor(2,E), "edge_weight": FloatTensor(E), "num_nodes": int}

图类型：
  G_spatial  — 8邻域高斯距离衰减图
  G_poi      — POI 余弦相似度 kNN 图
  G_flow     — 历史流量 Pearson 相关 kNN 图
  G_transit  — 轨道线路拓扑图（由 build_subway_features.py 生成后直接加载）
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ── G_spatial ─────────────────────────────────────────────────────────


def build_spatial_graph(H: int, W: int,
                        sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Moore 8 邻域，边权 = exp(-d² / σ²)。

    Parameters
    ----------
    H, W   : 网格行列数
    sigma  : 高斯带宽（格单位），默认 1.5

    Returns
    -------
    edge_index : (2, E) int64
    edge_weight: (E,)  float32
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
                        d   = math.sqrt(dr ** 2 + dc ** 2)
                        w   = math.exp(-(d ** 2) / (sigma ** 2))
                        rows.append(src); cols.append(dst)
                        weights.append(w)
    return (np.array([rows, cols], dtype=np.int64),
            np.array(weights, dtype=np.float32))


# ── G_poi ──────────────────────────────────────────────────────────────


POI_CATS = [
    "food", "shopping", "entertainment", "office", "residential",
    "education", "healthcare", "government", "tourism", "sports", "religious",
]


def build_poi_graph(poi_feat: pd.DataFrame,
                    k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    基于 POI 类别密度的余弦相似度 kNN 图。

    Parameters
    ----------
    poi_feat : poi_features_{city}.csv 加载后的 DataFrame
    k        : 每节点保留 top-k 近邻

    Returns
    -------
    edge_index, edge_weight
    """
    density_cols = [f"{c}_density" for c in POI_CATS]
    X = poi_feat[density_cols].values.astype(np.float32)
    X = np.log1p(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_n  = X / norms
    sim  = X_n @ X_n.T
    np.fill_diagonal(sim, -1.0)

    N  = sim.shape[0]
    k_ = min(k, N - 1)
    src_list, dst_list, w_list = [], [], []
    for i in range(N):
        top_k = np.argpartition(sim[i], -k_)[-k_:]
        for j in top_k:
            src_list.append(i); dst_list.append(j)
            w_list.append(float(sim[i, j]))

    return (np.array([src_list, dst_list], dtype=np.int64),
            np.array(w_list, dtype=np.float32))


# ── G_flow ─────────────────────────────────────────────────────────────


def build_flow_graph(flow_data: np.ndarray,
                     k: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    基于历史 inflow 时序的 Pearson 相关 kNN 图（仅保留正相关边）。

    Parameters
    ----------
    flow_data : (T, 2, H, W) float，取 channel 0（inflow）
    k         : 每节点保留 top-k 近邻

    Returns
    -------
    edge_index, edge_weight
    """
    T, C, H, W = flow_data.shape
    N = H * W
    X = flow_data[:, 0, :, :].reshape(T, N).astype(np.float64)

    mu   = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    Xn   = (X - mu) / std
    corr = (Xn.T @ Xn) / T
    np.fill_diagonal(corr, -1.0)

    k_ = min(k, N - 1)
    src_list, dst_list, w_list = [], [], []
    for i in range(N):
        top_k = np.argpartition(corr[i], -k_)[-k_:]
        for j in top_k:
            if corr[i, j] > 0:
                src_list.append(i); dst_list.append(j)
                w_list.append(float(corr[i, j]))

    return (np.array([src_list, dst_list], dtype=np.int64),
            np.array(w_list, dtype=np.float32))


# ── 存储 / 加载 ────────────────────────────────────────────────────────


def save_graph(edge_index: np.ndarray, edge_weight: np.ndarray,
               num_nodes: int, path: str | Path) -> None:
    """将图保存为 .pt 文件（PyTorch Geometric 兼容格式）。"""
    torch.save({
        "edge_index" : torch.from_numpy(edge_index).long(),
        "edge_weight": torch.from_numpy(edge_weight).float(),
        "num_nodes"  : num_nodes,
    }, path)


def load_graph(path: str | Path) -> dict:
    """加载 .pt 图文件，返回含 edge_index / edge_weight / num_nodes 的字典。"""
    return torch.load(path, weights_only=True)


def graph_stats(path: str | Path) -> str:
    """返回图的边数 / 节点数摘要字符串，用于日志输出。"""
    d = load_graph(path)
    E = d["edge_index"].shape[1]
    N = d["num_nodes"]
    return f"nodes={N}, edges={E}, avg_deg={E/N:.1f}"
