"""
公共工具函数

所有特征模块共享的底层工具：网格分配、Haversine 距离、归一化。
"""

import math
import numpy as np
import pandas as pd


def assign_to_grid(lon: float | np.ndarray,
                   lat: float | np.ndarray,
                   grid_meta: pd.DataFrame) -> int | np.ndarray:
    """
    将坐标点分配到最近网格，返回 grid_id（越界返回 -1）。

    支持标量或向量输入。grid_meta 需含列：
      lon_min, lon_max, lat_min, lat_max, grid_id
    """
    scalar = np.isscalar(lon)
    lons = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lats = np.atleast_1d(np.asarray(lat, dtype=np.float64))

    lo_min = grid_meta["lon_min"].values.astype(np.float64)
    lo_max = grid_meta["lon_max"].values.astype(np.float64)
    la_min = grid_meta["lat_min"].values.astype(np.float64)
    la_max = grid_meta["lat_max"].values.astype(np.float64)
    gids   = grid_meta["grid_id"].values

    result = np.full(len(lons), -1, dtype=np.int64)
    for i, (lo, la) in enumerate(zip(lons, lats)):
        mask = (lo_min <= lo) & (lo < lo_max) & (la_min <= la) & (la < la_max)
        hits = gids[mask]
        if len(hits):
            result[i] = hits[0]

    return int(result[0]) if scalar else result


def haversine_km(lon1: float, lat1: float,
                 lon2: float, lat2: float) -> float:
    """两点间 Haversine 球面距离（km）。"""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def haversine_matrix(lons: np.ndarray, lats: np.ndarray,
                     ref_lon: float, ref_lat: float) -> np.ndarray:
    """计算每个格子中心到参考点的 Haversine 距离（km），向量化。"""
    R = 6371.0
    lats_r = np.radians(lats)
    ref_lat_r = math.radians(ref_lat)
    d_lat = np.radians(lats - ref_lat)
    d_lon = np.radians(lons - ref_lon)
    a = (np.sin(d_lat / 2) ** 2
         + np.cos(lats_r) * math.cos(ref_lat_r) * np.sin(d_lon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max 归一化到 [0, 1]。"""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score 标准化。"""
    return (x - x.mean()) / (x.std() + 1e-8)
