"""
节点静态特征矩阵组装

将各分组特征文件合并为 (N, 41) float32 静态特征矩阵。

维度说明：
  [0:4]   地理位置
  [4:6]   到枢纽/商圈最短距离
  [6:11]  路网密度（5类）
  [11:22] POI 密度（11类）
  [22:27] 遥感指数（NDVI/NDBI/MNDWI/B4均值/B4标准差）
  [27:35] 历史流量统计（8维）
  [35:38] 公共交通密度（地铁站/入口/公交站）
  [38:40] 轨道线路（num_subway_lines log1p / is_transfer_hub）
  [40]    公交线路数（num_bus_routes log1p）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .base import minmax, zscore

POI_CATS = [
    "food", "shopping", "entertainment", "office", "residential",
    "education", "healthcare", "government", "tourism", "sports", "religious",
]

RING_THR_KM = [3.5, 7.0, 11.0, 16.0]


def _ring_id(dist_km: np.ndarray) -> np.ndarray:
    ring = np.ones(len(dist_km), dtype=np.float32)
    for i, thr in enumerate(RING_THR_KM):
        ring[dist_km > thr] = i + 2.0
    return ring


def hist_stats(flow_data: np.ndarray,
               timestamps_utc: np.ndarray,
               tz_offset: int) -> np.ndarray:
    """
    从流量数据计算每格 8 维历史统计特征。

    Parameters
    ----------
    flow_data      : (T, 2, H, W) float32
    timestamps_utc : (T,) datetime64[s]
    tz_offset      : 时区偏移小时数

    Returns
    -------
    np.ndarray, shape (N, 8)
    """
    T, C, H, W = flow_data.shape
    N = H * W
    inflow  = flow_data[:, 0, :, :].reshape(T, N).astype(np.float64)
    outflow = flow_data[:, 1, :, :].reshape(T, N).astype(np.float64)

    ts_utc   = pd.DatetimeIndex(timestamps_utc.astype("datetime64[s]"))
    ts_local = ts_utc + pd.Timedelta(hours=tz_offset)
    hour     = ts_local.hour.values
    weekday  = ts_local.dayofweek.values

    is_weekday = weekday < 5
    is_weekend = ~is_weekday
    is_night   = hour < 6
    is_wd_peak = is_weekday & ((hour == 7) | (hour == 8))

    wd_sum = is_weekday.sum()
    pk_sum = is_wd_peak.sum()
    we_sum = is_weekend.sum()

    hist_mean    = inflow.mean(axis=0)
    hist_p99     = np.percentile(inflow, 99, axis=0)
    global_p99   = hist_p99.max() + 1e-8
    zero_rate    = (inflow == 0).mean(axis=0)
    wd_mean      = inflow[is_weekday].mean(axis=0) if wd_sum else hist_mean
    wd_peak_mean = inflow[is_wd_peak].mean(axis=0) if pk_sum else wd_mean
    we_mean      = inflow[is_weekend].mean(axis=0) if we_sum else hist_mean
    night_mean   = inflow[is_night].mean(axis=0)
    flow_dir     = (inflow - outflow).mean(axis=0)
    flow_cv      = inflow.std(axis=0) / (hist_mean + 1e-8)

    feats = np.stack([
        minmax(hist_mean),
        hist_p99 / global_p99,
        zero_rate,
        np.clip(wd_peak_mean / (wd_mean + 1e-8), 0, 5),
        np.clip(we_mean      / (wd_mean + 1e-8), 0, 5),
        np.clip(night_mean   / (hist_mean + 1e-8), 0, 1),
        minmax(flow_dir),
        np.clip(flow_cv, 0, 5),
    ], axis=1).astype(np.float32)
    return feats


def build_node_features(grid_csv: str | Path,
                        spatial_csv: str | Path,
                        poi_csv: str | Path,
                        satellite_csv: str | Path,
                        transit_csv: str | Path,
                        subway_feat_csv: str | Path,
                        bus_feat_csv: str | Path,
                        flow_npz: str | Path,
                        tz_offset: int) -> np.ndarray:
    """
    组装 (N, 41) 节点静态特征矩阵。

    Parameters
    ----------
    各 *_csv / *_npz : 对应中间文件路径
    tz_offset        : 时区偏移小时数

    Returns
    -------
    np.ndarray, shape (N, 41), dtype float32, NaN=0 保证
    """
    grid    = pd.read_csv(grid_csv).set_index("grid_id").sort_index()
    spat    = pd.read_csv(spatial_csv,   index_col=0).sort_index()
    poi     = pd.read_csv(poi_csv,       index_col=0).sort_index()
    sat     = pd.read_csv(satellite_csv, index_col=0).sort_index()
    transit = pd.read_csv(transit_csv,   index_col=0).sort_index()
    sub_df  = pd.read_csv(subway_feat_csv, index_col=0).sort_index()
    bus_df  = pd.read_csv(bus_feat_csv,    index_col=0).sort_index()
    npz     = np.load(flow_npz, allow_pickle=False)

    # [0:4] 地理位置
    lat_norm = minmax(grid["center_lat"].values.astype(np.float32))
    lon_norm = minmax(grid["center_lon"].values.astype(np.float32))
    dist_c   = spat["dist_center_km"].values.astype(np.float32)
    ring     = _ring_id(dist_c) / 5.0
    geo      = np.stack([lat_norm, lon_norm, minmax(dist_c), ring], axis=1)

    # [4:6] 距离特征
    dist_f = np.stack([
        minmax(spat["dist_hub_min_km"].values.astype(np.float32)),
        minmax(spat["dist_commercial_min_km"].values.astype(np.float32)),
    ], axis=1)

    # [6:11] 路网
    road_cols = ["road_density_km_km2", "highway_density",
                 "local_density", "service_density", "active_density"]
    road = np.log1p(spat[road_cols].values.astype(np.float32))
    road = (road - road.mean(0)) / (road.std(0) + 1e-8)

    # [11:22] POI
    poi_cols = [f"{c}_density" for c in POI_CATS]
    poi_arr  = np.log1p(poi[poi_cols].values.astype(np.float32))
    poi_arr  = (poi_arr - poi_arr.mean(0)) / (poi_arr.std(0) + 1e-8)

    # [22:27] 遥感
    sat_arr = sat[["ndvi", "ndbi", "mndwi", "b4_mean", "b4_std"]].values.astype(np.float32)
    sat_arr = (sat_arr - sat_arr.mean(0)) / (sat_arr.std(0) + 1e-8)

    # [27:35] 历史统计
    hist = hist_stats(npz["data"], npz["timestamps"], tz_offset)

    # [35:38] 公共交通密度
    tr_cols = ["subway_station_density", "subway_entrance_density", "bus_stop_density"]
    tr_arr  = np.log1p(transit[tr_cols].values.astype(np.float32))
    tr_arr  = (tr_arr - tr_arr.mean(0)) / (tr_arr.std(0) + 1e-8)

    # [38:40] 轨道线路
    sub_raw  = sub_df[["num_subway_lines", "is_transfer_hub"]].values.astype(np.float32)
    sub_feat = np.stack([np.log1p(sub_raw[:, 0]), sub_raw[:, 1]], axis=1)

    # [40] 公交线路数
    bus_feat = np.log1p(bus_df["num_bus_routes"].values.astype(np.float32)).reshape(-1, 1)

    node_feat = np.concatenate(
        [geo, dist_f, road, poi_arr, sat_arr, hist, tr_arr, sub_feat, bus_feat],
        axis=1
    ).astype(np.float32)

    assert node_feat.shape[1] == 41, node_feat.shape
    return node_feat
