"""
统一流量数据集模块，支持 TaxiBJ 和 TaxiNYC。
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# TaxiBJ 各段步数
_BJ_SEG = dict(n13=4888, n14=4780, n15=5596, n16=7220)
_BJ_VAL_RATIO = 0.25   # BJ15 后 25% 作验证集


def load_taxibj(path: str = "data/processed/taxibj_clean.npz"):
    d = np.load(path, allow_pickle=False)
    return d["data"], d["timestamps"]   # (22484,2,32,32), (22484,)


def load_taxinyc(path: str = "data/processed/taxinyc_clean.npz"):
    d = np.load(path, allow_pickle=False)
    return d["data"], d["timestamps"]   # (17520,2,15,5), (17520,)


def split_chronological(data: np.ndarray,
                         train: float = 0.7,
                         val:   float = 0.1,
                         test:  float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按时序比例切分（7:1:2），适用于任意城市数据。
    不打乱顺序，确保测试集位于最后（时间最新）。
    """
    assert abs(train + val + test - 1.0) < 1e-6, "比例之和须为 1"
    T  = len(data)
    t1 = int(T * train)
    t2 = int(T * (train + val))
    return data[:t1], data[t1:t2], data[t2:]


def split_taxibj(data: np.ndarray,
                 train: float = 0.7,
                 val:   float = 0.1,
                 test:  float = 0.2):
    """TaxiBJ 7:1:2 时序切分。"""
    return split_chronological(data, train, val, test)


def split_taxinyc(data: np.ndarray,
                  train: float = 0.7,
                  val:   float = 0.1,
                  test:  float = 0.2):
    """TaxiNYC 7:1:2 时序切分。"""
    return split_chronological(data, train, val, test)


class FlowDataset(Dataset):
    """
    滑动窗口流量数据集。

    Parameters
    ----------
    data   : (T, 2, H, W) float32 — inflow / outflow
    T_in   : 输入时间步数
    T_out  : 预测时间步数
    mean   : 归一化均值（None 则从 data 计算）
    std    : 归一化标准差

    Item
    ----
    x : (N, T_in, 2)   归一化后的 inflow+outflow 序列
    y : (N, T_out)     归一化后的 inflow 预测目标
    """

    def __init__(self,
                 data: np.ndarray,
                 T_in: int = 12,
                 T_out: int = 12,
                 mean: float | None = None,
                 std:  float | None = None,
                 time_offset: int | None = None,
                 steps_per_day: int = 48,
                 timestamps: np.ndarray | None = None,
                 tz_offset: int = 0,
                 holidays: set | None = None,
                 makeup_workdays: set | None = None,
                 filter_gaps: bool = False,
                 include_time_features: bool = False,
                 dynamic_features: np.ndarray | None = None,
                 weather_features: np.ndarray | None = None,
                 return_future_time: bool = False):
        """
        time_offset : 该分割在完整数据集中的绝对起始步数。
                      不为 None 时，在 x 末尾追加 4 维正弦时间编码：
                      [sin/cos(hour_of_day), sin/cos(day_of_week)]
        steps_per_day : 一天包含多少个时间步（TaxiBJ 为 48，即 30 min/步）
        """
        if mean is None:
            mean = float(data.mean())
            std  = float(data.std())
        self.mean = mean
        self.std  = std

        norm = (data - mean) / (std + 1e-8)     # (T, 2, H, W)
        T, C, H, W = norm.shape
        flat = norm.reshape(T, C, H * W).transpose(0, 2, 1)  # (T, N, 2)
        self.flow  = torch.tensor(flat, dtype=torch.float32)
        self.T_in  = T_in
        self.T_out = T_out
        self.N     = H * W
        self.length = T - T_in - T_out + 1
        self.time_offset   = time_offset
        self.steps_per_day = steps_per_day
        self.time_features = None
        self.dynamic_features = None
        dynamic_valid = None
        if dynamic_features is not None:
            if (dynamic_features.ndim != 3 or
                    dynamic_features.shape[:2] != (T, self.N)):
                raise ValueError(
                    "dynamic_features must have shape (T, N, F), got "
                    f"{dynamic_features.shape}"
                )
            dynamic_valid = np.isfinite(dynamic_features).all(axis=(1, 2))
            safe = np.nan_to_num(dynamic_features, copy=True)
            self.dynamic_features = torch.from_numpy(
                safe.astype(np.float32, copy=False)
            )

        if include_time_features:
            if timestamps is None:
                raise ValueError("include_time_features=True requires timestamps")
            if len(timestamps) != T:
                raise ValueError("timestamps and data must have the same length")
            from src.features.temporal import build_temporal
            temporal = build_temporal(timestamps, tz_offset, holidays)
            weekend = temporal["is_weekend"].to_numpy(np.float32)
            holiday = temporal["is_holiday"].to_numpy(np.float32)
            local_dates = temporal.index.date
            makeup = np.array(
                [d in (makeup_workdays or set()) for d in local_dates],
                dtype=np.float32,
            )
            # Official makeup dates override the weekend rule.
            workday = np.where(makeup > 0, 1.0,
                               1.0 - np.maximum(weekend, holiday)).astype(np.float32)
            cols = temporal[[
                "slot_sin", "slot_cos", "weekday_sin", "weekday_cos",
                "is_weekend", "is_holiday",
            ]].to_numpy(np.float32)
            pieces = [cols, workday[:, None]]
            if makeup_workdays is not None:
                pieces.append(makeup[:, None])
            features = np.concatenate(pieces, axis=1)
            self.time_features = torch.from_numpy(features)  # (T, 7 or 8)

        self.weather_features = None
        self.return_future_time = return_future_time
        if return_future_time and self.time_features is None:
            raise ValueError("return_future_time=True requires time features")
        if weather_features is not None:
            if len(weather_features) != T or weather_features.ndim != 2:
                raise ValueError(
                    f"weather_features must have shape (T, F), got "
                    f"{weather_features.shape}"
                )
            if not np.isfinite(weather_features).all():
                raise ValueError("weather_features contains NaN or Inf")
            self.weather_features = torch.from_numpy(
                weather_features.astype(np.float32, copy=False)
            )

        self.valid_starts = np.arange(self.length, dtype=np.int64)
        if filter_gaps:
            if timestamps is None:
                raise ValueError("filter_gaps=True requires timestamps")
            ts = timestamps.astype("datetime64[s]").astype(np.int64)
            contiguous = np.diff(ts) == 30 * 60
            window_edges = T_in + T_out - 1
            bad = (~contiguous).astype(np.int32)
            prefix = np.concatenate([[0], np.cumsum(bad)])
            counts = prefix[window_edges:] - prefix[:-window_edges]
            self.valid_starts = np.flatnonzero(counts == 0).astype(np.int64)
            self.length = len(self.valid_starts)
        if dynamic_valid is not None:
            bad = (~dynamic_valid).astype(np.int32)
            prefix = np.concatenate([[0], np.cumsum(bad)])
            counts = prefix[T_in:] - prefix[:-T_in]
            valid_dynamic = counts == 0
            self.valid_starts = self.valid_starts[valid_dynamic[self.valid_starts]]
            self.length = len(self.valid_starts)

    def __len__(self):
        return self.length

    def _time_enc(self, start_abs: int) -> torch.Tensor:
        """正弦时间编码，返回 (T_in, 4)。"""
        t = torch.arange(self.T_in, dtype=torch.float32) + start_abs
        spd = self.steps_per_day
        hour = t % spd                    # 当天第几步（0~47）
        dow  = (t // spd) % 7            # 周几（0~6）
        enc  = torch.stack([
            torch.sin(2 * math.pi * hour / spd),
            torch.cos(2 * math.pi * hour / spd),
            torch.sin(2 * math.pi * dow  / 7),
            torch.cos(2 * math.pi * dow  / 7),
        ], dim=1)                         # (T_in, 4)
        return enc

    def __getitem__(self, idx):
        start = int(self.valid_starts[idx])
        x = self.flow[start : start + self.T_in]                                  # (T_in, N, 2)
        y = self.flow[start + self.T_in : start + self.T_in + self.T_out, :, 0]  # (T_out, N)

        if self.dynamic_features is not None:
            dynamic = self.dynamic_features[start : start + self.T_in]
            x = torch.cat([x, dynamic], dim=-1)

        if self.time_features is not None:
            enc = self.time_features[start : start + self.T_in]
            enc = enc.unsqueeze(1).expand(-1, self.N, -1)      # (T_in, N, 7)
            x = torch.cat([x, enc], dim=-1)                    # (T_in, N, 9)
        elif self.time_offset is not None:
            enc = self._time_enc(start + self.time_offset)     # (T_in, 4)
            enc = enc.unsqueeze(1).expand(-1, self.N, -1)      # (T_in, N, 4)
            x   = torch.cat([x, enc], dim=-1)                  # (T_in, N, 6)

        if self.weather_features is not None:
            weather = self.weather_features[start : start + self.T_in]
            weather = weather.unsqueeze(1).expand(-1, self.N, -1)
            x = torch.cat([x, weather], dim=-1)

        x = x.permute(1, 0, 2)
        y = y.permute(1, 0)
        if self.return_future_time:
            future_time = self.time_features[
                start + self.T_in : start + self.T_in + self.T_out
            ]
            return (x, future_time), y
        return x, y


def load_graphs(city: str = "bj",
                graph_names: list[str] | None = None,
                processed_dir: str = "data/processed",
                device: torch.device | None = None) -> dict:
    """
    加载城市图结构，返回 dict[name -> {"edge_index", "edge_weight", "num_nodes"}]。

    Parameters
    ----------
    city        : "bj" 或 "nyc"
    graph_names : 要加载的图名列表，默认全部四张
    """
    if graph_names is None:
        graph_names = ["spatial", "poi", "flow", "transit"]
    p = Path(processed_dir)
    graphs = {}
    for name in graph_names:
        path = p / f"graph_{name}_{city}.pt"
        g = torch.load(path, map_location=device or "cpu", weights_only=True)
        graphs[name] = g
    return graphs
