"""
统一流量数据集模块，支持 TaxiBJ 和 TaxiNYC。
"""
from pathlib import Path

import numpy as np
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
                 std:  float | None = None):
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.flow[idx : idx + self.T_in]                              # (T_in, N, 2)
        y = self.flow[idx + self.T_in : idx + self.T_in + self.T_out, :, 0]  # (T_out, N)
        return x.permute(1, 0, 2), y.permute(1, 0)                        # (N, T_in, 2), (N, T_out)


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
