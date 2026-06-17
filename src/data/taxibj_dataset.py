import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_taxibj(data_dir="data/raw/taxibj", years=(13, 14, 15, 16)):
    segments = []
    for y in years:
        with h5py.File(f"{data_dir}/BJ{y:02d}_M32x32_T30_InOut.h5", "r") as f:
            segments.append(f["data"][:])          # (T, 2, 32, 32)
    return np.concatenate(segments, axis=0)        # (T_total, 2, 32, 32)


def normalize(data, mean=None, std=None):
    if mean is None:
        mean = data.mean()
        std  = data.std()
    return (data - mean) / (std + 1e-8), mean, std


def build_spatial_edge_index(H=32, W=32, connectivity=8):
    """Return edge_index (2, E) for H×W grid with 4 or 8 connectivity."""
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    nid = rows * W + cols
    deltas = [(0,1),(0,-1),(1,0),(-1,0)]
    if connectivity == 8:
        deltas += [(1,1),(1,-1),(-1,1),(-1,-1)]
    src, dst = [], []
    for dr, dc in deltas:
        r2, c2 = rows + dr, cols + dc
        mask = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        src.extend(nid[mask].flatten().tolist())
        dst.extend((r2[mask] * W + c2[mask]).flatten().tolist())
    return torch.tensor([src, dst], dtype=torch.long)


class TaxiBJDataset(Dataset):
    """
    Sliding-window dataset over TaxiBJ flow data.
    Each sample: (x, y) where
      x: (N, T_in, 2)  — inflow/outflow for T_in steps
      y: (N, T_out)    — inflow for T_out steps (prediction target)
    N = H * W = 1024
    """
    def __init__(self, data, T_in=12, T_out=3, mean=None, std=None):
        # data: (T, 2, H, W)
        data_norm, self.mean, self.std = normalize(data, mean, std)
        T, C, H, W = data_norm.shape
        self.N = H * W
        # reshape to (T, N, C)
        self.flow = torch.tensor(
            data_norm.reshape(T, C, H * W).transpose(0, 2, 1),  # (T, N, C)
            dtype=torch.float32
        )
        self.T_in  = T_in
        self.T_out = T_out
        self.length = T - T_in - T_out

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.flow[idx : idx + self.T_in]          # (T_in, N, 2)
        y = self.flow[idx + self.T_in : idx + self.T_in + self.T_out, :, 0]  # (T_out, N)
        return x.permute(1, 0, 2), y.permute(1, 0)   # (N, T_in, 2), (N, T_out)
