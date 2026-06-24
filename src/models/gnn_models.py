"""
基础图神经网络模型：GCN 和 GAT。

将时间序列展平后作为节点特征输入图网络，捕捉空间依赖关系：
  x: (B, N, T_in, F_in) → 展平 → (B, N, T_in*F_in)
  → 2层图卷积（GCN 或 GAT）
  → Linear(hidden → T_out)
  → (B, N, T_out)

与 STGCN 的区别：无独立的时序卷积模块，时间信息完全展平为节点特征维度。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

from src.models.multi_graph_stgcn import _batch_edge_index


class _GNNBase(nn.Module):
    """GCN / GAT 的共享基类。"""

    def __init__(self, T_in: int, F_in: int, T_out: int, hidden: int):
        super().__init__()
        self.in_dim = T_in * F_in
        self.T_out  = T_out

    def _batch_forward(self, x: torch.Tensor,
                        graphs: dict | None) -> torch.Tensor:
        """
        x      : (B, N, T_in, F_in)
        graphs : 含 edge_index / edge_weight 的 dict（使用 "spatial" 图）
        """
        if graphs is None:
            raise ValueError(f"{self.__class__.__name__} 需要传入 graphs['spatial']")
        B, N, T, F = x.shape

        # 展平时序维度作为节点特征
        x_flat = x.reshape(B, N, T * F)                   # (B, N, T_in*F_in)

        g  = graphs["spatial"]
        ei = g["edge_index"]
        ew = g.get("edge_weight")

        # 批量复制边（每个 batch 样本独立一张图）
        ei_cat, ew_cat = _batch_edge_index(ei, ew, B, N)
        x_in = x_flat.reshape(B * N, T * F)               # (B*N, in_dim)

        out = self._gnn_forward(x_in, ei_cat, ew_cat)     # (B*N, T_out)
        return out.view(B, N, self.T_out)

    def _gnn_forward(self, x, ei, ew):
        raise NotImplementedError

    def forward(self, x: torch.Tensor,
                graphs: dict | None = None) -> torch.Tensor:
        return self._batch_forward(x, graphs)


# ── GCN ────────────────────────────────────────────────────────────────

class GCNPredictor(_GNNBase):
    """
    2层 GCNConv + Linear 输出层。

    Args
    ----
    T_in   : 输入时间步数
    F_in   : 每时步特征维度
    T_out  : 预测步数（48 = 24小时）
    hidden : 图卷积隐藏维度
    """

    def __init__(self, T_in: int = 12, F_in: int = 2, T_out: int = 48,
                 hidden: int = 64):
        super().__init__(T_in, F_in, T_out, hidden)
        self.conv1 = GCNConv(T_in * F_in, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc    = nn.Linear(hidden, T_out)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(0.1)

    def _gnn_forward(self, x, ei, ew):
        h = self.act(self.conv1(x,  ei, ew))   # (B*N, hidden)
        h = self.drop(h)
        h = self.act(self.conv2(h, ei, ew))    # (B*N, hidden)
        return self.fc(h)                       # (B*N, T_out)


# ── GAT ────────────────────────────────────────────────────────────────

class GATPredictor(_GNNBase):
    """
    2层 GATConv（多头注意力）+ Linear 输出层。

    Args
    ----
    T_in   : 输入时间步数
    F_in   : 每时步特征维度
    T_out  : 预测步数（48 = 24小时）
    hidden : 每个注意力头的维度
    heads  : 注意力头数（第1层多头，第2层单头聚合）
    """

    def __init__(self, T_in: int = 12, F_in: int = 2, T_out: int = 48,
                 hidden: int = 32, heads: int = 4):
        super().__init__(T_in, F_in, T_out, hidden)
        self.conv1 = GATConv(T_in * F_in, hidden, heads=heads,  dropout=0.1)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.1)
        self.fc    = nn.Linear(hidden, T_out)
        self.act   = nn.ELU()

    def _gnn_forward(self, x, ei, ew):
        h = self.act(self.conv1(x, ei))        # (B*N, hidden*heads)
        h = self.act(self.conv2(h, ei))        # (B*N, hidden)
        return self.fc(h)                       # (B*N, T_out)
