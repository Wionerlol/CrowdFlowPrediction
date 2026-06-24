"""
方案C 主模型：多图门控融合 STGCN。

架构：
  输入 x (B, N, T_in, F_in)
  → 4路并行 ChebGCN（每路独立处理每个时间步）
  → 门控网络（per-node Softmax 加权）
  → 融合后的空间表示送入 STGCN 时序卷积块 × 2
  → OutputModule → (B, N, T_out)

也包含单图 STGCN 包装器 SingleGraphSTGCN，与 MultiGraphSTGCN 接口一致，
方便 Trainer 统一调用。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv

from .stgcn import TemporalConv, OutputModule


# ── 工具：构建批量边索引 ────────────────────────────────────────────────

def _batch_edge_index(edge_index: torch.Tensor,
                      edge_weight: torch.Tensor | None,
                      batch_size: int,
                      N: int):
    """将单图的 edge_index / edge_weight 复制 batch_size 份，节点编号偏移。"""
    ei = edge_index.unsqueeze(0).expand(batch_size, -1, -1)        # (BT, 2, E)
    offsets = torch.arange(batch_size, device=edge_index.device).view(-1, 1) * N
    ei_cat = (ei + offsets.unsqueeze(-1)).view(2, -1)              # (2, BT*E)
    ew_cat = edge_weight.repeat(batch_size) if edge_weight is not None else None
    return ei_cat, ew_cat


# ── 门控多图空间卷积 ────────────────────────────────────────────────────

class GatedMultiGraphConv(nn.Module):
    """
    对 n_graphs 张图分别做 ChebConv，然后用门控网络加权融合。

    门控权重由输入节点特征直接计算（per-node per-sample Softmax），
    避免鸡生蛋问题（不依赖 GCN 输出来算权重）。
    """

    def __init__(self, in_ch: int, out_ch: int, K: int = 3, n_graphs: int = 4):
        super().__init__()
        self.gcns    = nn.ModuleList([
            ChebConv(in_ch, out_ch, K=K, normalization="sym")
            for _ in range(n_graphs)
        ])
        self.gate    = nn.Linear(in_ch, n_graphs)
        self.n_graphs = n_graphs

    def forward(self,
                x_flat: torch.Tensor,
                BT: int, N: int,
                edge_indices: list[torch.Tensor],
                edge_weights: list[torch.Tensor | None]) -> torch.Tensor:
        """
        Parameters
        ----------
        x_flat      : (BT*N, in_ch)
        BT          : batch × time_steps
        N           : 节点数
        edge_indices: list of (2, E_k) 各图的边索引（已在目标 device 上）
        edge_weights: list of (E_k,) or None

        Returns
        -------
        (BT*N, out_ch)
        """
        # 门控权重由输入特征决定
        gate_w = torch.softmax(self.gate(x_flat), dim=-1)  # (BT*N, n_graphs)

        hs = []
        for i, gcn in enumerate(self.gcns):
            ei_cat, ew_cat = _batch_edge_index(edge_indices[i], edge_weights[i], BT, N)
            h = gcn(x_flat, ei_cat, ew_cat)                # (BT*N, out_ch)
            hs.append(h)

        hs = torch.stack(hs, dim=-1)                        # (BT*N, out_ch, n_graphs)
        gate_w = gate_w.unsqueeze(1)                        # (BT*N, 1, n_graphs)
        return (hs * gate_w).sum(dim=-1)                    # (BT*N, out_ch)


# ── 多图时空卷积块 ──────────────────────────────────────────────────────

class MultiGraphSTConvBlock(nn.Module):
    """
    ST-Conv Block with gated multi-graph spatial convolution:
      TemporalGLU → GatedMultiGraphConv → TemporalGLU → LayerNorm
    """

    def __init__(self, in_ch: int, sp_ch: int, out_ch: int,
                 kernel_size: int = 3, K: int = 3, n_graphs: int = 4):
        super().__init__()
        self.tc1  = TemporalConv(in_ch,  sp_ch,  kernel_size)
        self.mgc  = GatedMultiGraphConv(sp_ch, sp_ch, K=K, n_graphs=n_graphs)
        self.tc2  = TemporalConv(sp_ch,  out_ch, kernel_size)
        self.norm = nn.LayerNorm(out_ch)
        self.act  = nn.ReLU()

    def forward(self, x: torch.Tensor,
                edge_indices: list[torch.Tensor],
                edge_weights: list[torch.Tensor | None]) -> torch.Tensor:
        # x: (B, N, C, T)
        B, N, _, T = x.shape

        h = self.act(self.tc1(x))                           # (B, N, sp_ch, T)
        _, _, Cs, T2 = h.shape

        # 展成 (BT, N, Cs) 送入空间卷积
        h = h.permute(0, 3, 1, 2).contiguous()             # (B, T, N, Cs)
        h_flat = h.view(B * T2 * N, Cs)

        h_flat = self.mgc(h_flat, B * T2, N, edge_indices, edge_weights)
        h = self.act(h_flat.view(B, T2, N, Cs))
        h = h.permute(0, 2, 3, 1).contiguous()             # (B, N, Cs, T)

        h = self.act(self.tc2(h))                           # (B, N, out_ch, T)
        h = self.norm(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return h


# ── 完整多图 STGCN ──────────────────────────────────────────────────────

class MultiGraphSTGCN(nn.Module):
    """
    四图门控融合 STGCN（方案C 主模型）。

    Args
    ----
    T_in      : 输入时间步数
    F_in      : 输入特征维度（2: inflow+outflow）
    T_out     : 预测步数
    n_ch      : (spatial_ch, out_ch) 两个 ST-Conv Block 的通道数
    K         : Chebyshev 阶数
    graph_names: 使用的图名称列表，决定 n_graphs
    """

    GRAPH_NAMES = ["spatial", "poi", "flow", "transit"]

    def __init__(self, T_in: int = 12, F_in: int = 2, T_out: int = 12,
                 n_ch: tuple[int, int] = (64, 16), K: int = 3,
                 graph_names: list[str] | None = None):
        super().__init__()
        self.graph_names = graph_names or self.GRAPH_NAMES
        n_graphs = len(self.graph_names)
        sp_ch, out_ch = n_ch
        self.block1 = MultiGraphSTConvBlock(F_in,    sp_ch,  out_ch, K=K, n_graphs=n_graphs)
        self.block2 = MultiGraphSTConvBlock(out_ch,  sp_ch,  out_ch, K=K, n_graphs=n_graphs)
        self.output = OutputModule(out_ch, T_out)

    def _extract_graphs(self, graphs: dict) -> tuple[list, list]:
        """从 graphs dict 中提取边索引和边权重列表。"""
        edge_indices = [graphs[k]["edge_index"] for k in self.graph_names]
        edge_weights = [graphs[k].get("edge_weight") for k in self.graph_names]
        return edge_indices, edge_weights

    def forward(self, x: torch.Tensor, graphs: dict | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (B, N, T_in, F_in)
        graphs : dict，键为图名，值为含 edge_index/edge_weight 的 dict

        Returns
        -------
        (B, N, T_out)
        """
        if graphs is None:
            raise ValueError("MultiGraphSTGCN 需要传入 graphs 参数")
        edge_indices, edge_weights = self._extract_graphs(graphs)

        h = x.permute(0, 1, 3, 2)                          # (B, N, F_in, T_in)
        h = self.block1(h, edge_indices, edge_weights)
        h = self.block2(h, edge_indices, edge_weights)
        return self.output(h)                               # (B, N, T_out)


# ── 单图 STGCN 包装器（接口统一）──────────────────────────────────────

class SingleGraphSTGCN(nn.Module):
    """
    复用 src.models.stgcn.STGCN，包装为 forward(x, graphs) 接口，
    仅使用 graphs["spatial"]。
    """

    def __init__(self, T_in: int = 12, F_in: int = 2, T_out: int = 12,
                 n_ch: tuple[int, int] = (64, 16), K: int = 3):
        super().__init__()
        from .stgcn import STGCN
        self._model = STGCN(T_in=T_in, F_in=F_in, T_out=T_out, n_ch=n_ch, K=K)

    def forward(self, x: torch.Tensor, graphs: dict | None = None) -> torch.Tensor:
        if graphs is None:
            raise ValueError("SingleGraphSTGCN 需要传入 graphs['spatial']")
        g = graphs["spatial"]
        return self._model(x, g["edge_index"], g.get("edge_weight"))
