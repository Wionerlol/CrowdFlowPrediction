"""
STGCN: Spatio-Temporal Graph Convolutional Networks
Yu et al., IJCAI 2018 — arxiv.org/abs/1709.04875

Architecture:
  Input → [ST-Conv Block] × 2 → Output Module → Prediction
  ST-Conv Block: Temporal-GLU → Spatial-ChebGCN → Temporal-GLU → LayerNorm
"""
import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv


class TemporalConv(nn.Module):
    """Gated temporal convolution (GLU) over the time axis."""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        # Conv1d along time: (B*N, in_ch, T) → (B*N, 2*out_ch, T)
        self.conv = nn.Conv1d(in_ch, 2 * out_ch, kernel_size,
                              padding=kernel_size // 2, bias=True)

    def forward(self, x):
        # x: (B, N, C, T)
        B, N, C, T = x.shape
        h = x.view(B * N, C, T)
        h = self.conv(h)                        # (B*N, 2*out_ch, T)
        p, q = h.chunk(2, dim=1)
        out = p * torch.sigmoid(q)              # GLU
        return out.view(B, N, -1, T)            # (B, N, out_ch, T)


class STConvBlock(nn.Module):
    """
    One Spatio-Temporal Convolutional Block:
      Temporal-GLU → Spatial ChebGCN (over all time steps) → Temporal-GLU → LayerNorm
    """
    def __init__(self, in_ch, sp_ch, out_ch, kernel_size=3, K=3):
        super().__init__()
        self.tc1    = TemporalConv(in_ch, sp_ch, kernel_size)
        self.gc     = ChebConv(sp_ch, sp_ch, K=K, normalization="sym")
        self.tc2    = TemporalConv(sp_ch, out_ch, kernel_size)
        self.norm   = nn.LayerNorm(out_ch)
        self.act    = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        # x: (B, N, C, T)
        B, N, _, T = x.shape

        # Temporal conv 1
        h = self.act(self.tc1(x))              # (B, N, sp_ch, T)
        _, _, Cs, T2 = h.shape

        # Spatial GCN — process each time step independently
        h = h.permute(0, 3, 1, 2).contiguous() # (B, T, N, Cs)
        h = h.view(B * T2, N, Cs)

        # PyG ChebConv expects (num_nodes, F); batch via loop over BT
        h_flat = h.view(B * T2 * N, Cs)
        # Build batch-aware edge_index: repeat edge_index for B*T graphs
        # For a fixed graph, shift node indices per sample
        ei_rep = edge_index.unsqueeze(0).expand(B * T2, -1, -1)  # (BT, 2, E)
        offsets = torch.arange(B * T2, device=x.device).view(-1, 1) * N
        ei_cat = (ei_rep + offsets.unsqueeze(-1)).view(2, -1)     # (2, BT*E)

        ew_cat = None
        if edge_weight is not None:
            ew_cat = edge_weight.repeat(B * T2)

        h_flat = self.gc(h_flat, ei_cat, ew_cat)  # (B*T*N, Cs)
        h = self.act(h_flat.view(B, T2, N, Cs))

        # Back to (B, N, Cs, T)
        h = h.permute(0, 2, 3, 1).contiguous()    # (B, N, Cs, T)

        # Temporal conv 2
        h = self.act(self.tc2(h))                  # (B, N, out_ch, T)

        # LayerNorm over channel dim
        h = self.norm(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return h


class OutputModule(nn.Module):
    """Temporal average pooling → linear prediction."""
    def __init__(self, in_ch, T_out):
        super().__init__()
        self.fc = nn.Linear(in_ch, T_out)

    def forward(self, x):
        # x: (B, N, C, T)
        h = x.mean(dim=-1)   # (B, N, C) — temporal average pool
        return self.fc(h)    # (B, N, T_out)


class STGCN(nn.Module):
    """
    Full STGCN model.

    Args:
        T_in:    number of input time steps (default 12)
        F_in:    input features per node (default 2: inflow+outflow)
        T_out:   prediction horizon (default 3 steps = 1.5h)
        n_ch:    [spatial_ch, out_ch] for the two ST-Conv blocks
        K:       Chebyshev order (default 3)
    """
    def __init__(self, T_in=12, F_in=2, T_out=3, n_ch=(64, 16), K=3):
        super().__init__()
        sp_ch, out_ch = n_ch
        self.block1 = STConvBlock(F_in,    sp_ch,  out_ch, kernel_size=3, K=K)
        self.block2 = STConvBlock(out_ch,  sp_ch,  out_ch, kernel_size=3, K=K)
        self.output = OutputModule(out_ch, T_out)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x:           (B, N, T_in, F_in)
        edge_index:  (2, E)
        Returns:     (B, N, T_out)
        """
        h = x.permute(0, 1, 3, 2)        # (B, N, F_in, T_in)
        h = self.block1(h, edge_index, edge_weight)
        h = self.block2(h, edge_index, edge_weight)
        return self.output(h)
