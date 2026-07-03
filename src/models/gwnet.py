"""
Graph WaveNet: Deep Learning on Graphs for Spatio-Temporal Graph Modeling
Wu et al., IJCAI 2019

关键改进（相比 STGCN）：
1. 膨胀因果时序卷积 —— RF 从 9 步扩至 64 步
2. 自适应邻接矩阵 —— 可学习节点嵌入推导隐式空间关系
3. Skip 连接 —— 6 个 block 输出直接汇入预测层

实现注意：
  空间卷积全部改用预计算稠密邻接矩阵 + einsum，
  彻底避免 ChebConv 的 BT×E 边索引复制，速度提升 3-5×。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _sym_norm_adj(edge_index: torch.Tensor,
                  edge_weight,
                  N: int) -> torch.Tensor:
    """
    对称归一化邻接矩阵 D^{-1/2}(A+I)D^{-1/2}，返回稠密 (N, N)。
    只需在 forward 第一次调用时计算一次，之后作为 buffer 缓存。
    """
    device = edge_index.device
    # 加自环
    sl = torch.arange(N, device=device).unsqueeze(0).expand(2, -1)
    ei = torch.cat([edge_index, sl], dim=1)

    ew = torch.ones(ei.shape[1], device=device)
    if edge_weight is not None:
        ew[:edge_index.shape[1]] = edge_weight.to(device)

    A = torch.zeros(N, N, device=device)
    A[ei[0], ei[1]] = ew
    D_inv_sqrt = A.sum(dim=1).clamp(min=1e-6).rsqrt()
    return D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)   # (N, N)


# ── 基础模块 ─────────────────────────────────────────────────────────────────

class DilatedCausalConv(nn.Module):
    """WaveNet 风格的门控膨胀因果卷积。"""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch * 2, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.causal_pad, 0))
        h1, h2 = self.conv(x).chunk(2, dim=1)
        return torch.tanh(h1) * torch.sigmoid(h2)


class DenseGCN(nn.Module):
    """
    单层稠密图卷积：out = A_hat @ h @ W
    A_hat (N,N) 预计算后传入，无 edge_index 批量复制。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.W = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, h: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        # h: (BT, N, C)   A_hat: (N, N)
        return self.W(torch.matmul(A_hat.unsqueeze(0), h))   # (BT, N, out_ch)


# ── 单图 WaveBlock ────────────────────────────────────────────────────────────

class WaveBlock(nn.Module):
    """单图版 WaveNet 块：膨胀因果卷积 → 预定义图 + 自适应图 → Skip + Residual。"""

    def __init__(self, residual_ch: int, skip_ch: int,
                 kernel_size: int = 2, dilation: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.temporal  = DilatedCausalConv(residual_ch, residual_ch, kernel_size, dilation)
        self.gc_pre    = DenseGCN(residual_ch, residual_ch)
        self.gc_ada    = nn.Linear(residual_ch, residual_ch, bias=False)
        self.skip_proj = nn.Linear(residual_ch, skip_ch)
        self.res_proj  = nn.Linear(residual_ch, residual_ch)
        self.norm      = nn.LayerNorm(residual_ch)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                A_hat: torch.Tensor,
                A_ada: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C, T)
        B, N, C, T = x.shape

        h = self.drop(self.temporal(x.view(B * N, C, T)))
        h = h.view(B, N, C, T).permute(0, 3, 1, 2).reshape(B * T, N, C)

        h = F.relu(self.gc_pre(h, A_hat) + self.gc_ada(torch.matmul(A_ada.unsqueeze(0), h)))
        h = h.view(B, T, N, C).permute(0, 2, 3, 1)   # (B, N, C, T)

        skip  = self.skip_proj(h.permute(0, 1, 3, 2))
        h_out = self.norm(self.res_proj(h.permute(0, 1, 3, 2)) + x.permute(0, 1, 3, 2))
        return h_out.permute(0, 1, 3, 2), skip


# ── 多图门控 WaveBlock ────────────────────────────────────────────────────────

class MultiGraphWaveBlock(nn.Module):
    """
    多图门控版 WaveNet 块。
    各预定义图各跑一路 DenseGCN + 自适应图，softmax 门控融合。
    """

    def __init__(self, residual_ch: int, skip_ch: int,
                 graph_names: list[str],
                 kernel_size: int = 2, dilation: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.graph_names  = graph_names
        num_branches      = len(graph_names) + 1   # 各预定义图 + 自适应图

        self.temporal  = DilatedCausalConv(residual_ch, residual_ch, kernel_size, dilation)
        self.gc_preds  = nn.ModuleDict({
            name: DenseGCN(residual_ch, residual_ch)
            for name in graph_names
        })
        self.gc_ada    = nn.Linear(residual_ch, residual_ch, bias=False)
        self.gate      = nn.Linear(residual_ch, num_branches)
        self.skip_proj = nn.Linear(residual_ch, skip_ch)
        self.res_proj  = nn.Linear(residual_ch, residual_ch)
        self.norm      = nn.LayerNorm(residual_ch)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                A_hats: dict[str, torch.Tensor],
                A_ada: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C, T = x.shape

        h = self.drop(self.temporal(x.view(B * N, C, T)))
        h = h.view(B, N, C, T).permute(0, 3, 1, 2).reshape(B * T, N, C)

        branches = [self.gc_preds[name](h, A_hats[name]) for name in self.graph_names]
        branches.append(self.gc_ada(torch.matmul(A_ada.unsqueeze(0), h)))

        gate    = torch.softmax(self.gate(h), dim=-1)          # (BT, N, num_branches)
        stacked = torch.stack(branches, dim=2)                 # (BT, N, num_branches, C)
        h = F.relu((gate.unsqueeze(-1) * stacked).sum(dim=2))  # (BT, N, C)
        h = h.view(B, T, N, C).permute(0, 2, 3, 1)            # (B, N, C, T)

        skip  = self.skip_proj(h.permute(0, 1, 3, 2))
        h_out = self.norm(self.res_proj(h.permute(0, 1, 3, 2)) + x.permute(0, 1, 3, 2))
        return h_out.permute(0, 1, 3, 2), skip


# ── 完整模型：单图 ───────────────────────────────────────────────────────────

class GraphWaveNet(nn.Module):
    """
    Graph WaveNet（单图版），接口：forward(x, graphs) → (B, N, T_out)

    空间图用 DenseGCN（预计算稠密邻接矩阵），自适应图用 E1@E2.T。
    dilations=[1,2,4,8,16,32]，RF=64，完整覆盖 T_in=48。
    """

    def __init__(self, N: int, T_in: int = 48, T_out: int = 48, F_in: int = 2,
                 residual_ch: int = 32, skip_ch: int = 64,
                 dilations: list[int] | None = None,
                 kernel_size: int = 2,
                 embed_dim: int = 10, dropout: float = 0.3):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        rf = 1 + (kernel_size - 1) * sum(dilations)
        print(f"[GraphWaveNet] N={N}  T_in={T_in}  T_out={T_out}  "
              f"RF={rf}  blocks={len(dilations)}")

        self.N = N
        self.E1 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)
        self.E2 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)
        self.input_proj = nn.Linear(F_in, residual_ch)
        self.blocks = nn.ModuleList([
            WaveBlock(residual_ch, skip_ch, kernel_size, d, dropout)
            for d in dilations
        ])
        self.fc1 = nn.Linear(skip_ch, skip_ch)
        self.fc2 = nn.Linear(skip_ch, T_out)
        # 缓存预计算的稠密邻接矩阵（第一次 forward 时填充）
        self.register_buffer("_A_hat", None)

    @property
    def adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)

    def _get_A_hat(self, graphs: dict) -> torch.Tensor:
        if self._A_hat is None:
            g = graphs['spatial']
            A = _sym_norm_adj(g['edge_index'], g.get('edge_weight'), self.N)
            self._A_hat = A
        return self._A_hat

    def forward(self, x: torch.Tensor, graphs: dict | None = None) -> torch.Tensor:
        if graphs is None or 'spatial' not in graphs:
            raise ValueError("GraphWaveNet 需要 graphs['spatial']")

        A_hat = self._get_A_hat(graphs)
        A_ada = self.adaptive_adj

        h = self.input_proj(x).permute(0, 1, 3, 2)   # (B, N, C, T)
        skip_sum = None
        for block in self.blocks:
            h, skip = block(h, A_hat, A_ada)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        out = F.relu(self.fc1(skip_sum[:, :, -1, :]))
        return self.fc2(out)   # (B, N, T_out)


# ── 完整模型：多图门控 ───────────────────────────────────────────────────────

class MultiGraphWaveNet(nn.Module):
    """
    多图门控 Graph WaveNet，接口：forward(x, graphs) → (B, N, T_out)

    各预定义图均预计算稠密邻接矩阵并缓存，空间卷积全部用 DenseGCN。
    """

    def __init__(self, N: int, T_in: int = 48, T_out: int = 48, F_in: int = 2,
                 residual_ch: int = 32, skip_ch: int = 64,
                 graph_names: list[str] | None = None,
                 dilations: list[int] | None = None,
                 kernel_size: int = 2,
                 embed_dim: int = 10, dropout: float = 0.3):
        super().__init__()
        if graph_names is None:
            graph_names = ["spatial", "poi", "flow", "transit"]
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        rf = 1 + (kernel_size - 1) * sum(dilations)
        print(f"[MultiGraphWaveNet] N={N}  T_in={T_in}  T_out={T_out}  "
              f"RF={rf}  blocks={len(dilations)}  graphs={graph_names}")

        self.N           = N
        self.graph_names = graph_names
        self.E1 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)
        self.E2 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)
        self.input_proj = nn.Linear(F_in, residual_ch)
        self.blocks = nn.ModuleList([
            MultiGraphWaveBlock(residual_ch, skip_ch, graph_names,
                                kernel_size, d, dropout)
            for d in dilations
        ])
        self.fc1 = nn.Linear(skip_ch, skip_ch)
        self.fc2 = nn.Linear(skip_ch, T_out)
        # 各图的稠密邻接矩阵缓存（第一次 forward 时填充）
        self._A_hats: dict[str, torch.Tensor] = {}

    @property
    def adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)

    def _get_A_hats(self, graphs: dict) -> dict[str, torch.Tensor]:
        if not self._A_hats:
            for name in self.graph_names:
                g = graphs[name]
                self._A_hats[name] = _sym_norm_adj(
                    g['edge_index'], g.get('edge_weight'), self.N
                )
        return self._A_hats

    def forward(self, x: torch.Tensor, graphs: dict | None = None) -> torch.Tensor:
        if graphs is None:
            raise ValueError("MultiGraphWaveNet 需要 graphs dict")

        A_hats = self._get_A_hats(graphs)
        A_ada  = self.adaptive_adj

        h = self.input_proj(x).permute(0, 1, 3, 2)   # (B, N, C, T)
        skip_sum = None
        for block in self.blocks:
            h, skip = block(h, A_hats, A_ada)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        out = F.relu(self.fc1(skip_sum[:, :, -1, :]))
        return self.fc2(out)   # (B, N, T_out)
