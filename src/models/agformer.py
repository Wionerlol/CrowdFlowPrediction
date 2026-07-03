"""
AGFormer: Adaptive Graph Transformer for Spatio-Temporal Prediction

核心改进（相比 STGCN）：
1. 时序自注意力替换时序卷积 —— 无感受野限制，T_in 内任意位置均可关联
2. 自适应图卷积 —— 预定义图(DenseGCN) + 可学习邻接矩阵，与 GWNet 相同策略
3. 位置编码 —— 正弦 PE 让注意力感知时间步位置
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gwnet import _sym_norm_adj, DenseGCN


# ── 位置编码 ─────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """固定正弦位置编码，加到时序嵌入上。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, :x.shape[1]]


# ── AGFormer Block ────────────────────────────────────────────────────────────

class AGFormerBlock(nn.Module):
    """
    单个 AGFormer 块：
      时序自注意力 → Adaptive GCN → FFN，每步都有残差 + LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # 时序自注意力
        self.attn   = nn.MultiheadAttention(d_model, n_heads,
                                            dropout=dropout, batch_first=True)
        self.norm1  = nn.LayerNorm(d_model)

        # 空间图卷积：预定义图 + 自适应图
        self.gc_pre = DenseGCN(d_model, d_model)
        self.gc_ada = nn.Linear(d_model, d_model, bias=False)
        self.norm2  = nn.LayerNorm(d_model)

        # FFN
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                A_hat: torch.Tensor,
                A_ada: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, N, T, d_model)
        A_hat : (N, N) 预计算稠密归一化邻接
        A_ada : (N, N) 自适应邻接
        """
        B, N, T, d = x.shape

        # ── 1. 时序自注意力（节点独立）────────────────────────────────
        h = x.view(B * N, T, d)
        attn_out, _ = self.attn(h, h, h)          # (BN, T, d)
        h = self.norm1(h + self.drop(attn_out))
        h = h.view(B, N, T, d)

        # ── 2. 自适应图卷积（时刻独立）───────────────────────────────
        # 展平为 (BT, N, d) 做图操作
        g = h.permute(0, 2, 1, 3).reshape(B * T, N, d)
        g_pre = self.gc_pre(g, A_hat)
        g_ada = self.gc_ada(torch.matmul(A_ada.unsqueeze(0), g))
        g = F.relu(g_pre + g_ada)
        g = g.view(B, T, N, d).permute(0, 2, 1, 3)   # (B, N, T, d)
        h = self.norm2(h + g)

        # ── 3. FFN ────────────────────────────────────────────────────
        h = self.norm3(h + self.ffn(h))
        return h


# ── 完整 AGFormer ─────────────────────────────────────────────────────────────

class AGFormer(nn.Module):
    """
    AGFormer 完整模型，接口：forward(x, graphs) → (B, N, T_out)

    时序：MultiheadAttention（全局感受野，不受卷积核限制）
    空间：预定义图 DenseGCN + 可学习自适应邻接矩阵
    """

    def __init__(self,
                 N: int,
                 T_in: int = 48,
                 T_out: int = 48,
                 F_in: int = 2,
                 d_model: int = 64,
                 n_heads: int = 4,
                 d_ff: int = 128,
                 num_layers: int = 3,
                 embed_dim: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.N     = N
        self.T_out = T_out

        # 自适应邻接矩阵嵌入
        self.E1 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)
        self.E2 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)

        self.input_proj = nn.Linear(F_in, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=T_in + 1)
        self.drop       = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            AGFormerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 取最后时刻 → 预测 T_out 步
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, T_out)

        # 稠密邻接矩阵缓存（第一次 forward 时计算）
        self.register_buffer("_A_hat", None)

        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[AGFormer] N={N}  T_in={T_in}  T_out={T_out}  "
              f"d={d_model}  heads={n_heads}  layers={num_layers}  "
              f"params={params:.3f}M")

    @property
    def adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)

    def _get_A_hat(self, graphs: dict) -> torch.Tensor:
        if self._A_hat is None:
            g = graphs['spatial']
            self._A_hat = _sym_norm_adj(g['edge_index'], g.get('edge_weight'), self.N)
        return self._A_hat

    def forward(self, x: torch.Tensor,
                graphs: dict | None = None) -> torch.Tensor:
        """
        x      : (B, N, T_in, F_in)
        graphs : 须含 'spatial' 键
        """
        if graphs is None or 'spatial' not in graphs:
            raise ValueError("AGFormer 需要 graphs['spatial']")

        A_hat = self._get_A_hat(graphs)
        A_ada = self.adaptive_adj

        # 输入投影 + 位置编码
        B, N, T, _ = x.shape
        h = self.drop(self.input_proj(x))           # (B, N, T, d_model)

        # 位置编码作用在时序维度
        h = h.view(B * N, T, -1)
        h = self.pos_enc(h)
        h = h.view(B, N, T, -1)

        # 逐 block 前向
        for block in self.blocks:
            h = block(h, A_hat, A_ada)              # (B, N, T, d_model)

        # 取最后时刻预测
        out = F.gelu(self.fc1(h[:, :, -1, :]))     # (B, N, d_model)
        return self.fc2(out)                         # (B, N, T_out)
