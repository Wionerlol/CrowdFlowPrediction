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
        # Attention matrices are only materialised for the Week7 XAI export.
        # The prediction/training hot path discards them, so asking PyTorch not
        # to build the O(B*N*T*T) tensor saves both latency and peak memory.
        self.capture_attention_weights = False

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
        attn_out, _ = self.attn(
            h, h, h, need_weights=self.capture_attention_weights
        )                                           # (BN, T, d)
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
                 graph_names: list[str] | None = None,
                 static_features: torch.Tensor | None = None,
                 context_fusion: str = "add",
                 flow_dim: int = 2,
                 time_dim: int | None = None,
                 weather_dim: int = 0,
                 future_time_dim: int = 0,
                 use_linear_output: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.N     = N
        self.T_out = T_out
        self.future_time_dim = future_time_dim
        self.use_linear_output = use_linear_output
        self.graph_names = graph_names or ["spatial"]
        if not self.graph_names:
            raise ValueError("graph_names must contain at least one graph")
        self.graph_logits = (nn.Parameter(torch.zeros(len(self.graph_names)))
                             if len(self.graph_names) > 1 else None)

        # 自适应邻接矩阵嵌入
        self.E1 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)
        self.E2 = nn.Parameter(torch.randn(N, embed_dim) * 0.01)

        self.context_fusion = context_fusion
        self.flow_dim = flow_dim
        self.weather_dim = weather_dim
        self.time_dim = (max(0, F_in - flow_dim - weather_dim)
                         if time_dim is None else time_dim)
        if flow_dim + self.time_dim + self.weather_dim != F_in:
            raise ValueError(
                f"F_in={F_in} must equal flow_dim={flow_dim} + "
                f"time_dim={self.time_dim} + "
                f"weather_dim={self.weather_dim}"
            )
        self.input_proj = nn.Linear(
            flow_dim if context_fusion == "gated" else F_in, d_model
        )
        self.time_proj = None
        if context_fusion == "gated" and self.time_dim:
            self.time_proj = nn.Sequential(
                nn.LayerNorm(self.time_dim),
                nn.Linear(self.time_dim, d_model),
            )
        self.weather_proj = None
        if context_fusion == "gated" and self.weather_dim:
            self.weather_proj = nn.Sequential(
                nn.LayerNorm(self.weather_dim),
                nn.Linear(self.weather_dim, d_model),
            )
        self.static_proj = None
        if static_features is not None:
            if static_features.ndim != 2 or static_features.shape[0] != N:
                raise ValueError(
                    f"static_features must have shape (N, F), got "
                    f"{tuple(static_features.shape)}"
                )
            self.register_buffer("static_features", static_features.float())
            self.static_proj = nn.Sequential(
                nn.LayerNorm(static_features.shape[1]),
                nn.Linear(static_features.shape[1], d_model),
            )
        else:
            self.register_buffer("static_features", None)
        n_contexts = (int(self.time_proj is not None) +
                      int(self.weather_proj is not None) +
                      int(self.static_proj is not None))
        self.context_merge = None
        self.context_gate = None
        if context_fusion == "gated" and n_contexts:
            self.context_merge = nn.Linear(n_contexts * d_model, d_model)
            self.context_gate = nn.Linear(2 * d_model, d_model)
        self.pos_enc    = SinusoidalPE(d_model, max_len=T_in + 1)
        self.drop       = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            AGFormerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 取最后时刻 → 预测 T_out 步
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, T_out)
        self.linear_head = nn.Linear(T_in, T_out) if use_linear_output else None
        self.future_time_proj = None
        self.future_gate = None
        self.future_out = None
        if future_time_dim:
            self.future_time_proj = nn.Sequential(
                nn.LayerNorm(future_time_dim),
                nn.Linear(future_time_dim, d_model),
            )
            self.future_gate = nn.Linear(2 * d_model, d_model)
            self.future_out = nn.Linear(d_model, 1)

        # 各预定义图的归一化稠密邻接缓存；融合权重保持可学习。
        for i in range(len(self.graph_names)):
            self.register_buffer(f"_A_hat_{i}", None)

        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[AGFormer] N={N}  T_in={T_in}  T_out={T_out}  "
              f"d={d_model}  heads={n_heads}  layers={num_layers}  "
              f"graphs={self.graph_names}  linear={use_linear_output}  "
              f"params={params:.3f}M")

    @property
    def adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)

    def _get_A_hat(self, graphs: dict) -> torch.Tensor:
        matrices = []
        for i, name in enumerate(self.graph_names):
            if name not in graphs:
                raise ValueError(f"AGFormer 缺少图结构 graphs['{name}']")
            cache_name = f"_A_hat_{i}"
            matrix = getattr(self, cache_name)
            if matrix is None:
                g = graphs[name]
                matrix = _sym_norm_adj(
                    g['edge_index'], g.get('edge_weight'), self.N
                )
                setattr(self, cache_name, matrix)
            matrices.append(matrix)
        if len(matrices) == 1:
            return matrices[0]
        weights = torch.softmax(self.graph_logits, dim=0)
        return sum(w * matrix for w, matrix in zip(weights, matrices))

    def forward(self, x: torch.Tensor,
                graphs: dict | None = None) -> torch.Tensor:
        """
        x      : (B, N, T_in, F_in)
        graphs : 须含 ``graph_names`` 指定的图
        """
        future_time = None
        if isinstance(x, (tuple, list)):
            x, future_time = x
        if graphs is None:
            raise ValueError("AGFormer 需要图结构")

        A_hat = self._get_A_hat(graphs)
        A_ada = self.adaptive_adj

        # 输入投影 + 位置编码
        B, N, T, _ = x.shape
        if self.context_fusion == "gated":
            h_flow = self.input_proj(x[..., :self.flow_dim])
            contexts = []
            if self.time_proj is not None:
                contexts.append(self.time_proj(
                    x[..., self.flow_dim:self.flow_dim + self.time_dim]
                ))
            if self.weather_proj is not None:
                start = self.flow_dim + self.time_dim
                contexts.append(self.weather_proj(x[..., start:start + self.weather_dim]))
            if self.static_proj is not None:
                static = self.static_proj(self.static_features).view(1, N, 1, -1)
                contexts.append(static.expand(B, -1, T, -1))
            if contexts:
                context = self.context_merge(torch.cat(contexts, dim=-1))
                gate = torch.sigmoid(self.context_gate(
                    torch.cat([h_flow, context], dim=-1)
                ))
                h = h_flow + gate * context
            else:
                h = h_flow
            h = self.drop(h)
        else:
            h = self.drop(self.input_proj(x))       # (B, N, T, d_model)
            if self.static_proj is not None:
                static = self.static_proj(self.static_features).view(1, N, 1, -1)
                h = h + static

        # 位置编码作用在时序维度
        h = h.view(B * N, T, -1)
        h = self.pos_enc(h)
        h = h.view(B, N, T, -1)

        # 逐 block 前向
        for block in self.blocks:
            h = block(h, A_hat, A_ada)              # (B, N, T, d_model)

        # 取最后时刻预测
        node_state = F.gelu(self.fc1(h[:, :, -1, :]))  # (B, N, d_model)
        if self.future_time_proj is not None:
            if future_time is None:
                raise ValueError("future_time context is required by this AGFormer")
            if future_time.shape[1:] != (self.T_out, self.future_time_dim):
                raise ValueError(
                    f"future_time must have shape (B, {self.T_out}, "
                    f"{self.future_time_dim}), got {tuple(future_time.shape)}"
                )
            future = self.future_time_proj(future_time)  # (B, T_out, d)
            node = node_state.unsqueeze(2).expand(-1, -1, self.T_out, -1)
            future = future.unsqueeze(1).expand(-1, N, -1, -1)
            gate = torch.sigmoid(self.future_gate(torch.cat([node, future], dim=-1)))
            decoded = node + gate * future
            out = self.future_out(F.gelu(decoded)).squeeze(-1)
        else:
            out = self.fc2(node_state)                    # (B, N, T_out)
        if self.linear_head is not None:
            out = out + self.linear_head(x[..., 0])
        return out
