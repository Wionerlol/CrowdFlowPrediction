"""
Spacetimeformer for large urban grids.

The original Spacetimeformer represents every (variable, time) pair as its own
token, then alternates:
  1. local attention over the timesteps of the same variable;
  2. global spatiotemporal attention over all (variable, time) tokens.

Full quadratic global attention over TaxiBJ's 1024 * 48 tokens is not practical
on a laptop GPU, so this implementation uses multi-head linear attention for
the global layer. This keeps the original local/global inductive bias without
using predefined graph edges.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearGlobalAttention(nn.Module):
    """Multi-head linear self-attention for long flattened spatiotemporal tokens."""

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D), where L = N * T.
        """
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, L, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Positive feature map used by common linear-attention variants.
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        kv = torch.einsum("bhld,bhle->bhde", k, v)
        k_sum = k.sum(dim=2)
        z = 1.0 / (torch.einsum("bhld,bhd->bhl", q, k_sum) + 1e-6)
        out = torch.einsum("bhld,bhde,bhl->bhle", q, kv, z)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.drop(self.out(out))


class LinearCrossAttention(nn.Module):
    """Multi-head linear cross-attention from target tokens to encoder tokens."""

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        query : (B, Lq, D)
        memory: (B, Lm, D)
        """
        B, Lq, D = query.shape
        Lm = memory.shape[1]
        q = self.q_proj(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, Lm, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, Lm, self.n_heads, self.head_dim).transpose(1, 2)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("bhld,bhle->bhde", k, v)
        k_sum = k.sum(dim=2)
        z = 1.0 / (torch.einsum("bhld,bhd->bhl", q, k_sum) + 1e-6)
        out = torch.einsum("bhld,bhde,bhl->bhle", q, kv, z)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.drop(self.out(out))


class EncoderBlock(nn.Module):
    """
    One Spacetimeformer-style block:
      local self-attention over each node's T-step temporal sequence;
      efficient global self-attention over all N*T spatiotemporal tokens;
      token FFN.
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()

        self.local_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.global_attn = LinearGlobalAttention(d_model, n_heads, dropout)

        self.norm_local = nn.LayerNorm(d_model)
        self.norm_global = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)

        self.ff_tokens = _FeedForward(d_model, d_ff, dropout)

    def forward(self,
                h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, N, T, D)
        """
        B, N, T, D = h.shape

        local_in = h.reshape(B * N, T, D)
        local_out, _ = self.local_attn(local_in, local_in, local_in, need_weights=False)
        h = self.norm_local((local_in + local_out).view(B, N, T, D))

        tokens = h.reshape(B, N * T, D)
        global_out = self.global_attn(tokens)
        tokens = self.norm_global(tokens + global_out)
        tokens = self.norm_ff(tokens + self.ff_tokens(tokens))

        return tokens.view(B, N, T, D)


class DecoderBlock(nn.Module):
    """
    Spacetimeformer-style decoder block:
      local self-attention over each node's target sequence;
      efficient cross-attention from target tokens to encoder tokens;
      target FFN.
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = LinearCrossAttention(d_model, n_heads, dropout)
        self.norm_local = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff_tokens = _FeedForward(d_model, d_ff, dropout)

    def forward(self,
                target: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        """
        target: (B, N, T_out, D)
        memory: (B, N, T_in, D)
        """
        B, N, T, D = target.shape
        local_in = target.reshape(B * N, T, D)
        local_out, _ = self.local_attn(local_in, local_in, local_in, need_weights=False)
        target = self.norm_local((local_in + local_out).view(B, N, T, D))

        target_tokens = target.reshape(B, N * T, D)
        memory_tokens = memory.reshape(B, memory.shape[1] * memory.shape[2], D)
        cross_out = self.cross_attn(target_tokens, memory_tokens)
        target_tokens = self.norm_cross(target_tokens + cross_out)
        target_tokens = self.norm_ff(target_tokens + self.ff_tokens(target_tokens))
        return target_tokens.view(B, N, T, D)


class Spacetimeformer(nn.Module):
    """
    Flattened-token Spacetimeformer approximation for TaxiBJ/TaxiNYC.

    Interface:
        forward(x, graphs=None) -> (B, N, T_out)
    """

    def __init__(self,
                 N: int,
                 T_in: int = 48,
                 T_out: int = 48,
                 F_in: int = 2,
                 d_model: int = 16,
                 n_heads: int = 2,
                 d_ff: int = 32,
                 num_layers: int = 2,
                 chunk_size: int = 256,
                 num_global_tokens: int = 8,
                 static_features: torch.Tensor | None = None,
                 flow_dim: int = 2,
                 time_dim: int | None = None,
                 future_time_dim: int = 0,
                 separate_value_time: bool = True,
                 use_given_embedding: bool = True,
                 use_linear_output: bool = False,
                 context_fusion: str = "add",
                 dropout: float = 0.1):
        super().__init__()
        self.N = N
        self.T_in = T_in
        self.T_out = T_out
        self.flow_dim = flow_dim
        self.time_dim = max(0, F_in - flow_dim) if time_dim is None else time_dim
        self.future_time_dim = future_time_dim
        self.separate_value_time = separate_value_time
        self.use_linear_output = use_linear_output
        self.context_fusion = context_fusion
        if context_fusion not in {"add", "gated"}:
            raise ValueError("context_fusion must be 'add' or 'gated'")
        if flow_dim + self.time_dim != F_in:
            raise ValueError(
                f"F_in={F_in} must equal flow_dim={flow_dim} + "
                f"time_dim={self.time_dim}"
            )

        self.input_proj = None
        self.flow_proj = None
        self.time_proj = None
        if separate_value_time:
            self.flow_proj = nn.Linear(flow_dim, d_model)
            if self.time_dim:
                self.time_proj = nn.Sequential(
                    nn.LayerNorm(self.time_dim),
                    nn.Linear(self.time_dim, d_model),
                )
        else:
            self.input_proj = nn.Linear(F_in, d_model)
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
        self.context_merge = None
        self.context_gate = None
        n_contexts = int(self.time_dim > 0) + int(static_features is not None)
        if context_fusion == "gated" and n_contexts:
            self.context_merge = nn.Linear(n_contexts * d_model, d_model)
            self.context_gate = nn.Linear(2 * d_model, d_model)
        self.node_emb = nn.Embedding(N, d_model)
        self.encoder_step_emb = nn.Embedding(T_in, d_model)
        self.decoder_step_emb = nn.Embedding(T_out, d_model)
        self.given_emb = nn.Embedding(2, d_model) if use_given_embedding else None
        self.target_value_emb = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.future_time_proj = None
        if future_time_dim:
            self.future_time_proj = nn.Sequential(
                nn.LayerNorm(future_time_dim),
                nn.Linear(future_time_dim, d_model),
            )
        self.drop = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        self.linear_head = nn.Linear(T_in, T_out) if use_linear_output else None

        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(
            f"[Spacetimeformer] N={N}  T_in={T_in}  T_out={T_out}  "
            f"d={d_model}  heads={n_heads}  layers={num_layers}  "
            f"encdec=True  local=temporal  global=linear  "
            f"params={params:.3f}M"
        )

    def forward(self,
                x: torch.Tensor,
                graphs: dict | None = None) -> torch.Tensor:
        """
        x: (B, N, T_in, F_in)
        returns: (B, N, T_out)
        """
        future_time = None
        if isinstance(x, (tuple, list)):
            x, future_time = x

        B, N, T, _ = x.shape
        if N != self.N:
            raise ValueError(f"Spacetimeformer expects N={self.N}, got {N}")
        if T != self.T_in:
            raise ValueError(f"Spacetimeformer expects T_in={self.T_in}, got {T}")
        if self.future_time_proj is not None:
            if future_time is None:
                raise ValueError("future_time is required when future_time_dim > 0")
            if future_time.shape[1:] != (self.T_out, self.future_time_dim):
                raise ValueError(
                    f"future_time must have shape (B, {self.T_out}, "
                    f"{self.future_time_dim}), got {tuple(future_time.shape)}"
                )

        node_ids = torch.arange(N, device=x.device).view(1, N, 1)
        enc_step_ids = torch.arange(T, device=x.device).view(1, 1, T)

        if self.separate_value_time:
            h = self.flow_proj(x[..., :self.flow_dim])
            contexts = []
            if self.time_proj is not None:
                contexts.append(
                    self.time_proj(x[..., self.flow_dim:self.flow_dim + self.time_dim])
                )
            if self.static_proj is not None:
                static = self.static_proj(self.static_features).view(1, N, 1, -1)
                contexts.append(static.expand(B, -1, T, -1))
            if self.context_fusion == "gated" and contexts:
                context = self.context_merge(torch.cat(contexts, dim=-1))
                gate = torch.sigmoid(self.context_gate(torch.cat([h, context], dim=-1)))
                h = h + gate * context
            else:
                for context in contexts:
                    h = h + context
        else:
            h = self.input_proj(x)
        if self.static_proj is not None and not self.separate_value_time:
            h = h + self.static_proj(self.static_features).view(1, N, 1, -1)
        h = h + self.node_emb(node_ids) + self.encoder_step_emb(enc_step_ids)
        if self.given_emb is not None:
            observed_id = torch.ones((), dtype=torch.long, device=x.device)
            h = h + self.given_emb(observed_id).view(1, 1, 1, -1)
        h = self.drop(h)

        for block in self.encoder_blocks:
            h = block(h)
        memory = self.encoder_norm(h)

        dec_step_ids = torch.arange(self.T_out, device=x.device).view(1, 1, self.T_out)
        target = self.target_value_emb.expand(B, N, self.T_out, -1)
        target = target + self.node_emb(node_ids) + self.decoder_step_emb(dec_step_ids)
        if self.future_time_proj is not None:
            future = self.future_time_proj(future_time).unsqueeze(1)
            target = target + future
        if self.static_proj is not None:
            target = target + self.static_proj(self.static_features).view(1, N, 1, -1)
        if self.given_emb is not None:
            target_id = torch.zeros((), dtype=torch.long, device=x.device)
            target = target + self.given_emb(target_id).view(1, 1, 1, -1)
        target = self.drop(target)

        for block in self.decoder_blocks:
            target = block(target, memory)
        target = self.decoder_norm(target)
        out = self.out_proj(target).squeeze(-1)
        if self.linear_head is not None:
            out = out + self.linear_head(x[..., 0])
        return out
