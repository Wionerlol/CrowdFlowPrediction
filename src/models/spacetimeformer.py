"""
Spacetimeformer-lite for large urban grids.

The original Spacetimeformer flattens every (node, time) pair into a token and
applies attention over the full token sequence. For TaxiBJ with N=1024 and
T_in=48, that would create 49,152 tokens per sample and make full attention
impractical on a 16 GB laptop GPU.

This variant keeps the core idea:
  1. Flatten (node, time) into tokens.
  2. Use local self-attention on contiguous token chunks.
  3. Use a small set of learnable global tokens to exchange long-range
     spatio-temporal information across all chunks.

Token order is node-major so local chunks mostly see full histories for nearby
nodes in the flattened sequence, while global tokens carry cross-node context.
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


class LocalGlobalBlock(nn.Module):
    """
    One flattened-token block:
      local self-attention inside fixed-size chunks
      global tokens attend to all data tokens
      data tokens attend back to updated global tokens
      token/global FFN
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 chunk_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.chunk_size = chunk_size

        self.local_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.global_read = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.global_write = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.norm_local = nn.LayerNorm(d_model)
        self.norm_g1 = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ff_t = nn.LayerNorm(d_model)
        self.norm_ff_g = nn.LayerNorm(d_model)

        self.ff_tokens = _FeedForward(d_model, d_ff, dropout)
        self.ff_globals = _FeedForward(d_model, d_ff, dropout)

    def _local_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L, D)
        returns: (B, L, D)
        """
        B, L, D = tokens.shape
        chunk = self.chunk_size
        pad = (chunk - (L % chunk)) % chunk
        if pad:
            pad_tokens = F.pad(tokens, (0, 0, 0, pad))
        else:
            pad_tokens = tokens

        L_pad = pad_tokens.shape[1]
        n_chunks = L_pad // chunk

        x = pad_tokens.view(B, n_chunks, chunk, D).reshape(B * n_chunks, chunk, D)
        attn_out, _ = self.local_attn(x, x, x, need_weights=False)
        x = attn_out.view(B, n_chunks, chunk, D).reshape(B, L_pad, D)
        if pad:
            x = x[:, :L]
        return x

    def forward(self,
                tokens: torch.Tensor,
                globals_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_out = self._local_attention(tokens)
        tokens = self.norm_local(tokens + local_out)

        # Global tokens summarize the entire flattened sequence.
        g_read, _ = self.global_read(globals_, tokens, tokens, need_weights=False)
        globals_ = self.norm_g1(globals_ + g_read)

        # Data tokens pull long-range context back from global tokens.
        g_write, _ = self.global_write(tokens, globals_, globals_, need_weights=False)
        tokens = self.norm_cross(tokens + g_write)

        tokens = self.norm_ff_t(tokens + self.ff_tokens(tokens))
        globals_ = self.norm_ff_g(globals_ + self.ff_globals(globals_))
        return tokens, globals_


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
                 dropout: float = 0.1):
        super().__init__()
        self.N = N
        self.T_in = T_in
        self.T_out = T_out

        self.input_proj = nn.Linear(F_in, d_model)
        self.node_emb = nn.Embedding(N, d_model)
        self.step_emb = nn.Embedding(T_in, d_model)
        self.drop = nn.Dropout(dropout)

        self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            LocalGlobalBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                chunk_size=chunk_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, T_out)

        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(
            f"[Spacetimeformer] N={N}  T_in={T_in}  T_out={T_out}  "
            f"d={d_model}  heads={n_heads}  layers={num_layers}  "
            f"chunk={chunk_size}  globals={num_global_tokens}  "
            f"params={params:.3f}M"
        )

    def forward(self,
                x: torch.Tensor,
                graphs: dict | None = None) -> torch.Tensor:
        """
        x: (B, N, T_in, F_in)
        returns: (B, N, T_out)
        """
        B, N, T, _ = x.shape
        if N != self.N:
            raise ValueError(f"Spacetimeformer expects N={self.N}, got {N}")
        if T != self.T_in:
            raise ValueError(f"Spacetimeformer expects T_in={self.T_in}, got {T}")

        node_ids = torch.arange(N, device=x.device).view(1, N, 1)
        step_ids = torch.arange(T, device=x.device).view(1, 1, T)

        h = self.input_proj(x)
        h = h + self.node_emb(node_ids) + self.step_emb(step_ids)
        h = self.drop(h)

        # Node-major flatten: token order is (node_1, t_1..t_T), then node_2, ...
        tokens = h.reshape(B, N * T, -1)
        globals_ = self.global_tokens.unsqueeze(0).expand(B, -1, -1)

        for block in self.blocks:
            tokens, globals_ = block(tokens, globals_)

        h = self.out_norm(tokens.view(B, N, T, -1))
        out = F.gelu(self.fc1(h[:, :, -1, :]))
        return self.fc2(out)
