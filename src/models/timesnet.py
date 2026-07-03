"""
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
Wu et al., ICLR 2023

核心思想：
  用 FFT 检测输入序列的 top-k 主周期，将 1D 时序 reshape 成 2D 矩阵，
  再用 2D Inception Conv 同时捕捉周期内模式（列方向）和周期间趋势（行方向）。

对 TaxiBJ（30 min 间隔）：T_in=48 主周期 = 1 天 (p=48)、半天 (p=24) 等。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _detect_periods(x: torch.Tensor, top_k: int) -> tuple[list[int], torch.Tensor]:
    """
    FFT 检测主周期，在批次和通道维度取均值再找峰值。
    x: (B, T, C)
    returns: periods (list of int), weights (softmax over amplitudes)
    """
    T = x.shape[1]
    xf  = torch.fft.rfft(x, dim=1)               # (B, T//2+1, C)
    amp = xf.abs().mean(dim=(0, 2))               # (T//2+1,)  均值幅度
    amp[0] = 0                                    # 去掉 DC 分量
    k = min(top_k, amp.shape[0] - 1)
    _, top_idx = amp.topk(k)
    # 频率索引 f → 周期 T//f（f 从 1 开始，f=0 已清零）
    periods = [max(1, T // max(1, int(idx))) for idx in top_idx]
    weights = F.softmax(amp[top_idx].float(), dim=0)
    return periods, weights


class Inception2D(nn.Module):
    """
    双尺度 2D Inception：并行 3×3 和 5×5 Conv，捕捉不同空间粒度的模式。
    输入：(B, d_model, H, W)
    输出：(B, H*W, d_model)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        half = d_ff // 2
        self.conv3 = nn.Conv2d(d_model, half, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(d_model, half, kernel_size=5, padding=2)
        self.proj  = nn.Linear(d_ff, d_model)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([F.gelu(self.conv3(x)),
                       F.gelu(self.conv5(x))], dim=1)   # (B, d_ff, H, W)
        B, C, H, W = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, H*W, d_ff)
        return self.norm(self.proj(h))                   # (B, H*W, d_model)


class TimesBlock(nn.Module):
    """
    单个 TimesBlock：
      1. FFT 检测 top-k 主周期
      2. 每个周期：1D→2D reshape → Inception2D → reshape 回 1D
      3. 按 FFT 幅度加权求和 + 残差 + LayerNorm
    """

    def __init__(self, T_in: int, d_model: int, d_ff: int, top_k: int):
        super().__init__()
        self.T_in  = T_in
        self.top_k = top_k
        self.conv  = Inception2D(d_model, d_ff)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BN, T, d)
        BN, T, d = x.shape
        periods, weights = _detect_periods(x, self.top_k)

        out = torch.zeros_like(x)
        for p, w in zip(periods, weights):
            # 填充使 T 能被 p 整除
            pad = (p - T % p) % p
            x_p = F.pad(x, (0, 0, 0, pad))          # (BN, T+pad, d)
            H   = (T + pad) // p
            # 1D → 2D → Inception → 2D → 1D
            x2d = x_p.view(BN, H, p, d).permute(0, 3, 1, 2)   # (BN, d, H, p)
            h2d = self.conv(x2d)                                 # (BN, H*p, d)
            out = out + w * h2d[:, :T, :]            # 截断 padding

        return self.norm(x + out)


class TimesNet(nn.Module):
    """
    TimesNet（纯时序，无图卷积），接口：forward(x, graphs=None) → (B, N, T_out)

    每个节点独立建模（节点维度折叠进 batch），保持和项目其他模型相同的接口。
    """

    def __init__(self,
                 T_in: int = 48,
                 T_out: int = 48,
                 F_in: int = 2,
                 d_model: int = 64,
                 d_ff: int = 128,
                 num_layers: int = 2,
                 top_k: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out

        self.input_proj = nn.Linear(F_in, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(T_in, d_model, d_ff, top_k)
            for _ in range(num_layers)
        ])
        self.drop   = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, T_out)

        params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[TimesNet] T_in={T_in}  T_out={T_out}  "
              f"d={d_model}  layers={num_layers}  top_k={top_k}  "
              f"params={params:.3f}M")

    def forward(self, x: torch.Tensor,
                graphs: dict | None = None) -> torch.Tensor:
        """
        x: (B, N, T_in, F_in)
        returns: (B, N, T_out)
        """
        B, N, T, F = x.shape

        # 节点折叠进 batch：(B*N, T, F)
        h = x.view(B * N, T, F)
        h = self.drop(self.input_proj(h))   # (BN, T, d)

        for block in self.blocks:
            h = block(h)                    # (BN, T, d)

        # 取最后一个时刻输出预测（seq2one 风格）
        out = self.output(h[:, -1, :])      # (BN, T_out)
        return out.view(B, N, self.T_out)
