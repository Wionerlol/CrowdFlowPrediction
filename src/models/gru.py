"""
节点共享 GRU 基线模型。

与 NodeLSTM 结构相同，将 LSTM 单元替换为 GRU，
用于与 LSTM 对比门控机制的差异。
"""
import torch
import torch.nn as nn


class NodeGRU(nn.Module):
    """
    Args
    ----
    T_in      : 输入时间步数
    F_in      : 每时步特征维度（2: inflow+outflow）
    T_out     : 预测步数（48 = 24小时）
    hidden    : GRU 隐藏维度
    num_layers: GRU 层数
    dropout   : dropout（num_layers>1 时生效）
    """

    def __init__(self, T_in: int = 12, F_in: int = 2, T_out: int = 48,
                 hidden: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=F_in,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, T_out)

    def forward(self, x: torch.Tensor, graphs: dict | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (B, N, T_in, F_in)
        graphs : 忽略

        Returns
        -------
        (B, N, T_out)
        """
        B, N, T, F = x.shape
        h = x.view(B * N, T, F)
        out, _ = self.gru(h)                   # (B*N, T_in, hidden)
        h_last = out[:, -1, :]                 # (B*N, hidden)
        pred = self.fc(h_last)                 # (B*N, T_out)
        return pred.view(B, N, -1)             # (B, N, T_out)
