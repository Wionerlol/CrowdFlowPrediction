"""
节点共享 LSTM 基线模型。

所有节点共用同一套 LSTM 权重，节点维度展入 batch 维：
  (B, N, T_in, F) → (B*N, T_in, F) → LSTM → Linear → (B, N, T_out)

这是纯时序基线，不引入任何空间信息，用于与 STGCN 对比空间建模的收益。
"""
import torch
import torch.nn as nn


class NodeLSTM(nn.Module):
    """
    Args
    ----
    T_in      : 输入时间步数
    F_in      : 每时步特征维度（2: inflow+outflow）
    T_out     : 预测步数
    hidden    : LSTM 隐藏维度
    num_layers: LSTM 层数
    dropout   : dropout（仅在 num_layers>1 时生效）
    """

    def __init__(self, T_in: int = 12, F_in: int = 2, T_out: int = 12,
                 hidden: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
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
        graphs : 忽略（保持与其他模型接口一致）

        Returns
        -------
        (B, N, T_out)
        """
        B, N, T, F = x.shape
        h = x.view(B * N, T, F)               # (B*N, T_in, F)
        out, _ = self.lstm(h)                  # (B*N, T_in, hidden)
        h_last = out[:, -1, :]                 # (B*N, hidden)
        pred = self.fc(h_last)                 # (B*N, T_out)
        return pred.view(B, N, -1)             # (B, N, T_out)
