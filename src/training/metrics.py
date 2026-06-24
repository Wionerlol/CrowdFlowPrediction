"""
时空预测评估指标：MAE / RMSE / MAPE，均支持 null_val 掩码。
"""
import torch


def masked_mae(pred: torch.Tensor, true: torch.Tensor,
               null_val: float = 0.0) -> torch.Tensor:
    mask = true.abs() > null_val
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return (pred[mask] - true[mask]).abs().mean()


def masked_rmse(pred: torch.Tensor, true: torch.Tensor,
                null_val: float = 0.0) -> torch.Tensor:
    mask = true.abs() > null_val
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return ((pred[mask] - true[mask]) ** 2).mean().sqrt()


def masked_mape(pred: torch.Tensor, true: torch.Tensor,
                null_val: float = 0.0, eps: float = 1.0) -> torch.Tensor:
    """MAPE (%)，eps 防止除零（默认 1.0 流量单位，与 STGCN 论文一致）。"""
    mask = true.abs() > null_val
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return ((pred[mask] - true[mask]).abs() / (true[mask].abs() + eps)).mean() * 100


def compute_metrics(pred: torch.Tensor, true: torch.Tensor,
                    null_val: float = 0.0) -> dict[str, float]:
    """
    一次性返回 MAE / RMSE / MAPE 三项指标（denormalized 后调用）。
    """
    return {
        "MAE":  masked_mae(pred,  true, null_val).item(),
        "RMSE": masked_rmse(pred, true, null_val).item(),
        "MAPE": masked_mape(pred, true, null_val).item(),
    }


def compute_horizon_metrics(pred: torch.Tensor, true: torch.Tensor,
                             horizons: list[int] | None = None,
                             null_val: float = 0.0) -> dict[str, dict[str, float]]:
    """
    按多个预测步长分别计算指标。

    Parameters
    ----------
    pred, true : (B, N, T_out)  反归一化后的张量
    horizons   : 要统计的步长列表，默认 [1, 3, 6, 12]（超出 T_out 则跳过）

    Returns
    -------
    {
      "h1":  {"MAE": ..., "RMSE": ..., "MAPE": ...},
      "h3":  {...},
      "avg": {...},   # 全部 T_out 步平均
    }
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]
    T_out = pred.shape[-1]
    result = {}
    for h in horizons:
        if h > T_out:
            continue
        p = pred[..., :h].reshape(-1)
        t = true[..., :h].reshape(-1)
        result[f"h{h}"] = compute_metrics(p, t, null_val)
    # 全步平均
    result["avg"] = compute_metrics(pred.reshape(-1), true.reshape(-1), null_val)
    return result
