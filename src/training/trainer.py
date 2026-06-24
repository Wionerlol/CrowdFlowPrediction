"""
统一训练器，支持所有 Week 3 模型（LSTM / SingleGraphSTGCN / MultiGraphSTGCN）。

所有模型需满足接口：
    pred = model(x, graphs)   # graphs 可为 None（LSTM）
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_horizon_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

try:
    import mlflow
    _MLFLOW = True
except ImportError:
    _MLFLOW = False


class Trainer:
    """
    Parameters
    ----------
    model      : 实现 forward(x, graphs) 接口的 nn.Module
    optimizer  : PyTorch 优化器
    scheduler  : 学习率调度器（可选）
    loss_fn    : 损失函数
    device     : 训练设备
    cfg        : 超参字典（用于日志记录）
    save_path  : 最优检查点保存路径
    log_dir    : TensorBoard 日志目录（可选）
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 loss_fn: nn.Module,
                 device: torch.device,
                 cfg: dict,
                 save_path: str = "outputs/checkpoints/model/best.pt",
                 log_dir: str | None = None):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn   = loss_fn
        self.device    = device
        self.cfg       = cfg
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if log_dir and _TB:
            self.writer = SummaryWriter(log_dir)

    # ── 设备迁移 ────────────────────────────────────────────────────────

    def _to_device(self, graphs: dict | None) -> dict | None:
        if graphs is None:
            return None
        out = {}
        for name, g in graphs.items():
            out[name] = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in g.items()
            }
        return out

    # ── 单 epoch 训练 ───────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader,
                     graphs: dict | None) -> float:
        self.model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x, graphs)
            loss = self.loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    # ── 评估 ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, graphs: dict | None,
                 mean: float, std: float) -> dict[str, Any]:
        """
        返回多时域指标字典，例如：
        {
          "h1":  {"MAE": ..., "RMSE": ..., "MAPE": ...},
          "h3":  {...},
          "avg": {...},
        }
        """
        self.model.eval()
        all_pred, all_true = [], []
        for x, y in loader:
            x = x.to(self.device)
            pred = self.model(x, graphs)                   # (B, N, T_out)
            all_pred.append(pred.cpu())
            all_true.append(y)

        pred_cat = torch.cat(all_pred, dim=0)              # (total_B, N, T_out)
        true_cat = torch.cat(all_true, dim=0)

        # 反归一化
        pred_real = pred_cat * std + mean
        true_real = true_cat * std + mean

        return compute_horizon_metrics(pred_real, true_real)

    # ── 主训练循环 ──────────────────────────────────────────────────────

    def fit(self,
            train_loader: DataLoader,
            val_loader:   DataLoader,
            test_loader:  DataLoader,
            graphs:       dict | None,
            mean: float,
            std:  float,
            epochs:   int = 100,
            patience: int = 15) -> dict[str, Any]:
        """
        训练直到早停，返回测试集最优指标。
        """
        graphs = self._to_device(graphs)

        best_val_mae   = float("inf")
        patience_cnt   = 0
        best_state     = None
        t_start        = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, graphs)
            val_metrics = self.evaluate(val_loader, graphs, mean, std)
            val_mae = val_metrics["avg"]["MAE"]

            if self.scheduler is not None:
                self.scheduler.step(val_mae)

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            eta = elapsed / epoch * (epochs - epoch)
            print(f"Epoch {epoch:03d}/{epochs}  "
                  f"loss={train_loss:.4f}  "
                  f"val_MAE={val_mae:.4f}  "
                  f"val_RMSE={val_metrics['avg']['RMSE']:.4f}  "
                  f"lr={lr:.2e}  ETA={eta/60:.1f}min")

            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("MAE/val",    val_mae,    epoch)
                self.writer.add_scalar("LR",         lr,         epoch)

            if _MLFLOW:
                try:
                    mlflow.log_metrics({"train_loss": train_loss,
                                        "val_mae": val_mae}, step=epoch)
                except Exception:
                    pass

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_cnt = 0
                best_state   = {k: v.cpu().clone()
                                for k, v in self.model.state_dict().items()}
                torch.save({
                    "epoch":    epoch,
                    "model":    best_state,
                    "mean":     mean,
                    "std":      std,
                    "val_mae":  val_mae,
                    "cfg":      self.cfg,
                }, self.save_path)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"早停于 epoch {epoch}（patience={patience}）")
                    break

        # ── 测试集评估 ───────────────────────────────────────────────
        if best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})
        test_metrics = self.evaluate(test_loader, graphs, mean, std)
        total_min = (time.time() - t_start) / 60

        print(f"\n{'='*55}")
        print(f"最佳验证 MAE : {best_val_mae:.4f}")
        for h, m in test_metrics.items():
            print(f"测试 {h:>4s}   MAE={m['MAE']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}  MAPE={m['MAPE']:.2f}%")
        print(f"总训练时长   : {total_min:.1f} min")
        print(f"检查点       : {self.save_path}")

        if self.writer:
            self.writer.close()

        return test_metrics
