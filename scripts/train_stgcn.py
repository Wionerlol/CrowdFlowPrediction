"""
STGCN 基线训练脚本 — TaxiBJ
用法: python scripts/train_stgcn.py [--epochs N] [--batch B] [--t_out T]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.taxibj_dataset import TaxiBJDataset, load_taxibj, build_spatial_edge_index
from src.models.stgcn import STGCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--t_in",    type=int,   default=12,  help="input steps (30-min each)")
    p.add_argument("--t_out",   type=int,   default=3,   help="prediction horizon")
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--conn",    type=int,   default=8,   help="4 or 8 connectivity")
    p.add_argument("--K",       type=int,   default=3,   help="Chebyshev order")
    p.add_argument("--patience",type=int,   default=15,  help="early stop patience")
    p.add_argument("--save",    type=str,   default="outputs/checkpoints/stgcn_best.pt")
    return p.parse_args()


def masked_mae(pred, true, null_val=0.0):
    mask = true.abs() > null_val
    return (pred[mask] - true[mask]).abs().mean()


def masked_rmse(pred, true, null_val=0.0):
    mask = true.abs() > null_val
    return ((pred[mask] - true[mask]) ** 2).mean().sqrt()


@torch.no_grad()
def evaluate(model, loader, edge_index, edge_weight, device, mean, std):
    model.eval()
    maes, rmses = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, edge_index, edge_weight)     # (B, N, T_out)
        # denormalize
        pred_real = pred * std + mean
        y_real    = y    * std + mean
        maes.append(masked_mae(pred_real, y_real).item())
        rmses.append(masked_rmse(pred_real, y_real).item())
    return np.mean(maes), np.mean(rmses)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    # ── 数据 ──────────────────────────────────────────────────────
    print("\n[1/4] 加载 TaxiBJ 数据...")
    raw = load_taxibj()                                # (22484, 2, 32, 32)

    # 时间切分：BJ13+BJ14+BJ15前75% → 训练，BJ15后25% → 验证，BJ16 → 测试
    #   BJ13=4888, BJ14=4780, BJ15=5596 (前75%=4197), BJ16=7220
    n13, n14, n15, n16 = 4888, 4780, 5596, 7220
    train_end = n13 + n14 + int(n15 * 0.75)           # ~13865
    val_end   = n13 + n14 + n15                        # ~15264

    train_raw = raw[:train_end]
    val_raw   = raw[train_end:val_end]
    test_raw  = raw[val_end:]

    train_ds = TaxiBJDataset(train_raw, args.t_in, args.t_out)
    val_ds   = TaxiBJDataset(val_raw,   args.t_in, args.t_out,
                              mean=train_ds.mean, std=train_ds.std)
    test_ds  = TaxiBJDataset(test_raw,  args.t_in, args.t_out,
                              mean=train_ds.mean, std=train_ds.std)
    mean, std = float(train_ds.mean), float(train_ds.std)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Normalization: mean={mean:.2f}, std={std:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── 图结构 ────────────────────────────────────────────────────
    print(f"\n[2/4] 构建空间图（{args.conn}-connectivity）...")
    edge_index = build_spatial_edge_index(32, 32, args.conn).to(device)
    print(f"  边数: {edge_index.shape[1]}")

    # ── 模型 ──────────────────────────────────────────────────────
    print("\n[3/4] 初始化 STGCN...")
    model = STGCN(T_in=args.t_in, F_in=2, T_out=args.t_out,
                  n_ch=(64, 16), K=args.K).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {params:.3f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    loss_fn = nn.HuberLoss()

    # ── 训练 ──────────────────────────────────────────────────────
    print(f"\n[4/4] 开始训练（{args.epochs} epochs）...")
    best_val_mae = float("inf")
    patience_cnt = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, edge_index)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        val_mae, val_rmse = evaluate(
            model, val_loader, edge_index, None, device, mean, std)
        scheduler.step(val_mae)

        elapsed = time.time() - t_start
        eta = elapsed / epoch * (args.epochs - epoch)
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"loss={np.mean(train_losses):.4f}  "
              f"val_MAE={val_mae:.4f}  val_RMSE={val_rmse:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"ETA={eta/60:.1f}min")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_cnt = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "mean": float(mean), "std": float(std),
                        "val_mae": float(val_mae), "val_rmse": float(val_rmse)}, args.save)
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"Early stop at epoch {epoch} (patience={args.patience})")
                break

    # ── 测试 ──────────────────────────────────────────────────────
    ckpt = torch.load(args.save, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_mae, test_rmse = evaluate(
        model, test_loader, edge_index, None, device, mean, std)

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*50}")
    print(f"最佳验证 MAE : {best_val_mae:.4f}")
    print(f"测试集 MAE   : {test_mae:.4f}")
    print(f"测试集 RMSE  : {test_rmse:.4f}")
    print(f"总训练时长   : {total_min:.1f} min")
    print(f"模型已保存   : {args.save}")


if __name__ == "__main__":
    main()
