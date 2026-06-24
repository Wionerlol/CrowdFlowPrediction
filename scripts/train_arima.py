"""
ARIMA 基线训练脚本。

对每个非零网格独立拟合 ARIMA(2,0,1)，用 joblib 并行。
只做 1步预测（ARIMA 多步退化严重，结果仅作参考基线）。

用法：
    python scripts/train_arima.py [--n_jobs 8] [--max_grids 200]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.flow_dataset import load_taxibj, split_taxibj
from src.training.metrics import compute_metrics


def fit_predict_one(series_train: np.ndarray,
                    series_test:  np.ndarray,
                    order=(2, 0, 1)) -> tuple[np.ndarray, bool]:
    """对单格时序拟合 ARIMA，滚动预测测试集（1步）。"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series_train, order=order)
        res   = model.fit()
        # 一步滚动预测：每次用真实值更新窗口
        preds = res.apply(series_test).fittedvalues
        # 只取 1步 ahead 预测
        forecast = np.roll(preds, 1)
        forecast[0] = series_train[-1]
        return forecast, True
    except Exception:
        return np.full(len(series_test), series_train.mean()), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs",    type=int, default=8,
                        help="并行进程数")
    parser.add_argument("--max_grids", type=int, default=200,
                        help="最多评估前 N 个非零格（全部跑约 15 min）")
    parser.add_argument("--order",     type=str, default="2,0,1",
                        help="ARIMA(p,d,q)")
    args = parser.parse_args()
    order = tuple(int(x) for x in args.order.split(","))

    print(f"ARIMA{order}  n_jobs={args.n_jobs}  max_grids={args.max_grids}")

    # ── 数据 ─────────────────────────────────────────────────────────
    print("[1/3] 加载 TaxiBJ...")
    data, _ = load_taxibj()                                # (22484,2,32,32)
    train_raw, _, test_raw = split_taxibj(data)
    # inflow only, flatten to (T, N)
    train_in = train_raw[:, 0, :, :].reshape(len(train_raw), -1)  # (T_tr, 1024)
    test_in  = test_raw[:, 0, :, :].reshape(len(test_raw),  -1)   # (T_te, 1024)
    mean = float(train_in.mean())
    std  = float(train_in.std()) + 1e-8

    # 选非零格
    nonzero_grids = np.where(train_in.mean(axis=0) > 0)[0]
    grids = nonzero_grids[:args.max_grids]
    print(f"  非零格: {len(nonzero_grids)}  本次评估: {len(grids)}")

    # ── 并行拟合 ──────────────────────────────────────────────────────
    print("[2/3] 拟合 ARIMA...")
    from joblib import Parallel, delayed

    warnings.filterwarnings("ignore")

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_predict_one)(
            (train_in[:, g] - mean) / std,
            (test_in[:, g]  - mean) / std,
            order,
        )
        for g in grids
    )

    # ── 指标 ─────────────────────────────────────────────────────────
    print("[3/3] 计算指标...")
    import torch
    all_pred, all_true = [], []
    for i, (pred_norm, ok) in enumerate(results):
        true_norm = (test_in[:, grids[i]] - mean) / std
        pred_real = pred_norm * std + mean
        true_real = true_norm * std + mean
        all_pred.append(pred_real)
        all_true.append(true_real)

    pred_t = torch.tensor(np.array(all_pred), dtype=torch.float32)
    true_t = torch.tensor(np.array(all_true), dtype=torch.float32)
    m = compute_metrics(pred_t.reshape(-1), true_t.reshape(-1))

    print(f"\n{'='*45}")
    print(f"ARIMA{order}  1步预测（{len(grids)} 非零格）")
    print(f"  MAE  = {m['MAE']:.4f}")
    print(f"  RMSE = {m['RMSE']:.4f}")
    print(f"  MAPE = {m['MAPE']:.2f}%")

    # 追加到结果表
    import csv, json
    result_path = Path("outputs/results/week3_metrics.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not result_path.exists()
    with open(result_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "city", "horizon",
                        "MAE", "RMSE", "MAPE", "config"])
        cfg_str = json.dumps({"model": "arima", "order": list(order),
                               "grids": len(grids)})
        w.writerow(["arima", "bj", "h1",
                    f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.2f}",
                    cfg_str])
    print(f"结果已追加到 {result_path}")


if __name__ == "__main__":
    main()
