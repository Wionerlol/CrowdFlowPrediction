"""
ARIMA 基线训练脚本。

对每个非零网格独立拟合 ARIMA(2,0,1)，滚动预测未来 24 小时（48步）。
用 joblib 并行加速。

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

T_OUT = 48   # 24小时 = 48步 × 30分钟


def fit_predict_arima(series_train: np.ndarray,
                       series_test:  np.ndarray,
                       order=(2, 0, 1)) -> np.ndarray:
    """
    对单格时序拟合 ARIMA，滚动预测测试集（每 T_OUT 步预测一次）。
    返回与 series_test 等长的预测数组。
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("请安装 statsmodels：pip install statsmodels")

    preds = np.full(len(series_test), np.nan)
    history = list(series_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start_i in range(0, len(series_test), T_OUT):
            try:
                model = ARIMA(history, order=order)
                res   = model.fit()
                fc    = res.forecast(steps=T_OUT)
            except Exception:
                fc = np.full(T_OUT, np.mean(history[-48:]))
            end_i = min(start_i + T_OUT, len(series_test))
            preds[start_i:end_i] = fc[:end_i - start_i]
            # 将真实值追加到历史（滚动更新）
            history.extend(series_test[start_i:end_i].tolist())

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs",    type=int, default=8)
    parser.add_argument("--max_grids", type=int, default=200,
                        help="评估的非零格数量（全部 1024 格约需 1~2 小时）")
    parser.add_argument("--order",     type=str, default="2,0,1",
                        help="ARIMA(p,d,q)")
    args = parser.parse_args()
    order = tuple(int(x) for x in args.order.split(","))

    print(f"ARIMA{order}  n_jobs={args.n_jobs}  max_grids={args.max_grids}  T_out={T_OUT}")

    # ── 数据 ─────────────────────────────────────────────────────────
    print("[1/3] 加载 TaxiBJ...")
    data, _ = load_taxibj()
    train_raw, _, test_raw = split_taxibj(data)
    train_in = train_raw[:, 0, :, :].reshape(len(train_raw), -1)  # (T_tr, 1024)
    test_in  = test_raw[:, 0, :, :].reshape(len(test_raw),  -1)

    mean = float(train_in.mean())
    std  = float(train_in.std()) + 1e-8

    nonzero = np.where(train_in.mean(axis=0) > 0)[0]
    grids   = nonzero[:args.max_grids]
    print(f"  非零格: {len(nonzero)}  本次评估: {len(grids)}")

    # ── 并行拟合 ──────────────────────────────────────────────────────
    print("[2/3] 拟合 ARIMA（滚动 48步 预测）...")
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_predict_arima)(
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
    for i, pred_norm in enumerate(results):
        true_norm = (test_in[:, grids[i]] - mean) / std
        L = min(len(pred_norm), len(true_norm))
        all_pred.append(pred_norm[:L] * std + mean)
        all_true.append(true_norm[:L] * std + mean)

    pred_t = torch.tensor(np.concatenate(all_pred), dtype=torch.float32)
    true_t = torch.tensor(np.concatenate(all_true), dtype=torch.float32)
    m = compute_metrics(pred_t, true_t)

    print(f"\n{'='*45}")
    print(f"ARIMA{order} 24h预测（{len(grids)} 非零格，滚动{T_OUT}步）")
    print(f"  MAE  = {m['MAE']:.4f}")
    print(f"  RMSE = {m['RMSE']:.4f}")
    print(f"  MAPE = {m['MAPE']:.2f}%")

    import csv, json
    result_path = Path("outputs/results/week3_metrics.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not result_path.exists()
    with open(result_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "city", "horizon", "MAE", "RMSE", "MAPE", "config"])
        cfg_str = json.dumps({"model": "arima", "order": list(order),
                               "grids": len(grids), "T_out": T_OUT})
        w.writerow(["arima", "bj", "avg",
                    f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.2f}",
                    cfg_str])
    print(f"结果已追加到 {result_path}")


if __name__ == "__main__":
    main()
