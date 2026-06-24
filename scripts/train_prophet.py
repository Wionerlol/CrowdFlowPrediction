"""
Prophet 基线训练脚本。

对每个非零网格独立拟合 Prophet，预测未来 24 小时（48步 × 30分钟）。
用 joblib 并行加速，建议 --max_grids 200 先评估代表性网格。

用法：
    python scripts/train_prophet.py [--n_jobs 8] [--max_grids 200]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.flow_dataset import load_taxibj, split_taxibj
from src.training.metrics import compute_metrics

T_OUT = 48   # 24小时 = 48步 × 30分钟


def fit_predict_prophet(series_train: np.ndarray,
                         series_test:  np.ndarray,
                         freq: str = "30min") -> np.ndarray:
    """
    Fit-once：在训练集拟合一次 Prophet，直接预测整个测试集长度。
    Prophet 学习的季节性分量可以外推到任意未来时间点，无需重新拟合。
    """
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet

    start = pd.Timestamp("2013-07-01")
    idx   = pd.date_range(start, periods=len(series_train), freq=freq)
    df_train = pd.DataFrame({"ds": idx, "y": series_train.astype(float)})

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_train)

    # 直接预测整个测试集时间范围
    future_idx = pd.date_range(
        idx[-1] + pd.Timedelta(freq), periods=len(series_test), freq=freq
    )
    future = pd.DataFrame({"ds": future_idx})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = model.predict(future)["yhat"].values

    return fc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs",    type=int, default=4)
    parser.add_argument("--max_grids", type=int, default=100,
                        help="评估的非零格数量（全部 1024 格约需数小时）")
    args = parser.parse_args()

    print(f"Prophet  n_jobs={args.n_jobs}  max_grids={args.max_grids}  T_out={T_OUT}")

    # 检查 Prophet 是否安装
    try:
        from prophet import Prophet
        print("  使用 prophet 包")
    except ImportError:
        try:
            from fbprophet import Prophet
            print("  使用 fbprophet 包")
        except ImportError:
            print("错误：未安装 Prophet。请运行：pip install prophet")
            sys.exit(1)

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
    print("[2/3] 拟合 Prophet（fit-once，直接外推测试集）...")
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_predict_prophet)(
            (train_in[:, g] - mean) / std,
            (test_in[:, g]  - mean) / std,
        )
        for g in grids
    )

    # ── 指标（对齐到整 T_OUT 窗口）──────────────────────────────────
    print("[3/3] 计算指标...")
    import torch
    all_pred, all_true = [], []
    for i, pred_norm in enumerate(results):
        true_norm = (test_in[:, grids[i]] - mean) / std
        # 截到相同长度
        L = min(len(pred_norm), len(true_norm))
        all_pred.append(pred_norm[:L] * std + mean)
        all_true.append(true_norm[:L] * std + mean)

    pred_t = torch.tensor(np.concatenate(all_pred), dtype=torch.float32)
    true_t = torch.tensor(np.concatenate(all_true), dtype=torch.float32)
    m = compute_metrics(pred_t, true_t)

    print(f"\n{'='*45}")
    print(f"Prophet 24h预测（{len(grids)} 非零格，fit-once）")
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
        cfg_str = json.dumps({"model": "prophet", "grids": len(grids), "T_out": T_OUT})
        w.writerow(["prophet", "bj", "all_history", T_OUT, "", "avg",
                    f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.2f}",
                    cfg_str])
    print(f"结果已追加到 {result_path}")


if __name__ == "__main__":
    main()
