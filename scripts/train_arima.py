"""
ARIMA 完整分析流程：
  阶段1  诊断分析  — ADF 检验确定 d，ACF/PACF 图建议 p/q，保存诊断图
  阶段2  参数选择  — AIC 网格搜索选最优 (p,d,q)，Ljung-Box 残差检验
  阶段3  滚动原点  — Rolling-origin 评估（扩展窗口，T_OUT 步预测）

用法：
    python scripts/train_arima.py               # 完整流程（推荐）
    python scripts/train_arima.py --diag_only   # 只做诊断，不训练
    python scripts/train_arima.py --n_jobs 8 --max_grids 200
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无界面环境
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.flow_dataset import load_taxibj, split_taxibj
from src.training.metrics import compute_metrics

warnings.filterwarnings("ignore")

T_OUT      = 48     # 预测 24 小时 = 48 步
DIAG_DIR   = Path("outputs/reports/arima_diagnostics")
RESULT_DIR = Path("outputs/results")


# ══════════════════════════════════════════════════════════
# 阶段1：诊断分析
# ══════════════════════════════════════════════════════════

def adf_test(series: np.ndarray) -> dict:
    """ADF 单位根检验。返回 {'stationary': bool, 'd': int, 'adf_stat': float, 'p_value': float}"""
    from statsmodels.tsa.stattools import adfuller

    def _adf(s):
        res = adfuller(s, autolag="AIC")
        return res[0], res[1]   # (statistic, p_value)

    stat, pval = _adf(series)
    if pval < 0.05:
        return {"stationary": True,  "d": 0, "adf_stat": stat, "p_value": pval}

    # 1阶差分
    diff1 = np.diff(series)
    stat1, pval1 = _adf(diff1)
    if pval1 < 0.05:
        return {"stationary": False, "d": 1, "adf_stat": stat1, "p_value": pval1}

    # 2阶差分
    diff2 = np.diff(diff1)
    stat2, pval2 = _adf(diff2)
    return {"stationary": False, "d": 2, "adf_stat": stat2, "p_value": pval2}


def plot_acf_pacf(series: np.ndarray, grid_id: int,
                   d: int, lags: int = 48) -> Path:
    """绘制 ACF / PACF 图，返回保存路径。"""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    s = series.copy()
    for _ in range(d):
        s = np.diff(s)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(s,  ax=axes[0], lags=lags, title=f"Grid {grid_id}  ACF  (d={d})")
    plot_pacf(s, ax=axes[1], lags=lags, title=f"Grid {grid_id}  PACF (d={d})")
    plt.tight_layout()

    out = DIAG_DIR / f"acf_pacf_grid{grid_id}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def run_diagnostics(train_in: np.ndarray,
                     sample_grids: list[int]) -> dict:
    """
    对采样格子运行 ADF 检验和 ACF/PACF 分析。
    返回每格的 ADF 结果，并保存诊断图。
    """
    import csv
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    adf_results = {}
    d_counts = {0: 0, 1: 0, 2: 0}

    print(f"\n[阶段1] ADF 检验（{len(sample_grids)} 个格子）")
    for gid in sample_grids:
        res = adf_test(train_in[:, gid])
        adf_results[gid] = res
        d_counts[res["d"]] = d_counts.get(res["d"], 0) + 1

    # 汇总 d 的分布
    total = len(sample_grids)
    print(f"  d=0（平稳）  : {d_counts.get(0,0):4d} 格（{d_counts.get(0,0)/total:.0%}）")
    print(f"  d=1（1阶差分）: {d_counts.get(1,0):4d} 格（{d_counts.get(1,0)/total:.0%}）")
    print(f"  d=2（2阶差分）: {d_counts.get(2,0):4d} 格（{d_counts.get(2,0)/total:.0%}）")

    # 推荐全局 d
    d_global = max(d_counts, key=d_counts.get)
    print(f"  → 推荐全局 d = {d_global}")

    # 保存 ADF 结果 CSV
    csv_path = DIAG_DIR / "adf_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grid_id", "stationary", "d", "adf_stat", "p_value"])
        for gid, r in adf_results.items():
            w.writerow([gid, r["stationary"], r["d"],
                        f"{r['adf_stat']:.4f}", f"{r['p_value']:.4f}"])
    print(f"  ADF 结果已保存：{csv_path}")

    # 对最活跃的 3 个格子绘制 ACF/PACF
    top3 = sorted(sample_grids,
                  key=lambda g: train_in[:, g].mean(), reverse=True)[:3]
    print(f"\n  绘制 ACF/PACF 图（Top-3 活跃格子：{top3}）")
    for gid in top3:
        path = plot_acf_pacf(train_in[:, gid], gid,
                              d=adf_results[gid]["d"])
        print(f"    Grid {gid} → {path}")

    return {"adf_results": adf_results, "d_global": d_global}


# ══════════════════════════════════════════════════════════
# 阶段2：AIC 网格搜索 + 残差检验
# ══════════════════════════════════════════════════════════

def select_order_aic(series: np.ndarray, d: int,
                      p_range=range(0, 4),
                      q_range=range(0, 3)) -> tuple[int, int, int]:
    """
    对给定 d，网格搜索 (p, q) 选 AIC 最小的组合。
    """
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_order = (1, d, 1)   # 默认回退值

    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(series, order=(p, d, q)).fit()
                if res.aic < best_aic:
                    best_aic   = res.aic
                    best_order = (p, d, q)
            except Exception:
                pass
    return best_order


def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> dict:
    """
    Ljung-Box 检验：H0 = 残差为白噪声。
    p_value > 0.05 → 接受 H0，残差无自相关，模型充分提取信息。
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    res = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    lb_stat  = float(res["lb_stat"].iloc[-1])
    lb_pvalue = float(res["lb_pvalue"].iloc[-1])
    return {"lb_stat": lb_stat, "lb_pvalue": lb_pvalue,
            "white_noise": lb_pvalue > 0.05}


def plot_residuals(residuals: np.ndarray, grid_id: int,
                    lb_result: dict) -> Path:
    """绘制残差诊断图（时序图 + 直方图 + ACF）。"""
    from statsmodels.graphics.tsaplots import plot_acf as _plot_acf

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(residuals[:200], linewidth=0.5)
    axes[0].axhline(0, color="red", linewidth=0.8)
    axes[0].set_title(f"Grid {grid_id} 残差时序图（前200步）")

    axes[1].hist(residuals, bins=40, edgecolor="white")
    axes[1].set_title("残差分布")

    _plot_acf(residuals, ax=axes[2], lags=30,
              title=f"残差 ACF\nLjung-Box p={lb_result['lb_pvalue']:.3f}"
                    f"（{'✅ 白噪声' if lb_result['white_noise'] else '⚠️ 有自相关'}）")

    plt.tight_layout()
    out = DIAG_DIR / f"residual_grid{grid_id}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fit_and_diagnose(series: np.ndarray, order: tuple,
                      grid_id: int) -> dict:
    """拟合 ARIMA，做残差诊断，返回诊断结果。"""
    from statsmodels.tsa.arima.model import ARIMA
    res = ARIMA(series, order=order).fit()
    residuals = res.resid
    lb = ljung_box_test(residuals)
    plot_residuals(residuals, grid_id, lb)
    return {"order": order, "aic": res.aic, **lb, "model": res}


# ══════════════════════════════════════════════════════════
# 阶段3：逐格 Rolling-origin 预测
# ══════════════════════════════════════════════════════════

def rolling_origin_predict(series_train: np.ndarray,
                            series_test:  np.ndarray,
                            order: tuple) -> np.ndarray:
    """
    Fit-once 预测：
      - 在完整训练集上拟合一次 ARIMA
      - 用 apply() 追加测试集观测更新状态（不重新估计参数）
      - 每次滚动 T_OUT 步后追加真实值，保持残差状态正确

    比每步重新拟合快 100x+，参数估计质量相同（都用全训练集）。
    返回与 series_test 等长的预测数组。
    """
    from statsmodels.tsa.arima.model import ARIMA

    preds = np.full(len(series_test), np.nan)
    try:
        res = ARIMA(series_train, order=order).fit()
    except Exception:
        return np.full(len(series_test), float(np.mean(series_train[-T_OUT:])))

    for start_i in range(0, len(series_test), T_OUT):
        end_i = min(start_i + T_OUT, len(series_test))
        steps = end_i - start_i
        try:
            fc = res.forecast(steps=steps)
            preds[start_i:end_i] = fc
            # 追加真实值更新残差状态，不重估参数
            res = res.append(series_test[start_i:end_i], refit=False)
        except Exception:
            preds[start_i:end_i] = np.mean(series_train[-T_OUT:])

    return preds


def fit_predict_one(args_tuple):
    """joblib 并行入口（单格）。"""
    gid, train_ser, test_ser, order = args_tuple
    preds = rolling_origin_predict(train_ser, test_ser, order)
    return gid, preds


# ══════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs",     type=int, default=8)
    parser.add_argument("--max_grids",  type=int, default=200,
                        help="最多评估 N 个非零格（全部约需 2~4 小时）")
    parser.add_argument("--diag_grids", type=int, default=30,
                        help="用于诊断分析的格子数量（ADF/AIC搜索）")
    parser.add_argument("--diag_only",  action="store_true",
                        help="只做诊断分析，不跑完整预测")
    args = parser.parse_args()

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────
    print("加载 TaxiBJ...")
    data, _ = load_taxibj()
    train_raw, _, test_raw = split_taxibj(data)
    train_in = train_raw[:, 0, :, :].reshape(len(train_raw), -1)  # (T_tr, 1024)
    test_in  = test_raw[:,  0, :, :].reshape(len(test_raw),  -1)

    mean = float(train_in.mean())
    std  = float(train_in.std()) + 1e-8

    nonzero = np.where(train_in.mean(axis=0) > 0)[0]
    # 采样格子：优先选择流量最大的（更有代表性）
    activity   = train_in[:, nonzero].mean(axis=0)
    sorted_idx = nonzero[np.argsort(-activity)]
    diag_grids = sorted_idx[:args.diag_grids].tolist()
    eval_grids = sorted_idx[:args.max_grids].tolist()
    print(f"  非零格: {len(nonzero)}  诊断格: {len(diag_grids)}  评估格: {len(eval_grids)}")

    # ══════════════════════════════════════════════════════
    # 阶段1：诊断分析
    # ══════════════════════════════════════════════════════
    diag = run_diagnostics(
        (train_in - mean) / std,
        diag_grids,
    )
    d_global = diag["d_global"]

    if args.diag_only:
        print("\n--diag_only 模式，流程结束。")
        return

    # ══════════════════════════════════════════════════════
    # 阶段2：AIC 选阶 + 残差检验（在最活跃格子上搜索）
    # ══════════════════════════════════════════════════════
    print(f"\n[阶段2] AIC 网格搜索最优 (p,{d_global},q)（在 Top-5 格子上搜索）")
    import csv as _csv
    order_csv = DIAG_DIR / "order_selection.csv"
    best_orders = []

    with open(order_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["grid_id", "p", "d", "q", "aic",
                    "lb_pvalue", "white_noise"])

        for gid in diag_grids[:5]:
            series_norm = (train_in[:, gid] - mean) / std
            order = select_order_aic(series_norm, d=d_global)
            best_orders.append(order)

            res = fit_and_diagnose(series_norm, order, gid)
            print(f"  Grid {gid:4d} → ARIMA{order}  "
                  f"AIC={res['aic']:.1f}  "
                  f"LB p={res['lb_pvalue']:.3f} "
                  f"({'✅ 白噪声' if res['white_noise'] else '⚠️ 有自相关'})")
            w.writerow([gid, order[0], order[1], order[2],
                        f"{res['aic']:.2f}",
                        f"{res['lb_pvalue']:.4f}", res["white_noise"]])

    # 用众数作为全局阶数
    from collections import Counter
    global_order = Counter(best_orders).most_common(1)[0][0]
    print(f"\n  → 全局阶数选定：ARIMA{global_order}")

    # 汇总参数分布图
    _plot_order_distribution(diag_grids, train_in, mean, std, d_global)

    # ══════════════════════════════════════════════════════
    # 阶段3：滚动原点预测
    # ══════════════════════════════════════════════════════
    print(f"\n[阶段3] Fit-once 滚动预测"
          f"（ARIMA{global_order}，{len(eval_grids)} 格，{T_OUT}步窗口）")

    from joblib import Parallel, delayed

    job_args = [
        (gid,
         (train_in[:, gid] - mean) / std,
         (test_in[:, gid]  - mean) / std,
         global_order)
        for gid in eval_grids
    ]
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(fit_predict_one)(a) for a in job_args
    )

    # ── 指标 ──────────────────────────────────────────────
    import torch
    all_pred, all_true = [], []
    for gid, pred_norm in results:
        true_norm = (test_in[:, gid] - mean) / std
        L = min(len(pred_norm), len(true_norm))
        all_pred.append(pred_norm[:L] * std + mean)
        all_true.append(true_norm[:L] * std + mean)

    pred_t = torch.tensor(np.concatenate(all_pred), dtype=torch.float32)
    true_t = torch.tensor(np.concatenate(all_true), dtype=torch.float32)
    m = compute_metrics(pred_t, true_t)

    print(f"\n{'='*50}")
    print(f"ARIMA{global_order}  Fit-once 滚动预测 (24h)")
    print(f"格子数量   : {len(eval_grids)}")
    print(f"测试集长度 : {len(test_in)} 步")
    print(f"  MAE  = {m['MAE']:.4f}")
    print(f"  RMSE = {m['RMSE']:.4f}")
    print(f"  MAPE = {m['MAPE']:.2f}%")

    # ── 保存结果 ──────────────────────────────────────────
    import csv, json
    result_path = RESULT_DIR / "week3_metrics.csv"
    write_header = not result_path.exists()
    with open(result_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "city", "T_in", "T_out", "tag",
                        "horizon", "MAE", "RMSE", "MAPE", "config"])
        cfg_str = json.dumps({"model": "arima", "order": list(global_order),
                               "grids": len(eval_grids), "T_out": T_OUT,
                               "d_method": "ADF", "pq_method": "AIC"})
        w.writerow(["arima", "bj", "all_history", T_OUT, "", "avg",
                    f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.2f}",
                    cfg_str])
    print(f"\n结果已追加到 {result_path}")
    print(f"诊断图已保存到 {DIAG_DIR}/")


def _plot_order_distribution(grids, train_in, mean, std, d):
    """对所有诊断格子快速估算 (p,q)，绘制分布图。"""
    from statsmodels.tsa.arima.model import ARIMA

    orders = []
    for gid in grids:
        s = (train_in[:, gid] - mean) / std
        try:
            # 快速 AIC 搜索（小范围）
            best, best_aic = (1, d, 1), np.inf
            for p in range(0, 4):
                for q in range(0, 3):
                    if p == 0 and q == 0:
                        continue
                    try:
                        aic = ARIMA(s, order=(p, d, q)).fit().aic
                        if aic < best_aic:
                            best_aic = aic
                            best = (p, d, q)
                    except Exception:
                        pass
            orders.append(best)
        except Exception:
            pass

    if not orders:
        return

    ps = [o[0] for o in orders]
    qs = [o[2] for o in orders]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(ps, bins=range(0, 6), edgecolor="white")
    axes[0].set_xlabel("p（AR 阶数）")
    axes[0].set_title("p 分布（AIC 选择）")
    axes[1].hist(qs, bins=range(0, 5), edgecolor="white")
    axes[1].set_xlabel("q（MA 阶数）")
    axes[1].set_title("q 分布（AIC 选择）")
    plt.suptitle(f"ARIMA 参数分布（d={d}，{len(orders)} 格）")
    plt.tight_layout()

    out = DIAG_DIR / "param_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  参数分布图已保存：{out}")


if __name__ == "__main__":
    main()
