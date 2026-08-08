"""Generate Week 4 result figures and render the Markdown report to HTML/PDF.

The figures are derived from preserved experiment artifacts.  No model training
is started by this script.  Run from any directory with:

    .venv/bin/python scripts/generate_week4_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from markdown import markdown
from matplotlib import font_manager
from weasyprint import HTML


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = REPORT_DIR / "week4_figures"
METRICS_PATH = ROOT / "outputs" / "results" / "week3_metrics.csv"
MULTISEED_RUNS_PATH = ROOT / "outputs" / "multiseed" / "runs.csv"
ENSEMBLE_PATH = (
    ROOT / "outputs" / "results" / "agformer_gate_context_linear_ensemble_4seed.json"
)
PERIODIC_BASELINE_PATH = (
    ROOT / "outputs" / "results" / "agformer_periodic_subset_baseline.json"
)
MULTIGRAPH_WEIGHTS_PATH = (
    ROOT / "outputs" / "results" / "agformer_multigraph_weights.json"
)
MARKDOWN_PATH = REPORT_DIR / "week4_report.md"
HTML_PATH = REPORT_DIR / "week4_report.html"
PDF_PATH = REPORT_DIR / "week4_report.pdf"

COLORS = {
    "blue": "#2563A6",
    "light_blue": "#87ACD1",
    "green": "#2E8B57",
    "orange": "#D97706",
    "red": "#C2413B",
    "gray": "#8A94A3",
    "light_gray": "#D7DEE7",
    "dark": "#25364A",
}
METRICS = ["MAE", "RMSE", "MAPE"]
HORIZONS = ["h1", "h3", "h6", "h12", "avg"]
HORIZON_LABELS = ["0.5h", "1.5h", "3h", "6h", "24h"]


def display_metric(metric: str) -> str:
    return "MAPE (%)" if metric == "MAPE" else metric


def configure_plotting() -> None:
    bundled_cjk_fonts = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/msttcorefonts/msyh.ttc"),
    ]
    selected = None
    for path in bundled_cjk_fonts:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            selected = font_manager.FontProperties(fname=str(path)).get_name()
            break

    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    if selected is None:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        selected = next((font for font in preferred if font in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlecolor": COLORS["dark"],
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "grid.color": "#DDE4EC",
            "grid.linewidth": 0.7,
        }
    )


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    required = [
        METRICS_PATH,
        MULTISEED_RUNS_PATH,
        ENSEMBLE_PATH,
        PERIODIC_BASELINE_PATH,
        MULTIGRAPH_WEIGHTS_PATH,
        MARKDOWN_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Week 4 artifacts: {missing}")

    metrics = pd.read_csv(METRICS_PATH)
    multiseed = pd.read_csv(MULTISEED_RUNS_PATH)
    ensemble = json.loads(ENSEMBLE_PATH.read_text(encoding="utf-8"))
    periodic = json.loads(PERIODIC_BASELINE_PATH.read_text(encoding="utf-8"))
    return metrics, multiseed, ensemble, periodic


def tagged_metrics(metrics: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = metrics.loc[metrics["tag"].fillna("") == tag].copy()
    if rows.empty:
        raise ValueError(f"No metrics found for tag={tag}")
    rows["horizon"] = pd.Categorical(rows["horizon"], HORIZONS, ordered=True)
    return rows.sort_values("horizon")


def avg_row(metrics: pd.DataFrame, tag: str) -> pd.Series:
    rows = tagged_metrics(metrics, tag)
    row = rows.loc[rows["horizon"] == "avg"]
    if len(row) != 1:
        raise ValueError(f"Expected one avg row for tag={tag}, got {len(row)}")
    return row.iloc[0]


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_values(ax: plt.Axes, bars, fmt: str = ".2f") -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            format(value, fmt),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["dark"],
        )


def plot_baseline_multiseed(metrics: pd.DataFrame, multiseed: pd.DataFrame) -> None:
    groups = [
        ("LSTM\ngap-aware", "lstm_gapaware", COLORS["orange"]),
        ("AGFormer\nTime+Static", "agformer_context", COLORS["blue"]),
    ]
    gated = avg_row(metrics, "gate_context_seed42")
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    rng = np.random.default_rng(42)

    for ax, metric in zip(axes, METRICS):
        for x, (label, experiment, color) in enumerate(groups):
            values = multiseed.loc[multiseed["experiment"] == experiment, metric].to_numpy()
            if len(values) != 3:
                raise ValueError(f"Expected 3 seeds for {experiment}, got {len(values)}")
            jitter = rng.uniform(-0.065, 0.065, size=len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                s=42,
                color=color,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.errorbar(
                x,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="D",
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                markersize=6,
                capsize=4,
                linewidth=1.5,
                zorder=4,
            )

        ax.scatter(
            2,
            float(gated[metric]),
            marker="*",
            s=115,
            color=COLORS["green"],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.annotate(
            f"{float(gated[metric]):.2f}",
            (2, float(gated[metric])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
        ax.set_xticks([0, 1, 2], [groups[0][0], groups[1][0], "AGFormer\ngated seed42"])
        ax.set_title(f"avg {display_metric(metric)} ↓")
        ax.grid(axis="y")
        ax.margins(y=0.18)

    fig.suptitle("LSTM 与 AGFormer 基线：多随机种子稳定性", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "圆点为独立 seed；菱形及误差棒为 mean ± sample std；星形为 gated context 单模型 seed42。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, "fig3_1_baseline_multiseed.png")


def plot_spacetimeformer_ablation(metrics: pd.DataFrame) -> None:
    variants = [
        ("Realtime context", "spacetimeformer_realtime_context_seed42_resume"),
        ("参数匹配", "spacetimeformer_realtime_context_matched_seed42"),
        ("原版对齐 + future", "spacetimeformer_original_aligned_seed42"),
        ("原版对齐 no future", "spacetimeformer_original_aligned_no_future_time_seed42"),
        ("no future + linear", "spacetimeformer_original_aligned_no_future_time_linear_seed42"),
        ("no future + linear + gated", "spacetimeformer_original_aligned_no_future_time_linear_gated_seed42"),
    ]
    values = np.array(
        [[float(avg_row(metrics, tag)[metric]) for metric in METRICS] for _, tag in variants]
    )
    labels = [label for label, _ in variants]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.0))

    for index, (ax, metric) in enumerate(zip(axes, METRICS)):
        column = values[:, index]
        best = int(np.argmin(column))
        colors = [COLORS["blue"] if i == best else COLORS["light_blue"] for i in range(len(labels))]
        y = np.arange(len(labels))
        bars = ax.barh(y, column, color=colors, height=0.62)
        ax.set_yticks(y, labels if index == 0 else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_title(f"avg {display_metric(metric)} ↓")
        ax.grid(axis="x")
        lower = max(0.0, float(column.min()) - max(1.2, float(np.ptp(column)) * 0.35))
        upper = float(column.max()) + max(0.8, float(np.ptp(column)) * 0.25)
        ax.set_xlim(lower, upper)
        for bar, value in zip(bars, column):
            ax.text(
                value + (upper - lower) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=8,
            )

    fig.suptitle("Spacetimeformer 结构对齐与组件消融", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "全部为 seed42；蓝色标出该指标下最优 Spacetimeformer 版本。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save_figure(fig, "fig3_2_spacetimeformer_ablation.png")


def plot_feature_ablation(metrics: pd.DataFrame, periodic: dict) -> None:
    full_variants = [
        ("AGFormer baseline", "gate_context_seed42", COLORS["dark"]),
        ("Calendar", "calendar_context_seed42", COLORS["light_blue"]),
        ("Calendar + Weather", "calendar_weather_seed42", COLORS["blue"]),
        ("Multi-graph", "agformer_multigraph_context_seed42", COLORS["green"]),
    ]
    full_values = [float(avg_row(metrics, tag)["MAE"]) for _, tag, _ in full_variants]
    matched_baseline = float(periodic["metrics"]["avg"]["MAE"])
    periodic_mae = float(avg_row(metrics, "agformer_periodic_lags_seed42")["MAE"])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    bars = ax.bar(
        np.arange(len(full_variants)),
        full_values,
        color=[color for _, _, color in full_variants],
        width=0.66,
    )
    ax.set_xticks(np.arange(len(full_variants)), [label for label, _, _ in full_variants], rotation=12)
    ax.set_ylabel("avg MAE ↓")
    ax.set_title("同一 4,022 测试窗口")
    ax.set_ylim(min(full_values) - 0.8, max(full_values) + 0.8)
    ax.grid(axis="y")
    annotate_values(ax, bars, ".3f")

    ax = axes[1]
    matched_values = [matched_baseline, periodic_mae]
    bars = ax.bar(
        [0, 1],
        matched_values,
        color=[COLORS["dark"], COLORS["orange"]],
        width=0.58,
    )
    ax.set_xticks([0, 1], ["Matched baseline", "Periodic lags"])
    ax.set_ylabel("avg MAE ↓")
    ax.set_title(f"周期滞后匹配子集（{int(periodic['windows']):,} 窗口）")
    ax.set_ylim(min(matched_values) - 0.8, max(matched_values) + 0.8)
    ax.grid(axis="y")
    annotate_values(ax, bars, ".3f")
    delta = periodic_mae - matched_baseline
    ax.text(
        0.5,
        max(matched_values) + 0.45,
        f"ΔMAE = +{delta:.3f}",
        ha="center",
        fontsize=9,
        color=COLORS["red"],
        fontweight="bold",
    )

    fig.suptitle("外生特征、图结构与周期滞后实验", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "周期滞后会减少有效窗口，因而单独与相同 3,433 窗口上的原 AGFormer 基线比较。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, "fig3_3_feature_ablation.png")


def plot_graph_weights() -> None:
    result = json.loads(MULTIGRAPH_WEIGHTS_PATH.read_text(encoding="utf-8"))
    names = list(result["graph_names"])
    weights = np.asarray(result["weights_percent"], dtype=np.float64)
    if len(names) != len(weights) or not np.isclose(weights.sum(), 100.0, atol=1e-4):
        raise ValueError("Invalid multi-graph weight artifact")
    display_names = {"spatial": "Spatial", "poi": "POI", "flow": "Flow", "transit": "Transit"}
    labels = [display_names.get(name, name) for name in names]
    colors = [COLORS["light_blue"], COLORS["orange"], COLORS["blue"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    bars = ax.bar(labels, weights, color=colors, width=0.62)
    ax.set_ylabel("学习权重（%）")
    ax.set_title("Multi-graph AGFormer 的全局图融合权重")
    ax.set_ylim(0, 101)
    ax.grid(axis="y")
    for bar, value in zip(bars, weights):
        ax.annotate(
            f"{value:.2f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold" if value == weights.max() else "normal",
        )
    fig.text(
        0.5,
        0.01,
        "权重由 best checkpoint 中 graph_logits 经 softmax 计算；Flow graph 占比 92.18%。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "fig3_4_multigraph_weights.png")


def horizon_values(metrics: pd.DataFrame, tag: str, metric: str) -> np.ndarray:
    rows = tagged_metrics(metrics, tag).set_index("horizon")
    return np.array([float(rows.loc[h, metric]) for h in HORIZONS])


def plot_linear_horizon(metrics: pd.DataFrame) -> None:
    variants = [
        ("原 AGFormer gated", "gate_context_seed42", COLORS["gray"]),
        ("AGFormer + linear", "agformer_gate_context_linear_seed42", COLORS["blue"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9))
    for ax, metric in zip(axes, METRICS):
        for label, tag, color in variants:
            ax.plot(
                HORIZON_LABELS,
                horizon_values(metrics, tag, metric),
                marker="o",
                linewidth=2.1,
                markersize=5,
                label=label,
                color=color,
            )
        ax.set_title(f"累计 {display_metric(metric)} ↓")
        ax.set_xlabel("累计预测范围")
        ax.grid(axis="y")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Linear residual head 对不同预测范围的影响（seed42）", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        "24h 对应完整 48 步平均；其余点分别为前 1/3/6/12 步累计平均。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, "fig4_1_linear_horizon.png")


def plot_linear_multiseed(ensemble: dict) -> None:
    singles = ensemble["singles"]
    seeds = [int(item["seed"]) for item in singles]
    order = np.argsort(seeds)
    seeds = [seeds[i] for i in order]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    for ax, metric in zip(axes, METRICS):
        values = np.array([float(singles[i]["metrics"]["avg"][metric]) for i in order])
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        x = np.arange(len(seeds))
        ax.scatter(x, values, s=62, color=COLORS["blue"], edgecolor="white", linewidth=0.7, zorder=3)
        ax.axhline(mean, color=COLORS["green"], linewidth=1.8, label="mean")
        ax.fill_between(
            [-0.45, len(seeds) - 0.55],
            mean - std,
            mean + std,
            color=COLORS["green"],
            alpha=0.12,
            label="± population std",
        )
        for xi, value in zip(x, values):
            ax.annotate(f"{value:.2f}", (xi, value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, [f"seed{s}" for s in seeds])
        ax.set_title(f"avg {display_metric(metric)}: {mean:.2f} ± {std:.2f}")
        ax.grid(axis="y")
        ax.margins(y=0.2)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("AGFormer + linear 的四随机种子复验", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save_figure(fig, "fig4_2_linear_multiseed.png")


def plot_ensemble_horizon(ensemble: dict) -> None:
    singles = ensemble["singles"]
    ensemble_metrics = ensemble["ensemble"]["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.0))

    for ax, metric in zip(axes, METRICS):
        for index, item in enumerate(singles):
            values = [float(item["metrics"][h][metric]) for h in HORIZONS]
            ax.plot(
                HORIZON_LABELS,
                values,
                color=COLORS["light_gray"],
                linewidth=1.25,
                marker="o",
                markersize=3,
                label="单 seed" if index == 0 else None,
                zorder=1,
            )
        ensemble_values = [float(ensemble_metrics[h][metric]) for h in HORIZONS]
        ax.plot(
            HORIZON_LABELS,
            ensemble_values,
            color=COLORS["blue"],
            linewidth=2.7,
            marker="D",
            markersize=5,
            label="4-seed ensemble",
            zorder=3,
        )
        ax.set_title(f"累计 {display_metric(metric)} ↓")
        ax.set_xlabel("累计预测范围")
        ax.grid(axis="y")
    axes[0].legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("四个单模型与均值 Ensemble 的预测误差", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        f"同一测试集：{int(ensemble['windows']):,} 个窗口；灰线为 seed42/7/21/123，蓝线为预测均值。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, "fig5_1_ensemble_horizon.png")


def plot_residual_quantiles(ensemble: dict) -> None:
    quantiles = ["p50", "p90", "p95", "p99"]
    labels = ["P50", "P90", "P95", "P99"]
    x = np.arange(len(quantiles))
    fig, ax = plt.subplots(figsize=(9.2, 4.5))

    for index, item in enumerate(ensemble["singles"]):
        values = [float(item["residual_stats"][q]) for q in quantiles]
        ax.plot(
            x,
            values,
            color=COLORS["light_gray"],
            linewidth=1.3,
            marker="o",
            markersize=3.5,
            label="单 seed" if index == 0 else None,
        )
    ensemble_values = [float(ensemble["ensemble"]["residual_stats"][q]) for q in quantiles]
    ax.plot(
        x,
        ensemble_values,
        color=COLORS["blue"],
        linewidth=2.8,
        marker="D",
        markersize=6,
        label="4-seed ensemble",
        zorder=3,
    )
    for xi, value in zip(x, ensemble_values):
        ax.annotate(
            f"{value:.1f}",
            (xi, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=COLORS["blue"],
            fontweight="bold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("绝对残差 |y - y_pred|")
    ax.set_title("测试集绝对残差尾部分位数")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=9)
    fig.text(
        0.5,
        0.01,
        "分位数来自最多 5,000,000 个等距抽样残差；mean/std 使用全部测试残差精确计算。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_figure(fig, "fig5_2_residual_quantiles.png")


def plot_final_model_selection(
    metrics: pd.DataFrame,
    multiseed: pd.DataFrame,
    ensemble: dict,
) -> None:
    multiseed_means = multiseed.groupby("experiment")[METRICS].mean()
    candidate_specs = [
        ("LSTM 3-seed", multiseed_means.loc["lstm_gapaware"]),
        ("AGFormer T+S 3-seed", multiseed_means.loc["agformer_context"]),
        ("AGFormer gated s42", avg_row(metrics, "gate_context_seed42")),
        (
            "STF linear s42",
            avg_row(metrics, "spacetimeformer_original_aligned_no_future_time_linear_seed42"),
        ),
        (
            "STF lin+gate s42",
            avg_row(
                metrics,
                "spacetimeformer_original_aligned_no_future_time_linear_gated_seed42",
            ),
        ),
        ("AGFormer linear s42", avg_row(metrics, "agformer_gate_context_linear_seed42")),
        ("AGFormer linear ensemble", ensemble["ensemble"]["metrics"]["avg"]),
    ]
    labels = [label for label, _ in candidate_specs]
    numbered_labels = [f"[{index}] {label}" for index, label in enumerate(labels, start=1)]
    mae = np.array([float(row["MAE"]) for _, row in candidate_specs])
    rmse = np.array([float(row["RMSE"]) for _, row in candidate_specs])
    mape = np.array([float(row["MAPE"]) for _, row in candidate_specs])
    baseline_index = labels.index("AGFormer gated s42")
    ensemble_index = labels.index("AGFormer linear ensemble")
    delta_mae = (mae - mae[baseline_index]) / mae[baseline_index] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    scatter = ax.scatter(
        mae,
        rmse,
        c=mape,
        cmap="RdYlGn_r",
        s=[205 if i == ensemble_index else 130 for i in range(len(labels))],
        edgecolor=[COLORS["blue"] if i == ensemble_index else "white" for i in range(len(labels))],
        linewidth=[2.4 if i == ensemble_index else 0.9 for i in range(len(labels))],
        zorder=3,
    )
    for index, (x, y) in enumerate(zip(mae, rmse), start=1):
        ax.text(x, y, str(index), ha="center", va="center", fontsize=7.4, fontweight="bold")
    ax.set_xlabel("avg MAE ↓")
    ax.set_ylabel("avg RMSE ↓")
    ax.set_title("绝对误差与峰值误差权衡")
    ax.grid()
    ax.annotate(
        "更优方向",
        xy=(mae.min() - 0.05, rmse.min() - 0.08),
        xytext=(mae.min() + 0.75, rmse.min() + 2.0),
        arrowprops={"arrowstyle": "->", "color": COLORS["green"], "lw": 1.5},
        color=COLORS["green"],
        fontsize=8.5,
        fontweight="bold",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    colorbar.set_label("avg MAPE (%) ↓")

    ax = axes[1]
    y = np.arange(len(labels))
    colors = [
        COLORS["green"] if value < -1e-9 else COLORS["red"] if value > 1e-9 else COLORS["gray"]
        for value in delta_mae
    ]
    bars = ax.barh(y, delta_mae, color=colors, height=0.62)
    ax.axvline(0, color=COLORS["dark"], linewidth=1)
    ax.set_yticks(y, numbered_labels)
    ax.invert_yaxis()
    ax.set_xlabel("相对 AGFormer gated seed42 的 MAE 变化（%）")
    ax.set_title("主指标相对变化")
    ax.grid(axis="x")
    for bar, value in zip(bars, delta_mae):
        x = value + (0.25 if value >= 0 else -0.25)
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8,
        )
    ax.margins(x=0.18)

    fig.suptitle("Week5 预测残差基线的最终候选模型对比", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "左右图编号对应；全部使用相同 4,022 个 gap-aware 测试窗口；3-seed 表示指标均值而非预测 ensemble。",
        ha="center",
        fontsize=8.5,
        color="#5B6573",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, "fig6_1_final_model_selection.png")


REPORT_CSS = r"""
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @bottom-center {
    content: "Week 4 模型训练与预测基线选型报告 · " counter(page) " / " counter(pages);
    font-size: 8.5pt;
    color: #666;
  }
}
html { font-family: "Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans", sans-serif; color: #202124; }
body { font-size: 10.5pt; line-height: 1.58; }
h1 { font-size: 23pt; margin: 0 0 14pt; padding-bottom: 8pt; border-bottom: 2px solid #1f4e79; color: #17365d; }
h2 { font-size: 17pt; margin-top: 22pt; padding-bottom: 4pt; border-bottom: 1px solid #d0d7de; color: #1f4e79; page-break-after: avoid; }
h3 { font-size: 13pt; margin-top: 16pt; color: #333; page-break-after: avoid; }
p { margin: 7pt 0; }
ul, ol { margin-top: 5pt; margin-bottom: 8pt; padding-left: 22pt; }
li { margin: 2.5pt 0; }
code { font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace; background: #f6f8fa; padding: 1pt 3pt; border-radius: 3px; font-size: 9.2pt; }
pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 5px; padding: 8pt; overflow-wrap: break-word; white-space: pre-wrap; font-size: 9pt; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0 13pt; font-size: 9.2pt; page-break-inside: avoid; }
th, td { border: 1px solid #c8d1da; padding: 5pt 6pt; vertical-align: top; }
th { background: #eaf2f8; color: #17365d; font-weight: 700; }
tr:nth-child(even) td { background: #fbfcfd; }
blockquote { margin: 8pt 0; padding: 5pt 10pt; border-left: 3px solid #9ab6d3; background: #f7fbff; color: #333; }
hr { border: 0; border-top: 1px solid #d0d7de; margin: 16pt 0; }
strong { color: #111; }
img { display: block; width: 100%; height: auto; margin: 10pt auto 4pt; page-break-inside: avoid; }
.figure-caption { margin: 2pt 8pt 13pt; color: #59636e; font-size: 8.6pt; text-align: center; page-break-before: avoid; }
"""


def render_report() -> None:
    source = MARKDOWN_PATH.read_text(encoding="utf-8")
    body = markdown(
        source,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Week 4 模型训练与预测基线选型报告</title>
<style>{REPORT_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")
    HTML(string=document, base_url=str(REPORT_DIR)).write_pdf(PDF_PATH)


def main() -> None:
    configure_plotting()
    metrics, multiseed, ensemble, periodic = load_artifacts()
    plot_baseline_multiseed(metrics, multiseed)
    plot_spacetimeformer_ablation(metrics)
    plot_feature_ablation(metrics, periodic)
    plot_graph_weights()
    plot_linear_horizon(metrics)
    plot_linear_multiseed(ensemble)
    plot_ensemble_horizon(ensemble)
    plot_residual_quantiles(ensemble)
    plot_final_model_selection(metrics, multiseed, ensemble)
    render_report()
    figure_count = len(list(FIGURE_DIR.glob("*.png")))
    print(f"Generated {figure_count} figures in {FIGURE_DIR.relative_to(ROOT)}")
    print(f"Rendered {HTML_PATH.relative_to(ROOT)}")
    print(f"Rendered {PDF_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
