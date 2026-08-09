"""Generate the three mentor-facing Week7 reports as Markdown, HTML and PDF.

The script reads frozen Week7 evaluation, interpretability and performance
artifacts. It does not train models or rerun inference benchmarks.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from markdown import markdown
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_ROOT = REPORT_DIR / "week7_figures"
EVAL_DIR = ROOT / "outputs" / "week7_evaluation" / "final_system_v1"
XAI_DIR = ROOT / "outputs" / "week7_interpretability" / "final_system_v1"
PERF_DIR = ROOT / "outputs" / "week7_performance" / "final_system_v5"
ENSEMBLE_PATH = ROOT / "outputs" / "results" / "agformer_gate_context_linear_ensemble_4seed.json"

COLORS = {
    "blue": "#2563A6",
    "light_blue": "#8BB4DB",
    "green": "#2E8B57",
    "orange": "#D97706",
    "red": "#C2413B",
    "purple": "#7C5AA6",
    "gray": "#7C8796",
    "dark": "#25364A",
    "grid": "#DDE4EC",
}


REPORT_CSS = r"""
@page { size: A4; margin: 18mm 17mm 18mm 17mm; @bottom-center { content: counter(page); font-size: 8pt; color: #667085; } }
@font-face { font-family: CJK; src: local("Noto Sans CJK SC"), local("Noto Sans CJK JP"), local("Microsoft YaHei"); }
html { font-family: CJK, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; color: #1f2937; font-size: 10pt; line-height: 1.62; }
body { max-width: 100%; margin: 0; }
h1 { color: #173A5E; font-size: 24pt; line-height: 1.25; margin: 8mm 0 3mm; padding-bottom: 4mm; border-bottom: 2px solid #2563A6; }
h2 { color: #1F5A8A; font-size: 16pt; margin: 8mm 0 3mm; border-bottom: 1px solid #D8E3ED; padding-bottom: 1.5mm; page-break-after: avoid; }
h3 { color: #2D5E83; font-size: 12pt; margin: 5mm 0 2mm; page-break-after: avoid; }
p { margin: 2.2mm 0; text-align: justify; }
ul, ol { margin: 2mm 0 3mm 6mm; padding-left: 5mm; }
li { margin: 1mm 0; }
blockquote { margin: 4mm 0; padding: 3mm 4mm; border-left: 4px solid #2E8B57; background: #F2F8F5; color: #244536; }
table { width: 100%; border-collapse: collapse; margin: 3mm 0 5mm; font-size: 8.5pt; page-break-inside: avoid; }
th { background: #EAF2F8; color: #173A5E; font-weight: 700; }
th, td { border: 1px solid #CCD8E3; padding: 1.7mm 1.8mm; vertical-align: top; }
tr:nth-child(even) td { background: #FAFCFD; }
img { display: block; max-width: 100%; width: 100%; height: auto; margin: 4mm auto 1mm; page-break-inside: avoid; }
.figure-caption { color: #5D6875; font-size: 8.4pt; text-align: center; margin: 0 5mm 5mm; page-break-before: avoid; }
.meta { color: #5D6875; margin: 0 0 8mm; }
.decision { border: 1px solid #9EC5AE; background: #F4FAF6; padding: 3mm 4mm; margin: 4mm 0; }
code { background: #F1F4F7; padding: 0.2mm 1mm; border-radius: 2px; font-size: 8.5pt; }
strong { color: #142F4B; }
"""


def configure_plotting() -> None:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ]
    selected = None
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            selected = font_manager.FontProperties(fname=str(path)).get_name()
            break
    if selected is None:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        selected = next(
            (name for name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "DejaVu Sans"] if name in installed),
            "DejaVu Sans",
        )
    plt.rcParams.update({
        "font.family": selected,
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["dark"],
        "axes.labelcolor": COLORS["dark"],
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
    })


def require(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Week7 report inputs: {missing}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    selected = frame.loc[:, columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_evaluation_figures() -> None:
    target = FIGURE_ROOT / "evaluation"
    horizon = pd.read_csv(EVAL_DIR / "prediction_metrics_by_horizon.csv")
    slots = pd.read_csv(EVAL_DIR / "prediction_h1_by_time_slot.csv")
    nodes = pd.read_csv(EVAL_DIR / "prediction_h1_by_node.csv")
    timestep = pd.read_csv(EVAL_DIR / "anomaly_timestep_metrics.csv")
    events = pd.read_csv(EVAL_DIR / "anomaly_event_metrics.csv")
    summary = load_json(EVAL_DIR / "summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    x = np.arange(len(horizon))
    axes[0].plot(x, horizon["MAE"], marker="o", linewidth=2.2, color=COLORS["blue"], label="MAE")
    axes[0].plot(x, horizon["RMSE"], marker="s", linewidth=2.2, color=COLORS["red"], label="RMSE")
    axes[0].set_xticks(x, horizon["scope"])
    axes[0].set_xlabel("累计预测范围")
    axes[0].set_ylabel("流量误差")
    axes[0].set_title("预测时域扩大时误差变化")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    quantiles = summary["prediction"]["residual_quantiles"]
    labels = list(quantiles)
    values = [quantiles[key] for key in labels]
    bars = axes[1].bar(labels, values, color=[COLORS["light_blue"], COLORS["blue"], COLORS["orange"], COLORS["red"]])
    axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axes[1].set_ylabel("绝对残差 / flow units")
    axes[1].set_title("h1 节点残差分位数（含零流量格）")
    axes[1].grid(axis="y")
    fig.tight_layout()
    save_figure(fig, target / "prediction_horizon_residual.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    axes[0].plot(slots["slot"], slots["MAE"], color=COLORS["blue"], linewidth=2)
    worst = slots.loc[slots["MAE"].idxmax()]
    axes[0].scatter([worst["slot"]], [worst["MAE"]], color=COLORS["red"], zorder=3)
    axes[0].annotate(
        f"最高：{worst['local_time']} / {worst['MAE']:.2f}",
        (worst["slot"], worst["MAE"]), xytext=(12, -28), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["red"]}, fontsize=8,
    )
    axes[0].set_xticks(np.arange(0, 48, 6), [f"{hour:02d}:00" for hour in range(0, 24, 3)])
    axes[0].set_xlabel("北京时间")
    axes[0].set_ylabel("h1 MAE")
    axes[0].set_title("不同时段的短期预测误差")
    axes[0].grid(axis="y")
    axes[1].hist(nodes["MAE"], bins=30, color=COLORS["light_blue"], edgecolor="white")
    axes[1].axvline(nodes["MAE"].median(), color=COLORS["blue"], linestyle="--", label=f"中位数 {nodes['MAE'].median():.2f}")
    axes[1].axvline(nodes["MAE"].quantile(0.9), color=COLORS["orange"], linestyle="--", label=f"P90 {nodes['MAE'].quantile(0.9):.2f}")
    axes[1].set_xlabel("网格 h1 MAE")
    axes[1].set_ylabel("网格数量")
    axes[1].set_title("1024 个网格的误差分布")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y")
    fig.tight_layout()
    save_figure(fig, target / "prediction_temporal_spatial.png")

    methods = ["statistical", "reconstruction", "prediction", "fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, reference in zip(axes, ["level_A", "level_B"]):
        view = timestep.loc[timestep["label_reference"] == reference].set_index("method").loc[methods]
        x = np.arange(len(methods)); width = 0.25
        for offset, metric, color in [(-width, "precision", COLORS["blue"]), (0, "recall", COLORS["orange"]), (width, "f1", COLORS["green"])]:
            ax.bar(x + offset, view[metric], width, label=metric, color=color)
        ax.set_xticks(x, ["统计", "重构", "预测", "融合"])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{reference} timestep 一致性")
        ax.grid(axis="y")
    axes[0].set_ylabel("指标值")
    axes[1].legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    save_figure(fig, target / "anomaly_timestep_methods.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(events)); width = 0.24
    for offset, metric, color in [(-width, "event_precision", COLORS["blue"]), (0, "event_recall", COLORS["orange"]), (width, "event_f1", COLORS["green"])]:
        bars = ax.bar(x + offset, events[metric], width, label=metric.replace("event_", ""), color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x, events["label_reference"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("事件级指标")
    ax.set_title("融合异常检测的时空事件级评估")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y")
    fig.tight_layout()
    save_figure(fig, target / "anomaly_event_metrics.png")


def copy_interpretability_figures() -> None:
    target = FIGURE_ROOT / "interpretability"
    target.mkdir(parents=True, exist_ok=True)
    names = [
        "attention_temporal_heatmap.png", "attention_spatial_summary.png",
        "shap_feature_importance.png", "shap_group_importance.png",
        "anomaly_cause_distribution.png", "anomaly_weather_context.png",
    ]
    for name in names:
        shutil.copy2(XAI_DIR / name, target / name)


def plot_performance_figures() -> None:
    target = FIGURE_ROOT / "performance"
    variants = pd.read_csv(PERF_DIR / "pipeline_variants.csv")
    api = pd.read_csv(PERF_DIR / "api_comparison.csv")
    sweep = pd.read_csv(PERF_DIR / "batch_sweep.csv")

    order = ["baseline_fp32", "optimized_cold_fp32", "optimized_warm_fp32", "optimized_warm_fp16"]
    variants = variants.set_index("name").loc[order].reset_index()
    labels = ["Baseline\nFP32", "Cold\nFP32", "Warm\nFP32", "Warm\nFP16"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.1))
    metrics = [
        ("pipeline_seconds", "端到端耗时 / s", COLORS["blue"]),
        ("peak_rss_mib", "进程树峰值 RSS / MiB", COLORS["green"]),
        ("cuda_peak_allocated_mib", "CUDA 峰值分配 / MiB", COLORS["purple"]),
    ]
    for ax, (column, title, color) in zip(axes, metrics):
        bars = ax.bar(np.arange(len(variants)), variants[column], color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=7.5)
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=7.5)
        ax.set_title(title)
        ax.grid(axis="y")
    fig.tight_layout()
    save_figure(fig, target / "pipeline_time_memory.png")

    stages = ["model_inference_seconds", "forecast_inference_seconds", "statistical_seconds", "reconstruction_seconds"]
    stage_labels = ["AGFormer 推理", "独立 forecast", "统计检测", "重构检测"]
    base = variants.iloc[0]
    warm = variants.loc[variants["name"] == "optimized_warm_fp32"].iloc[0]
    x = np.arange(len(stages)); width = 0.35
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(x - width / 2, [base[s] for s in stages], width, label="Baseline FP32", color=COLORS["gray"])
    ax.bar(x + width / 2, [warm[s] for s in stages], width, label="Optimized warm FP32", color=COLORS["blue"])
    ax.set_xticks(x, stage_labels)
    ax.set_ylabel("耗时 / s")
    ax.set_title("关键阶段耗时变化")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    fig.tight_layout()
    save_figure(fig, target / "pipeline_stage_comparison.png")

    fig, ax1 = plt.subplots(figsize=(8.8, 4.1))
    ax1.plot(sweep["batch_size"], sweep["latency_seconds_median"], marker="o", color=COLORS["blue"], linewidth=2.2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("24 窗口推理中位耗时 / s", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax1.set_xticks(sweep["batch_size"])
    ax1.grid(axis="y")
    ax2 = ax1.twinx()
    ax2.plot(sweep["batch_size"], sweep["cuda_peak_allocated_mib_max"], marker="s", color=COLORS["red"], linewidth=2.0)
    ax2.set_ylabel("CUDA 峰值分配 / MiB", color=COLORS["red"])
    ax2.tick_params(axis="y", labelcolor=COLORS["red"])
    ax1.set_title("Micro-batch 延迟与显存权衡")
    fig.tight_layout()
    save_figure(fig, target / "batch_sweep.png")

    api = api.sort_values("mean_speedup")
    colors = [COLORS["green"] if value >= 1 else COLORS["orange"] for value in api["mean_speedup"]]
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    bars = ax.barh(api["endpoint"], api["mean_speedup"], color=colors)
    ax.axvline(1.0, color=COLORS["dark"], linewidth=1, linestyle="--")
    ax.bar_label(bars, fmt="%.2fx", padding=4, fontsize=8)
    ax.set_xlabel("平均延迟加速比（>1 表示更快）")
    ax.set_title("API 热路径优化前后对比")
    ax.grid(axis="x")
    fig.tight_layout()
    save_figure(fig, target / "api_speedup.png")


def build_evaluation_report() -> str:
    summary = load_json(EVAL_DIR / "summary.json")
    ensemble = load_json(ENSEMBLE_PATH)
    horizons = pd.read_csv(EVAL_DIR / "prediction_metrics_by_horizon.csv")
    slots = pd.read_csv(EVAL_DIR / "prediction_h1_by_time_slot.csv")
    nodes = pd.read_csv(EVAL_DIR / "prediction_h1_by_node.csv")
    timestep = pd.read_csv(EVAL_DIR / "anomaly_timestep_metrics.csv")
    events = pd.read_csv(EVAL_DIR / "anomaly_event_metrics.csv")
    singles = pd.DataFrame([
        {"seed": row["seed"], **row["metrics"]["avg"]} for row in ensemble["singles"]
    ])
    fusion = timestep.loc[timestep["method"] == "fusion", [
        "label_reference", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"
    ]]
    worst_slots = slots.nlargest(5, "MAE")[["local_time", "MAE", "RMSE", "MAPE"]]
    worst_nodes = nodes.nlargest(10, "MAE")[["grid_id", "MAE", "RMSE", "MAPE", "valid_count"]]
    h1 = horizons.iloc[0]
    avg = horizons.iloc[-1]
    best_single = singles.loc[singles["MAE"].idxmin()]
    gain = (best_single["MAE"] - avg["MAE"]) / best_single["MAE"] * 100
    return f"""# Week7 最终模型性能评估报告

<div class="meta">TaxiBJ · AGFormer gated+linear 四随机种子集成 · 生成日期 {datetime.now():%Y-%m-%d}</div>

> **结论摘要：** 最终模型在 4,022 个严格 gap-aware 测试窗口上取得 h1 MAE **{h1['MAE']:.4f}**、24 小时平均 MAE **{avg['MAE']:.4f}**。四 seed 均值集成相对最佳单模型的 24 小时平均 MAE 再降低 **{gain:.2f}%**。异常融合在 Level-A 弱标签上 timestep F1 为 **{fusion.iloc[0]['f1']:.4f}**、事件 F1 为 **{events.iloc[0]['event_f1']:.4f}**；这些异常指标只表示弱规则一致性，不是真实业务准确率。

## 1. 评估目标与固定口径

- 城市与数据：TaxiBJ，32×32 网格，共 1,024 个节点；
- 输入/输出：过去 48 个半小时步预测未来 48 步，即 24 小时到 24 小时；
- 测试窗口：4,022 个，使用真实时间戳并过滤跨缺口窗口；
- 最终模型：AGFormer，`d_model=32`、4 个注意力头、2 层，输入含流量、时间上下文及 33 维无泄漏静态节点特征，并使用 linear residual head；
- 集成成员：seed 42、7、21、123，预测按算术平均融合；
- 预测指标：对真实值非零位置计算 MAE/RMSE/MAPE，MAPE 使用 `eps=1.0`；
- 模型选择在验证集完成，本报告只汇报固定测试集结果。

## 2. 最终预测精度

{markdown_table(horizons, ['scope', 'forecast_hours', 'MAE', 'RMSE', 'MAPE', 'test_windows'])}

`h3/h6/h12` 是前 3/6/12 步累计指标，而不是单独第 3/6/12 步；`avg` 汇总全部 48 个预测步。

![预测时域与残差](week7_figures/evaluation/prediction_horizon_residual.png)
<div class="figure-caption">图1　随着预测时域扩大，误差逐步上升；右图展示 h1 节点绝对残差的长尾。</div>

四个单模型的 24 小时平均结果如下。最佳单模型 seed42 的 MAE 为 {best_single['MAE']:.4f}，ensemble 为 {avg['MAE']:.4f}，说明多 seed 平均主要降低了随机初始化带来的方差。

{markdown_table(singles, ['seed', 'MAE', 'RMSE', 'MAPE'])}

## 3. 时间与空间误差结构

![时段与网格误差](week7_figures/evaluation/prediction_temporal_spatial.png)
<div class="figure-caption">图2　左：h1 MAE 的北京时间分布；右：1,024 个网格的 h1 MAE 分布。</div>

网格 MAE 中位数为 **{nodes['MAE'].median():.4f}**，P90 为 **{nodes['MAE'].quantile(0.9):.4f}**。最困难网格为 grid {int(nodes.loc[nodes['MAE'].idxmax(), 'grid_id'])}，MAE **{nodes['MAE'].max():.4f}**。时段上 16:00 的误差最高，明显高于相邻时段，提示后续应针对突变日或特殊日期检查该时段的样本构成。

误差最高的时段：

{markdown_table(worst_slots, ['local_time', 'MAE', 'RMSE', 'MAPE'])}

误差最高的网格：

{markdown_table(worst_nodes, ['grid_id', 'MAE', 'RMSE', 'MAPE', 'valid_count'])}

## 4. 异常检测 timestep 评估

![异常方法比较](week7_figures/evaluation/anomaly_timestep_methods.png)
<div class="figure-caption">图3　统计、重构、预测误差和最终融合对两套弱标签的 timestep precision/recall/F1。</div>

最终融合结果：

{markdown_table(fusion, ['label_reference', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'])}

Level-A 的 F1 更高，Level-B 的 recall 和 balanced accuracy 更高但 precision 较低。融合结果优于任一单类检测器的 F1，说明三类证据具有互补性；但统计检测同时参与弱标签构造和最终融合（权重 0.70），因此这些数值存在规则相关性。

## 5. 时空事件级评估

![事件级评估](week7_figures/evaluation/anomaly_event_metrics.png)
<div class="figure-caption">图4　事件匹配同时要求时间重叠以及空间 IoU 或质心距离满足阈值。</div>

{markdown_table(events, ['label_reference', 'event_precision', 'event_recall', 'event_f1', 'true_event_count', 'pred_event_count', 'mean_detection_delay_hours'])}

最终双向融合产生 131 个事件。Level-A 参考下事件 F1 为 {events.iloc[0]['event_f1']:.4f}；Level-B 参考更严格、事件数更少，recall 达 {events.iloc[1]['event_recall']:.4f}，但 precision 下降。匹配采用一对一最大基数匹配，并在非连续 30 分钟时间缺口处强制断开事件。

## 6. 综合判断

1. 最终 AGFormer ensemble 可作为 Week5/Week6 预测误差检测和在线预测服务的固定模型；
2. h1 表现明显优于长时域，Web 端展示 24 小时预测时应同时提示不确定性随时域增长；
3. 网格和时段误差具有显著异质性，grid 157 和 16:00 时段值得后续专项诊断；
4. 异常融合提升了弱标签一致性，但 TaxiBJ 无权威异常真值，不能将 precision/recall 解释为现实事件检测准确率。

## 7. 评估边界与复现

- 预测指标是固定测试集结果，不代表其他城市或未来年份的泛化能力；
- h1 残差分位数图包含零流量格，仅用于描述检测阈值面对的残差分布，不应与 masked MAE 混用；
- 异常 weak label 来自项目统计规则，且最终融合包含统计分支；
- 原始结果位于 `outputs/week7_evaluation/final_system_v1/`，配置为 `configs/week7_evaluation.yaml`；
- 输入结果哈希、依赖版本和环境信息记录在该目录的 `summary.json`。
"""


def build_interpretability_report() -> str:
    summary = load_json(XAI_DIR / "summary.json")
    lag = pd.read_csv(XAI_DIR / "attention_last_query_by_lag.csv")
    feature = pd.read_csv(XAI_DIR / "shap_feature_importance.csv")
    groups = pd.read_csv(XAI_DIR / "shap_group_importance.csv")
    causes = pd.read_csv(XAI_DIR / "anomaly_cause_summary.csv")
    top_lag = lag.sort_values(["layer", "attention_weight"], ascending=[True, False]).groupby("layer").head(5)
    top_features = feature.sort_values("mean_abs_shap", ascending=False).head(15)
    attn = summary["attention"]
    shap = summary["shap"]
    cause = summary["anomaly_causes"]
    history_share = float(groups.loc[groups["feature_group"].isin(["inflow_history", "outflow_history"]), "importance_share"].sum())
    return f"""# Week7 模型可解释性分析报告

<div class="meta">最终 AGFormer 四 seed ensemble · 注意力、TreeSHAP 与异常上下文证据 · 生成日期 {datetime.now():%Y-%m-%d}</div>

> **结论摘要：** 跨四个 checkpoint、四个日期和各层聚合后的非线性 Transformer-GCN 平均时间注意力画像较为分散，其归一化熵为 **{attn['mean_temporal_attention_entropy']:.4f}**；TreeSHAP 代理显示历史流量合计贡献占 **{history_share:.2%}**，日历上下文占 {float(groups.loc[groups['feature_group']=='time_calendar', 'importance_share'].iloc[0]):.2%}。131 个异常事件中有 82 个找到天气或日历候选证据，49 个没有外部上下文匹配。所有结论均为模型行为或相关证据，不是因果证明。

## 1. 解释对象与三类方法

| 分析 | 解释对象 | 方法 | 主要边界 |
| --- | --- | --- | --- |
| 时间/空间注意力 | 非线性 Transformer-GCN 分支 | 原生 head-averaged attention 与 adaptive adjacency | 不覆盖 parallel linear residual head |
| 特征重要性 | 完整四 seed ensemble 的 h1 输出 | 时间留出的 ExtraTrees 行为代理 + TreeSHAP | 解释代理近似，不是直接模型梯度 |
| 异常原因 | 131 个融合异常事件 | 天气、日历、形态和可选外部事件匹配 | 候选证据，不是校准概率或因果结论 |

模型输入规则与最终评估一致：TaxiBJ、4,022 个 gap-aware 测试窗口、48 步历史流量、7 个时间/日历通道和 33 维无泄漏静态节点特征。

## 2. 时间注意力分析

注意力抽样覆盖测试集首、中、末四个日期，窗口索引为 {attn['sample_window_indices']}。这是一组定性覆盖样本，不是全测试集的统计估计。

![时间注意力](week7_figures/interpretability/attention_temporal_heatmap.png)
<div class="figure-caption">图1　两层 AGFormer 非线性分支的平均时间注意力矩阵及最后 query 的 lag profile。</div>

最后 query 权重最高的滞后：

{markdown_table(top_lag, ['layer', 'lag_steps', 'lag_hours', 'attention_weight'])}

- 最近 3 小时注意力质量占比：**{attn['mean_recent_attention_mass']:.4f}**；
- 平均时间注意力归一化熵：**{attn['mean_temporal_attention_entropy']:.4f}**。

接近 1 的熵说明**聚合后的平均 attention profile** 未集中在少数时间步；不同模型或窗口各自集中的注意力也可能在平均后呈现分散，因此不能将该结论外推到每个样本。第一层的聚合画像较关注约 9.5–13 小时前的输入，第二层同时关注最近时刻与 8.5–9.5 小时前信息。由于最终模型存在 linear residual head，不能据此断言最终预测主要由这些滞后驱动。

## 3. 空间自适应邻接分析

![空间注意力](week7_figures/interpretability/attention_spatial_summary.png)
<div class="figure-caption">图2　非线性分支 adaptive adjacency 的节点集中度、熵及网络摘要。</div>

- adaptive adjacency 平均行熵：**{attn['mean_adaptive_row_entropy']:.4f}**；
- 每个目标节点 top-k 权重平均集中度：**{attn['mean_adaptive_top_k_concentration']:.4f}**。

该图显示模型学习到了非均匀节点关系，但只覆盖 learned adaptive graph。预定义 spatial GCN 和 parallel linear residual head 是独立路径，因此这里不是完整空间贡献分解。

## 4. TreeSHAP 全局特征重要性

代理模型使用前 80% 测试窗口拟合、后 20% 时间留出验证，验证 R² 为 **{shap['validation_r2']:.4f}**、MAE 为 **{shap['validation_mae_flow_units']:.4f}** flow units。SHAP 平均加法误差为 {shap['mean_additivity_abs_error']:.8f}，说明 TreeSHAP 分解本身数值一致。

![SHAP 特征](week7_figures/interpretability/shap_feature_importance.png)
<div class="figure-caption">图3　完整 ensemble h1 行为代理的单特征 mean |SHAP| 排名。</div>

{markdown_table(top_features, ['importance_rank', 'feature', 'feature_group', 'mean_abs_shap', 'mean_signed_shap'])}

排名靠前的特征集中在最近 1–4 个半小时步和输入窗口末端的 lag 46–48，说明模型行为同时依赖短期惯性和接近前一日同一位置的历史边界信息。

![SHAP 分组](week7_figures/interpretability/shap_group_importance.png)
<div class="figure-caption">图4　按流量、日历与静态节点特征组汇总的 mean |SHAP| 占比。</div>

{markdown_table(groups, ['feature_group', 'feature_count', 'mean_abs_shap', 'importance_share'])}

历史 inflow/outflow 合计占 {history_share:.2%}，日历上下文占 {float(groups.loc[groups['feature_group']=='time_calendar', 'importance_share'].iloc[0]):.2%}。静态特征单组占比较小，但这不等于静态特征“无用”：代理输入压缩、门控交互以及模型对不同节点的条件化效应都会降低简单全局 mean |SHAP|。

## 5. 异常上下文证据

![异常上下文](week7_figures/interpretability/anomaly_cause_distribution.png)
<div class="figure-caption">图5　131 个融合事件的首要上下文候选分布。</div>

{markdown_table(causes, ['primary_context_candidate', 'primary_evidence_type', 'event_count', 'mean_max_score'])}

![天气上下文](week7_figures/interpretability/anomaly_weather_context.png)
<div class="figure-caption">图6　异常事件对应的温度、风速与检测分数分布。</div>

- 天气覆盖率：{cause['weather_coverage_rate']:.2%}；
- 节假日事件：{cause['holiday_event_count']} 个；周末事件：{cause['weekend_event_count']} 个；调休工作日：{cause['makeup_workday_event_count']} 个；
- 未配置外部新闻/活动表，因此匹配新闻事件数为 0；
- `event_type` 只保留为检测器形态标签，不冒充现实原因。

## 6. 综合解释

1. 完整模型仍以历史流量为主，linear residual head 与最近时刻特征共同强化了短期惯性；
2. 非线性注意力分支呈高熵、分散式聚合，更像对多个时间位置的补充校正，而不是单一关键 lag 选择器；
3. learned adaptive graph 存在节点选择性，但不能替代预定义空间图的解释；
4. 天气和日历能为部分异常提供合理上下文，49 个事件没有外部匹配，报告应保持“未知”而不是强行分类原因。

## 7. 解释边界与复现

- Attention 不等于特征贡献，更不等于因果关系；
- TreeSHAP 解释的是高拟合度代理，必须与 R²/MAE 一起阅读；
- 代理的日历特征只取上下文最后时刻，流量特征保留完整 48 步；
- 当前无新闻表，任何现实事件归因均未得到外部来源验证；
- 原始 CSV、NPZ、图片、模型和 manifest 位于 `outputs/week7_interpretability/final_system_v1/`，正式配置为 `configs/week7_interpretability.yaml`。
"""


def build_performance_report() -> str:
    summary = load_json(PERF_DIR / "summary.json")
    variants = pd.read_csv(PERF_DIR / "pipeline_variants.csv")
    api = pd.read_csv(PERF_DIR / "api_comparison.csv")
    sweep = pd.read_csv(PERF_DIR / "batch_sweep.csv")
    columns = [
        "name", "pipeline_seconds", "model_inference_seconds", "forecast_inference_seconds",
        "statistical_seconds", "peak_rss_mib", "cuda_peak_allocated_mib",
    ]
    base = variants.loc[variants["name"] == "baseline_fp32"].iloc[0]
    warm = variants.loc[variants["name"] == "optimized_warm_fp32"].iloc[0]
    speedup = base["pipeline_seconds"] / warm["pipeline_seconds"]
    rss_reduction = 1 - warm["peak_rss_mib"] / base["peak_rss_mib"]
    cuda_reduction = 1 - warm["cuda_peak_allocated_mib"] / base["cuda_peak_allocated_mib"]
    fp16 = summary["fp16_validation"]
    quant = summary["quantization"]
    rec = summary["recommendation"]
    api_view = api.copy()
    api_view["result"] = np.where(api_view["mean_speedup"] >= 1, "faster", "slower/noise")
    return f"""# Week7 系统性能测试报告

<div class="meta">Week6 生产流水线与 FastAPI 服务 · 固定 TaxiBJ 24 窗口基准 · 生成日期 {datetime.now():%Y-%m-%d}</div>

> **结论摘要：** 生产可部署的 FP32 执行优化将暖启动流水线从 **{base['pipeline_seconds']:.3f}s** 降至 **{warm['pipeline_seconds']:.3f}s**，端到端加速 **{speedup:.2f}×**；进程树 RSS 汇总下降 {rss_reduction:.1%}，CUDA 峰值分配下降 {cuda_reduction:.1%}。FP16 精度门槛通过但本轮速度门槛未通过，动态 INT8 只保留为 CPU fallback 候选，生产继续使用 FP32。

## 1. 测试对象与控制变量

- 固定输入：测试集末尾 24 个连续有效窗口；
- 固定模型：AGFormer gated+linear 四 seed ensemble；
- 固定输出：h1 异常残差、48 步同 origin 预测、三类异常分数、融合掩码与事件；
- baseline：batch=8、2 个 DataLoader workers、分别执行 h1 与 forecast、每次重拟合统计量；
- optimized：batch=8、worker=0、共享 full-sequence 推理、统计缓存、关闭未使用注意力矩阵物化；
- 每个 pipeline variant 单次顺序执行；API 每端点 2 次 warm-up + 100 次本机顺序请求。

## 2. 端到端耗时与资源

{markdown_table(variants, columns)}

![流水线资源](week7_figures/performance/pipeline_time_memory.png)
<div class="figure-caption">图1　各执行策略的端到端耗时、进程树 RSS 汇总和 CUDA 峰值分配。</div>

暖启动 FP32 是正式生产候选：端到端加速 {speedup:.2f}×，同时减少 worker 带来的进程树 RSS。冷启动 FP32 仍需构建统计缓存，因此只获得较小收益。

## 3. 瓶颈定位与代码优化

![阶段对比](week7_figures/performance/pipeline_stage_comparison.png)
<div class="figure-caption">图2　baseline 与 optimized warm FP32 的关键阶段耗时。</div>

| 瓶颈 | 修改 | 结果 |
| --- | --- | --- |
| h1 与 Web forecast 重复加载四个 checkpoint 并重复前向 | 24 个窗口一次输出完整序列，h1 与最新 forecast 共享 | 独立 forecast 阶段由 {base['forecast_inference_seconds']:.3f}s 降至近 0 |
| 每次实时调用都拟合 inflow/outflow slot statistics | 以数据 SHA256、train_len、时区与规则参数构建指纹缓存 | 统计阶段由 {base['statistical_seconds']:.3f}s 降至 {warm['statistical_seconds']:.3f}s |
| MultiheadAttention 默认返回但调用方丢弃权重 | 普通预测 `need_weights=False`，XAI 时显式开启 | FP32 fusion mask 保持完全一致 |
| 24 窗口仍启动 2 个 DataLoader workers | 实测后改为 worker=0 | 减少进程树 RSS 与启动开销 |
| API 每次重新转换事件时间并循环映射坐标 | mtime-aware 派生缓存 + vectorized merge | 历史事件/统计端点明显加速 |

## 4. Batch size 扫描

![Batch sweep](week7_figures/performance/batch_sweep.png)
<div class="figure-caption">图3　24 个窗口、四模型 ensemble 在不同 micro-batch 下的延迟与 CUDA 显存。</div>

{markdown_table(sweep, ['batch_size', 'latency_seconds_median', 'latency_seconds_min', 'latency_seconds_max', 'cuda_peak_allocated_mib_max', 'max_abs_diff_vs_first_batch'])}

batch=8 的中位延迟最低；batch=24 将峰值显存推至约 2,074 MiB，速度反而变慢。因此正式配置选择 batch=8，而不是盲目使用最大 batch。

## 5. 数值一致性门禁

- FP32 scores 的 fusion mask 不一致率：**{summary['comparisons']['optimized_fp32']['fusion_mask_mismatch_rate']}**；
- FP32 h1 `pred_flow` 最大绝对差：**{summary['comparisons']['optimized_fp32']['numeric']['pred_flow']['max_abs']:.8f}**；
- forecast 的 city/member/origin/timestamps 完全一致，target indices/step 完全一致；
- 48×1024 forecast 最大绝对差：**{summary['comparisons']['optimized_forecast_fp32']['numeric']['pred_flow']['max_abs']:.8f}**，低于 0.001 门槛；
- 正式判断：`deploy_fp32_execution_optimizations={rec['deploy_fp32_execution_optimizations']}`。

## 6. FP16 与动态 INT8

FP16 对四个时期的 256 个窗口进行了补充核验：masked MAE 从 {fp16['fp32_masked_mae']:.6f} 变为 {fp16['fp16_masked_mae']:.6f}，变化 {fp16['masked_mae_delta']:.6f}；融合掩码不一致率为 {fp16['fusion_mask_mismatch_rate']}。质量门槛通过，但同轮模型推理加速比仅 **{rec['fp16_speedup']:.4f}×**，低于 1.05，因此 `deploy_fp16=False`。

CPU dynamic INT8 microbenchmark 的速度比为 {quant['speedup']:.3f}×，state_dict 仅缩小 {quant['size_reduction']:.2%}，且只覆盖单模型、单窗口和 Linear 动态量化。当前生产推理在 CUDA 上运行，因此不部署 INT8，只把它记录为 CPU fallback 候选。

## 7. API 响应性能

![API speedup](week7_figures/performance/api_speedup.png)
<div class="figure-caption">图4　优化后平均延迟相对 Week7 初始固定 API 基线的比值。</div>

{markdown_table(api_view, ['endpoint', 'mean_ms_baseline', 'mean_ms_optimized', 'p95_ms_baseline', 'p95_ms_optimized', 'mean_speedup', 'result'])}

历史事件查询和统计端点分别约加速 {float(api.loc[api['endpoint']=='historical_anomalies_200', 'mean_speedup'].iloc[0]):.2f}× 与 {float(api.loc[api['endpoint']=='historical_statistics', 'mean_speedup'].iloc[0]):.2f}×。热力图和 48×1024 forecast 的完整 JSON 响应本轮略慢，说明它们主要受 Python 对象构造、JSON 序列化和 gzip 压缩限制；不应宣称 API 全面加速。

优化后 API 冷启动 {summary['api']['resources']['startup_seconds']:.3f}s，空闲 RSS {summary['api']['resources']['idle_rss_mib']:.2f} MiB，基准期间峰值 RSS {summary['api']['resources']['peak_rss_mib']:.2f} MiB。

## 8. 测量边界与后续加固

1. pipeline variant 各运行一次且顺序固定，小差异可能受系统调度影响；
2. `peak_rss_mib` 是进程树 RSS 求和，共享页可能重复计算，不等同于物理 PSS；
3. API 是单客户端 loopback，不包含公网、反向代理、浏览器渲染和并发压力；
4. 统计缓存按当前单进程顺序运行设计；并行 writer 场景应增加 file lock 或唯一临时文件；
5. FP16/INT8 都由显式质量与速度门槛控制，本轮生产配置保持 FP32；
6. 正式结果位于 `outputs/week7_performance/final_system_v5/`，配置为 `configs/week7_performance.yaml`。
"""


def render(name: str, title: str, source: str) -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"{name}.md"
    html_path = REPORT_DIR / f"{name}.html"
    pdf_path = REPORT_DIR / f"{name}.pdf"
    md_path.write_text(source, encoding="utf-8")
    body = markdown(source, extensions=["extra", "tables", "fenced_code", "sane_lists"], output_format="html5")
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title><style>{REPORT_CSS}</style></head><body>{body}</body></html>"""
    html_path.write_text(document, encoding="utf-8")
    rendered = HTML(string=document, base_url=str(REPORT_DIR)).render()
    rendered.write_pdf(pdf_path)
    return len(rendered.pages)


def main() -> None:
    required = [
        EVAL_DIR / "summary.json", EVAL_DIR / "prediction_metrics_by_horizon.csv",
        EVAL_DIR / "prediction_h1_by_time_slot.csv", EVAL_DIR / "prediction_h1_by_node.csv",
        EVAL_DIR / "anomaly_timestep_metrics.csv", EVAL_DIR / "anomaly_event_metrics.csv",
        XAI_DIR / "summary.json", XAI_DIR / "shap_feature_importance.csv",
        XAI_DIR / "shap_group_importance.csv", XAI_DIR / "attention_last_query_by_lag.csv",
        XAI_DIR / "anomaly_cause_summary.csv", PERF_DIR / "summary.json",
        PERF_DIR / "pipeline_variants.csv", PERF_DIR / "api_comparison.csv",
        PERF_DIR / "batch_sweep.csv", ENSEMBLE_PATH,
    ]
    require(required)
    configure_plotting()
    plot_evaluation_figures()
    copy_interpretability_figures()
    plot_performance_figures()
    reports = [
        ("week7_final_model_evaluation_report", "Week7 最终模型性能评估报告", build_evaluation_report()),
        ("week7_model_interpretability_report", "Week7 模型可解释性分析报告", build_interpretability_report()),
        ("week7_system_performance_report", "Week7 系统性能测试报告", build_performance_report()),
    ]
    for name, title, source in reports:
        pages = render(name, title, source)
        print(f"Rendered outputs/reports/{name}.pdf ({pages} pages)")
    print(f"Figures: {len(list(FIGURE_ROOT.rglob('*.png')))}")


if __name__ == "__main__":
    main()
