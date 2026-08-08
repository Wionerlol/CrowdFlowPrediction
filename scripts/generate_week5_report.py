"""Generate Week 5 evaluation figures and render the report to HTML/PDF.

All figures are derived from compact, versioned Week 5 artifacts.  The script
does not read ``scores.npz`` and never launches model training.

Run from any directory with:

    .venv/bin/python scripts/generate_week5_report.py
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
FIGURE_DIR = REPORT_DIR / "week5_figures"
ANOMALY_DIR = ROOT / "outputs" / "week5_anomaly"

SUMMARY_PATH = ANOMALY_DIR / "summary.json"
FUSION_TUNING_PATH = ANOMALY_DIR / "tuned_fusion" / "fusion_tuning_summary.json"
TIMESTEP_METRICS_PATH = ANOMALY_DIR / "evaluation" / "directional_timestep_metrics.csv"
EVENT_METRICS_PATH = ANOMALY_DIR / "evaluation" / "event_level_metrics.csv"
EVENTS_PATH = ANOMALY_DIR / "events.csv"
EXPLAINABILITY_PATH = ANOMALY_DIR / "event_explainability.csv"
AUDIT_PATH = ANOMALY_DIR / "audit" / "event_audit_sample_50.csv"
MARKDOWN_PATH = REPORT_DIR / "week5_report.md"
HTML_PATH = REPORT_DIR / "week5_report.html"
PDF_PATH = REPORT_DIR / "week5_report.pdf"

COLORS = {
    "blue": "#2563A6",
    "light_blue": "#8BB4DB",
    "green": "#2E8B57",
    "orange": "#D97706",
    "red": "#C2413B",
    "purple": "#7C5AA6",
    "gray": "#8A94A3",
    "light_gray": "#D7DEE7",
    "dark": "#25364A",
}
METHOD_LABELS = {
    "statistical": "Statistical",
    "reconstruction": "Reconstruction",
    "prediction": "Prediction",
    "fusion": "Fusion",
}
TYPE_LABELS = {
    "mass_gathering": "Mass gathering",
    "traffic_jam": "Traffic jam",
    "sudden_incident": "Sudden incident",
}


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

    if selected is None:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        preferred = [
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "Microsoft YaHei",
            "WenQuanYi Micro Hei",
            "DejaVu Sans",
        ]
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


def load_artifacts() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        SUMMARY_PATH,
        FUSION_TUNING_PATH,
        TIMESTEP_METRICS_PATH,
        EVENT_METRICS_PATH,
        EVENTS_PATH,
        EXPLAINABILITY_PATH,
        AUDIT_PATH,
        MARKDOWN_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Week 5 artifacts: {missing}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    tuning = json.loads(FUSION_TUNING_PATH.read_text(encoding="utf-8"))
    timestep = pd.read_csv(TIMESTEP_METRICS_PATH)
    event = pd.read_csv(EVENT_METRICS_PATH)
    events = pd.read_csv(EVENTS_PATH)
    explainability = pd.read_csv(EXPLAINABILITY_PATH)
    audit = pd.read_csv(AUDIT_PATH)
    return summary, tuning, timestep, event, events, explainability, audit


def validate_artifacts(
    summary: dict,
    tuning: dict,
    timestep: pd.DataFrame,
    event: pd.DataFrame,
    events: pd.DataFrame,
    explainability: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    if len(events) != summary["events"]["count"]:
        raise ValueError("events.csv count does not match summary.json")
    if len(explainability) != len(events):
        raise ValueError("event_explainability.csv must contain one row per event")
    selected = tuning["variants"]["high_precision_ge_0_70"]
    if selected["weights"] != summary["fusion"]["weights"]:
        raise ValueError("Selected fusion weights differ between tuning and main summary")
    if selected["decision_threshold"] != summary["fusion"]["decision_threshold"]:
        raise ValueError("Selected fusion threshold differs between tuning and main summary")
    if len(timestep.loc[timestep["method"] == "fusion"]) != 6:
        raise ValueError("Expected six directional timestep rows for fusion")
    if len(event.loc[event["method"] == "fusion"]) != 6:
        raise ValueError("Expected six directional event rows for fusion")
    expected_audit_groups = {
        "high_confidence": 20,
        "borderline": 15,
        "false_positive_by_weak_label": 15,
    }
    if audit["sample_group"].value_counts().to_dict() != expected_audit_groups:
        raise ValueError("Audit sample is not the documented 20/15/15 split")


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, fmt: str = ".3f", offset: int = 4) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            format(value, fmt),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=COLORS["dark"],
        )


def plot_fusion_selection(tuning: dict) -> None:
    variants = [
        ("best_f1_unconstrained", "Best validation F1"),
        ("high_precision_ge_0_70", "Selected high precision"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1.1, 1.1, 0.8]})
    x = np.arange(len(variants))
    width = 0.23

    ax = axes[0]
    for idx, metric in enumerate(["precision", "recall", "f1"]):
        values = [tuning["variants"][key]["selection_metrics_val"][metric] for key, _ in variants]
        bars = ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            label=metric.capitalize(),
            color=[COLORS["blue"], COLORS["orange"], COLORS["green"]][idx],
        )
        annotate_bars(ax, bars)
    ax.set_xticks(x, [label.replace(" ", "\n") for _, label in variants])
    ax.set_ylim(0, 0.86)
    ax.set_title("Validation Level-A")
    ax.set_ylabel("Metric")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    ax = axes[1]
    for idx, level in enumerate(["level_A", "level_B"]):
        values = [tuning["variants"][key]["test_metrics"][level]["f1"] for key, _ in variants]
        bars = ax.bar(
            x + (idx - 0.5) * 0.31,
            values,
            0.31,
            label=level.replace("level_", "Level-"),
            color=[COLORS["light_blue"], COLORS["purple"]][idx],
        )
        annotate_bars(ax, bars)
    ax.set_xticks(x, [label.replace(" ", "\n") for _, label in variants])
    ax.set_ylim(0, 0.62)
    ax.set_title("Test F1")
    ax.grid(axis="y")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    counts = [tuning["variants"][key]["events"]["count"] for key, _ in variants]
    bars = ax.bar(x, counts, color=[COLORS["gray"], COLORS["blue"]], width=0.58)
    annotate_bars(ax, bars, fmt=".0f")
    ax.set_xticks(x, [label.replace(" ", "\n") for _, label in variants])
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_ylabel("Detected events")
    ax.set_title("Review workload")
    ax.grid(axis="y")

    fig.suptitle("融合参数选择：验证集目标、测试集表现与事件规模", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "fig4_1_fusion_selection.png")


def plot_method_comparison(summary: dict) -> None:
    methods = ["statistical", "reconstruction", "prediction", "fusion"]
    labels = [METHOD_LABELS[item] for item in methods]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    x = np.arange(len(methods))
    width = 0.23
    metric_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

    for ax, level, title in zip(axes[:2], ["level_A", "level_B"], ["Level-A", "Level-B"]):
        for idx, metric in enumerate(["precision", "recall", "f1"]):
            values = [summary["metrics_vs_weak_labels"][method][level][metric] for method in methods]
            bars = ax.bar(x + (idx - 1) * width, values, width, label=metric.capitalize(), color=metric_colors[idx])
            if metric == "f1":
                annotate_bars(ax, bars)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{title} timestep metrics")
        ax.grid(axis="y")
    axes[0].set_ylabel("Metric")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper center")

    rates = [summary["method_time_positive_rates"][method] * 100 for method in methods]
    weak_a = summary["weak_label_positive_rates"]["level_A"] * 100
    weak_b = summary["weak_label_positive_rates"]["level_B"] * 100
    bars = axes[2].bar(x, rates, color=[COLORS["gray"], COLORS["purple"], COLORS["orange"], COLORS["blue"]])
    annotate_bars(axes[2], bars, fmt=".1f")
    axes[2].axhline(weak_a, color=COLORS["green"], linestyle="--", label=f"Weak A {weak_a:.1f}%")
    axes[2].axhline(weak_b, color=COLORS["red"], linestyle=":", label=f"Weak B {weak_b:.1f}%")
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylabel("Positive windows (%)")
    axes[2].set_title("Detection coverage")
    axes[2].grid(axis="y")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("三类异常检测方法与融合结果对比", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "fig5_1_method_comparison.png")


def plot_direction_heatmap(timestep: pd.DataFrame) -> None:
    rows = timestep.loc[timestep["method"] == "fusion"].copy()
    directions = ["high", "low", "bidirectional"]
    levels = ["level_A", "level_B"]
    metrics = ["precision", "recall", "f1"]
    matrix = np.array(
        [
            [
                float(rows.loc[(rows["direction"] == direction) & (rows["label_level"] == level), metric].iloc[0])
                for level in levels
                for metric in metrics
            ]
            for direction in directions
        ]
    )

    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] >= 0.62 else COLORS["dark"],
                fontsize=9,
                fontweight="bold" if col in {2, 5} else "normal",
            )
    ax.set_xticks(
        np.arange(6),
        ["A Precision", "A Recall", "A F1", "B Precision", "B Recall", "B F1"],
    )
    ax.set_yticks(np.arange(3), ["High flow", "Low flow", "Bidirectional"])
    ax.axvline(2.5, color="white", linewidth=2.5)
    ax.set_title("Fusion 的方向性 timestep 指标")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_label("Metric")
    fig.tight_layout()
    save_figure(fig, "fig5_2_direction_heatmap.png")


def plot_timestep_event_comparison(timestep: pd.DataFrame, event: pd.DataFrame) -> None:
    step_rows = timestep.loc[timestep["method"] == "fusion"]
    event_rows = event.loc[event["method"] == "fusion"]
    directions = ["high", "low", "bidirectional"]
    levels = ["level_A", "level_B"]
    direction_labels = {"high": "High", "low": "Low", "bidirectional": "Bi-dir"}
    labels = [f"{direction_labels[direction]} {level[-1]}" for direction in directions for level in levels]
    step_f1 = []
    event_f1 = []
    delays = []
    for direction in directions:
        for level in levels:
            selector = (step_rows["direction"] == direction) & (step_rows["label_level"] == level)
            event_selector = (event_rows["direction"] == direction) & (event_rows["label_level"] == level)
            step_f1.append(float(step_rows.loc[selector, "f1"].iloc[0]))
            event_f1.append(float(event_rows.loc[event_selector, "event_f1"].iloc[0]))
            delays.append(float(event_rows.loc[event_selector, "mean_detection_delay_hours"].iloc[0]))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})
    bars1 = axes[0].bar(x - 0.18, step_f1, 0.36, label="Timestep F1", color=COLORS["light_blue"])
    bars2 = axes[0].bar(x + 0.18, event_f1, 0.36, label="Event F1", color=COLORS["blue"])
    annotate_bars(axes[0], bars1)
    annotate_bars(axes[0], bars2)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 0.78)
    axes[0].set_ylabel("F1")
    axes[0].set_title("逐时间步与事件级 F1")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, fontsize=8)

    bars = axes[1].bar(x, delays, color=[COLORS["green"], COLORS["purple"]] * 3, width=0.62)
    annotate_bars(axes[1], bars, fmt=".2f")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Mean delay (hours)")
    axes[1].set_title("事件平均检测延迟")
    axes[1].grid(axis="y")
    axes[1].set_ylim(0, max(delays) * 1.25)

    fig.suptitle("Fusion：timestep 与 event-level 评估对照", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "fig5_3_timestep_event.png")


def plot_event_profile(events: pd.DataFrame) -> None:
    type_order = ["mass_gathering", "traffic_jam", "sudden_incident"]
    type_counts = events["event_type"].value_counts().reindex(type_order, fill_value=0)
    severity_counts = events["severity"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    wedges, _, _ = axes[0].pie(
        type_counts.to_numpy(),
        labels=[TYPE_LABELS[item] for item in type_order],
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=[COLORS["blue"], COLORS["orange"], COLORS["light_blue"]],
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    axes[0].legend(
        wedges,
        [f"{TYPE_LABELS[item]} ({type_counts[item]})" for item in type_order],
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    axes[0].set_title("Event type")

    bars = axes[1].bar(
        severity_counts.index.astype(str),
        severity_counts.to_numpy(),
        color=[COLORS["light_blue"], COLORS["green"], COLORS["orange"], COLORS["red"]],
        width=0.62,
    )
    annotate_bars(axes[1], bars, fmt=".0f")
    axes[1].set_xlabel("Severity level")
    axes[1].set_ylabel("Events")
    axes[1].set_title("Alert severity")
    axes[1].grid(axis="y")
    axes[1].set_ylim(0, max(severity_counts) * 1.18)

    fig.suptitle("131 个融合异常事件的类型与等级构成", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_figure(fig, "fig6_1_event_profile.png")


def plot_event_duration(events: pd.DataFrame) -> None:
    type_order = ["mass_gathering", "traffic_jam", "sudden_incident"]
    values = [events.loc[events["event_type"] == item, "duration_hours"].to_numpy() for item in type_order]
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    box = ax.boxplot(values, patch_artist=True, widths=0.48, showfliers=False, medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(box["boxes"], [COLORS["blue"], COLORS["orange"], COLORS["light_blue"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
    for index, group in enumerate(values, start=1):
        jitter = rng.uniform(-0.10, 0.10, size=len(group))
        ax.scatter(np.full(len(group), index) + jitter, group, s=24, color=COLORS["dark"], alpha=0.52, edgecolor="white", linewidth=0.35)
        ax.text(index, max(group) + 0.45, f"n={len(group)}", ha="center", fontsize=8.5, color=COLORS["dark"])
    ax.set_xticks(np.arange(1, 4), [TYPE_LABELS[item] for item in type_order])
    ax.set_ylabel("Duration (hours)")
    ax.set_title("不同规则分类事件的持续时间分布")
    ax.grid(axis="y")
    ax.set_ylim(0, max(max(group) for group in values) * 1.14)
    fig.tight_layout()
    save_figure(fig, "fig6_2_event_duration.png")


def plot_explainability(explainability: pd.DataFrame) -> None:
    methods = ["statistical", "reconstruction", "prediction"]
    types = ["mass_gathering", "traffic_jam", "sudden_incident"]
    dominant = explainability["dominant_method"].value_counts().reindex(methods, fill_value=0)
    means = explainability.groupby("event_type")[[f"{method}_score" for method in methods]].mean().reindex(types)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), gridspec_kw={"width_ratios": [0.8, 1.35]})
    bars = axes[0].bar(
        [METHOD_LABELS[item] for item in methods],
        dominant.to_numpy(),
        color=[COLORS["blue"], COLORS["purple"], COLORS["orange"]],
        width=0.62,
    )
    annotate_bars(axes[0], bars, fmt=".0f")
    axes[0].set_ylabel("Events")
    axes[0].set_title("Dominant method")
    axes[0].grid(axis="y")
    axes[0].set_ylim(0, max(dominant) * 1.16)

    x = np.arange(len(types))
    width = 0.24
    for idx, method in enumerate(methods):
        bars = axes[1].bar(
            x + (idx - 1) * width,
            means[f"{method}_score"].to_numpy(),
            width,
            label=METHOD_LABELS[method],
            color=[COLORS["blue"], COLORS["purple"], COLORS["orange"]][idx],
        )
        annotate_bars(axes[1], bars)
    axes[1].set_xticks(x, [TYPE_LABELS[item].replace(" ", "\n") for item in types])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Mean event score")
    axes[1].set_title("各事件类型的平均分支置信度")
    axes[1].grid(axis="y")
    axes[1].legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )

    fig.suptitle("事件级可解释性：主导方法与分支分数组成", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    save_figure(fig, "fig7_1_method_contribution.png")


def plot_audit_sample(audit: pd.DataFrame) -> None:
    group_order = ["high_confidence", "borderline", "false_positive_by_weak_label"]
    group_labels = ["High confidence", "Borderline", "Weak-label FP\ncandidates"]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    for index, group in enumerate(group_order):
        values = audit.loc[audit["sample_group"] == group, "max_score"].to_numpy()
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        axes[0].scatter(np.full(len(values), index) + jitter, values, s=36, alpha=0.7, color=[COLORS["blue"], COLORS["orange"], COLORS["gray"]][index], edgecolor="white", linewidth=0.5)
        axes[0].plot([index - 0.18, index + 0.18], [np.median(values)] * 2, color=COLORS["dark"], linewidth=2)
    axes[0].set_xticks(np.arange(3), group_labels)
    axes[0].set_ylabel("Event max fusion score")
    axes[0].set_title("Audit sample score distribution")
    axes[0].grid(axis="y")

    overlap_a = [audit.loc[audit["sample_group"] == group, "weak_A_overlap"].astype(float).mean() * 100 for group in group_order]
    overlap_b = [audit.loc[audit["sample_group"] == group, "weak_B_overlap"].astype(float).mean() * 100 for group in group_order]
    x = np.arange(3)
    bars1 = axes[1].bar(x - 0.18, overlap_a, 0.36, label="Weak Level-A", color=COLORS["green"])
    bars2 = axes[1].bar(x + 0.18, overlap_b, 0.36, label="Weak Level-B", color=COLORS["purple"])
    annotate_bars(axes[1], bars1, fmt=".0f")
    annotate_bars(axes[1], bars2, fmt=".0f")
    axes[1].set_xticks(x, group_labels)
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("Samples overlapping weak labels (%)")
    axes[1].set_title("Weak-label overlap")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("50 条待人工核验事件的抽样结构", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "fig7_2_audit_sample.png")


REPORT_CSS = r"""
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @bottom-center {
    content: "Week 5 异常检测模型实现与融合评估报告 · " counter(page) " / " counter(pages);
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


def render_report() -> int:
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
<title>Week 5 异常检测模型实现与融合评估报告</title>
<style>{REPORT_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(document, encoding="utf-8")
    rendered = HTML(string=document, base_url=str(REPORT_DIR)).render()
    rendered.write_pdf(PDF_PATH)
    return len(rendered.pages)


def main() -> None:
    configure_plotting()
    summary, tuning, timestep, event, events, explainability, audit = load_artifacts()
    validate_artifacts(summary, tuning, timestep, event, events, explainability, audit)
    plot_fusion_selection(tuning)
    plot_method_comparison(summary)
    plot_direction_heatmap(timestep)
    plot_timestep_event_comparison(timestep, event)
    plot_event_profile(events)
    plot_event_duration(events)
    plot_explainability(explainability)
    plot_audit_sample(audit)
    page_count = render_report()
    figure_count = len(list(FIGURE_DIR.glob("*.png")))
    print(f"Generated {figure_count} figures in {FIGURE_DIR.relative_to(ROOT)}")
    print(f"Rendered {HTML_PATH.relative_to(ROOT)}")
    print(f"Rendered {PDF_PATH.relative_to(ROOT)} ({page_count} pages)")


if __name__ == "__main__":
    main()
