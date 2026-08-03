"""Sample Week 5 anomaly events for human sanity-check auditing.

TaxiBJ has no ground-truth anomaly labels, so this script creates a compact
manual-review CSV from fusion events:

- high-confidence detections;
- borderline detections near the fusion decision threshold;
- detections that look like false positives against the statistical weak label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="outputs/week5_anomaly/events.csv")
    parser.add_argument("--scores", default="outputs/week5_anomaly/scores.npz")
    parser.add_argument("--summary", default="outputs/week5_anomaly/summary.json")
    parser.add_argument("--output-dir", default="outputs/week5_anomaly/audit")
    parser.add_argument("--high-confidence", type=int, default=20)
    parser.add_argument("--borderline", type=int, default=15)
    parser.add_argument("--weak-false-positive", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_decision_threshold(summary_path: str | Path) -> float:
    path = Path(summary_path)
    if not path.exists():
        return 0.5
    summary = json.loads(path.read_text(encoding="utf-8"))
    return float(summary.get("fusion", {}).get("decision_threshold", 0.5))


def event_overlaps(labels: np.ndarray, start: int, end: int) -> bool:
    labels = np.asarray(labels, dtype=bool)
    start = max(int(start), 0)
    end = min(int(end), len(labels) - 1)
    if end < start:
        return False
    return bool(labels[start:end + 1].any())


def region_label(row: pd.Series) -> str:
    centroid_r = row.get("main_centroid_row")
    centroid_c = row.get("main_centroid_col")
    top_cells = str(row.get("top_cells", ""))
    if pd.notna(centroid_r) and pd.notna(centroid_c):
        base = f"centroid=({float(centroid_r):.1f},{float(centroid_c):.1f})"
    else:
        base = "centroid=unknown"
    if top_cells and top_cells != "nan":
        return f"{base}; top_cells={top_cells}"
    return base


def take_rows(
    df: pd.DataFrame,
    count: int,
    *,
    selected: set[int],
) -> pd.DataFrame:
    if count <= 0 or len(df) == 0:
        return df.iloc[:0].copy()
    candidates = df[~df["event_id"].isin(selected)].copy()
    out = candidates.head(count).copy()
    selected.update(int(x) for x in out["event_id"].tolist())
    return out


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.events)
    if len(events) == 0:
        raise RuntimeError(f"No events found in {args.events}")
    scores = np.load(args.scores)
    threshold = load_decision_threshold(args.summary)

    labels_a = scores["weak_label_A"].astype(bool) if "weak_label_A" in scores.files else None
    labels_b = scores["weak_label_B"].astype(bool) if "weak_label_B" in scores.files else None
    events["weak_A_overlap"] = [
        event_overlaps(labels_a, row.start_index, row.end_index) if labels_a is not None else False
        for row in events.itertuples(index=False)
    ]
    events["weak_B_overlap"] = [
        event_overlaps(labels_b, row.start_index, row.end_index) if labels_b is not None else False
        for row in events.itertuples(index=False)
    ]
    events["borderline_distance"] = (events["max_score"].astype(float) - threshold).abs()
    events["region"] = events.apply(region_label, axis=1)

    selected: set[int] = set()
    groups: list[pd.DataFrame] = []

    high = events.sort_values(
        ["max_score", "duration_steps", "union_cells"],
        ascending=[False, False, False],
    )
    high = take_rows(high, args.high_confidence, selected=selected)
    high["sample_group"] = "high_confidence"
    groups.append(high)

    borderline = events.sort_values(
        ["borderline_distance", "max_score"],
        ascending=[True, False],
    )
    borderline = take_rows(borderline, args.borderline, selected=selected)
    borderline["sample_group"] = "borderline"
    groups.append(borderline)

    weak_fp = events[~events["weak_A_overlap"]].sort_values(
        ["max_score", "duration_steps", "union_cells"],
        ascending=[False, False, False],
    )
    weak_fp = take_rows(weak_fp, args.weak_false_positive, selected=selected)
    weak_fp["sample_group"] = "false_positive_by_weak_label"
    groups.append(weak_fp)

    requested = args.high_confidence + args.borderline + args.weak_false_positive
    sampled = pd.concat(groups, ignore_index=True)
    if len(sampled) < requested:
        remainder = events.sort_values(
            ["max_score", "duration_steps", "union_cells"],
            ascending=[False, False, False],
        )
        fill = take_rows(remainder, requested - len(sampled), selected=selected)
        fill["sample_group"] = "fallback_fill"
        sampled = pd.concat([sampled, fill], ignore_index=True)

    sampled["sample_rank"] = np.arange(1, len(sampled) + 1)
    sampled["manual_is_real_anomaly"] = ""
    sampled["manual_event_class"] = ""
    sampled["manual_notes"] = ""

    review_cols = [
        "sample_rank",
        "sample_group",
        "event_id",
        "start_time",
        "end_time",
        "duration_hours",
        "region",
        "event_type",
        "severity",
        "max_score",
        "dominant_method",
        "statistical_score",
        "reconstruction_score",
        "prediction_score",
        "mean_inflow",
        "max_inflow",
        "mean_outflow",
        "max_outflow",
        "in_out_imbalance",
        "max_z",
        "max_prediction_residual",
        "union_cells",
        "cluster_count",
        "main_aspect_ratio",
        "road_density_mean",
        "duration_trend_slope",
        "weak_A_overlap",
        "weak_B_overlap",
        "top_cells",
        "manual_is_real_anomaly",
        "manual_event_class",
        "manual_notes",
    ]
    sampled = ensure_columns(sampled, review_cols)

    csv_path = output_dir / "event_audit_sample_50.csv"
    sampled.to_csv(csv_path, index=False)
    summary = {
        "events": str(args.events),
        "scores": str(args.scores),
        "decision_threshold": threshold,
        "requested": {
            "high_confidence": args.high_confidence,
            "borderline": args.borderline,
            "false_positive_by_weak_label": args.weak_false_positive,
        },
        "actual_counts": sampled["sample_group"].value_counts().to_dict(),
        "output_csv": str(csv_path),
    }
    summary_path = output_dir / "event_audit_sample_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
