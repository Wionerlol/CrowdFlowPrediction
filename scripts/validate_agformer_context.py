"""Run the gap-aware AGFormer baseline and gated-context validation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs/results/week3_metrics.csv"
RUNS = [
    ("agformer", "gate_flow_seed42", ROOT / "configs/agformer_gate_flow.yaml"),
    ("agformer", "gate_time_seed42", ROOT / "configs/agformer_gate_time.yaml"),
    ("agformer", "gate_static_seed42", ROOT / "configs/agformer_gate_static.yaml"),
    ("agformer", "gate_context_seed42", ROOT / "configs/agformer_gate_context.yaml"),
    ("lstm", "gapaware_seed42", ROOT / "configs/lstm_gapaware.yaml"),
]


def complete(model: str, tag: str) -> bool:
    if not RESULTS.exists():
        return False
    df = pd.read_csv(RESULTS)
    rows = df[(df["model"] == model) & (df["tag"].fillna("") == tag)]
    return set(rows["horizon"]) >= {"h1", "h3", "h6", "h12", "avg"}


def summarize() -> None:
    if not RESULTS.exists():
        return
    df = pd.read_csv(RESULTS)
    pairs = {(model, tag) for model, tag, _ in RUNS}
    mask = pd.Series(
        [(model, tag) in pairs
         for model, tag in zip(df["model"], df["tag"].fillna(""))],
        index=df.index,
    )
    out = df[mask & (df["horizon"] == "avg")]
    path = ROOT / "outputs/context_gate/summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out[["model", "tag", "MAE", "RMSE", "MAPE"]].to_csv(path, index=False)


def main() -> None:
    for model, tag, config in RUNS:
        if complete(model, tag):
            print(f"SKIP completed: {model}/{tag}", flush=True)
            continue
        print(f"RUN {tag}: {config}", flush=True)
        subprocess.run(
            [sys.executable, "-u", "scripts/train.py", "--config", str(config)],
            cwd=ROOT,
            check=True,
        )
        summarize()
    summarize()
    print("Validation complete: outputs/context_gate/summary.csv", flush=True)


if __name__ == "__main__":
    main()
