"""Run the seed-42 calendar/weather AGFormer screening experiment."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs/results/week3_metrics.csv"
RUNS = [
    ("calendar_context_seed42", ROOT / "configs/agformer_calendar_context.yaml"),
    ("calendar_weather_seed42", ROOT / "configs/agformer_calendar_weather.yaml"),
]


def complete(tag: str) -> bool:
    if not RESULTS.exists():
        return False
    df = pd.read_csv(RESULTS)
    rows = df[(df["model"] == "agformer") & (df["tag"].fillna("") == tag)]
    return set(rows["horizon"]) >= {"h1", "h3", "h6", "h12", "avg"}


def summarize() -> None:
    if not RESULTS.exists():
        return
    df = pd.read_csv(RESULTS)
    tags = {tag for tag, _ in RUNS}
    out = df[(df["model"] == "agformer") &
             (df["tag"].isin(tags)) & (df["horizon"] == "avg")]
    path = ROOT / "outputs/weather_ablation/summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out[["model", "tag", "MAE", "RMSE", "MAPE"]].to_csv(path, index=False)


def main() -> None:
    for tag, config in RUNS:
        if complete(tag):
            print(f"SKIP completed: {tag}", flush=True)
            continue
        print(f"RUN {tag}: {config}", flush=True)
        subprocess.run(
            [sys.executable, "-u", "scripts/train.py", "--config", str(config)],
            cwd=ROOT,
            check=True,
        )
        summarize()
    summarize()
    print("Weather screening complete: outputs/weather_ablation/summary.csv", flush=True)


if __name__ == "__main__":
    main()
