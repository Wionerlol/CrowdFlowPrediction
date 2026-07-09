"""Run the reproducible 2x2 time/static context ablation.

Each model is evaluated with flow only, time encoding only, leakage-free static
features only, and both contexts. Runs are sequential to bound GPU/host memory.
Completed variants are detected in week3_metrics.csv and skipped on resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs/results/week3_metrics.csv"
RUN_ROOT = ROOT / "outputs/context_ablation"
VARIANTS = {
    "flow": (False, False),
    "time": (True, False),
    "static": (False, True),
    "time_static": (True, True),
}


def is_complete(model: str, tag: str) -> bool:
    if not RESULTS.exists():
        return False
    df = pd.read_csv(RESULTS)
    rows = df[(df["model"] == model) & (df["tag"].fillna("") == tag)]
    return set(rows["horizon"]) >= {"h1", "h3", "h6", "h12", "avg"}


def make_config(model: str, variant: str, seed: int) -> Path:
    base_path = ROOT / f"configs/{model}.yaml"
    cfg = yaml.safe_load(base_path.read_text())
    use_time, use_static = VARIANTS[variant]
    tag = f"ctx_{variant}_seed{seed}"
    cfg.update({
        "seed": seed,
        "time_enc": use_time,
        "F_in": 6 if use_time else 2,
        "use_static": use_static,
        "static_dim": 33 if use_static else 0,
        "tag": tag,
        "save_path": f"outputs/checkpoints/context_ablation/{model}_{variant}/best.pt",
        "log_dir": f"outputs/logs/context_ablation/{model}_{variant}",
    })
    config_dir = RUN_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out = config_dir / f"{model}_{variant}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def summarize(seed: int) -> None:
    if not RESULTS.exists():
        return
    df = pd.read_csv(RESULTS)
    tags = {f"ctx_{v}_seed{seed}" for v in VARIANTS}
    out = df[(df["tag"].isin(tags)) & (df["horizon"] == "avg")].copy()
    if out.empty:
        return
    cols = ["model", "tag", "T_in", "T_out", "MAE", "RMSE", "MAPE"]
    summary = RUN_ROOT / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    out[cols].sort_values(["model", "tag"]).to_csv(summary, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["agformer", "spacetimeformer"],
                        choices=["agformer", "spacetimeformer"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS),
                        choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for model in args.models:
        for variant in args.variants:
            tag = f"ctx_{variant}_seed{args.seed}"
            if is_complete(model, tag):
                print(f"SKIP completed: {model}/{variant}", flush=True)
                continue
            config = make_config(model, variant, args.seed)
            print(f"RUN {model}/{variant}: {config}", flush=True)
            subprocess.run(
                [sys.executable, "-u", "scripts/train.py", "--config", str(config)],
                cwd=ROOT,
                check=True,
            )
            summarize(args.seed)

    summarize(args.seed)
    print(f"Ablation summary: {RUN_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
