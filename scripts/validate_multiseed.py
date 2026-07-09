"""Reproduce the best AGFormer and gap-aware LSTM with three seeds."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs/results/week3_metrics.csv"
OUT = ROOT / "outputs/multiseed"
SEEDS = (42, 123, 2026)
MODELS = {
    "agformer_context": {
        "model": "agformer",
        "base": ROOT / "configs/agformer_gate_context.yaml",
        "tag": "gate_context_seed{seed}",
    },
    "lstm_gapaware": {
        "model": "lstm",
        "base": ROOT / "configs/lstm_gapaware.yaml",
        "tag": "gapaware_seed{seed}",
    },
}


def is_complete(model: str, tag: str) -> bool:
    if not RESULTS.exists():
        return False
    df = pd.read_csv(RESULTS)
    rows = df[(df["model"] == model) & (df["tag"].fillna("") == tag)]
    return set(rows["horizon"]) >= {"h1", "h3", "h6", "h12", "avg"}


def make_config(name: str, spec: dict, seed: int) -> Path:
    cfg = yaml.safe_load(spec["base"].read_text())
    tag = spec["tag"].format(seed=seed)
    cfg.update({
        "seed": seed,
        "tag": tag,
        "save_path": f"outputs/checkpoints/multiseed/{name}/seed{seed}/best.pt",
        "log_dir": f"outputs/logs/multiseed/{name}/seed{seed}",
    })
    config_dir = OUT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}_seed{seed}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def summarize() -> None:
    if not RESULTS.exists():
        return
    df = pd.read_csv(RESULTS)
    records = []
    for name, spec in MODELS.items():
        for seed in SEEDS:
            tag = spec["tag"].format(seed=seed)
            row = df[(df["model"] == spec["model"]) &
                     (df["tag"].fillna("") == tag) &
                     (df["horizon"] == "avg")]
            if not row.empty:
                item = row.iloc[-1]
                records.append({
                    "experiment": name, "model": spec["model"],
                    "seed": seed, "MAE": item["MAE"],
                    "RMSE": item["RMSE"], "MAPE": item["MAPE"],
                })
    if not records:
        return
    runs = pd.DataFrame(records).sort_values(["experiment", "seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "runs.csv", index=False)
    stats = runs.groupby(["experiment", "model"])[["MAE", "RMSE", "MAPE"]].agg(
        ["mean", "std", "count"]
    )
    stats.to_csv(OUT / "summary.csv")


def main() -> None:
    for name, spec in MODELS.items():
        for seed in SEEDS:
            tag = spec["tag"].format(seed=seed)
            if is_complete(spec["model"], tag):
                print(f"SKIP completed: {name}/seed{seed}", flush=True)
                continue
            config = make_config(name, spec, seed)
            print(f"RUN {name}/seed{seed}: {config}", flush=True)
            subprocess.run(
                [sys.executable, "-u", "scripts/train.py", "--config", str(config)],
                cwd=ROOT,
                check=True,
            )
            summarize()
    summarize()
    print(f"Multi-seed summary: {OUT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
