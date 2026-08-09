#!/usr/bin/env python3
"""Week7 AGFormer Optuna entry point; training requires an explicit --run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.week7 import load_optimization_config, run_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/week7_optuna_agformer.yaml",
        help="Optuna YAML configuration",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Actually launch training trials; without this flag only validate and print the plan",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Optional trial-count override")
    parser.add_argument("--timeout-hours", type=float, default=None, help="Optional time-limit override")
    args = parser.parse_args()

    cfg = load_optimization_config(args.config)
    if args.n_trials is not None:
        if args.n_trials <= 0:
            parser.error("--n-trials must be positive")
        cfg["n_trials"] = args.n_trials
    if args.timeout_hours is not None:
        if args.timeout_hours <= 0:
            parser.error("--timeout-hours must be positive")
        cfg["timeout_hours"] = args.timeout_hours

    printable = {
        "study_name": cfg["study_name"],
        "base_config": cfg["base_config"],
        "objective": cfg["objective"],
        "n_trials": cfg["n_trials"],
        "timeout_hours": cfg["timeout_hours"],
        "n_jobs": cfg["n_jobs"],
        "enqueue_incumbent_trial": cfg.get("enqueue_incumbent_trial", True),
        "search_keys": sorted(cfg["search_space"]),
        "test_evaluated_during_search": False,
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if not args.run:
        print("\n配置校验通过。未提供 --run，因此没有创建 study、加载数据或启动训练。")
        return

    study = run_optimization(cfg)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation MAE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
