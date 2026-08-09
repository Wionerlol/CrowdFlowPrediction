#!/usr/bin/env python3
"""Train one Optuna promotion candidate on train/validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.week7 import (
    load_optimization_config,
    load_promotion_config,
    run_validation_promotion,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-config", default="configs/week7_optuna_agformer.yaml")
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument(
        "--run-promotion", action="store_true",
        help="Explicitly launch validation-only full-budget training",
    )
    args = parser.parse_args()
    optimization = load_optimization_config(args.optimization_config)
    candidate = load_promotion_config(args.candidate_config, optimization)
    print(json.dumps({
        "tag": candidate["tag"],
        "role": candidate["optimization_role"],
        "epochs": candidate["epochs"],
        "test_dataset_constructed": False,
        "test_evaluated": False,
    }, ensure_ascii=False, indent=2))
    if not args.run_promotion:
        print("\n晋级配置校验通过。未提供 --run-promotion，因此没有加载数据或训练。")
        return
    print(json.dumps(run_validation_promotion(optimization, candidate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
