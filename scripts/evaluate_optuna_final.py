#!/usr/bin/env python3
"""Explicit final-test gate for one confirmed Optuna promotion checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optimization.week7 import (
    evaluate_final_test,
    load_optimization_config,
    load_promotion_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-config", default="configs/week7_optuna_agformer.yaml")
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument(
        "--run-final-test", action="store_true",
        help="Explicitly load the frozen 4,022-window test set and evaluate the checkpoint",
    )
    args = parser.parse_args()
    optimization = load_optimization_config(args.optimization_config)
    candidate = load_promotion_config(args.candidate_config, optimization)
    if not args.run_final_test:
        print(json.dumps({
            "tag": candidate["tag"],
            "checkpoint": candidate["save_path"],
            "test_evaluated": False,
        }, ensure_ascii=False, indent=2))
        print("\n未提供 --run-final-test，因此没有加载测试数据或执行模型推理。")
        return
    print(json.dumps(evaluate_final_test(optimization, candidate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
