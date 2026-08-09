"""Benchmark and report Week7 system performance optimizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.week7 import json_safe, resolve_path  # noqa: E402
from src.performance.week7 import run_performance_optimization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/week7_performance.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    summary_path = resolve_path(cfg["output_dir"]) / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} already exists; use a new output_dir or --overwrite intentionally"
        )
    summary = run_performance_optimization(cfg, config_path=args.config)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
