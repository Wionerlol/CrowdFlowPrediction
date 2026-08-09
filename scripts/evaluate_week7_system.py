"""Run the Week7 comprehensive final-system evaluation without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.week7 import json_safe, resolve_path, run_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/week7_evaluation.yaml")
    parser.add_argument("--skip-api-benchmark", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing evaluation summary for the same tag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    output_dir = resolve_path(cfg["output_dir"])
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir / 'summary.json'} already exists; use a new tag/output_dir "
            "or pass --overwrite intentionally"
        )
    summary = run_evaluation(
        cfg,
        config_path=args.config,
        skip_api_benchmark=args.skip_api_benchmark,
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
