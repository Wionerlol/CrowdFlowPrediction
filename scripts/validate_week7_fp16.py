"""Validate FP16 inference on stratified TaxiBJ test windows without training."""

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
from src.performance.week7 import validate_fp16_segments  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/week7_performance.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    validation_cfg = cfg["fp16_validation"]
    output_dir = resolve_path(cfg["output_dir"]) / "fp16_validation"
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = validate_fp16_segments(
        validation_cfg, cfg["gates"], output_dir=output_dir
    )
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
