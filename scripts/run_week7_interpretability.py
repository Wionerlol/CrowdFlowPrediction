"""Generate Week7 attention, SHAP and anomaly-cause interpretability artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interpretability.week7 import json_safe, resolve_path, run_interpretability  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/week7_interpretability.yaml")
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing summary for the same output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    output_dir = resolve_path(cfg["output_dir"])
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir / 'summary.json'} already exists; use --overwrite intentionally"
        )
    summary = run_interpretability(
        cfg,
        config_path=args.config,
        device_override=args.device,
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
