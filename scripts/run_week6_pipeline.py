"""Run Week 6 end-to-end batch or realtime pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.week6_pipeline import Week6Pipeline, json_default, load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/week6_pipeline.yaml")
    parser.add_argument("--mode", choices=["batch", "realtime"], default="batch")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-anomaly", action="store_true", help="batch mode: reuse existing Week5 outputs")
    parser.add_argument("--smoke", action="store_true", help="batch mode: pass --smoke to Week5 runner")
    parser.add_argument("--window-count", type=int, default=None, help="realtime mode micro-batch window count")
    parser.add_argument("--window-start", type=int, default=None, help="realtime mode start; negative counts from end")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pipeline_config(args.config)
    pipeline = Week6Pipeline(cfg, config_path=args.config, device=args.device)
    if args.mode == "batch":
        summary = pipeline.run_batch(skip_anomaly=args.skip_anomaly, smoke=args.smoke)
    else:
        summary = pipeline.run_realtime(
            window_count=args.window_count,
            window_start=args.window_start,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
