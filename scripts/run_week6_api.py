"""Start the Week 6 FastAPI inference/query service."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/week6_web.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    api_cfg = cfg.get("api", {})
    os.environ["WEEK6_WEB_CONFIG"] = args.config
    uvicorn.run(
        "src.api.main:app",
        host=args.host or str(api_cfg.get("host", "127.0.0.1")),
        port=args.port or int(api_cfg.get("port", 8000)),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
