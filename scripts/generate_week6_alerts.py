"""Generate Week 6 three-level alerts from Week 5 anomaly events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.alerts.levels import (
    ALERT_EXTRA_COLUMNS,
    alert_levels_from_config,
    annotate_event_alerts,
    summarize_alerts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/week6_alerts.yaml")
    parser.add_argument("--input-events", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    input_events = Path(args.input_events or cfg["input_events"])
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(input_events)
    if len(events) == 0:
        alerts = pd.DataFrame(columns=list(events.columns) + ALERT_EXTRA_COLUMNS)
    else:
        alerts = annotate_event_alerts(events, levels=alert_levels_from_config(cfg.get("levels", {})))

    alerts_path = output_dir / "alerts.csv"
    summary_path = output_dir / "alert_summary.json"
    config_path = output_dir / "config_used.yaml"

    alerts.to_csv(alerts_path, index=False)
    summary = {
        "input_events": str(input_events),
        "alerts": str(alerts_path),
        "config": str(config_path),
        "level_mapping": {
            name: {
                "id": int(item["id"]),
                "name_zh": item["name_zh"],
                "week5_severities": item["week5_severities"],
                "popup": bool(item["popup"]),
                "sound": bool(item["sound"]),
                "ui_behavior": item["ui_behavior"],
            }
            for name, item in cfg["levels"].items()
        },
        **summarize_alerts(alerts),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {alerts_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
