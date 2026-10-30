from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import yaml

from src.api.main import create_app


class Week6ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        grid = pd.DataFrame(
            {
                "grid_id": [0, 1, 2, 3],
                "row": [0, 0, 1, 1],
                "col": [0, 1, 0, 1],
                "center_lon": [116.1, 116.2, 116.1, 116.2],
                "center_lat": [39.9, 39.9, 40.0, 40.0],
            }
        )
        grid_path = root / "grid.csv"
        grid.to_csv(grid_path, index=False)

        timestamps = np.asarray(["2016-04-01T08:00:00", "2016-04-01T08:30:00"], dtype="datetime64[s]")
        score_kwargs = {
            "timestamps": timestamps,
            "true_flow": np.asarray([[10, 20, 30, 40], [11, 21, 31, 41]], dtype=np.float32),
            "pred_flow": np.asarray([[12, 19, 29, 39], [13, 22, 33, 44]], dtype=np.float32),
            "fusion_score": np.asarray([[0.1, 0.2, 0.3, 0.4], [0.2, 0.6, 0.8, 0.9]], dtype=np.float32),
            "fusion_mask": np.asarray([[False, False, False, False], [False, True, True, True]]),
        }
        realtime_scores = root / "realtime_scores.npz"
        historical_scores = root / "historical_scores.npz"
        np.savez_compressed(realtime_scores, **score_kwargs)
        np.savez_compressed(historical_scores, **score_kwargs)

        forecast_path = root / "forecast.npz"
        np.savez_compressed(
            forecast_path,
            city=np.asarray("bj"),
            origin_time=np.asarray(np.datetime64("2016-04-01T07:30:00", "s")),
            step_minutes=np.asarray(30, dtype=np.int64),
            timestamps=np.asarray(
                ["2016-04-01T08:00:00", "2016-04-01T08:30:00", "2016-04-01T09:00:00"],
                dtype="datetime64[s]",
            ),
            pred_flow=np.asarray([[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]], dtype=np.float32),
            true_flow=np.asarray([[9, 19, 29, 39], [10, 20, 30, 40], [11, 21, 31, 41]], dtype=np.float32),
            member_tags=np.asarray(["seed42", "seed7"]),
        )

        events = pd.DataFrame(
            [
                {
                    "event_id": 1,
                    "start_time": "2016-04-01T08:00:00",
                    "end_time": "2016-04-01T09:00:00",
                    "event_type": "mass_gathering",
                    "event_type_zh": "大型集会",
                    "severity": 4,
                    "week5_severity": 4,
                    "alert_level_id": 3,
                    "alert_level": "urgent",
                    "alert_level_zh": "紧急",
                    "alert_popup": True,
                    "alert_sound": True,
                    "max_score": 0.9,
                    "duration_hours": 1.5,
                    "union_cells": 3,
                    "cluster_count": 1,
                    "main_centroid_row": 1.0,
                    "main_centroid_col": 1.0,
                    "center_lon": 0.0,
                    "center_lat": 0.0,
                    "statistical_score": 1.0,
                    "reconstruction_score": 0.7,
                    "prediction_score": 0.4,
                    "alert_message": "紧急预警测试",
                    "alert_action_hint": "立即处置",
                },
                {
                    "event_id": 2,
                    "start_time": "2016-03-30T10:00:00",
                    "end_time": "2016-03-30T10:30:00",
                    "event_type": "sudden_incident",
                    "event_type_zh": "突发事件",
                    "severity": 2,
                    "week5_severity": 2,
                    "alert_level_id": 1,
                    "alert_level": "general",
                    "alert_level_zh": "一般",
                    "alert_popup": False,
                    "alert_sound": False,
                    "max_score": 0.65,
                    "duration_hours": 0.5,
                    "union_cells": 2,
                    "cluster_count": 1,
                    "main_centroid_row": np.nan,
                    "main_centroid_col": np.nan,
                    "top_cells": "r0|c0|score=0.650|flow=10.0|z=3.2",
                    "center_lon": 0.0,
                    "center_lat": 0.0,
                    "statistical_score": 0.8,
                    "reconstruction_score": 0.5,
                    "prediction_score": 0.3,
                    "alert_message": "一般预警测试",
                    "alert_action_hint": "继续观察",
                },
            ]
        )
        historical_alerts = root / "historical_alerts.csv"
        realtime_alerts = root / "realtime_alerts.csv"
        events.to_csv(historical_alerts, index=False)
        events.iloc[:1].to_csv(realtime_alerts, index=False)
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps({"mode": "realtime", "events": {"count": 1}}), encoding="utf-8")

        config = {
            "city": "bj",
            "api": {"cors_origins": []},
            "artifacts": {
                "grid_meta": str(grid_path),
                "realtime_summary": str(summary_path),
                "realtime_scores": str(realtime_scores),
                "forecast_24h": str(forecast_path),
                "realtime_alerts": str(realtime_alerts),
                "historical_scores": str(historical_scores),
                "historical_alerts": str(historical_alerts),
            },
            "anomaly": {"decision_threshold": 0.56, "min_event_cells": 2},
        }
        config_path = root / "web.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.client = TestClient(create_app(config_path))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_status_and_heatmap_region(self) -> None:
        status = self.client.get("/api/v1/status")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "ok")

        response = self.client.get(
            "/api/v1/heatmap",
            params={"source": "realtime", "value_type": "predicted", "row_min": 0, "row_max": 0},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["timestamp"], "2016-04-01T08:30:00")
        self.assertEqual(body["cell_count"], 2)
        self.assertEqual([cell["value"] for cell in body["cells"]], [13.0, 22.0])

    def test_same_origin_forecast_query(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            json={"city": "bj", "horizon_steps": 2, "grid_ids": [1, 2]},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["origin_time"], "2016-04-01T07:30:00")
        self.assertEqual(body["horizon_steps"], 2)
        self.assertEqual(body["frames"][1]["values"], [21.0, 31.0])
        self.assertNotIn("true_flow", body)

    def test_anomaly_detection_and_event_match(self) -> None:
        response = self.client.post(
            "/api/v1/anomaly/detect",
            json={
                "city": "bj",
                "source": "realtime",
                "timestamp": "2016-04-01T08:30:00",
                "decision_threshold": 0.56,
                "min_event_cells": 2,
                "top_k": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["is_anomaly"])
        self.assertEqual(body["anomaly_cell_count"], 3)
        self.assertEqual(body["pipeline_matched_event_count"], 1)
        self.assertEqual(body["top_cells"][0]["grid_id"], 3)

        regional = self.client.post(
            "/api/v1/anomaly/detect",
            json={
                "city": "bj",
                "source": "realtime",
                "timestamp": "2016-04-01T08:30:00",
                "decision_threshold": 0.1,
                "min_event_cells": 1,
                "row_min": 0,
                "row_max": 0,
            },
        )
        self.assertEqual(regional.status_code, 200)
        self.assertEqual(regional.json()["pipeline_matched_event_count"], 0)

    def test_history_filters_detail_and_statistics(self) -> None:
        filtered = self.client.get("/api/v1/anomalies", params={"alert_level": "urgent"})
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(filtered.json()["total"], 1)
        self.assertEqual(filtered.json()["events"][0]["center_lon"], 116.2)
        self.assertEqual(filtered.json()["events"][0]["coordinate_source"], "main_centroid")

        all_events = self.client.get("/api/v1/anomalies")
        self.assertEqual(all_events.status_code, 200)
        body = all_events.json()
        self.assertEqual(
            body["map_alignment"],
            {
                "aligned": 2,
                "main_centroid": 1,
                "top_cell_fallback": 1,
                "unavailable": 0,
            },
        )
        fallback = next(item for item in body["events"] if item["event_id"] == 2)
        self.assertEqual(fallback["center_lon"], 116.1)
        self.assertEqual(fallback["center_lat"], 39.9)
        self.assertEqual(fallback["map_grid_id"], 0)
        self.assertEqual(fallback["coordinate_source"], "top_cell_fallback")
        self.assertEqual(fallback["map_alignment_status"], "aligned")

        regional = self.client.get("/api/v1/anomalies", params={"grid_ids": "0"})
        self.assertEqual(regional.status_code, 200)
        self.assertEqual(regional.json()["total"], 1)
        self.assertEqual(regional.json()["events"][0]["event_id"], 2)

        detail = self.client.get("/api/v1/anomalies/1")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["alert_level"], "urgent")

        stats = self.client.get("/api/v1/statistics")
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.json()["total_events"], 2)
        self.assertEqual(stats.json()["by_alert_level"], {"urgent": 1, "general": 1})

    def test_invalid_region_is_rejected(self) -> None:
        response = self.client.get("/api/v1/heatmap", params={"row_min": 3, "row_max": 1})
        self.assertEqual(response.status_code, 422)

        invalid_time = self.client.get("/api/v1/heatmap", params={"timestamp": "not-a-time"})
        self.assertEqual(invalid_time.status_code, 422)


if __name__ == "__main__":
    unittest.main()
