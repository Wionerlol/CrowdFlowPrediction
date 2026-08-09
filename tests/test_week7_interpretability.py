"""Regression tests for Week7 interpretability helpers."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from src.interpretability.week7 import (
    _event_weather_rows,
    _match_external_events,
    candidate_cause,
    evenly_spaced_indices,
    surrogate_feature_names,
)


class Week7InterpretabilityTest(unittest.TestCase):
    def test_surrogate_schema_contains_all_model_inputs_without_leakage_stats(self) -> None:
        names, groups = surrogate_feature_names(48)

        self.assertEqual(len(names), 48 * 2 + 7 + 33)
        self.assertEqual(len(groups), len(names))
        self.assertEqual(names[0], "inflow_lag_48")
        self.assertEqual(names[47], "inflow_lag_1")
        self.assertIn("is_holiday", names)
        self.assertIn("subway_station_density", names)
        self.assertNotIn("historical_flow_mean", names)

    def test_evenly_spaced_indices_are_bounded_and_unique(self) -> None:
        indices = evenly_spaced_indices(10, 4)

        self.assertEqual(indices.tolist(), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices(2, 5).tolist(), [0, 1])

    def test_weather_evidence_can_rank_above_calendar_candidate(self) -> None:
        weather = pd.DataFrame(
            {
                "precipitation": [2.5],
                "rain_1h": [1.5],
                "snow_1h": [0.0],
                "wind_speed": [3.0],
                "temperature": [10.0],
                "weather_code": [61.0],
            }
        )
        external = pd.DataFrame(
            columns=["start_time", "end_time", "title", "category", "source", "url"]
        )
        local_start = pd.Timestamp("2016-02-08T08:00:00", tz="Asia/Shanghai")

        result = candidate_cause(
            event={
                "event_type": "sudden_incident",
                "duration_hours": 1.0,
                "cluster_count": 2,
                "road_density_mean": 0.2,
            },
            weather_rows=weather,
            local_start=local_start,
            holidays={local_start.date()},
            makeup_workdays=set(),
            weather_thresholds={
                "wind_p95": 8.0,
                "temperature_p05": -10.0,
                "temperature_p95": 35.0,
            },
            external_matches=external,
        )

        self.assertEqual(result["primary_context_candidate"], "heavy_precipitation")
        self.assertTrue(result["is_holiday"])
        candidates = json.loads(result["context_candidates_json"])
        self.assertIn("official_holiday", {item["cause"] for item in candidates})
        self.assertEqual(result["context_status"], "candidate_evidence_not_causal_proof")

    def test_explicit_external_event_has_highest_evidence_score(self) -> None:
        external = pd.DataFrame(
            [
                {
                    "start_time": pd.Timestamp("2016-01-01", tz="UTC"),
                    "end_time": pd.Timestamp("2016-01-01T01:00", tz="UTC"),
                    "title": "Verified event",
                    "category": "public_event",
                    "source": "Example source",
                    "url": "https://example.com/event",
                }
            ]
        )
        local_start = pd.Timestamp("2016-01-01T08:00:00", tz="Asia/Shanghai")

        result = candidate_cause(
            event={"event_type": "mass_gathering", "duration_hours": 2, "cluster_count": 1},
            weather_rows=pd.DataFrame(),
            local_start=local_start,
            holidays=set(),
            makeup_workdays=set(),
            weather_thresholds={
                "wind_p95": 8.0,
                "temperature_p05": -10.0,
                "temperature_p95": 35.0,
            },
            external_matches=external,
        )

        self.assertEqual(
            result["primary_context_candidate"], "external_citywide_time_cooccurrence"
        )
        self.assertEqual(result["external_event_status"], "matched")
        self.assertIn("Verified event", result["external_event_titles"])

    def test_external_event_spatial_constraint_rejects_remote_event(self) -> None:
        external = pd.DataFrame(
            [
                {
                    "start_time": pd.Timestamp("2016-01-01", tz="UTC"),
                    "end_time": pd.Timestamp("2016-01-01T01:00", tz="UTC"),
                    "title": "Remote event",
                    "category": "public_event",
                    "source": "Example source",
                    "url": "https://example.com/event",
                    "center_row": 25,
                    "center_col": 25,
                    "radius_cells": 2,
                }
            ]
        )

        matches = _match_external_events(
            external,
            start_utc=pd.Timestamp("2016-01-01T00:30", tz="UTC"),
            end_utc=pd.Timestamp("2016-01-01T00:30", tz="UTC"),
            tolerance=pd.Timedelta(hours=1),
            event_row=2,
            event_col=2,
        )

        self.assertTrue(matches.empty)

    def test_weather_alignment_uses_utc_event_interval(self) -> None:
        weather = pd.DataFrame(
            {
                "utc": pd.to_datetime(
                    ["2016-01-01T00:00Z", "2016-01-01T00:30Z", "2016-01-01T01:00Z"]
                ),
                "temperature": [1.0, 2.0, 3.0],
            }
        )

        selected = _event_weather_rows(
            weather,
            pd.Timestamp("2016-01-01T00:30", tz="UTC"),
            pd.Timestamp("2016-01-01T01:00", tz="UTC"),
        )

        self.assertEqual(selected["temperature"].tolist(), [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
