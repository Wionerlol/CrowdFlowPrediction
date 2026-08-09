"""Regression tests for the Week7 final-system evaluation helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.week7 import (
    _NodeEvent,
    _maximum_cardinality_matches,
    classification_metrics,
    forecast_metrics,
    kth_largest_time_score,
    spatiotemporal_event_metrics,
    summarize_latencies,
)


class Week7EvaluationTest(unittest.TestCase):
    def test_forecast_metrics_match_project_mask_and_mape_rules(self) -> None:
        true = np.asarray([[0.0, 1.0, 3.0], [2.0, 4.0, 0.0]], dtype=np.float32)
        pred = np.asarray([[99.0, 2.0, 1.0], [4.0, 2.0, 88.0]], dtype=np.float32)

        metrics = forecast_metrics(pred, true)

        self.assertAlmostEqual(metrics["MAE"], 1.75)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(13 / 4))
        expected_mape = np.mean([1 / 2, 2 / 4, 2 / 3, 2 / 5]) * 100
        self.assertAlmostEqual(metrics["MAPE"], expected_mape)
        self.assertEqual(metrics["valid_count"], 4)

    def test_kth_largest_score_matches_minimum_cell_decision(self) -> None:
        scores = np.asarray(
            [[0.1, 0.9, 0.7, 0.4], [0.8, 0.6, 0.5, 0.2]], dtype=np.float32
        )

        result = kth_largest_time_score(scores, min_cells=2)

        np.testing.assert_allclose(result, [0.7, 0.6])

    def test_classification_metrics_include_accuracy_and_ranking_metrics(self) -> None:
        target = np.asarray([1, 1, 0, 0], dtype=bool)
        pred = np.asarray([1, 0, 1, 0], dtype=bool)
        score = np.asarray([0.9, 0.7, 0.6, 0.1], dtype=np.float32)

        metrics = classification_metrics(pred, target, score)

        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)

    def test_spatiotemporal_event_matching_requires_time_and_space(self) -> None:
        timestamps = np.asarray(
            ["2016-01-01T00:00", "2016-01-01T00:30", "2016-01-01T01:00"],
            dtype="datetime64[m]",
        )
        true_mask = np.asarray(
            [[True, False, False, False], [True, False, False, False], [False] * 4]
        )
        matched_pred = true_mask.copy()
        remote_pred = np.asarray(
            [[False, False, False, True], [False, False, False, True], [False] * 4]
        )

        matched = spatiotemporal_event_metrics(
            matched_pred,
            true_mask,
            timestamps,
            min_cells=1,
            grid_shape=(2, 2),
            tolerance_steps=0,
            spatial_iou_threshold=0.5,
            centroid_distance_threshold=0.5,
        )
        remote = spatiotemporal_event_metrics(
            remote_pred,
            true_mask,
            timestamps,
            min_cells=1,
            grid_shape=(2, 2),
            tolerance_steps=0,
            spatial_iou_threshold=0.5,
            centroid_distance_threshold=0.5,
        )

        self.assertEqual(matched["event_f1"], 1.0)
        self.assertEqual(matched["mean_detection_delay_steps"], 0.0)
        self.assertEqual(remote["event_f1"], 0.0)
        self.assertEqual(remote["matched_true_events"], 0)

    def test_event_builder_does_not_bridge_timestamp_gap(self) -> None:
        timestamps = np.asarray(
            ["2016-01-01T00:00", "2016-01-01T00:30", "2016-01-01T03:00"],
            dtype="datetime64[m]",
        )
        mask = np.asarray([[True, False], [True, False], [True, False]])

        metrics = spatiotemporal_event_metrics(
            mask,
            mask,
            timestamps,
            min_cells=1,
            grid_shape=(1, 2),
            tolerance_steps=0,
            spatial_iou_threshold=0.5,
            centroid_distance_threshold=0.5,
        )

        self.assertEqual(metrics["pred_event_count"], 2)
        self.assertEqual(metrics["true_event_count"], 2)
        self.assertEqual(metrics["event_f1"], 1.0)

    def test_event_matching_uses_maximum_cardinality(self) -> None:
        event = _NodeEvent(start=0, end=0, union_mask=np.asarray([True]))
        candidates = [
            [(0, event, 1.0, 0.0), (1, event, 0.5, 0.0)],
            [(0, event, 1.0, 0.0)],
        ]

        matches = _maximum_cardinality_matches(candidates)

        self.assertEqual(len(matches), 2)
        self.assertEqual({item[0] for item in matches.values()}, {0, 1})

    def test_empty_event_masks_are_supported(self) -> None:
        timestamps = np.asarray(["2016-01-01T00:00"], dtype="datetime64[m]")
        mask = np.zeros((1, 4), dtype=bool)

        metrics = spatiotemporal_event_metrics(
            mask,
            mask,
            timestamps,
            min_cells=1,
            grid_shape=(2, 2),
            tolerance_steps=1,
            spatial_iou_threshold=0.5,
            centroid_distance_threshold=0.5,
        )

        self.assertEqual(metrics["pred_event_count"], 0)
        self.assertEqual(metrics["true_event_count"], 0)
        self.assertEqual(metrics["event_f1"], 0.0)

    def test_latency_summary_uses_millisecond_percentiles(self) -> None:
        summary = summarize_latencies([10.0, 20.0, 30.0, 40.0])

        self.assertAlmostEqual(summary["mean_ms"], 25.0)
        self.assertAlmostEqual(summary["p50_ms"], 25.0)
        self.assertAlmostEqual(summary["p95_ms"], 38.5)
        self.assertAlmostEqual(summary["max_ms"], 40.0)
        self.assertAlmostEqual(summary["sequential_requests_per_second"], 40.0)


if __name__ == "__main__":
    unittest.main()
