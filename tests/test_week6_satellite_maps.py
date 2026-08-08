"""Tests for the georeferenced Landsat basemap used by the Week6 UI."""

import unittest

from PIL import Image

from src.visualization.satellite_maps import (
    SATELLITE_IMAGE,
    event_satellite_map_html,
    flow_satellite_map_html,
    plotly_satellite_layer,
    satellite_metadata,
)


class Week6SatelliteMapTest(unittest.TestCase):
    def test_asset_matches_taxibj_bounds(self) -> None:
        metadata = satellite_metadata()
        self.assertEqual(
            metadata["bounds"],
            {"west": 116.25, "south": 39.75, "east": 116.75, "north": 40.25},
        )
        with Image.open(SATELLITE_IMAGE) as image:
            self.assertEqual(image.format, "WEBP")
            self.assertEqual(
                image.size,
                (metadata["size"]["width"], metadata["size"]["height"]),
            )

    def test_flow_map_embeds_bitmap_and_heatmap(self) -> None:
        cells = [
            {
                "grid_id": 0,
                "center_lon": 116.4,
                "center_lat": 39.9,
                "value": 120.0,
                "anomaly_score": 0.75,
                "is_anomaly": True,
            }
        ]
        html = flow_satellite_map_html(cells, height=430)
        self.assertIn("BitmapLayer", html)
        self.assertIn("HeatmapLayer", html)
        self.assertIn("data:image/webp;base64,", html)
        self.assertIn("height: 430px", html)

    def test_event_map_embeds_bitmap_and_event_points(self) -> None:
        events = [
            {
                "event_id": 7,
                "center_lon": 116.4,
                "center_lat": 39.9,
                "alert_level": "urgent",
                "alert_level_zh": "紧急",
                "event_type_zh": "突发事件",
                "start_time": "2016-04-01T08:00:00",
                "max_score": 0.8,
            }
        ]
        html = event_satellite_map_html(events, height=440)
        self.assertIn("BitmapLayer", html)
        self.assertIn("ScatterplotLayer", html)
        self.assertIn("height: 440px", html)

    def test_plotly_layer_uses_georeferenced_image_corners(self) -> None:
        layer = plotly_satellite_layer(opacity=0.65)
        self.assertEqual(layer["sourcetype"], "image")
        self.assertTrue(layer["source"].startswith("data:image/webp;base64,"))
        self.assertEqual(
            layer["coordinates"],
            [
                [116.25, 40.25],
                [116.75, 40.25],
                [116.75, 39.75],
                [116.25, 39.75],
            ],
        )
        self.assertEqual(layer["opacity"], 0.65)


if __name__ == "__main__":
    unittest.main()
