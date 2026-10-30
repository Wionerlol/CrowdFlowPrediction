"""Tests for the georeferenced Landsat basemap used by the Week6 UI."""

import unittest

from PIL import Image

from src.visualization.satellite_maps import (
    RING_ROADS_GEOJSON,
    SATELLITE_IMAGE,
    event_satellite_map_html,
    flow_satellite_map_html,
    plotly_ring_roads_layer,
    plotly_satellite_layer,
    satellite_caption,
    satellite_metadata,
)


class Week6SatelliteMapTest(unittest.TestCase):
    def test_asset_matches_calibrated_display_bounds(self) -> None:
        metadata = satellite_metadata()
        self.assertEqual(
            metadata["bounds"],
            {"west": 116.125, "south": 39.671875, "east": 116.625, "north": 40.171875},
        )
        self.assertEqual(metadata["calibration"]["row_offset"], 5)
        self.assertEqual(metadata["calibration"]["col_offset"], -8)
        self.assertTrue(RING_ROADS_GEOJSON.exists())
        with Image.open(SATELLITE_IMAGE) as image:
            self.assertEqual(image.format, "WEBP")
            self.assertGreaterEqual(image.width, 2_000)
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
        self.assertIn("GeoJsonLayer", html)
        self.assertIn('"relation_id": 18649348', html)
        self.assertIn("data:image/webp;base64,", html)
        self.assertNotIn("@@=data:image", html)
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
                "coordinate_source": "top_cell_fallback",
                "is_selected": True,
            }
        ]
        html = event_satellite_map_html(events, height=440)
        self.assertIn("BitmapLayer", html)
        self.assertIn("ScatterplotLayer", html)
        self.assertIn("GeoJsonLayer", html)
        self.assertIn("top_cell_fallback", html)
        self.assertIn("height: 440px", html)

    def test_plotly_layer_uses_georeferenced_image_corners(self) -> None:
        layer = plotly_satellite_layer(opacity=0.65)
        self.assertEqual(layer["sourcetype"], "image")
        self.assertTrue(layer["source"].startswith("data:image/webp;base64,"))
        self.assertEqual(
            layer["coordinates"],
            [
                [116.125, 40.171875],
                [116.625, 40.171875],
                [116.625, 39.671875],
                [116.125, 39.671875],
            ],
        )
        self.assertEqual(layer["opacity"], 0.65)

    def test_plotly_ring_layer_uses_local_geojson(self) -> None:
        layer = plotly_ring_roads_layer(opacity=0.7)
        self.assertEqual(layer["sourcetype"], "geojson")
        self.assertEqual(layer["type"], "line")
        self.assertEqual(layer["source"]["type"], "FeatureCollection")
        names = {feature["properties"]["name"] for feature in layer["source"]["features"]}
        self.assertEqual(names, {"二环路", "北京三环路", "北京四环路", "五环路"})

    def test_caption_reports_current_ui_offsets(self) -> None:
        caption = satellite_caption(row_offset=7, col_offset=-6)
        self.assertIn("向南 7 格", caption)
        self.assertIn("向西 6 格", caption)
        self.assertIn("不改变训练节点或模型输入", caption)


if __name__ == "__main__":
    unittest.main()
