"""Regression tests for the display-only TaxiBJ map calibration."""

import unittest

import pandas as pd

from src.visualization.geo_calibration import (
    DEFAULT_COL_OFFSET,
    DEFAULT_ROW_OFFSET,
    adjust_display_frame,
    build_display_grid_frame,
    calibrated_bounds,
)


class Week6DisplayCalibrationTest(unittest.TestCase):
    def test_default_bounds_move_overlay_southwest_without_changing_extent(self) -> None:
        bounds = calibrated_bounds()
        self.assertEqual(
            bounds,
            {"west": 116.125, "south": 39.671875, "east": 116.625, "north": 40.171875},
        )
        self.assertAlmostEqual(bounds["east"] - bounds["west"], 0.5)
        self.assertAlmostEqual(bounds["north"] - bounds["south"], 0.5)

    def test_display_grid_preserves_node_identity_and_recomputes_only_geography(self) -> None:
        source = pd.DataFrame(
            {
                "grid_id": [0, 1023],
                "row": [0, 31],
                "col": [0, 31],
                "center_lon": [116.2578125, 116.7421875],
                "center_lat": [40.2421875, 39.7578125],
                "lon_min": [116.25, 116.734375],
                "lon_max": [116.265625, 116.75],
                "lat_min": [40.234375, 39.75],
                "lat_max": [40.25, 39.765625],
                "area_km2": [2.3, 2.3],
            }
        )

        display = build_display_grid_frame(source)

        self.assertEqual(display["grid_id"].tolist(), source["grid_id"].tolist())
        self.assertEqual(display["row"].tolist(), source["row"].tolist())
        self.assertEqual(display["col"].tolist(), source["col"].tolist())
        self.assertEqual(display["area_km2"].tolist(), source["area_km2"].tolist())
        self.assertAlmostEqual(display.loc[0, "center_lon"], 116.1328125)
        self.assertAlmostEqual(display.loc[0, "center_lat"], 40.1640625)
        self.assertAlmostEqual(display.loc[1, "center_lon"], 116.6171875)
        self.assertAlmostEqual(display.loc[1, "center_lat"], 39.6796875)

    def test_ui_adjustment_is_relative_to_default_display_grid(self) -> None:
        frame = pd.DataFrame({"center_lon": [116.4], "center_lat": [39.9]})
        unchanged = adjust_display_frame(
            frame,
            row_offset=DEFAULT_ROW_OFFSET,
            col_offset=DEFAULT_COL_OFFSET,
        )
        shifted = adjust_display_frame(
            frame,
            row_offset=DEFAULT_ROW_OFFSET + 1,
            col_offset=DEFAULT_COL_OFFSET + 1,
        )

        self.assertEqual(unchanged.to_dict("records"), frame.to_dict("records"))
        self.assertAlmostEqual(shifted.loc[0, "center_lon"], 116.415625)
        self.assertAlmostEqual(shifted.loc[0, "center_lat"], 39.884375)


if __name__ == "__main__":
    unittest.main()
