from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.models.agformer import AGFormer
from src.pipeline.week6_pipeline import load_or_fit_statistical_context


class Week7PerformanceTest(unittest.TestCase):
    def test_attention_weight_elision_preserves_eval_output(self) -> None:
        torch.manual_seed(7)
        model = AGFormer(
            N=4, T_in=3, T_out=2, F_in=2, d_model=8, n_heads=2,
            d_ff=16, num_layers=1, dropout=0.0,
        ).eval()
        x = torch.randn(2, 4, 3, 2)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 1, 2, 3, 0]],
            dtype=torch.long,
        )
        graphs = {"spatial": {"edge_index": edge_index}}
        model.blocks[0].capture_attention_weights = True
        with_weights = model(x, graphs)
        model.blocks[0].capture_attention_weights = False
        without_weights = model(x, graphs)
        torch.testing.assert_close(with_weights, without_weights, rtol=1e-5, atol=1e-6)

    def test_statistical_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            data = rng.uniform(0, 100, size=(16, 2, 2, 2)).astype(np.float32)
            timestamps = np.arange(
                np.datetime64("2016-01-01T00:00", "m"),
                np.datetime64("2016-01-05T00:00", "m"),
                np.timedelta64(6, "h"),
            ).astype("datetime64[s]")
            source = root / "source.npz"
            np.savez_compressed(source, data=data, timestamps=timestamps)
            cfg = {
                "data_path": str(source),
                "steps_per_day": 4,
                "tz_offset": 0,
                "statistical": {
                    "sigma_z": 3.0,
                    "iqr_factor": 1.5,
                    "min_std": 1.0,
                    "weak_label_min_cells": 2,
                    "level_b_outflow_z": 3.0,
                    "level_b_min_outflow_cells": 2,
                },
            }
            cache = root / "stats.npz"
            first = load_or_fit_statistical_context(
                cfg, data, timestamps, len(data), cache_path=cache
            )
            second = load_or_fit_statistical_context(
                cfg, data, timestamps, len(data), cache_path=cache
            )
            self.assertFalse(first[3])
            self.assertTrue(second[3])
            np.testing.assert_array_equal(first[2], second[2])
            np.testing.assert_allclose(first[0].mean, second[0].mean)
            cfg["statistical"]["sigma_z"] = 2.9
            invalidated = load_or_fit_statistical_context(
                cfg, data, timestamps, len(data), cache_path=cache
            )
            self.assertFalse(invalidated[3])


if __name__ == "__main__":
    unittest.main()
