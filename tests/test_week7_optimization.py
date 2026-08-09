"""Tests for Week7 Optuna configuration without running model training."""

from __future__ import annotations

import copy
import subprocess
import sys
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.optimization.week7 import (
    incumbent_search_params,
    load_optimization_config,
    load_promotion_config,
    remaining_study_budget,
    sample_trial_config,
    validate_optimization_config,
    validation_mae,
)


class FakeTrial:
    number = 3

    def __init__(self) -> None:
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        preferred = {
            "d_model": 48,
            "n_heads": 8,
            "d_ff_multiplier": 3,
            "batch": 4,
            "num_layers": 2,
            "embed_dim": 10,
        }
        value = preferred.get(name, choices[0])
        if value not in choices:
            value = choices[0]
        return value

    def suggest_float(self, name, low, high, **kwargs):
        del name, high, kwargs
        return low

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class FakeState:
    def __init__(self, name):
        self.name = name


class StoredTrial:
    def __init__(self, state):
        self.state = FakeState(state)


class Week7OptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_optimization_config("configs/week7_optuna_agformer.yaml")

    def test_configuration_is_validation_only_and_single_gpu_safe(self) -> None:
        self.assertEqual(self.cfg["n_jobs"], 1)
        self.assertTrue(self.cfg["require_cuda"])
        self.assertEqual(self.cfg["resource_guard"]["allowed_batches"], [4, 8])
        self.assertNotIn("test", self.cfg["objective"].lower())

    def test_sample_resolves_compatible_heads_and_unique_paths(self) -> None:
        trial = FakeTrial()
        sampled = sample_trial_config(trial, self.cfg)

        self.assertEqual(sampled["d_model"], 48)
        self.assertEqual(sampled["n_heads"], 8)
        self.assertEqual(sampled["d_ff"], 144)
        self.assertEqual(sampled["batch"], 4)
        self.assertEqual(sampled["seed"], 42)
        self.assertIn("trial_0003", sampled["save_path"])
        self.assertIsNone(sampled["log_dir"])
        self.assertEqual(trial.user_attrs["resolved_d_ff"], 144)

    def test_incumbent_is_represented_in_the_search_schema(self) -> None:
        params = incumbent_search_params(self.cfg["_base_config"])

        self.assertEqual(params["d_model"], 32)
        self.assertEqual(params["n_heads"], 4)
        self.assertEqual(params["d_ff_multiplier"], 2)
        self.assertEqual(params["batch"], 8)
        self.assertEqual(params["lr"], 0.001)

    def test_data_rules_cannot_drift(self) -> None:
        broken = copy.deepcopy(self.cfg)
        broken["_base_config"]["filter_gaps"] = False

        with self.assertRaisesRegex(ValueError, "frozen Week7 data rules"):
            validate_optimization_config(broken)

    def test_parallel_gpu_trials_are_rejected(self) -> None:
        broken = copy.deepcopy(self.cfg)
        broken["n_jobs"] = 2

        with self.assertRaisesRegex(ValueError, "n_jobs must be 1"):
            validate_optimization_config(broken)

    def test_resume_budget_counts_terminal_and_stale_running_trials(self) -> None:
        trials = [StoredTrial("COMPLETE") for _ in range(8)]
        trials += [StoredTrial("PRUNED"), StoredTrial("RUNNING"), StoredTrial("WAITING")]

        remaining, seconds, consumed = remaining_study_budget(
            trials, total_trials=30, started_at_unix=100.0,
            timeout_hours=18, now_unix=3700.0,
        )

        self.assertEqual(consumed, 10)
        self.assertEqual(remaining, 20)
        self.assertEqual(seconds, 17 * 3600)

    def test_dynamic_head_distribution_is_rejected_before_optuna(self) -> None:
        broken = copy.deepcopy(self.cfg)
        broken["search_space"]["d_model"]["choices"] = [24, 32, 50]

        with self.assertRaisesRegex(ValueError, "changing categorical distribution"):
            validate_optimization_config(broken)

    def test_promotion_config_requires_validation_only_marker(self) -> None:
        import tempfile
        from pathlib import Path
        import yaml

        candidate = dict(self.cfg["_base_config"])
        candidate.update({
            "save_path": "outputs/week7_optimization/week7_agformer_optuna_v1/promotion/demo/best.pt",
            "optimization_role": "challenger",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.yaml"
            path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_only"):
                load_promotion_config(path, self.cfg)

    def test_generic_training_entrypoint_rejects_promotion_config(self) -> None:
        import tempfile
        from pathlib import Path
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.yaml"
            path.write_text(
                yaml.safe_dump({"optimization_validation_only": True}), encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, "scripts/train.py", "--config", str(path)],
                text=True, capture_output=True, check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("validation-only", completed.stderr)

    def test_promotion_and_final_test_clis_default_to_no_execution(self) -> None:
        import tempfile
        from pathlib import Path
        import yaml

        candidate = dict(self.cfg["_base_config"])
        candidate.update({
            "tag": "dry_run_candidate",
            "epochs": 50,
            "patience": 10,
            "num_workers": 2,
            "save_path": "outputs/week7_optimization/week7_agformer_optuna_v1/promotion/dry/best.pt",
            "log_dir": None,
            "optimization_validation_only": True,
            "optimization_role": "challenger",
            "optimization_source_trial": 1,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.yaml"
            path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
            promotion = subprocess.run(
                [sys.executable, "scripts/train_optuna_promotion.py", "--candidate-config", str(path)],
                text=True, capture_output=True, check=False,
            )
            final_test = subprocess.run(
                [sys.executable, "scripts/evaluate_optuna_final.py", "--candidate-config", str(path)],
                text=True, capture_output=True, check=False,
            )

        self.assertEqual(promotion.returncode, 0, promotion.stderr)
        self.assertIn("没有加载数据或训练", promotion.stdout)
        self.assertEqual(final_test.returncode, 0, final_test.stderr)
        self.assertIn("没有加载测试数据", final_test.stdout)

    def test_unapproved_batch_size_is_rejected(self) -> None:
        broken = copy.deepcopy(self.cfg)
        broken["search_space"]["batch"]["choices"] = [4, 16]

        with self.assertRaisesRegex(ValueError, "allowed_batches"):
            validate_optimization_config(broken)

    def test_validation_objective_uses_denormalized_nonzero_mask(self) -> None:
        class Echo(torch.nn.Module):
            def forward(self, x, graphs):
                del graphs
                return x

        prediction = torch.tensor([[[100.0, 1.0, 4.0]]])
        target = torch.tensor([[[-0.5, 2.0, 2.0]]])
        loader = DataLoader(TensorDataset(prediction, target), batch_size=1)

        value = validation_mae(
            Echo(), loader, {}, torch.device("cpu"), mean=1.0, std=2.0
        )

        # Denormalized target is [0, 5, 5]. The large first error is masked;
        # the remaining absolute errors are 2 and 4.
        self.assertAlmostEqual(value, 3.0)


if __name__ == "__main__":
    unittest.main()
