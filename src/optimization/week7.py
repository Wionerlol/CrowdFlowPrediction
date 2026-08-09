"""Safe Optuna search for the final Week4/Week7 AGFormer configuration.

The optimization objective uses the fixed chronological validation split only.
It deliberately does not construct or evaluate a test dataset, preventing test
metrics from influencing trial selection.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.flow_dataset import FlowDataset, load_graphs, load_taxibj, split_taxibj
from src.features.temporal import load_bj_holidays, load_bj_makeup_workdays
ROOT = Path(__file__).resolve().parents[2]

# These fields define the frozen data/feature/evaluation protocol. Allowing
# Optuna to change them would make trials incomparable or introduce leakage.
FROZEN_BASE_VALUES: dict[str, Any] = {
    "model": "agformer",
    "city": "bj",
    "N": 1024,
    "T_in": 48,
    "T_out": 48,
    "F_in": 9,
    "context_fusion": "gated",
    "use_linear_output": True,
    "use_static": True,
    "static_dim": 33,
    "time_enc": True,
    "real_time_context": True,
    "filter_gaps": True,
}

SEARCH_KEYS = {
    "lr", "batch", "d_model", "n_heads", "d_ff_multiplier",
    "num_layers", "dropout", "weight_decay", "embed_dim",
}


def incumbent_search_params(base: dict[str, Any]) -> dict[str, Any]:
    """Map the current production config into the Optuna parameter schema."""

    d_model = int(base["d_model"])
    d_ff = int(base["d_ff"])
    if d_ff % d_model:
        raise ValueError("Incumbent d_ff must be an integer multiple of d_model")
    return {
        "lr": float(base["lr"]),
        "batch": int(base["batch"]),
        "d_model": d_model,
        "n_heads": int(base["n_heads"]),
        "d_ff_multiplier": d_ff // d_model,
        "num_layers": int(base["num_layers"]),
        "dropout": float(base["dropout"]),
        "weight_decay": float(base["weight_decay"]),
        "embed_dim": int(base["embed_dim"]),
    }


@dataclass(frozen=True)
class PreparedData:
    """Datasets and normalization shared by all trials in one process."""

    train: FlowDataset
    val: FlowDataset
    mean: float
    std: float
    train_windows: int
    val_windows: int


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_optimization_config(path: str | Path) -> dict[str, Any]:
    """Load optimization and base-model YAML files without starting training."""

    optimization_path = _resolve(path)
    with optimization_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Optimization config must be a mapping: {optimization_path}")
    base_path = _resolve(cfg.get("base_config", ""))
    with base_path.open(encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, dict):
        raise ValueError(f"Base config must be a mapping: {base_path}")
    cfg["_config_path"] = str(optimization_path)
    cfg["_base_config_path"] = str(base_path)
    cfg["_base_config"] = base
    validate_optimization_config(cfg)
    return cfg


def validate_optimization_config(cfg: dict[str, Any]) -> None:
    """Reject unsafe or incomparable search definitions early."""

    base = cfg.get("_base_config")
    if not isinstance(base, dict):
        raise ValueError("Loaded optimization config must include _base_config")
    mismatches = {
        key: (base.get(key), expected)
        for key, expected in FROZEN_BASE_VALUES.items()
        if base.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Base config violates frozen Week7 data rules: {mismatches}")
    unsupported = {
        "use_weather": base.get("use_weather", False),
        "future_time_context": base.get("future_time_context", False),
        "periodic_lags": bool(base.get("periodic_lags", [])),
        "makeup_calendar": base.get("makeup_calendar", False),
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError(f"This fixed optimization protocol does not support: {enabled}")

    if int(cfg.get("n_jobs", 1)) != 1:
        raise ValueError("n_jobs must be 1; concurrent GPU trials are intentionally disabled")
    if int(cfg.get("n_trials", 0)) <= 0 or float(cfg.get("timeout_hours", 0)) <= 0:
        raise ValueError("n_trials and timeout_hours must be positive")
    if int(cfg.get("max_epochs", 0)) <= 0 or int(cfg.get("patience", 0)) <= 0:
        raise ValueError("max_epochs and patience must be positive")

    space = cfg.get("search_space", {})
    missing = SEARCH_KEYS - set(space)
    extra = set(space) - SEARCH_KEYS
    if missing or extra:
        raise ValueError(f"Unexpected search-space keys; missing={missing}, extra={extra}")
    batches = list(space["batch"].get("choices", []))
    allowed = list(cfg.get("resource_guard", {}).get("allowed_batches", []))
    if not batches or any(batch not in allowed for batch in batches):
        raise ValueError("Every batch choice must be listed in resource_guard.allowed_batches")
    for d_model in space["d_model"].get("choices", []):
        valid_heads = [h for h in space["n_heads"].get("choices", []) if d_model % h == 0]
        if not valid_heads:
            raise ValueError(f"d_model={d_model} has no compatible n_heads choice")
    head_sets = {
        tuple(h for h in space["n_heads"]["choices"] if d_model % h == 0)
        for d_model in space["d_model"]["choices"]
    }
    if len(head_sets) != 1:
        raise ValueError(
            "All d_model choices must share the same compatible n_heads choices; "
            "Optuna does not support a changing categorical distribution"
        )
    incumbent = incumbent_search_params(base)
    for name in ["batch", "d_model", "n_heads", "d_ff_multiplier", "num_layers", "embed_dim"]:
        if incumbent[name] not in space[name]["choices"]:
            raise ValueError(f"Incumbent {name}={incumbent[name]} is outside the search space")
    for name in ["lr", "dropout", "weight_decay"]:
        if not float(space[name]["low"]) <= incumbent[name] <= float(space[name]["high"]):
            raise ValueError(f"Incumbent {name}={incumbent[name]} is outside the search space")

    storage = str(cfg.get("storage", ""))
    if not storage.startswith("sqlite:///outputs/week7_optimization/"):
        raise ValueError("storage must be a project-local SQLite URL under outputs/week7_optimization")
    output_dir = Path(str(cfg.get("output_dir", "")))
    if output_dir.is_absolute() or output_dir.parts[:2] != ("outputs", "week7_optimization"):
        raise ValueError("output_dir must be relative and under outputs/week7_optimization")


def _suggest_float(trial: Any, name: str, spec: dict[str, Any]) -> float:
    kwargs = {"log": bool(spec.get("log", False))}
    if "step" in spec:
        kwargs["step"] = float(spec["step"])
        kwargs.pop("log", None)
    return float(trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash config, implementation and immutable model-input artifacts."""

    config = {
        key: value for key, value in cfg.items()
        if not key.startswith("_") and key not in {"storage", "output_dir"}
    }
    files = [
        _resolve(cfg["_config_path"]),
        _resolve(cfg["_base_config_path"]),
        ROOT / "src" / "optimization" / "week7.py",
        ROOT / "scripts" / "train.py",
        ROOT / "src" / "models" / "agformer.py",
        ROOT / "src" / "data" / "flow_dataset.py",
        ROOT / "src" / "features" / "temporal.py",
        ROOT / "src" / "training" / "metrics.py",
        ROOT / "data" / "raw" / "taxibj" / "BJ_Holiday.txt",
        ROOT / "data" / "processed" / "taxibj_clean.npz",
        ROOT / "data" / "processed" / "node_features_bj.npy",
        ROOT / "data" / "processed" / "graph_spatial_bj.pt",
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot fingerprint missing protocol artifacts: {missing}")
    payload = {
        "config": config,
        "artifacts": {str(path.relative_to(ROOT)): _sha256(path) for path in files},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def remaining_study_budget(
    trials: list[Any], total_trials: int, started_at_unix: float,
    timeout_hours: float, now_unix: float,
) -> tuple[int, float, int]:
    """Return remaining trials/time; RUNNING trials count as consumed after restart."""

    consumed_states = {"COMPLETE", "PRUNED", "FAIL", "RUNNING"}
    consumed = sum(trial.state.name in consumed_states for trial in trials)
    remaining_trials = max(0, int(total_trials) - consumed)
    elapsed = max(0.0, float(now_unix) - float(started_at_unix))
    remaining_seconds = max(0.0, float(timeout_hours) * 3600.0 - elapsed)
    return remaining_trials, remaining_seconds, consumed


def sample_trial_config(trial: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Sample one valid AGFormer configuration with head divisibility enforced."""

    space = cfg["search_space"]
    d_model = int(trial.suggest_categorical("d_model", space["d_model"]["choices"]))
    compatible_heads = [
        int(head) for head in space["n_heads"]["choices"] if d_model % int(head) == 0
    ]
    n_heads = int(trial.suggest_categorical("n_heads", compatible_heads))
    multiplier = int(trial.suggest_categorical(
        "d_ff_multiplier", space["d_ff_multiplier"]["choices"]
    ))
    sampled = {
        "lr": _suggest_float(trial, "lr", space["lr"]),
        "batch": int(trial.suggest_categorical("batch", space["batch"]["choices"])),
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_model * multiplier,
        "d_ff_multiplier": multiplier,
        "num_layers": int(trial.suggest_categorical("num_layers", space["num_layers"]["choices"])),
        "dropout": _suggest_float(trial, "dropout", space["dropout"]),
        "weight_decay": _suggest_float(trial, "weight_decay", space["weight_decay"]),
        "embed_dim": int(trial.suggest_categorical("embed_dim", space["embed_dim"]["choices"])),
    }
    trial_cfg = dict(cfg["_base_config"])
    trial_cfg.update({key: value for key, value in sampled.items() if key != "d_ff_multiplier"})
    trial_cfg.update({
        "seed": int(cfg["trial_seed"]),
        "epochs": int(cfg["max_epochs"]),
        "patience": int(cfg["patience"]),
        "num_workers": int(cfg["num_workers"]),
        "tag": f"{cfg['tag']}_trial_{trial.number:04d}",
        "save_path": str(Path(cfg["output_dir"]) / "trials" / f"trial_{trial.number:04d}" / "best.pt"),
        "log_dir": None,
    })
    trial.set_user_attr("resolved_d_ff", sampled["d_ff"])
    trial.set_user_attr("fixed_trial_seed", int(cfg["trial_seed"]))
    return trial_cfg


def resolve_frozen_trial_config(trial: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a config from a completed trial without asking Optuna to sample."""

    params = trial.params
    resolved = dict(cfg["_base_config"])
    resolved.update({
        "lr": float(params["lr"]),
        "batch": int(params["batch"]),
        "d_model": int(params["d_model"]),
        "n_heads": int(params["n_heads"]),
        "d_ff": int(params["d_model"]) * int(params["d_ff_multiplier"]),
        "num_layers": int(params["num_layers"]),
        "dropout": float(params["dropout"]),
        "weight_decay": float(params["weight_decay"]),
        "embed_dim": int(params["embed_dim"]),
        "seed": int(cfg["trial_seed"]),
        "num_workers": int(cfg["num_workers"]),
    })
    return resolved


def prepare_data(base_cfg: dict[str, Any]) -> PreparedData:
    """Build only train/validation datasets using the final-model data rules."""

    data, timestamps = load_taxibj()
    train_raw, val_raw, _ = split_taxibj(data)
    n_train, n_val = len(train_raw), len(val_raw)
    train_ts = timestamps[:n_train]
    val_ts = timestamps[n_train:n_train + n_val]
    holidays = load_bj_holidays()
    makeup_workdays = load_bj_makeup_workdays() if base_cfg.get("makeup_calendar", False) else None
    common: dict[str, Any] = {
        "tz_offset": 8,
        "holidays": holidays,
        "timestamps": train_ts,
        "filter_gaps": True,
        "include_time_features": True,
    }
    if makeup_workdays is not None:
        common["makeup_workdays"] = makeup_workdays
    train_ds = FlowDataset(
        train_raw, base_cfg["T_in"], base_cfg["T_out"], **common
    )
    val_kwargs = dict(common)
    val_kwargs["timestamps"] = val_ts
    val_ds = FlowDataset(
        val_raw, base_cfg["T_in"], base_cfg["T_out"],
        train_ds.mean, train_ds.std, **val_kwargs,
    )
    expected_features = int(train_ds[0][0].shape[-1])
    if expected_features != int(base_cfg["F_in"]):
        raise ValueError(
            f"Dataset produced F_in={expected_features}, expected {base_cfg['F_in']}"
        )
    return PreparedData(
        train=train_ds,
        val=val_ds,
        mean=float(train_ds.mean),
        std=float(train_ds.std),
        train_windows=len(train_ds),
        val_windows=len(val_ds),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, (tuple, list)):
        return type(value)(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


@torch.no_grad()
def validation_mae(
    model: nn.Module,
    loader: DataLoader,
    graphs: dict[str, Any],
    device: torch.device,
    mean: float,
    std: float,
) -> float:
    """Compute the same denormalized, zero-masked avg MAE as normal training."""

    model.eval()
    absolute_error = 0.0
    valid_count = 0
    for x, y in loader:
        x = _move(x, device)
        y = y.to(device, non_blocking=True)
        pred = model(x, graphs)
        pred_real = pred * std + mean
        true_real = y * std + mean
        mask = true_real.abs() > 0.0
        absolute_error += (pred_real[mask] - true_real[mask]).abs().sum().item()
        valid_count += int(mask.sum().item())
    if valid_count == 0:
        raise RuntimeError("Validation set contains no non-zero targets")
    return absolute_error / valid_count


class AGFormerObjective:
    """Callable Optuna objective with pruning, checkpointing and OOM cleanup."""

    def __init__(self, cfg: dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.data = prepare_data(cfg["_base_config"])
        graph_names = cfg["_base_config"].get("graphs", ["spatial"])
        self.graphs = load_graphs("bj", graph_names, device)

    def __call__(self, trial: Any) -> float:
        trial_cfg = sample_trial_config(trial, self.cfg)
        return self.run_config(trial, trial_cfg)

    def run_config(self, trial: Any, trial_cfg: dict[str, Any]) -> float:
        """Train one config against validation only; shared by search/promotion."""

        # Local import avoids coupling environment validation/tests to scripts.
        from scripts.train import build_model

        _seed_everything(int(trial_cfg["seed"]))
        generator = torch.Generator().manual_seed(int(trial_cfg["seed"]))
        train_loader = DataLoader(
            self.data.train,
            batch_size=trial_cfg["batch"],
            shuffle=True,
            num_workers=trial_cfg["num_workers"],
            pin_memory=True,
            generator=generator,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            self.data.val,
            batch_size=trial_cfg["batch"],
            shuffle=False,
            num_workers=trial_cfg["num_workers"],
            pin_memory=True,
            persistent_workers=False,
        )
        model: nn.Module | None = None
        checkpoint = _resolve(trial_cfg["save_path"])
        metrics_path = checkpoint.parent / "history.jsonl"
        config_path = checkpoint.parent / "config.yaml"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(trial_cfg, sort_keys=False), encoding="utf-8")
        start = time.perf_counter()
        best_mae = float("inf")
        best_epoch = 0
        wait = 0
        history: list[dict[str, float | int]] = []
        try:
            model = build_model(trial_cfg).to(self.device)
            parameter_millions = sum(p.numel() for p in model.parameters()) / 1e6
            trial.set_user_attr("parameter_millions", parameter_millions)
            limit = float(self.cfg["resource_guard"]["max_parameter_millions"])
            if parameter_millions > limit:
                import optuna
                trial.set_user_attr("resource_pruned", "parameter_limit")
                raise optuna.TrialPruned(
                    f"{parameter_millions:.3f}M parameters exceeds {limit:.3f}M"
                )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=trial_cfg["lr"],
                weight_decay=trial_cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )
            loss_fn = nn.HuberLoss()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            for epoch in range(1, int(trial_cfg["epochs"]) + 1):
                model.train()
                losses = []
                for x, y in train_loader:
                    x = _move(x, self.device)
                    y = y.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    pred = model(x, self.graphs)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
                val_mae = validation_mae(
                    model, val_loader, self.graphs, self.device,
                    self.data.mean, self.data.std,
                )
                if not np.isfinite(val_mae):
                    raise FloatingPointError(f"Non-finite validation MAE at epoch {epoch}")
                scheduler.step(val_mae)
                row: dict[str, float | int] = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_mae": val_mae,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(row)
                metrics_path.write_text(
                    "\n".join(json.dumps(item, ensure_ascii=False) for item in history) + "\n",
                    encoding="utf-8",
                )
                trial.report(val_mae, step=epoch)
                if val_mae < best_mae:
                    best_mae = val_mae
                    best_epoch = epoch
                    wait = 0
                    torch.save({
                        "epoch": epoch,
                        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "mean": self.data.mean,
                        "std": self.data.std,
                        "val_mae": val_mae,
                        "cfg": trial_cfg,
                        "optuna_trial": trial.number,
                    }, checkpoint)
                else:
                    wait += 1
                if trial.should_prune():
                    import optuna
                    trial.set_user_attr("best_val_mae", best_mae)
                    trial.set_user_attr("best_epoch", best_epoch)
                    raise optuna.TrialPruned(f"MedianPruner stopped trial at epoch {epoch}")
                if wait >= int(trial_cfg["patience"]):
                    break

            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("elapsed_seconds", time.perf_counter() - start)
            if self.device.type == "cuda":
                trial.set_user_attr(
                    "peak_cuda_mib", torch.cuda.max_memory_allocated(self.device) / 1024**2
                )
            return best_mae
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or not self.cfg["resource_guard"].get("prune_cuda_oom", True):
                raise
            import optuna
            trial.set_user_attr("resource_pruned", "cuda_oom")
            raise optuna.TrialPruned("CUDA out of memory") from exc
        finally:
            del model, train_loader, val_loader
            gc.collect()
            if self.device.type == "cuda" and self.cfg["resource_guard"].get("empty_cuda_cache_after_trial", True):
                torch.cuda.empty_cache()


class StandaloneTrial:
    """Minimal non-pruning trial adapter for full-budget promotion training."""

    def __init__(self, number: int = -1):
        self.number = number
        self.user_attrs: dict[str, Any] = {}

    def set_user_attr(self, name: str, value: Any) -> None:
        self.user_attrs[name] = value

    def report(self, value: float, step: int) -> None:
        del value, step

    def should_prune(self) -> bool:
        return False


def load_promotion_config(path: str | Path, optimization_cfg: dict[str, Any]) -> dict[str, Any]:
    """Load an exported candidate and enforce its validation-only contract."""

    candidate_path = _resolve(path)
    with candidate_path.open(encoding="utf-8") as handle:
        candidate = yaml.safe_load(handle)
    if not isinstance(candidate, dict):
        raise ValueError(f"Promotion config must be a mapping: {candidate_path}")
    if candidate.get("optimization_validation_only") is not True:
        raise ValueError("Promotion config must set optimization_validation_only=true")
    mismatches = {
        key: (candidate.get(key), expected)
        for key, expected in FROZEN_BASE_VALUES.items()
        if candidate.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Promotion config violates frozen Week7 data rules: {mismatches}")
    save_path = Path(str(candidate.get("save_path", "")))
    expected_root = Path(optimization_cfg["output_dir"]) / "promotion"
    if save_path.is_absolute() or not save_path.is_relative_to(expected_root):
        raise ValueError(f"Promotion save_path must be under {expected_root}")
    return candidate


def run_validation_promotion(
    optimization_cfg: dict[str, Any], candidate_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Train one promoted config without constructing or evaluating a test set."""

    current_fingerprint = protocol_fingerprint(optimization_cfg)
    if candidate_cfg.get("optimization_protocol_fingerprint") != current_fingerprint:
        raise RuntimeError(
            "Promotion protocol fingerprint differs from current config/data/code"
        )
    if optimization_cfg.get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this optimization config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_mib = free_bytes / 1024**2
        minimum = float(
            optimization_cfg["resource_guard"].get("min_free_cuda_mib_before_study", 0)
        )
        if free_mib < minimum:
            raise RuntimeError(
                f"Only {free_mib:.0f} MiB CUDA memory is free; "
                f"the configured minimum is {minimum:.0f} MiB"
            )
    objective = AGFormerObjective(optimization_cfg, device)
    trial = StandaloneTrial(int(candidate_cfg.get("optimization_source_trial", -1)))
    best_mae = objective.run_config(trial, candidate_cfg)
    result = {
        "tag": candidate_cfg["tag"],
        "role": candidate_cfg["optimization_role"],
        "best_validation_mae": best_mae,
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "elapsed_seconds": trial.user_attrs.get("elapsed_seconds"),
        "peak_cuda_mib": trial.user_attrs.get("peak_cuda_mib"),
        "train_windows": objective.data.train_windows,
        "validation_windows": objective.data.val_windows,
        "test_dataset_constructed": False,
        "test_evaluated": False,
        "checkpoint": candidate_cfg["save_path"],
    }
    summary_path = _resolve(candidate_cfg["save_path"]).parent / "validation_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_final_test(
    optimization_cfg: dict[str, Any], candidate_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Explicitly evaluate one promoted checkpoint on the frozen test set."""

    from scripts.train import build_model
    from src.training.metrics import compute_horizon_metrics

    checkpoint_path = _resolve(candidate_cfg["save_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Promotion checkpoint does not exist: {checkpoint_path}")
    output_path = checkpoint_path.parent / "final_test_metrics.json"
    if output_path.exists():
        raise RuntimeError(
            f"Final test was already evaluated for this checkpoint: {output_path}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if optimization_cfg.get("require_cuda", True) and device.type != "cuda":
        raise RuntimeError("CUDA is required by this optimization config")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint.get("cfg", {})
    identity_keys = list(FROZEN_BASE_VALUES) + [
        "d_model", "n_heads", "d_ff", "num_layers", "embed_dim",
        "dropout", "lr", "weight_decay", "batch", "seed", "tag",
    ]
    mismatches = {
        key: (checkpoint_cfg.get(key), candidate_cfg.get(key))
        for key in identity_keys
        if checkpoint_cfg.get(key) != candidate_cfg.get(key)
    }
    if mismatches:
        raise ValueError(f"Checkpoint/config mismatch blocks final test: {mismatches}")
    current_fingerprint = protocol_fingerprint(optimization_cfg)
    if candidate_cfg.get("optimization_protocol_fingerprint") != current_fingerprint:
        raise RuntimeError("Final-test protocol fingerprint differs from current config/data/code")
    data, timestamps = load_taxibj()
    train_raw, val_raw, test_raw = split_taxibj(data)
    test_start = len(train_raw) + len(val_raw)
    test_ts = timestamps[test_start:]
    test_ds = FlowDataset(
        test_raw, candidate_cfg["T_in"], candidate_cfg["T_out"],
        float(checkpoint["mean"]), float(checkpoint["std"]),
        timestamps=test_ts, tz_offset=8, holidays=load_bj_holidays(),
        filter_gaps=True, include_time_features=True,
    )
    if len(test_ds) != 4022:
        raise ValueError(f"Frozen final test must contain 4022 windows, got {len(test_ds)}")
    loader = DataLoader(
        test_ds, batch_size=int(candidate_cfg["batch"]), shuffle=False,
        num_workers=int(candidate_cfg["num_workers"]), pin_memory=device.type == "cuda",
    )
    model = build_model(candidate_cfg).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    graphs = load_graphs("bj", candidate_cfg.get("graphs", ["spatial"]), device)
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(_move(x, device), graphs)
            predictions.append(pred.cpu())
            targets.append(y)
    pred = torch.cat(predictions) * float(checkpoint["std"]) + float(checkpoint["mean"])
    true = torch.cat(targets) * float(checkpoint["std"]) + float(checkpoint["mean"])
    metrics = compute_horizon_metrics(pred, true)
    result = {
        "tag": candidate_cfg["tag"],
        "checkpoint": str(candidate_cfg["save_path"]),
        "test_windows": len(test_ds),
        "explicit_final_test_gate": True,
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_study(cfg: dict[str, Any]) -> Any:
    """Create or resume the configured SQLite-backed Optuna study."""

    import optuna

    sampler_cfg = cfg["sampler"]
    sampler = optuna.samplers.TPESampler(
        seed=int(cfg["optimization_seed"]),
        n_startup_trials=int(sampler_cfg["n_startup_trials"]),
        multivariate=bool(sampler_cfg.get("multivariate", True)),
    )
    pruner_cfg = cfg["pruner"]
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_cfg["n_startup_trials"]),
        n_warmup_steps=int(pruner_cfg["n_warmup_steps"]),
        interval_steps=int(pruner_cfg["interval_steps"]),
    )
    relative_db = cfg["storage"].removeprefix("sqlite:///")
    storage = f"sqlite:///{_resolve(relative_db)}"
    return optuna.create_study(
        study_name=cfg["study_name"],
        storage=storage,
        direction=cfg["direction"],
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )


def export_study(study: Any, cfg: dict[str, Any]) -> None:
    """Persist trials and full-fidelity configs for the best validation candidates."""

    output = _resolve(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    promotion = cfg["promotion"]
    completed = sorted(
        (trial for trial in study.trials if trial.value is not None and trial.state.name == "COMPLETE"),
        key=lambda trial: float(trial.value),
    )
    if not completed:
        raise RuntimeError("Study contains no completed trials to export")
    promotion_dir = output / "promotion" / "configs"
    promotion_dir.mkdir(parents=True, exist_ok=True)
    incumbent = dict(cfg["_base_config"])
    incumbent_tag = f"{cfg['tag']}_incumbent_seed{cfg['trial_seed']}"
    incumbent.update({
        "seed": int(cfg["trial_seed"]),
        "epochs": int(promotion["full_epochs"]),
        "patience": int(promotion["full_patience"]),
        "num_workers": int(cfg["num_workers"]),
        "tag": incumbent_tag,
        "save_path": str(Path(cfg["output_dir"]) / "promotion" / incumbent_tag / "best.pt"),
        "log_dir": str(Path(cfg["output_dir"]) / "promotion" / incumbent_tag / "tensorboard"),
        "optimization_validation_only": True,
        "optimization_role": "incumbent_control",
        "optimization_protocol_fingerprint": study.user_attrs.get("protocol_fingerprint"),
    })
    (promotion_dir / "incumbent_control.yaml").write_text(
        yaml.safe_dump(incumbent, sort_keys=False), encoding="utf-8"
    )
    incumbent_params = incumbent_search_params(cfg["_base_config"])
    challengers = [
        trial for trial in completed
        if trial.user_attrs.get("role") != "incumbent_control"
        and trial.params != incumbent_params
    ]
    candidates = []
    for rank, trial in enumerate(challengers[:int(promotion["top_k"])], start=1):
        candidate = resolve_frozen_trial_config(trial, cfg)
        candidate_tag = f"{cfg['tag']}_rank{rank}_seed{cfg['trial_seed']}"
        candidate.update({
            "epochs": int(promotion["full_epochs"]),
            "patience": int(promotion["full_patience"]),
            "tag": candidate_tag,
            "save_path": str(Path(cfg["output_dir"]) / "promotion" / candidate_tag / "best.pt"),
            "log_dir": str(Path(cfg["output_dir"]) / "promotion" / candidate_tag / "tensorboard"),
            "optimization_validation_only": True,
            "optimization_role": "challenger",
            "optimization_source_trial": trial.number,
            "optimization_protocol_fingerprint": study.user_attrs.get("protocol_fingerprint"),
        })
        config_name = f"rank{rank}_trial{trial.number:04d}.yaml"
        (promotion_dir / config_name).write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
        candidates.append({
            "rank": rank,
            "trial": trial.number,
            "validation_mae": trial.value,
            "config": str(Path(cfg["output_dir"]) / "promotion" / "configs" / config_name),
        })
    best = completed[0]
    if candidates:
        best_challenger = challengers[0]
        (output / "best_config.yaml").write_text(
            (promotion_dir / f"rank1_trial{best_challenger.number:04d}.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    summary = {
        "study_name": study.study_name,
        "direction": cfg["direction"],
        "best_trial": best.number,
        "best_validation_mae": best.value,
        "best_params": best.params,
        "best_is_incumbent": best.user_attrs.get("role") == "incumbent_control",
        "promotion_candidates": candidates,
        "incumbent_control_config": str(
            Path(cfg["output_dir"]) / "promotion" / "configs" / "incumbent_control.yaml"
        ),
        "train_windows": study.user_attrs.get("train_windows"),
        "validation_windows": study.user_attrs.get("validation_windows"),
        "test_evaluated_during_search": False,
        "promotion": promotion,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_optimization(cfg: dict[str, Any]) -> Any:
    """Run the study. Callers must guard this behind explicit user intent."""

    import optuna

    if cfg.get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this optimization config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mib = free_bytes / 1024**2
        minimum = float(cfg["resource_guard"].get("min_free_cuda_mib_before_study", 0))
        if free_mib < minimum:
            raise RuntimeError(
                f"Only {free_mib:.0f} MiB CUDA memory is free; "
                f"the configured minimum is {minimum:.0f} MiB"
            )
    output = _resolve(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    study = build_study(cfg)
    fingerprint = protocol_fingerprint(cfg)
    stored_fingerprint = study.user_attrs.get("protocol_fingerprint")
    if stored_fingerprint is not None and stored_fingerprint != fingerprint:
        raise RuntimeError(
            "Study protocol fingerprint differs from the current config/data/code; "
            "use a new study_name and output_dir"
        )
    if stored_fingerprint is None and study.trials:
        raise RuntimeError("Existing study has trials but no protocol fingerprint; refusing unsafe resume")
    study.set_user_attr("protocol_fingerprint", fingerprint)
    started_at = study.user_attrs.get("budget_started_at_unix")
    if started_at is None:
        started_at = time.time()
        study.set_user_attr("budget_started_at_unix", started_at)
    if cfg.get("enqueue_incumbent_trial", True) and not study.trials:
        study.enqueue_trial(
            incumbent_search_params(cfg["_base_config"]),
            user_attrs={"role": "incumbent_control"},
        )
    remaining_trials, remaining_seconds, consumed = remaining_study_budget(
        study.trials, int(cfg["n_trials"]), float(started_at),
        float(cfg["timeout_hours"]), time.time(),
    )
    study.set_user_attr("consumed_trial_budget", consumed)
    study.set_user_attr("remaining_trial_budget_at_resume", remaining_trials)
    study.set_user_attr("remaining_seconds_at_resume", remaining_seconds)
    if remaining_trials > 0 and remaining_seconds > 0:
        objective = AGFormerObjective(cfg, device)
        for key, value in {
            "train_windows": objective.data.train_windows,
            "validation_windows": objective.data.val_windows,
        }.items():
            existing = study.user_attrs.get(key)
            if existing is not None and existing != value:
                raise RuntimeError(f"Study {key}={existing} differs from current value {value}")
            study.set_user_attr(key, value)
    else:
        objective = None
    study.set_user_attr("test_evaluated_during_search", False)
    if device.type == "cuda":
        study.set_user_attr("cuda_total_mib", total_bytes / 1024**2)
        study.set_user_attr("cuda_free_mib_at_start", free_bytes / 1024**2)
    if objective is not None:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=remaining_seconds,
            n_jobs=1,
            gc_after_trial=True,
            catch=(FloatingPointError,),
        )
    export_study(study, cfg)
    if cfg.get("keep_only_best_checkpoint", True):
        best_dir = output / "trials" / f"trial_{study.best_trial.number:04d}"
        for trial_dir in (output / "trials").glob("trial_*"):
            if trial_dir != best_dir and trial_dir.is_dir():
                checkpoint = trial_dir / "best.pt"
                if checkpoint.exists():
                    checkpoint.unlink()
    return study
