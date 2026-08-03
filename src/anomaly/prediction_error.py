"""Prediction-error anomaly scoring using Week 4 forecasting checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.train import build_model
from src.data.flow_dataset import FlowDataset, load_graphs, load_taxibj, load_taxinyc
from src.data.flow_dataset import split_taxibj, split_taxinyc
from src.features.temporal import load_bj_holidays, load_bj_makeup_workdays, us_federal_holidays


ROOT = Path(__file__).resolve().parents[2]


def clean_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove cached AGFormer adjacency buffers from checkpoint state."""

    return {k: v for k, v in state.items() if not k.startswith("_A_hat")}


def load_city_data(city: str) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if city == "bj":
        data, timestamps = load_taxibj()
        splits = split_taxibj(data)
    elif city == "nyc":
        data, timestamps = load_taxinyc()
        splits = split_taxinyc(data)
    else:
        raise ValueError(f"Unsupported city: {city}")
    return data, timestamps, splits


def split_offsets(splits: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, int]:
    train_raw, val_raw, _ = splits
    return {"train": 0, "val": len(train_raw), "test": len(train_raw) + len(val_raw)}


def get_split_arrays(
    city: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return raw split data, split timestamps, and absolute offset."""

    data, timestamps, splits = load_city_data(city)
    offsets = split_offsets(splits)
    names = {"train": 0, "val": 1, "test": 2}
    if split not in names:
        raise ValueError("split must be one of train/val/test")
    raw = splits[names[split]]
    start = offsets[split]
    ts = timestamps[start:start + len(raw)]
    return raw, ts, start


def _dataset_kwargs(cfg: dict[str, Any], timestamps: np.ndarray, city: str) -> dict[str, Any]:
    time_enc = cfg.get("time_enc", False)
    real_time = cfg.get("real_time_context", False)
    filter_gaps = cfg.get("filter_gaps", False)
    kwargs: dict[str, Any] = {"timestamps": timestamps, "filter_gaps": filter_gaps}
    if time_enc and real_time:
        years = sorted({int(str(t)[:4]) for t in timestamps})
        kwargs.update({
            "include_time_features": True,
            "tz_offset": 8 if city == "bj" else -5,
            "holidays": load_bj_holidays() if city == "bj" else us_federal_holidays(years),
        })
        if city == "bj" and cfg.get("makeup_calendar", False):
            kwargs["makeup_workdays"] = load_bj_makeup_workdays()
    elif time_enc:
        # Week 4 best uses real_time_context. This branch is kept for older ckpts.
        kwargs = {"time_offset": 0}
    return kwargs


def build_split_dataset(
    cfg: dict[str, Any],
    split: str,
    *,
    mean: float,
    std: float,
) -> tuple[FlowDataset, np.ndarray, np.ndarray]:
    """Build a FlowDataset and its h1 absolute target indices."""

    city = cfg.get("city", "bj")
    raw, timestamps, offset = get_split_arrays(city, split)
    ds = FlowDataset(
        raw,
        cfg["T_in"],
        cfg["T_out"],
        mean,
        std,
        **_dataset_kwargs(cfg, timestamps, city),
    )
    target_abs = offset + ds.valid_starts + cfg["T_in"]
    target_ts = timestamps[ds.valid_starts + cfg["T_in"]]
    return ds, target_abs.astype(np.int64), target_ts


def load_model_from_checkpoint(
    tag: str,
    *,
    device: torch.device,
    checkpoint_root: str | Path = ROOT / "outputs/checkpoints",
) -> tuple[torch.nn.Module, dict[str, Any], float, float, Path]:
    """Load one Week 4 checkpoint by tag."""

    path = Path(checkpoint_root) / tag / "best.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    incompatible = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("_A_hat")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {tag}: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    model.eval()
    return model, cfg, float(ckpt["mean"]), float(ckpt["std"]), path


@torch.no_grad()
def predict_ensemble_horizon(
    tags: list[str],
    *,
    split: str = "test",
    horizon: int = 1,
    batch_size: int | None = None,
    num_workers: int | None = None,
    device: torch.device | str = "cpu",
    max_windows: int | None = None,
    window_start: int = 0,
) -> dict[str, Any]:
    """Predict one horizon with a mean ensemble.

    Returns arrays with shape ``(num_windows, N)`` for ``pred``, ``true`` and
    ``residual``. ``horizon=1`` means the first future 30-minute step.
    """

    if not tags:
        raise ValueError("At least one checkpoint tag is required")
    if horizon < 1:
        raise ValueError("horizon is 1-indexed and must be >= 1")

    device = torch.device(device)
    first = torch.load(
        ROOT / "outputs/checkpoints" / tags[0] / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_cfg = first["cfg"]
    if horizon > int(base_cfg["T_out"]):
        raise ValueError(f"horizon={horizon} exceeds T_out={base_cfg['T_out']}")
    mean = float(first["mean"])
    std = float(first["std"])

    ds, target_abs, target_ts = build_split_dataset(base_cfg, split, mean=mean, std=std)
    eval_ds = ds
    if max_windows is not None and int(max_windows) <= 0:
        raise ValueError(f"max_windows must be positive when provided, got {max_windows}")
    start = int(window_start)
    if start < 0:
        start = max(len(ds) + start, 0)
    if start >= len(ds):
        raise ValueError(f"window_start={window_start} resolves to {start}, exceeds {len(ds)}")
    stop = len(ds) if max_windows is None else min(start + int(max_windows), len(ds))
    if stop <= start:
        raise ValueError(
            f"Selected empty window range [{start}, {stop}) from split={split}, len={len(ds)}"
        )
    if start != 0 or stop != len(ds):
        eval_ds = Subset(ds, range(start, stop))
        target_abs = target_abs[start:stop]
        target_ts = target_ts[start:stop]
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size or int(base_cfg.get("batch", 8)),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else base_cfg.get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )
    graphs = load_graphs(
        base_cfg.get("city", "bj"),
        base_cfg.get("graphs", ["spatial"]),
        device=device,
    )

    pred_sum = None
    true_ref = None
    members = []
    for tag in tags:
        model, cfg, ckpt_mean, ckpt_std, path = load_model_from_checkpoint(tag, device=device)
        if abs(ckpt_mean - mean) > 1e-6 or abs(ckpt_std - std) > 1e-6:
            raise RuntimeError(f"Checkpoint {tag} uses different normalization stats")
        preds = []
        true = []
        for x, y in loader:
            out = model(x.to(device), graphs)
            preds.append(out[..., horizon - 1].cpu())
            true.append(y[..., horizon - 1])
        pred = torch.cat(preds, dim=0) * std + mean
        true_cat = torch.cat(true, dim=0) * std + mean
        pred_sum = pred if pred_sum is None else pred_sum + pred
        if true_ref is None:
            true_ref = true_cat
        members.append({
            "tag": tag,
            "checkpoint": str(path.relative_to(ROOT)),
            "cfg": cfg,
        })

    assert pred_sum is not None and true_ref is not None
    pred_ensemble = (pred_sum / len(tags)).numpy().astype(np.float32)
    true_np = true_ref.numpy().astype(np.float32)
    residual = np.abs(pred_ensemble - true_np).astype(np.float32)
    return {
        "split": split,
        "horizon": horizon,
        "target_indices": target_abs,
        "timestamps": target_ts,
        "pred": pred_ensemble,
        "true": true_np,
        "residual": residual,
        "members": members,
        "cfg": base_cfg,
    }
