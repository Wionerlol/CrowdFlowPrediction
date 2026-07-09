"""Evaluate the original AGFormer checkpoint on periodic-lag-valid test windows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train import build_model
from src.data.flow_dataset import FlowDataset, load_graphs, load_taxibj, split_taxibj
from src.features.temporal import load_bj_holidays
from src.training.metrics import compute_horizon_metrics


def periodic_valid(timestamps: np.ndarray, lags: list[int]) -> np.ndarray:
    ts = timestamps.astype("datetime64[s]")
    valid = np.ones(len(ts), dtype=bool)
    for lag in lags:
        target = ts - np.timedelta64(30 * lag, "m")
        pos = np.searchsorted(ts, target)
        found = pos < len(ts)
        found[found] &= ts[pos[found]] == target[found]
        valid &= found
    return valid


@torch.no_grad()
def main() -> None:
    checkpoint_path = ROOT / "outputs/checkpoints/context_gate/agformer_context/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    data, timestamps = load_taxibj()
    train_raw, val_raw, test_raw = split_taxibj(data)
    n_train, n_val = len(train_raw), len(val_raw)
    test_ts = timestamps[n_train + n_val:]

    test_ds = FlowDataset(
        test_raw, cfg["T_in"], cfg["T_out"], checkpoint["mean"], checkpoint["std"],
        timestamps=test_ts, tz_offset=8, holidays=load_bj_holidays(),
        filter_gaps=True, include_time_features=True,
    )
    valid_time = periodic_valid(timestamps, [48, 336])[n_train + n_val:]
    bad = (~valid_time).astype(np.int32)
    prefix = np.concatenate([[0], np.cumsum(bad)])
    valid_encoder = (prefix[cfg["T_in"]:] - prefix[:-cfg["T_in"]]) == 0
    keep = [i for i, start in enumerate(test_ds.valid_starts)
            if valid_encoder[start]]
    subset = Subset(test_ds, keep)
    if len(subset) != 3433:
        raise RuntimeError(f"Expected 3433 matched windows, got {len(subset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = dict(checkpoint["model"])
    state.pop("_A_hat", None)  # Old single-graph cache; recomputed by current model.
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or incompatible.missing_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.eval()
    graphs = load_graphs("bj", ["spatial"], device=device)
    loader = DataLoader(subset, batch_size=cfg["batch"], shuffle=False,
                        num_workers=2, pin_memory=True)
    pred, true = [], []
    for x, y in loader:
        pred.append(model(x.to(device), graphs).cpu())
        true.append(y)
    pred = torch.cat(pred) * checkpoint["std"] + checkpoint["mean"]
    true = torch.cat(true) * checkpoint["std"] + checkpoint["mean"]
    metrics = compute_horizon_metrics(pred, true)
    result = {"checkpoint": str(checkpoint_path), "windows": len(subset),
              "metrics": metrics}
    out = ROOT / "outputs/results/agformer_periodic_subset_baseline.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
