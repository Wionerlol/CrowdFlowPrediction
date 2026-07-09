"""Evaluate AGFormer+linear multi-seed checkpoints and their mean ensemble."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train import build_model
from src.data.flow_dataset import FlowDataset, load_graphs, load_taxibj, split_taxibj
from src.features.temporal import load_bj_holidays
from src.training.metrics import compute_horizon_metrics


TAGS = [
    "agformer_gate_context_linear_seed42",
    "agformer_gate_context_linear_seed7",
    "agformer_gate_context_linear_seed21",
    "agformer_gate_context_linear_seed123",
]


def _clean_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if not k.startswith("_A_hat")}


def _seed_from_tag(tag: str) -> int | None:
    match = re.search(r"seed(\d+)$", tag)
    return int(match.group(1)) if match else None


def residual_stats(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    resid = (pred - true).abs().reshape(-1).float()
    # torch.quantile has size limits on very large tensors. Use a deterministic
    # evenly-spaced sample for residual percentiles; mean/std remain exact.
    max_q = 5_000_000
    if resid.numel() > max_q:
        idx = torch.linspace(0, resid.numel() - 1, max_q, dtype=torch.long)
        q_source = resid[idx]
    else:
        q_source = resid
    q = torch.quantile(q_source, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    return {
        "mean": float(resid.mean()),
        "std": float(resid.std(unbiased=False)),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
    }


def load_test_loader(cfg: dict[str, Any],
                     mean: float,
                     std: float) -> tuple[DataLoader, int]:
    data, timestamps = load_taxibj()
    train_raw, val_raw, test_raw = split_taxibj(data)
    n_train, n_val = len(train_raw), len(val_raw)
    test_ts = timestamps[n_train + n_val:]

    test_ds = FlowDataset(
        test_raw,
        cfg["T_in"],
        cfg["T_out"],
        mean,
        std,
        timestamps=test_ts,
        tz_offset=8,
        holidays=load_bj_holidays(),
        filter_gaps=cfg.get("filter_gaps", False),
        include_time_features=cfg.get("time_enc", False),
    )
    loader = DataLoader(
        test_ds,
        batch_size=cfg.get("batch", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )
    return loader, len(test_ds)


@torch.no_grad()
def predict_one(tag: str,
                loader: DataLoader,
                graphs: dict[str, Any],
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ckpt_path = ROOT / "outputs" / "checkpoints" / tag / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    incompatible = model.load_state_dict(_clean_state(ckpt["model"]), strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("_A_hat")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {tag}: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    model.eval()

    preds, true = [], []
    for x, y in loader:
        pred = model(x.to(device), graphs).cpu()
        preds.append(pred)
        true.append(y)

    pred = torch.cat(preds) * ckpt["std"] + ckpt["mean"]
    true_cat = torch.cat(true) * ckpt["std"] + ckpt["mean"]
    meta = {
        "tag": tag,
        "seed": _seed_from_tag(tag),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "cfg": cfg,
    }
    return pred, true_cat, meta


def append_csv(tag: str, cfg: dict[str, Any], metrics: dict[str, dict[str, float]]) -> None:
    result_path = ROOT / "outputs/results/week3_metrics.csv"
    write_header = not result_path.exists()
    with result_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "city", "T_in", "T_out", "tag",
                             "horizon", "MAE", "RMSE", "MAPE", "config"])
        for h, m in metrics.items():
            writer.writerow([
                cfg.get("model", "agformer_ensemble"),
                cfg.get("city", "bj"),
                cfg["T_in"],
                cfg["T_out"],
                tag,
                h,
                f"{m['MAE']:.4f}",
                f"{m['RMSE']:.4f}",
                f"{m['MAPE']:.2f}",
                json.dumps(cfg, ensure_ascii=False),
            ])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    first = torch.load(
        ROOT / "outputs/checkpoints" / TAGS[0] / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    base_cfg = first["cfg"]
    loader, windows = load_test_loader(base_cfg, first["mean"], first["std"])
    graphs = load_graphs("bj", base_cfg.get("graphs", ["spatial"]), device=device)

    singles = []
    ensemble_sum = None
    true_ref = None
    for tag in TAGS:
        pred, true, meta = predict_one(tag, loader, graphs, device)
        metrics = compute_horizon_metrics(pred, true)
        stats = residual_stats(pred, true)
        singles.append({**meta, "metrics": metrics, "residual_stats": stats})
        ensemble_sum = pred if ensemble_sum is None else ensemble_sum + pred
        true_ref = true
        print(f"{tag} avg={metrics['avg']}")

    assert ensemble_sum is not None and true_ref is not None
    ensemble_pred = ensemble_sum / len(TAGS)
    ensemble_metrics = compute_horizon_metrics(ensemble_pred, true_ref)
    ensemble_stats = residual_stats(ensemble_pred, true_ref)

    ensemble_tag = "agformer_gate_context_linear_ensemble_4seed"
    ensemble_cfg = dict(base_cfg)
    ensemble_cfg.update({
        "model": "agformer_ensemble",
        "tag": ensemble_tag,
        "ensemble_members": TAGS,
        "ensemble_method": "mean",
    })
    append_csv(ensemble_tag, ensemble_cfg, ensemble_metrics)

    result = {
        "windows": windows,
        "members": TAGS,
        "singles": singles,
        "ensemble": {
            "tag": ensemble_tag,
            "method": "mean",
            "metrics": ensemble_metrics,
            "residual_stats": ensemble_stats,
        },
    }
    out = ROOT / "outputs/results/agformer_gate_context_linear_ensemble_4seed.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["ensemble"], ensure_ascii=False, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
