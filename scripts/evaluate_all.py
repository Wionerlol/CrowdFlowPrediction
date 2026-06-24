"""
汇总所有模型在 TaxiBJ 测试集上的对比指标。

用法：
    python scripts/evaluate_all.py [--city bj]

会从 outputs/checkpoints/ 自动发现所有已训练的模型，
逐一加载并在测试集上评估，最终打印对比表并保存到 outputs/results/。
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.flow_dataset import (
    FlowDataset, load_graphs, load_taxibj, load_taxinyc,
    split_taxibj, split_taxinyc,
)
from src.training.metrics import compute_horizon_metrics
from src.training.trainer import Trainer


MODEL_REGISTRY = {
    "lstm": {
        "cls_path": "src.models.lstm.NodeLSTM",
        "graphs":   None,
    },
    "stgcn": {
        "cls_path": "src.models.multi_graph_stgcn.SingleGraphSTGCN",
        "graphs":   ["spatial"],
    },
    "stgcn_multigraph": {
        "cls_path": "src.models.multi_graph_stgcn.MultiGraphSTGCN",
        "graphs":   ["spatial", "poi", "flow", "transit"],
    },
}


def load_checkpoint(ckpt_path: Path, device: torch.device):
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def build_model_from_ckpt(model_name: str, ckpt: dict) -> nn.Module:
    cfg = ckpt.get("cfg", {})
    T_in  = cfg.get("T_in",  12)
    T_out = cfg.get("T_out", 12)

    if model_name == "lstm":
        from src.models.lstm import NodeLSTM
        m = NodeLSTM(T_in=T_in, T_out=T_out,
                     hidden=cfg.get("hidden", 64),
                     num_layers=cfg.get("num_layers", 2))
    elif model_name == "stgcn":
        from src.models.multi_graph_stgcn import SingleGraphSTGCN
        m = SingleGraphSTGCN(T_in=T_in, T_out=T_out,
                             n_ch=(cfg.get("sp_ch", 64), cfg.get("out_ch", 16)),
                             K=cfg.get("K", 3))
    elif model_name == "stgcn_multigraph":
        from src.models.multi_graph_stgcn import MultiGraphSTGCN
        m = MultiGraphSTGCN(T_in=T_in, T_out=T_out,
                            n_ch=(cfg.get("sp_ch", 64), cfg.get("out_ch", 16)),
                            K=cfg.get("K", 3),
                            graph_names=cfg.get("graphs"))
    else:
        raise ValueError(f"未知模型: {model_name}")
    return m


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader,
                   graphs: dict | None, mean: float, std: float,
                   device: torch.device) -> dict:
    model.eval()
    all_pred, all_true = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x, graphs)
        all_pred.append(pred.cpu())
        all_true.append(y)
    pred_cat = torch.cat(all_pred) * std + mean
    true_cat = torch.cat(all_true) * std + mean
    return compute_horizon_metrics(pred_cat, true_cat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city",   default="bj")
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    city = args.city
    print(f"评估城市: {city} | 设备: {device}\n")

    # ── 测试数据 ─────────────────────────────────────────────────────
    if city == "bj":
        data, _ = load_taxibj()
        train_raw, _, test_raw = split_taxibj(data)
    else:
        data, _ = load_taxinyc()
        train_raw, _, test_raw = split_taxinyc(data)

    # 用训练集统计量归一化
    train_ds = FlowDataset(train_raw, T_in=12, T_out=12)
    mean, std = train_ds.mean, train_ds.std
    test_ds  = FlowDataset(test_raw, T_in=12, T_out=12, mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=args.batch,
                              shuffle=False, num_workers=2)

    # ── 发现检查点 ───────────────────────────────────────────────────
    ckpt_root = Path("outputs/checkpoints")
    results   = {}

    for model_name, meta in MODEL_REGISTRY.items():
        ckpt_path = ckpt_root / model_name / "best.pt"
        if not ckpt_path.exists():
            print(f"  [{model_name}] 检查点不存在，跳过")
            continue

        print(f"  [{model_name}] 加载 {ckpt_path}")
        ckpt   = load_checkpoint(ckpt_path, device)
        model  = build_model_from_ckpt(model_name, ckpt).to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in ckpt["model"].items()})

        graph_names = meta["graphs"]
        graphs = (load_graphs(city=city, graph_names=graph_names, device=device)
                  if graph_names else None)

        metrics = evaluate_model(model, test_loader, graphs, mean, std, device)
        results[model_name] = metrics
        print(f"    avg  MAE={metrics['avg']['MAE']:.4f}  "
              f"RMSE={metrics['avg']['RMSE']:.4f}  "
              f"MAPE={metrics['avg']['MAPE']:.2f}%")

    # ── 打印对比表 ───────────────────────────────────────────────────
    if not results:
        print("\n没有找到任何已训练模型，请先运行 train.py。")
        return

    horizons = ["h1", "h3", "h6", "avg"]
    header_h  = [f"{h}:MAE" for h in horizons] + [f"{h}:RMSE" for h in horizons]
    print(f"\n{'='*80}")
    print(f"{'模型':<22}" + "".join(f"{h:>10}" for h in header_h))
    print("-" * 80)
    for model_name, metrics in results.items():
        row = ""
        for h in horizons:
            if h in metrics:
                row += f"{metrics[h]['MAE']:>10.4f}"
            else:
                row += f"{'—':>10}"
        for h in horizons:
            if h in metrics:
                row += f"{metrics[h]['RMSE']:>10.4f}"
            else:
                row += f"{'—':>10}"
        print(f"{model_name:<22}{row}")
    print("=" * 80)

    # ── 保存 CSV ─────────────────────────────────────────────────────
    import csv
    out_path = Path(f"outputs/results/week3_eval_{city}.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "horizon", "MAE", "RMSE", "MAPE"])
        for model_name, metrics in results.items():
            for h, m in metrics.items():
                w.writerow([model_name, h,
                             f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.2f}"])
    print(f"\n完整结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
