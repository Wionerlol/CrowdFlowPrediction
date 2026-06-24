"""
统一训练入口。

用法：
    python scripts/train.py --config configs/lstm.yaml
    python scripts/train.py --config configs/stgcn.yaml
    python scripts/train.py --config configs/stgcn_multigraph.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.flow_dataset import (
    FlowDataset, load_graphs, load_taxibj, load_taxinyc,
    split_taxibj, split_taxinyc,
)
from src.training.trainer import Trainer


def build_model(cfg: dict) -> nn.Module:
    model_name = cfg["model"]
    T_in  = cfg["T_in"]
    T_out = cfg["T_out"]

    if model_name == "lstm":
        from src.models.lstm import NodeLSTM
        return NodeLSTM(
            T_in=T_in, T_out=T_out,
            hidden=cfg["hidden"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    if model_name == "gru":
        from src.models.gru import NodeGRU
        return NodeGRU(
            T_in=T_in, T_out=T_out,
            hidden=cfg["hidden"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
    if model_name == "gcn":
        from src.models.gnn_models import GCNPredictor
        return GCNPredictor(
            T_in=T_in, T_out=T_out,
            hidden=cfg["hidden"],
        )
    if model_name == "gat":
        from src.models.gnn_models import GATPredictor
        return GATPredictor(
            T_in=T_in, T_out=T_out,
            hidden=cfg["hidden"],
            heads=cfg.get("heads", 4),
        )
    if model_name == "stgcn":
        from src.models.multi_graph_stgcn import SingleGraphSTGCN
        return SingleGraphSTGCN(
            T_in=T_in, T_out=T_out,
            n_ch=(cfg["sp_ch"], cfg["out_ch"]),
            K=cfg["K"],
        )
    if model_name == "stgcn_multigraph":
        from src.models.multi_graph_stgcn import MultiGraphSTGCN
        return MultiGraphSTGCN(
            T_in=T_in, T_out=T_out,
            n_ch=(cfg["sp_ch"], cfg["out_ch"]),
            K=cfg["K"],
            graph_names=cfg.get("graphs"),
        )
    raise ValueError(f"未知模型：{model_name}")


# 无需图的模型
_NO_GRAPH_MODELS = {"lstm", "gru"}

def build_graphs(cfg: dict, device: torch.device) -> dict | None:
    if cfg["model"] in _NO_GRAPH_MODELS:
        return None
    city  = cfg["city"]
    names = cfg.get("graphs", ["spatial"])
    return load_graphs(city=city, graph_names=names, device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--device", default=None, help="cuda / cpu（默认自动选择）")
    parser.add_argument("--t_in",   type=int, default=None, help="覆盖 config 中的 T_in")
    parser.add_argument("--batch",  type=int, default=None, help="覆盖 config 中的 batch")
    parser.add_argument("--tag",    type=str, default=None, help="运行标签（写入结果 CSV）")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI 参数覆盖 yaml
    if args.t_in  is not None: cfg["T_in"]  = args.t_in
    if args.batch is not None: cfg["batch"] = args.batch
    if args.tag   is not None: cfg["tag"]   = args.tag

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"模型: {cfg['model']} | 城市: {cfg['city']}")

    # ── 数据 ─────────────────────────────────────────────────────────
    city = cfg["city"]
    print("\n[1/4] 加载数据...")
    if city == "bj":
        data, _ = load_taxibj()
        train_raw, val_raw, test_raw = split_taxibj(data)
    else:
        data, _ = load_taxinyc()
        train_raw, val_raw, test_raw = split_taxinyc(data)

    T_in  = cfg["T_in"]
    T_out = cfg["T_out"]
    train_ds = FlowDataset(train_raw, T_in, T_out)
    val_ds   = FlowDataset(val_raw,   T_in, T_out, train_ds.mean, train_ds.std)
    test_ds  = FlowDataset(test_raw,  T_in, T_out, train_ds.mean, train_ds.std)
    mean, std = train_ds.mean, train_ds.std
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  归一化: mean={mean:.2f}  std={std:.2f}")

    num_workers = min(4, __import__("os").cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"],
                               shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch"],
                               shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch"],
                               shuffle=False, num_workers=num_workers, pin_memory=True)

    # ── 图结构 ────────────────────────────────────────────────────────
    print(f"\n[2/4] 加载图结构...")
    graphs = build_graphs(cfg, device)
    if graphs:
        for name, g in graphs.items():
            print(f"  G_{name}: {g['edge_index'].shape[1]} 条边")
    else:
        print("  无图（LSTM 模式）")

    # ── 模型 ──────────────────────────────────────────────────────────
    print(f"\n[3/4] 初始化模型...")
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {params:.3f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    loss_fn = nn.HuberLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        cfg=cfg,
        save_path=cfg["save_path"],
        log_dir=cfg.get("log_dir"),
    )

    # ── 训练 ──────────────────────────────────────────────────────────
    print(f"\n[4/4] 开始训练（最多 {cfg['epochs']} epochs，patience={cfg['patience']}）...\n")

    try:
        import mlflow
        mlflow.set_experiment("week3_baselines")
        with mlflow.start_run(run_name=f"{cfg['model']}_{city}"):
            mlflow.log_params(cfg)
            test_metrics = trainer.fit(
                train_loader, val_loader, test_loader,
                graphs, mean, std,
                epochs=cfg["epochs"], patience=cfg["patience"],
            )
            for h, m in test_metrics.items():
                mlflow.log_metrics({f"test_{h}_{k}": v for k, v in m.items()})
    except Exception:
        # MLflow 不可用时直接训练
        test_metrics = trainer.fit(
            train_loader, val_loader, test_loader,
            graphs, mean, std,
            epochs=cfg["epochs"], patience=cfg["patience"],
        )

    # ── 保存结果 ──────────────────────────────────────────────────────
    import csv, json
    result_path = Path("outputs/results/week3_metrics.csv")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not result_path.exists()
    with open(result_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "city", "T_in", "T_out", "tag",
                        "horizon", "MAE", "RMSE", "MAPE", "config"])
        tag = cfg.get("tag", "")
        for h, m in test_metrics.items():
            w.writerow([cfg["model"], city,
                        cfg["T_in"], cfg["T_out"], tag, h,
                        f"{m['MAE']:.4f}", f"{m['RMSE']:.4f}", f"{m['MAPE']:.2f}",
                        json.dumps(cfg)])
    print(f"\n结果已追加到 {result_path}")


if __name__ == "__main__":
    main()
