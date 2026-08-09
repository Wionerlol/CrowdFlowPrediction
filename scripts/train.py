"""
统一训练入口。

用法：
    python scripts/train.py --config configs/lstm.yaml
    python scripts/train.py --config configs/stgcn.yaml
    python scripts/train.py --config configs/stgcn_multigraph.yaml
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
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


def build_periodic_features(data: np.ndarray, timestamps: np.ndarray,
                            mean: float, std: float,
                            lags: list[int], include_deltas: bool) -> np.ndarray:
    """Build exact-timestamp lag covariates using observations from the past."""
    ts = timestamps.astype("datetime64[s]")
    raw = data.reshape(len(data), 2, -1).transpose(0, 2, 1).astype(np.float32)
    pieces = []
    for lag in lags:
        target = ts - np.timedelta64(30 * lag, "m")
        pos = np.searchsorted(ts, target)
        valid = pos < len(ts)
        valid[valid] &= ts[pos[valid]] == target[valid]
        lagged = np.full_like(raw, np.nan)
        lagged[valid] = (raw[pos[valid]] - mean) / (std + 1e-8)
        pieces.append(lagged)
        if include_deltas:
            delta = np.full_like(raw, np.nan)
            delta[valid] = (raw[valid] - raw[pos[valid]]) / (std + 1e-8)
            pieces.append(delta)
    return np.concatenate(pieces, axis=-1)


def load_static_features(cfg: dict) -> torch.Tensor | None:
    """Load leakage-free static node features for contextual ablations.

    The generated 41-D matrix contains eight full-dataset historical flow
    statistics at columns [27:35]. Those are deliberately excluded here so
    validation/test information cannot leak into a strict ablation.
    """
    if not cfg.get("use_static", False):
        return None
    path = Path(cfg.get(
        "static_path", f"data/processed/node_features_{cfg['city']}.npy"
    ))
    features = np.load(path, allow_pickle=False)
    if features.ndim != 2 or features.shape[0] != cfg["N"]:
        raise ValueError(f"Unexpected static feature shape at {path}: {features.shape}")
    features = np.concatenate([features[:, :27], features[:, 35:]], axis=1)
    if features.shape[1] != 33 or not np.isfinite(features).all():
        raise ValueError(f"Invalid leakage-free static features: {features.shape}")
    return torch.from_numpy(features.astype(np.float32, copy=False))


def build_model(cfg: dict) -> nn.Module:
    model_name = cfg["model"]
    T_in  = cfg["T_in"]
    T_out = cfg["T_out"]
    static_features = load_static_features(cfg)

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
            F_in=cfg.get("F_in", 2),
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
    if model_name == "agformer":
        from src.models.agformer import AGFormer
        return AGFormer(
            N=cfg["N"],
            T_in=T_in, T_out=T_out,
            F_in=cfg.get("F_in", 2),
            d_model=cfg.get("d_model", 64),
            n_heads=cfg.get("n_heads", 4),
            d_ff=cfg.get("d_ff", 128),
            num_layers=cfg.get("num_layers", 3),
            embed_dim=cfg.get("embed_dim", 10),
            graph_names=cfg.get("graphs"),
            static_features=static_features,
            context_fusion=cfg.get("context_fusion", "add"),
            flow_dim=cfg.get("flow_dim", 2),
            time_dim=cfg.get("time_dim"),
            weather_dim=cfg.get("weather_dim", 0),
            future_time_dim=cfg.get("future_time_dim", 0),
            use_linear_output=cfg.get("use_linear_output", False),
            dropout=cfg.get("dropout", 0.1),
        )
    if model_name == "timesnet":
        from src.models.timesnet import TimesNet
        return TimesNet(
            T_in=T_in, T_out=T_out,
            d_model=cfg.get("d_model", 64),
            d_ff=cfg.get("d_ff", 128),
            num_layers=cfg.get("num_layers", 2),
            top_k=cfg.get("top_k", 3),
            dropout=cfg.get("dropout", 0.1),
        )
    if model_name == "spacetimeformer":
        from src.models.spacetimeformer import Spacetimeformer
        return Spacetimeformer(
            N=cfg["N"],
            T_in=T_in, T_out=T_out,
            F_in=cfg.get("F_in", 2),
            d_model=cfg.get("d_model", 16),
            n_heads=cfg.get("n_heads", 2),
            d_ff=cfg.get("d_ff", 32),
            num_layers=cfg.get("num_layers", 2),
            chunk_size=cfg.get("chunk_size", 256),
            num_global_tokens=cfg.get("num_global_tokens", 8),
            static_features=static_features,
            flow_dim=cfg.get("flow_dim", 2),
            time_dim=cfg.get("time_dim"),
            future_time_dim=cfg.get("future_time_dim", 0),
            separate_value_time=cfg.get("separate_value_time", True),
            use_given_embedding=cfg.get("use_given_embedding", True),
            use_linear_output=cfg.get("use_linear_output", False),
            context_fusion=cfg.get("context_fusion", "add"),
            dropout=cfg.get("dropout", 0.1),
        )
    if model_name == "gwnet":
        from src.models.gwnet import GraphWaveNet
        return GraphWaveNet(
            N=cfg["N"],
            T_in=T_in, T_out=T_out,
            F_in=cfg.get("F_in", 2),
            residual_ch=cfg.get("residual_ch", 32),
            skip_ch=cfg.get("skip_ch", 64),
            dilations=cfg.get("dilations"),
            kernel_size=cfg.get("kernel_size", 2),
            embed_dim=cfg.get("embed_dim", 10),
            dropout=cfg.get("dropout", 0.3),
        )
    if model_name == "gwnet_multigraph":
        from src.models.gwnet import MultiGraphWaveNet
        return MultiGraphWaveNet(
            N=cfg["N"],
            T_in=T_in, T_out=T_out,
            residual_ch=cfg.get("residual_ch", 32),
            skip_ch=cfg.get("skip_ch", 64),
            graph_names=cfg.get("graphs"),
            dilations=cfg.get("dilations"),
            kernel_size=cfg.get("kernel_size", 2),
            embed_dim=cfg.get("embed_dim", 10),
            dropout=cfg.get("dropout", 0.3),
        )
    raise ValueError(f"未知模型：{model_name}")


# 无显式图结构的模型
_NO_GRAPH_MODELS = {"lstm", "gru", "timesnet", "spacetimeformer"}

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
    if cfg.get("optimization_validation_only", False):
        raise ValueError(
            "This Optuna promotion config is validation-only. Use "
            "scripts/train_optuna_promotion.py; test evaluation requires the "
            "separate scripts/evaluate_optuna_final.py --run-final-test gate."
        )

    # CLI 参数覆盖 yaml
    if args.t_in  is not None: cfg["T_in"]  = args.t_in
    if args.batch is not None: cfg["batch"] = args.batch
    if args.tag   is not None: cfg["tag"]   = args.tag

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        data, timestamps = load_taxibj()
        train_raw, val_raw, test_raw = split_taxibj(data)
    else:
        data, timestamps = load_taxinyc()
        train_raw, val_raw, test_raw = split_taxinyc(data)

    n_train, n_val = len(train_raw), len(val_raw)
    train_ts = timestamps[:n_train]
    val_ts = timestamps[n_train:n_train + n_val]
    test_ts = timestamps[n_train + n_val:]

    T_in  = cfg["T_in"]
    T_out = cfg["T_out"]
    # 时间编码：time_enc=true 时向 x 追加 4 维正弦特征，F_in 变为 6
    time_enc = cfg.get("time_enc", False)
    t_offset_train = 0
    t_offset_val   = len(train_raw)
    t_offset_test  = len(train_raw) + len(val_raw)
    real_time = cfg.get("real_time_context", False)
    filter_gaps = cfg.get("filter_gaps", False)
    if real_time:
        from src.features.temporal import (
            load_bj_holidays, load_bj_makeup_workdays, us_federal_holidays,
        )
        years = sorted({int(str(t)[:4]) for t in timestamps})
        holidays = load_bj_holidays() if city == "bj" else us_federal_holidays(years)
        makeup_workdays = (load_bj_makeup_workdays()
                           if city == "bj" and cfg.get("makeup_calendar", False)
                           else None)
        tz_offset = 8 if city == "bj" else -5
        common = {"tz_offset": tz_offset, "holidays": holidays,
                  "filter_gaps": filter_gaps}
        if makeup_workdays is not None:
            common["makeup_workdays"] = makeup_workdays
        tenc_kw_train = {**common, "timestamps": train_ts,
                         "include_time_features": True} if time_enc else {
            "timestamps": train_ts, "filter_gaps": filter_gaps}
        tenc_kw_val = {**common, "timestamps": val_ts,
                       "include_time_features": True} if time_enc else {
            "timestamps": val_ts, "filter_gaps": filter_gaps}
        tenc_kw_test = {**common, "timestamps": test_ts,
                        "include_time_features": True} if time_enc else {
            "timestamps": test_ts, "filter_gaps": filter_gaps}
    else:
        tenc_kw_train = {"time_offset": t_offset_train} if time_enc else {}
        tenc_kw_val   = {"time_offset": t_offset_val}   if time_enc else {}
        tenc_kw_test  = {"time_offset": t_offset_test}  if time_enc else {}

    if cfg.get("use_weather", False):
        from src.data.weather_features import build_weather_features
        weather_path = cfg.get(
            "weather_path", f"data/processed/weather_{city}_merged.csv"
        )
        train_weather, weather_stats = build_weather_features(train_ts, weather_path)
        val_weather, _ = build_weather_features(val_ts, weather_path, weather_stats)
        test_weather, _ = build_weather_features(test_ts, weather_path, weather_stats)
        tenc_kw_train["weather_features"] = train_weather
        tenc_kw_val["weather_features"] = val_weather
        tenc_kw_test["weather_features"] = test_weather
    if cfg.get("future_time_context", False):
        if not time_enc or not real_time:
            raise ValueError("future_time_context requires real timestamp time encoding")
        tenc_kw_train["return_future_time"] = True
        tenc_kw_val["return_future_time"] = True
        tenc_kw_test["return_future_time"] = True

    periodic_lags = cfg.get("periodic_lags", [])
    if periodic_lags:
        train_mean, train_std = float(train_raw.mean()), float(train_raw.std())
        periodic = build_periodic_features(
            data, timestamps, train_mean, train_std,
            [int(x) for x in periodic_lags],
            bool(cfg.get("periodic_deltas", False)),
        )
        tenc_kw_train["dynamic_features"] = periodic[:n_train]
        tenc_kw_val["dynamic_features"] = periodic[n_train:n_train + n_val]
        tenc_kw_test["dynamic_features"] = periodic[n_train + n_val:]

    train_ds = FlowDataset(train_raw, T_in, T_out, **tenc_kw_train)
    val_ds   = FlowDataset(val_raw,   T_in, T_out, train_ds.mean, train_ds.std, **tenc_kw_val)
    test_ds  = FlowDataset(test_raw,  T_in, T_out, train_ds.mean, train_ds.std, **tenc_kw_test)
    mean, std = train_ds.mean, train_ds.std
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  归一化: mean={mean:.2f}  std={std:.2f}")
    if time_enc:
        print(f"  时间编码: 已启用（F_in={cfg.get('F_in')}，"
              f"{'真实时间戳' if real_time else '序号推算'}）")
    if filter_gaps:
        print("  间断过滤: 已启用（窗口内必须全为连续30分钟）")
    if cfg.get("makeup_calendar", False):
        print("  调休日历: 已启用（官方周末补班覆盖）")
    if cfg.get("use_weather", False):
        print(f"  天气特征: 已启用（weather_dim={cfg.get('weather_dim')}）")
    if cfg.get("future_time_context", False):
        print(f"  未来时间解码: 已启用（future_time_dim={cfg.get('future_time_dim')}）")
    if periodic_lags:
        print(f"  周期滞后: {periodic_lags}，差值={cfg.get('periodic_deltas', False)}")

    num_workers = cfg.get("num_workers", min(4, __import__("os").cpu_count() or 1))
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"],
                               shuffle=True, num_workers=num_workers, pin_memory=True,
                               generator=loader_generator)
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
        print("  无显式图结构")

    # ── 模型 ──────────────────────────────────────────────────────────
    print(f"\n[3/4] 初始化模型...")
    model = build_model(cfg)
    resume_from = cfg.get("resume_from")
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "val_mae" in ckpt:
            cfg["initial_best_val_mae"] = float(ckpt["val_mae"])
        print(f"  恢复模型: {resume_from}"
              f"（epoch={ckpt.get('epoch', '?')}, "
              f"val_MAE={ckpt.get('val_mae', float('nan')):.4f}）")
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
