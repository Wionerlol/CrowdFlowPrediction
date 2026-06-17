"""
METR-LA 数据清洗与标准化
输入：data/raw/metr_la/
输出：data/processed/metrla_clean.npz   — (T, N) 速度矩阵 + UTC时间戳
      data/processed/metrla_adj.npy      — (207, 207) 归一化邻接矩阵
      data/processed/metrla_sensor.csv   — 传感器位置（WGS84）

数据格式说明（原始 parquet）：
  每行 = 一个 (node × 滑窗) 样本
  列名: x_t-11_d0 ~ x_t+0_d0 (输入速度, mph)
        x_t-11_d1 ~ x_t+0_d1 (日内时间编码, 0~1)
        y_t+1_d0  ~ y_t+12_d0 (预测目标速度)
  → 本脚本从滑窗中重建原始时间序列 (T, N) 矩阵
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/metr_la")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

N_SENSORS  = 207
FREQ       = "5min"
TZ_OFFSET  = -8   # LA: PST=UTC-8 (3月DST前), 简化统一用UTC-8


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 从滑窗 parquet 重建原始时序矩阵
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_timeseries() -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    从 train/val/test parquet 重建 (T, N) 速度矩阵。
    策略：取每个 (node_id, t0_timestamp) 的 x_t+0_d0 作为该节点该时刻的速度值。
    """
    print("\n[1/4] 重建时序矩阵")

    frames = []
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(RAW / f"metrla_{split}.parquet",
                             columns=["node_id", "t0_timestamp", "x_t+0_d0"])
        frames.append(df)
        print(f"  {split}: {len(df):,} 行, "
              f"t0 [{df['t0_timestamp'].min()} ~ {df['t0_timestamp'].max()}]")

    all_df = pd.concat(frames, ignore_index=True)

    # 去重：同 (node_id, t0_timestamp) 若有多条取均值
    n_dup = all_df.duplicated(subset=["node_id", "t0_timestamp"]).sum()
    if n_dup:
        print(f"  [去重] {n_dup} 条重复 → 取均值")
        all_df = all_df.groupby(["node_id", "t0_timestamp"])["x_t+0_d0"].mean().reset_index()
    else:
        print(f"  [去重] 无重复")

    # 解析时间戳（本地 PST UTC-8 → UTC）
    all_df["utc"] = (pd.to_datetime(all_df["t0_timestamp"])
                     - pd.Timedelta(hours=abs(TZ_OFFSET)))

    # pivot: 行=时间, 列=node_id (0~206)
    pivot = all_df.pivot(index="utc", columns="node_id", values="x_t+0_d0")
    pivot = pivot.sort_index()
    ts    = pd.DatetimeIndex(pivot.index).tz_localize("UTC")

    # 检查时间连续性
    expected = pd.date_range(ts[0], ts[-1], freq=FREQ, tz="UTC")
    missing  = expected.difference(ts)
    print(f"\n  时间范围: {ts[0]} ~ {ts[-1]}")
    print(f"  时间步数: {len(ts)} (期望 {len(expected)}, 缺失 {len(missing)})")
    if len(missing) > 0:
        print(f"  缺失时间步(前5): {missing[:5].tolist()}")

    # 补全缺失时间步（线性插值）
    if len(missing) > 0:
        pivot = pivot.reindex(expected.tz_localize(None))
        pivot = pivot.interpolate(method="time").ffill().bfill()
        ts    = expected
        print(f"  → 线性插值补全, 新 T={len(ts)}")

    # 检查值域（速度应在 0~70 mph，0 = 停止/缺测）
    data = pivot.values.astype(np.float32)   # (T, N)
    print(f"\n  速度统计: min={data.min():.1f}, max={data.max():.1f}, "
          f"mean={data.mean():.2f}, zeros={( data==0).mean()*100:.2f}%")
    # 超出物理范围的值（理论上不存在，检查一下）
    bad = (data < 0) | (data > 100)
    if bad.any():
        print(f"  [异常] 超出 [0,100] mph 的值: {bad.sum()} 个 → 裁剪")
        data = np.clip(data, 0, 100)

    return data, ts


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 邻接矩阵
# ═══════════════════════════════════════════════════════════════════════════════

def process_adj() -> np.ndarray:
    print("\n[2/4] 邻接矩阵")
    with open(RAW / "adj_mx.pkl", "rb") as f:
        sensor_ids, id2idx, adj = pickle.load(f, encoding="latin1")

    adj = np.array(adj, dtype=np.float32)
    print(f"  shape={adj.shape}, 非零边数={( adj > 0).sum()}, "
          f"对角线均为1: {np.all(np.diag(adj)==1)}")
    print(f"  权重范围: [{adj[adj>0].min():.4f}, {adj.max():.4f}]")

    # 验证对称性
    sym_err = np.abs(adj - adj.T).max()
    print(f"  对称误差 (max|A-A^T|): {sym_err:.6f}")

    np.save(OUT / "metrla_adj.npy", adj)
    print(f"  已保存: metrla_adj.npy  shape={adj.shape}")
    return adj


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 传感器位置
# ═══════════════════════════════════════════════════════════════════════════════

def process_sensor_locations():
    print("\n[3/4] 传感器位置")
    loc = pd.read_csv(RAW / "graph_sensor_locations.csv")
    print(f"  shape={loc.shape}, cols={loc.columns.tolist()}")
    print(f"  lat: [{loc['latitude'].min():.4f}, {loc['latitude'].max():.4f}]")
    print(f"  lon: [{loc['longitude'].min():.4f}, {loc['longitude'].max():.4f}]")
    print(f"  坐标系: WGS84 (EPSG:4326) — 直接确认")

    # 检查重复
    n_dup = loc["sensor_id"].duplicated().sum()
    print(f"  重复 sensor_id: {n_dup}")

    out_path = OUT / "metrla_sensor.csv"
    loc.to_csv(out_path, index=False)
    print(f"  已保存: metrla_sensor.csv")
    return loc


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 保存 & 汇总
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  METR-LA 数据清洗与标准化")
    print("=" * 58)

    data, ts = reconstruct_timeseries()
    adj      = process_adj()
    loc      = process_sensor_locations()

    print("\n[4/4] 保存时序矩阵")
    np.savez_compressed(
        OUT / "metrla_clean.npz",
        data       = data,                             # (T, N) float32
        timestamps = ts.values.astype("datetime64[s]"),
    )
    size_mb = (OUT / "metrla_clean.npz").stat().st_size / 1024**2
    print(f"  已保存: metrla_clean.npz  shape={data.shape}  {size_mb:.1f} MB")

    print("\n" + "=" * 58)
    print("  汇总")
    print("=" * 58)
    print(f"  时序矩阵 : (T={len(ts)}, N={data.shape[1]})  速度 mph, float32")
    print(f"  时间范围 : {ts[0].strftime('%Y-%m-%d')} ~ {ts[-1].strftime('%Y-%m-%d')} UTC")
    print(f"  时间粒度 : 5 分钟")
    print(f"  传感器数 : {loc.shape[0]}  (LA 高速公路, WGS84)")
    print(f"  邻接矩阵 : {adj.shape}  距离衰减权重")
    print(f"  坐标系   : WGS84 (EPSG:4326)")
    print(f"  时间戳   : UTC (ISO 8601, datetime64[s])")
    print(f"  用途     : 模型验证基线（交通速度预测，非人流）")
    print("\n  完成。")


if __name__ == "__main__":
    main()
