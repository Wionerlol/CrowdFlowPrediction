"""
第二周 Task 1：多源数据清洗与标准化
处理对象：TaxiBJ (BJ13-BJ16) / TaxiNYC / OpenWeather (北京+纽约)
输出目录：data/processed/

产出文件：
  taxibj_clean.npz          — 清洗后流量数组 + UTC时间戳
  taxinyc_clean.npz         — 清洗后流量数组 + UTC时间戳
  weather_bj_30min.csv      — 北京气象，30min粒度，UTC对齐
  weather_nyc_30min.csv     — 纽约气象，30min粒度，UTC对齐
  grid_coords_bj.npy        — (1024,2) 网格中心 WGS84 lon/lat
  grid_coords_nyc.npy       — (75,2)   网格中心 WGS84 lon/lat
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import pandas as pd

# ── 路径配置 ─────────────────────────────────────────────────────────────────
RAW   = Path("data/raw")
OUT   = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# TaxiBJ 网格地理范围（WGS84）
BJ_LON = (116.25, 116.75)   # 经度 西→东
BJ_LAT = (39.75,  40.25)    # 纬度 南→北
BJ_H, BJ_W = 32, 32

# TaxiNYC 网格地理范围（WGS84，曼哈顿核心区）
NYC_LON = (-74.0166, -73.9004)
NYC_LAT = (40.6996,  40.9196)
NYC_H, NYC_W = 15, 5

OUTLIER_ZSCORE = 5.0   # 每格 z-score 超过此值视为异常


# ═══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def parse_taxibj_dates(raw_dates: np.ndarray, tz_offset_hours: int = 8) -> pd.DatetimeIndex:
    """
    将 TaxiBJ/TaxiNYC 的日期字符串 YYYYMMDDSS 转换为 UTC DatetimeIndex。
    SS = 1-based 30min 步编号（01=00:00, 02=00:30, ..., 48=23:30 本地时）。
    """
    strs = [s.decode() if isinstance(s, bytes) else s for s in raw_dates]
    timestamps = []
    for s in strs:
        date_part = s[:8]          # YYYYMMDD
        step      = int(s[8:10])   # 01-48
        local_dt  = pd.Timestamp(date_part) + pd.Timedelta(minutes=30 * (step - 1))
        utc_dt    = local_dt - pd.Timedelta(hours=tz_offset_hours)
        timestamps.append(utc_dt)
    return pd.DatetimeIndex(timestamps, name="utc")


def check_duplicates(ts: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    dupes = ts.duplicated()
    n = dupes.sum()
    if n:
        print(f"  [!] {name}: {n} 重复时间步 → 保留首次出现")
        return ts[~dupes], ~dupes
    print(f"  [ok] {name}: 无重复时间步")
    return ts, np.ones(len(ts), dtype=bool)


def check_gaps(ts: pd.DatetimeIndex, freq: str, name: str):
    expected = pd.date_range(ts[0], ts[-1], freq=freq)
    missing  = expected.difference(ts)
    if len(missing):
        print(f"  [!] {name}: {len(missing)} 个缺失时间步")
        for m in missing[:5]:
            print(f"       {m}")
        if len(missing) > 5:
            print(f"       ...（共 {len(missing)} 个）")
    else:
        print(f"  [ok] {name}: 时间序列连续，无缺口")
    return missing


def fill_gaps(data: np.ndarray, ts: pd.DatetimeIndex,
              freq: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """在 data (T, C, H, W) 中线性插值补全缺失时间步。"""
    full_ts  = pd.date_range(ts[0], ts[-1], freq=freq)
    if len(full_ts) == len(ts):
        return data, ts

    T, C, H, W = data.shape
    df = pd.DataFrame(
        data.reshape(T, -1),
        index=ts
    )
    df = df.reindex(full_ts).interpolate(method="time").fillna(0)
    return df.values.reshape(-1, C, H, W), full_ts


def detect_outliers(data: np.ndarray, threshold: float = OUTLIER_ZSCORE):
    """
    逐网格 z-score 异常检测。
    data: (T, C, H, W)
    返回 outlier_mask (T, C, H, W) bool
    注：TaxiBJ 是聚合数据，无原始轨迹漂移点，此处以统计异常代替。
    """
    mean = data.mean(axis=0, keepdims=True)   # (1, C, H, W)
    std  = data.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-6, 1e-6, std)
    z    = np.abs((data - mean) / std)
    return z > threshold


def fix_outliers(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """将异常值替换为前后线性插值。"""
    T, C, H, W = data.shape
    out = data.copy().astype(np.float32)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                bad_idx = np.where(mask[:, c, h, w])[0]
                if len(bad_idx) == 0:
                    continue
                series = out[:, c, h, w].copy()
                series[bad_idx] = np.nan
                sr = pd.Series(series).interpolate(method="linear",
                                                    limit_direction="both")
                out[:, c, h, w] = sr.values
    return out


def build_grid_coords(lon_range, lat_range, H, W) -> np.ndarray:
    """
    生成 H×W 网格的中心点坐标 (WGS84)。
    返回 (H*W, 2) 数组，列为 [lon, lat]，行顺序为 row-major (由北→南, 由西→东)。
    """
    lons = np.linspace(lon_range[0], lon_range[1], W + 1)
    lats = np.linspace(lat_range[1], lat_range[0], H + 1)   # 北→南
    lon_centers = (lons[:-1] + lons[1:]) / 2
    lat_centers = (lats[:-1] + lats[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    coords = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)
    return coords.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TaxiBJ 清洗
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_taxibj():
    print("\n" + "="*60)
    print("【1/4】TaxiBJ 数据清洗")
    print("="*60)

    segments, all_dates = [], []
    for y in [13, 14, 15, 16]:
        p = RAW / f"taxibj/BJ{y:02d}_M32x32_T30_InOut.h5"
        with h5py.File(p, "r") as f:
            seg   = f["data"][:]
            dates = f["date"][:]
        ts = parse_taxibj_dates(dates, tz_offset_hours=8)
        print(f"\n  BJ{y:02d}: T={len(seg)}, "
              f"UTC [{ts[0]}] ~ [{ts[-1]}], "
              f"max={seg.max():.0f}")
        segments.append(seg)
        all_dates.append(ts)

    data = np.concatenate(segments, axis=0).astype(np.float32)  # (T,2,32,32)
    ts   = all_dates[0].append(all_dates[1:])
    print(f"\n  合并后: T={len(data)}, shape={data.shape}")

    # ── 去重 ──────────────────────────────────────────────────────────────────
    print("\n  [去重]")
    _, keep_mask = check_duplicates(ts, "TaxiBJ")
    data = data[keep_mask]
    ts   = ts[keep_mask]

    # ── 时间连续性检查 ─────────────────────────────────────────────────────────
    print("\n  [时间连续性]")
    # BJ 数据跨年且有季节性缺口（BJ13: 7-10月, BJ14: 3-6月 等），不做全局补全
    # 只在每个年份段内检查
    year_bounds = [(all_dates[i][0], all_dates[i][-1]) for i in range(4)]
    for i, (t0, t1) in enumerate(year_bounds):
        seg_mask  = (ts >= t0) & (ts <= t1)
        seg_ts    = ts[seg_mask]
        missing   = check_gaps(seg_ts, "30min", f"BJ{[13,14,15,16][i]:02d}")

    # ── 异常值检测与修复 ───────────────────────────────────────────────────────
    print("\n  [异常值检测] z-score 阈值 =", OUTLIER_ZSCORE)
    mask = detect_outliers(data)
    n_out = mask.sum()
    frac  = n_out / data.size * 100
    print(f"  异常点: {n_out} / {data.size} ({frac:.3f}%)")
    if n_out > 0:
        print(f"  最大异常 z-score: {((data - data.mean(axis=0,keepdims=True)) / (data.std(axis=0,keepdims=True)+1e-6)).max():.1f}")
        data = fix_outliers(data, mask)
        print(f"  已用线性插值修复")

    # ── 基础统计 ──────────────────────────────────────────────────────────────
    print(f"\n  [统计]")
    print(f"  inflow  — mean={data[:,0].mean():.2f}, std={data[:,0].std():.2f}, "
          f"max={data[:,0].max():.0f}, zeros={( data[:,0]==0).mean()*100:.1f}%")
    print(f"  outflow — mean={data[:,1].mean():.2f}, std={data[:,1].std():.2f}, "
          f"max={data[:,1].max():.0f}, zeros={( data[:,1]==0).mean()*100:.1f}%")

    # ── 网格坐标映射（WGS84）──────────────────────────────────────────────────
    coords = build_grid_coords(BJ_LON, BJ_LAT, BJ_H, BJ_W)
    np.save(OUT / "grid_coords_bj.npy", coords)
    print(f"\n  网格中心坐标 (WGS84) 已保存: grid_coords_bj.npy  shape={coords.shape}")
    print(f"  lon [{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]  "
          f"lat [{coords[:,1].min():.4f}, {coords[:,1].max():.4f}]")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        OUT / "taxibj_clean.npz",
        data       = data,
        timestamps = ts.values.astype("datetime64[s]"),
    )
    print(f"\n  已保存: taxibj_clean.npz  shape={data.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TaxiNYC 清洗
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_taxinyc():
    print("\n" + "="*60)
    print("【2/4】TaxiNYC 数据清洗")
    print("="*60)

    with h5py.File(RAW / "taxinyc/NYC2014.h5", "r") as f:
        data  = f["data"][:].astype(np.float32)   # (17520,2,15,5)
        dates = f["date"][:]

    # NYC 使用 UTC-5（东部标准时间），夏令时期间 UTC-4，简化处理统一用 UTC-5
    ts = parse_taxibj_dates(dates, tz_offset_hours=-5)
    print(f"  shape={data.shape}, UTC [{ts[0]}] ~ [{ts[-1]}]")
    print(f"  max={data.max():.0f}, zeros={(data==0).mean()*100:.1f}%")

    # ── 去重 ──────────────────────────────────────────────────────────────────
    print("\n  [去重]")
    _, keep_mask = check_duplicates(ts, "TaxiNYC")
    data = data[keep_mask]
    ts   = ts[keep_mask]

    # ── 时间连续性 ─────────────────────────────────────────────────────────────
    print("\n  [时间连续性]")
    check_gaps(ts, "30min", "TaxiNYC")

    # ── 异常值检测 ─────────────────────────────────────────────────────────────
    print("\n  [异常值检测] z-score 阈值 =", OUTLIER_ZSCORE)
    mask  = detect_outliers(data)
    n_out = mask.sum()
    frac  = n_out / data.size * 100
    print(f"  异常点: {n_out} / {data.size} ({frac:.3f}%)")
    if n_out > 0:
        data = fix_outliers(data, mask)
        print(f"  已用线性插值修复")

    # ── 统计 ──────────────────────────────────────────────────────────────────
    print(f"\n  [统计]")
    print(f"  inflow  — mean={data[:,0].mean():.2f}, std={data[:,0].std():.2f}, max={data[:,0].max():.0f}")
    print(f"  outflow — mean={data[:,1].mean():.2f}, std={data[:,1].std():.2f}, max={data[:,1].max():.0f}")

    # ── 网格坐标映射（WGS84）──────────────────────────────────────────────────
    coords = build_grid_coords(NYC_LON, NYC_LAT, NYC_H, NYC_W)
    np.save(OUT / "grid_coords_nyc.npy", coords)
    print(f"\n  网格中心坐标 (WGS84) 已保存: grid_coords_nyc.npy  shape={coords.shape}")
    print(f"  lon [{coords[:,0].min():.4f}, {coords[:,0].max():.4f}]  "
          f"lat [{coords[:,1].min():.4f}, {coords[:,1].max():.4f}]")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        OUT / "taxinyc_clean.npz",
        data       = data,
        timestamps = ts.values.astype("datetime64[s]"),
    )
    print(f"\n  已保存: taxinyc_clean.npz  shape={data.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OpenWeather 气象数据清洗
# ═══════════════════════════════════════════════════════════════════════════════

WEATHER_MAIN_MAP = {
    "Clear": 0, "Clouds": 1, "Rain": 2, "Drizzle": 3,
    "Thunderstorm": 4, "Snow": 5, "Mist": 6, "Fog": 7,
    "Haze": 8, "Smoke": 9, "Dust": 10, "Sand": 11,
    "Squall": 12, "Tornado": 13,
}

DROP_COLS = ["timezone", "city_name", "lat", "lon",
             "visibility", "sea_level", "grnd_level",
             "wind_gust", "rain_3h", "snow_3h",
             "feels_like", "temp_min", "temp_max",
             "weather_description", "weather_icon", "dt"]

KEEP_COLS = ["temp", "dew_point", "pressure", "humidity",
             "wind_speed", "wind_deg", "rain_1h", "snow_1h",
             "clouds_all", "weather_id", "weather_main"]


def preprocess_weather(city: str, csv_path: Path, out_name: str,
                       clip_start: str, clip_end: str):
    print(f"\n  [{city}]  {csv_path.name}")
    df = pd.read_csv(csv_path)

    # 解析时间，设为索引（UTC）
    df["utc"] = pd.to_datetime(df["dt_iso"].str.replace(r"\s+\+\d+ UTC$", "", regex=True),
                                utc=True)
    df = df.set_index("utc").sort_index()

    # 去重
    n_dup = df.index.duplicated().sum()
    if n_dup:
        print(f"    [去重] {n_dup} 条重复 → 删除")
        df = df[~df.index.duplicated(keep="first")]
    else:
        print(f"    [去重] 无重复")

    # 保留有效字段
    df = df[[c for c in KEEP_COLS if c in df.columns]]

    # 填充降水 NaN → 0（无降水即为0）
    for col in ["rain_1h", "snow_1h"]:
        if col in df.columns:
            n_nan = df[col].isna().sum()
            df[col] = df[col].fillna(0.0)
            print(f"    [{col}] {n_nan} 个 NaN → 0")

    # weather_main → 整数编码
    if "weather_main" in df.columns:
        df["weather_code"] = df["weather_main"].map(WEATHER_MAIN_MAP).fillna(15).astype(int)
        df = df.drop(columns=["weather_main"])

    # 其他字段缺失：前向填充
    n_nan_before = df.isna().sum().sum()
    df = df.ffill().bfill()
    n_nan_after  = df.isna().sum().sum()
    if n_nan_before > 0:
        print(f"    [插值] 其他字段 {n_nan_before} 个 NaN → 前向/后向填充，剩余 {n_nan_after}")

    # 时间连续性检查（1H 粒度）
    expected_1h = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
    missing_1h  = expected_1h.difference(df.index)
    if len(missing_1h):
        print(f"    [时间gap] {len(missing_1h)} 个缺失小时 → reindex 后插值")
        df = df.reindex(expected_1h).interpolate(method="time").ffill().bfill()
    else:
        print(f"    [时间] 1H 序列连续，无缺口")

    # 重采样 1H → 30min（前向填充，天气在30min内不变）
    df_30min = df.resample("30min").ffill()

    # 裁剪到与流量数据对齐的时间范围
    df_30min = df_30min.loc[clip_start:clip_end]
    print(f"    [裁剪] UTC {df_30min.index[0]} ~ {df_30min.index[-1]}, "
          f"T={len(df_30min)}")

    # 保存
    out_path = OUT / out_name
    df_30min.to_csv(out_path)
    print(f"    已保存: {out_name}  shape={df_30min.shape}")
    return df_30min


def preprocess_weather_all():
    print("\n" + "="*60)
    print("【3/4】OpenWeather 气象数据清洗")
    print("="*60)

    ow_dir = RAW / "openweather"
    bj_csv  = next(ow_dir.glob("Beijing*.csv"))
    nyc_csv = next(ow_dir.glob("New_York*.csv"))

    # TaxiBJ 时间范围（UTC）：BJ13起始 2013-07-01 00:00 UTC+8 → 2013-06-30 16:00 UTC
    #                         BJ16终止 ~2016-04-10
    preprocess_weather(
        city="北京", csv_path=bj_csv, out_name="weather_bj_30min.csv",
        clip_start="2013-06-30 16:00:00+00:00",
        clip_end  ="2016-04-10 23:30:00+00:00",
    )

    # TaxiNYC 时间范围：2014全年 UTC-5 → UTC: 2014-01-01 05:00 ~ 2014-12-31 23:30
    preprocess_weather(
        city="纽约", csv_path=nyc_csv, out_name="weather_nyc_30min.csv",
        clip_start="2014-01-01 05:00:00+00:00",
        clip_end  ="2015-01-01 04:30:00+00:00",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 数据质量汇总报告
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "="*60)
    print("【4/4】处理结果汇总")
    print("="*60)

    files = [
        "taxibj_clean.npz", "taxinyc_clean.npz",
        "grid_coords_bj.npy", "grid_coords_nyc.npy",
        "weather_bj_30min.csv", "weather_nyc_30min.csv",
    ]
    for fname in files:
        p = OUT / fname
        if p.exists():
            size_mb = p.stat().st_size / 1024**2
            print(f"  ✓ {fname:<30} {size_mb:>7.1f} MB")
        else:
            print(f"  ✗ {fname}  (未生成)")

    # 验证 taxibj
    bj = np.load(OUT / "taxibj_clean.npz")
    ts_bj = pd.DatetimeIndex(bj["timestamps"])
    print(f"\n  TaxiBJ : T={len(ts_bj)}, "
          f"UTC {ts_bj[0].strftime('%Y-%m-%d')} ~ {ts_bj[-1].strftime('%Y-%m-%d')}")

    # 验证 taxinyc
    nyc = np.load(OUT / "taxinyc_clean.npz")
    ts_nyc = pd.DatetimeIndex(nyc["timestamps"])
    print(f"  TaxiNYC: T={len(ts_nyc)}, "
          f"UTC {ts_nyc[0].strftime('%Y-%m-%d')} ~ {ts_nyc[-1].strftime('%Y-%m-%d')}")

    # 坐标范围
    bj_coords  = np.load(OUT / "grid_coords_bj.npy")
    nyc_coords = np.load(OUT / "grid_coords_nyc.npy")
    print(f"\n  网格坐标系统: WGS84 (EPSG:4326)")
    print(f"  TaxiBJ  网格中心: lon [{bj_coords[:,0].min():.4f}, {bj_coords[:,0].max():.4f}]"
          f"  lat [{bj_coords[:,1].min():.4f}, {bj_coords[:,1].max():.4f}]")
    print(f"  TaxiNYC 网格中心: lon [{nyc_coords[:,0].min():.4f}, {nyc_coords[:,0].max():.4f}]"
          f"  lat [{nyc_coords[:,1].min():.4f}, {nyc_coords[:,1].max():.4f}]")

    print(f"\n  时间戳标准: UTC (ISO 8601)")
    print(f"  漂移点过滤: 适用原始 GPS 轨迹；本数据集为预聚合格点流量，"
          f"已以逐格 z-score>{OUTLIER_ZSCORE} 方式替代")
    print("\n  全部完成。")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    preprocess_taxibj()
    preprocess_taxinyc()
    preprocess_weather_all()
    print_summary()
