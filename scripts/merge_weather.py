"""
气象数据合并：ERA5 (Open-Meteo) + OpenWeather
规则：
  - 重叠变量以 ERA5 为准
  - 所有变量全部保留，不丢弃
输出：
  data/processed/weather_bj_merged.csv
  data/processed/weather_nyc_merged.csv

最终 12 维特征：
  ERA5 (4)  : temperature, wind_speed, precipitation, weather_code(WMO)
  OW 补充(8): dew_point, pressure, humidity, wind_deg,
              rain_1h, snow_1h, clouds_all, weather_id
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")


def load_era5(h5_path: Path, tz_offset: int) -> pd.DataFrame:
    """
    读取 ERA5 h5，将本地时转换为 UTC，返回 DataFrame（UTC 索引）。
    tz_offset: 本地时 = UTC + tz_offset（北京+8, 纽约-5）
    """
    with h5py.File(h5_path, "r") as f:
        times = [t.decode() for t in f["time"][:]]
        df = pd.DataFrame({
            "temperature"  : f["temperature"][:].astype(np.float32),
            "wind_speed"   : f["wind_speed"][:].astype(np.float32),
            "precipitation": f["precipitation"][:].astype(np.float32),
            "weather_code" : f["weather_code"][:].astype(np.int16),
        })

    # YYYYMMDDHHII → local datetime → UTC
    local_ts = pd.to_datetime(times, format="%Y%m%d%H%M")
    utc_ts   = local_ts - pd.Timedelta(hours=tz_offset)
    df.index = utc_ts.tz_localize("UTC")
    df.index.name = "utc"
    return df


def load_openweather(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df.index.name = "utc"
    return df


def merge(era5: pd.DataFrame, ow: pd.DataFrame) -> pd.DataFrame:
    """
    以 UTC 时间戳为键做 outer join，重叠列以 ERA5 为准。
    """
    # OW 中与 ERA5 重叠的列：temp→temperature, wind_speed, weather_code
    ow = ow.copy()
    ow = ow.rename(columns={"temp": "_ow_temp"})          # 重命名避免冲突
    ow = ow.drop(columns=["wind_speed", "weather_code"],   # 以ERA5为准，OW版本丢弃
                 errors="ignore")

    # outer join
    merged = era5.join(ow, how="outer")

    # 对非 ERA5 列做前向/后向填充（极少量边界缺口）
    ow_cols = [c for c in merged.columns if c not in era5.columns]
    merged[ow_cols] = merged[ow_cols].ffill().bfill()

    # ERA5 列同样填充（应无缺失，以防万一）
    era5_cols = list(era5.columns)
    merged[era5_cols] = merged[era5_cols].ffill().bfill()

    # 丢弃重命名的 OW temp（已被 ERA5 temperature 替代）
    merged = merged.drop(columns=["_ow_temp"], errors="ignore")

    return merged


def process_city(city: str, h5_name: str, ow_name: str,
                 tz_offset: int, out_name: str):
    print(f"\n{'='*55}")
    print(f"  {city}")
    print(f"{'='*55}")

    era5 = load_era5(RAW / "weather" / h5_name, tz_offset)
    ow   = load_openweather(OUT / ow_name)

    print(f"  ERA5  : T={len(era5)},  {era5.index[0]} ~ {era5.index[-1]}")
    print(f"  OW    : T={len(ow)},  {ow.index[0]} ~ {ow.index[-1]}")

    merged = merge(era5, ow)
    print(f"  Merged: T={len(merged)}, cols={merged.columns.tolist()}")

    # 缺失率报告
    nan_rates = merged.isna().mean() * 100
    if nan_rates.any():
        print("  NaN (>0%):", nan_rates[nan_rates > 0].round(2).to_dict())
    else:
        print("  NaN: 无")

    out_path = OUT / out_name.replace("_30min", "_merged")
    merged.to_csv(out_path)
    print(f"  已保存: {out_path.name}  shape={merged.shape}")
    return merged


if __name__ == "__main__":
    bj = process_city(
        city       = "北京",
        h5_name    = "beijing_weather_historical.h5",
        ow_name    = "weather_bj_30min.csv",
        tz_offset  = 8,
        out_name   = "weather_bj_30min.csv",
    )

    nyc = process_city(
        city       = "纽约",
        h5_name    = "newyork_weather_historical.h5",
        ow_name    = "weather_nyc_30min.csv",
        tz_offset  = -5,
        out_name   = "weather_nyc_30min.csv",
    )

    print("\n\n【字段说明】")
    print("  ERA5 来源  : temperature(°C), wind_speed(m/s), "
          "precipitation(mm), weather_code(WMO)")
    print("  OW 补充    : dew_point(°C), pressure(hPa), humidity(%), "
          "wind_deg(°), rain_1h(mm), snow_1h(mm), clouds_all(%), weather_id")
    print("  合计       : 12 维气象特征")
    print("\n  完成。")
