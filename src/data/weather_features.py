"""Leakage-safe weather alignment and feature encoding."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CONTINUOUS_COLUMNS = [
    "temperature", "dew_point", "pressure", "humidity", "wind_speed",
    "precipitation", "rain_1h", "snow_1h", "clouds_all",
]
WEATHER_GROUPS = ("clear", "cloud", "fog", "drizzle", "rain", "snow", "storm")


def _weather_group(code: float) -> int:
    code = int(code)
    if code == 0:
        return 0
    if 1 <= code <= 3:
        return 1
    if 45 <= code <= 48:
        return 2
    if 51 <= code <= 57:
        return 3
    if 61 <= code <= 67:
        return 4
    if 71 <= code <= 77:
        return 5
    return 6


def build_weather_features(
    timestamps_utc: np.ndarray,
    csv_path: str | Path,
    stats: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Align weather exactly to timestamps and encode 18 features.

    Continuous statistics are fitted only when ``stats`` is None. Validation
    and test calls must reuse the training statistics returned by the first
    call.
    """
    weather = pd.read_csv(csv_path)
    weather_ts = pd.DatetimeIndex(pd.to_datetime(weather["utc"], utc=True))
    if weather_ts.duplicated().any():
        raise ValueError(f"Duplicate weather timestamps in {csv_path}")
    target = pd.DatetimeIndex(
        timestamps_utc.astype("datetime64[s]")
    ).tz_localize("UTC")
    positions = weather_ts.get_indexer(target)
    if (positions < 0).any():
        missing = target[positions < 0][:5].tolist()
        raise ValueError(f"Weather does not cover flow timestamps: {missing}")
    aligned = weather.iloc[positions]
    if aligned.isna().any().any():
        raise ValueError("Aligned weather contains missing values")

    continuous = aligned[CONTINUOUS_COLUMNS].to_numpy(np.float32)
    # Precipitation variables are zero-inflated and right-skewed.
    for name in ("precipitation", "rain_1h", "snow_1h"):
        j = CONTINUOUS_COLUMNS.index(name)
        continuous[:, j] = np.log1p(np.clip(continuous[:, j], 0, None))
    wind = np.deg2rad(aligned["wind_deg"].to_numpy(np.float32))
    base = np.concatenate(
        [continuous, np.sin(wind)[:, None], np.cos(wind)[:, None]], axis=1
    ).astype(np.float32)

    if stats is None:
        mean = base.mean(axis=0)
        std = base.std(axis=0)
        std[std < 1e-6] = 1.0
        stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    scaled = (base - stats["mean"]) / stats["std"]

    groups = np.array(
        [_weather_group(v) for v in aligned["weather_code"]], dtype=np.int64
    )
    one_hot = np.eye(len(WEATHER_GROUPS), dtype=np.float32)[groups]
    features = np.concatenate([scaled, one_hot], axis=1).astype(np.float32)
    assert features.shape[1] == 18
    return features, stats
