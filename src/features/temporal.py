"""
时间特征编码

将时间戳序列编码为 12 维周期特征向量，供 GNN / Transformer 使用。

输出列（12维）：
  slot_sin/cos   — 日内 30 分钟步（周期 48）
  hour_sin/cos   — 小时（周期 24）
  weekday_sin/cos — 星期（周期 7）
  month_sin/cos  — 月份（周期 12）
  doy_sin/cos    — 年内天数（周期 366）
  is_weekend     — 二值
  is_holiday     — 二值
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── 节假日 ─────────────────────────────────────────────────────────────


def load_bj_holidays(txt_path: str | Path | None = None) -> set:
    """加载北京节假日集合（BJ_Holiday.txt），返回 date 对象集合。"""
    if txt_path is None:
        txt_path = _PROJECT_ROOT / "data/raw/taxibj/BJ_Holiday.txt"
    holidays = set()
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    holidays.add(pd.Timestamp(line).date())
                except ValueError:
                    pass
    return holidays


def us_federal_holidays(years: list[int]) -> set:
    """生成美国联邦假日集合（固定 + 浮动），返回 date 对象集合。"""
    holidays = set()
    for y in years:
        # 固定假日
        for m, d in [(1, 1), (7, 4), (11, 11), (12, 25)]:
            holidays.add(pd.Timestamp(y, m, d).date())
        # 浮动假日（月内第 N 个星期 X）
        for m, wday, n in [(1, 0, 3), (2, 0, 3), (5, 0, -1),
                           (9, 0, 1), (10, 0, 2), (11, 3, 4)]:
            holidays.add(_nth_weekday(y, m, wday, n))
        # 独立日补班
        ind = pd.Timestamp(y, 7, 4)
        if ind.weekday() == 5:
            holidays.add(pd.Timestamp(y, 7, 3).date())
        elif ind.weekday() == 6:
            holidays.add(pd.Timestamp(y, 7, 5).date())
    return holidays


def _nth_weekday(year: int, month: int, weekday: int, n: int):
    """第 n 个星期 weekday（0=周一）；n=-1 表示最后一个。"""
    import calendar
    cal = calendar.monthcalendar(year, month)
    days = [week[weekday] for week in cal if week[weekday] != 0]
    return pd.Timestamp(year, month, days[n]).date()


# ── 主函数 ─────────────────────────────────────────────────────────────


def build_temporal(timestamps_utc: np.ndarray,
                   tz_offset: int,
                   holidays: set | None = None) -> pd.DataFrame:
    """
    将 UTC 时间戳数组编码为 12 维时间特征矩阵。

    Parameters
    ----------
    timestamps_utc : (T,) datetime64[s] UTC
    tz_offset      : 时区偏移小时数（北京 +8，纽约 -5）
    holidays       : date 对象集合，None 时不标记节假日

    Returns
    -------
    pd.DataFrame, shape (T, 12)，index 为本地时间
    """
    ts_utc   = pd.DatetimeIndex(timestamps_utc.astype("datetime64[s]"))
    ts_local = ts_utc + pd.Timedelta(hours=tz_offset)

    hour    = ts_local.hour.values.astype(np.float32)
    slot    = (hour * 2 + ts_local.minute.values / 30).astype(np.float32)
    weekday = ts_local.dayofweek.values.astype(np.float32)
    month   = (ts_local.month.values - 1).astype(np.float32)
    doy     = (ts_local.dayofyear.values - 1).astype(np.float32)

    def _sc(x, period):
        r = 2 * math.pi * x / period
        return np.sin(r).astype(np.float32), np.cos(r).astype(np.float32)

    s_slot, c_slot = _sc(slot, 48)
    s_hour, c_hour = _sc(hour, 24)
    s_wd,   c_wd   = _sc(weekday, 7)
    s_mon,  c_mon  = _sc(month, 12)
    s_doy,  c_doy  = _sc(doy, 366)

    is_weekend = (ts_local.dayofweek.values >= 5).astype(np.float32)

    if holidays:
        dates = ts_local.date
        is_holiday = np.array([d in holidays for d in dates], dtype=np.float32)
    else:
        is_holiday = np.zeros(len(ts_local), dtype=np.float32)

    df = pd.DataFrame({
        "slot_sin": s_slot, "slot_cos": c_slot,
        "hour_sin": s_hour, "hour_cos": c_hour,
        "weekday_sin": s_wd, "weekday_cos": c_wd,
        "month_sin": s_mon, "month_cos": c_mon,
        "doy_sin": s_doy, "doy_cos": c_doy,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
    }, index=ts_local)

    return df
