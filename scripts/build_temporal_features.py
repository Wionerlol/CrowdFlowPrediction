"""
时间特征工程
输出：
  data/processed/temporal_features_bj.csv   — (22484, 13)
  data/processed/temporal_features_nyc.csv  — (17520, 13)

特征列：
  slot_sin/cos   : 当日第几个30min步（0~47）的周期编码
  hour_sin/cos   : 小时（0~23）周期编码
  weekday_sin/cos: 星期（0~6）周期编码
  month_sin/cos  : 月份（1~12）周期编码
  doy_sin/cos    : 年内第几天（1~366）周期编码
  is_weekend     : 0/1
  is_holiday     : 0/1（北京用 BJ_Holiday.txt；纽约用美国联邦假日）
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/processed")


def load_bj_holidays() -> set:
    lines = Path("data/raw/taxibj/BJ_Holiday.txt").read_text().splitlines()
    return {pd.Timestamp(d.strip()).date() for d in lines if d.strip()}


def us_federal_holidays(years) -> set:
    """生成美国联邦假日集合（固定规则 + 浮动规则）"""
    holidays = set()
    for y in years:
        fixed = [
            f"{y}-01-01",  # New Year's Day
            f"{y}-07-04",  # Independence Day
            f"{y}-11-11",  # Veterans Day
            f"{y}-12-25",  # Christmas
        ]
        for d in fixed:
            holidays.add(pd.Timestamp(d).date())

        # 浮动假日（规则计算）
        # MLK Day: Jan 第3个周一
        holidays.add(_nth_weekday(y, 1, 0, 3))   # 0=Monday
        # Presidents Day: Feb 第3个周一
        holidays.add(_nth_weekday(y, 2, 0, 3))
        # Memorial Day: May 最后一个周一
        holidays.add(_last_weekday(y, 5, 0))
        # Labor Day: Sep 第1个周一
        holidays.add(_nth_weekday(y, 9, 0, 1))
        # Columbus Day: Oct 第2个周一
        holidays.add(_nth_weekday(y, 10, 0, 2))
        # Thanksgiving: Nov 第4个周四
        holidays.add(_nth_weekday(y, 11, 3, 4))  # 3=Thursday
    return holidays


def _nth_weekday(year, month, weekday, n) -> pd.Timestamp.date:
    first = pd.Timestamp(year=year, month=month, day=1)
    diff  = (weekday - first.dayofweek) % 7
    return (first + pd.Timedelta(days=diff + 7*(n-1))).date()


def _last_weekday(year, month, weekday) -> pd.Timestamp.date:
    last = pd.Timestamp(year=year, month=month,
                        day=pd.Timestamp(year=year, month=month, day=1).days_in_month)
    diff = (last.dayofweek - weekday) % 7
    return (last - pd.Timedelta(days=diff)).date()


def build_temporal(timestamps_utc: np.ndarray,
                   tz_offset: int,
                   holidays: set,
                   label: str) -> pd.DataFrame:
    ts_local = (pd.DatetimeIndex(timestamps_utc.astype("datetime64[s]"))
                + pd.Timedelta(hours=tz_offset))

    two_pi = 2 * np.pi
    hour      = ts_local.hour.values
    minute    = ts_local.minute.values
    weekday   = ts_local.dayofweek.values          # 0=Mon … 6=Sun
    month     = ts_local.month.values              # 1-12
    doy       = ts_local.dayofyear.values          # 1-366
    time_slot = hour * 2 + (minute == 30).astype(int)  # 0-47

    is_weekend = (weekday >= 5).astype(np.int8)
    is_holiday = np.array([int(d in holidays)
                           for d in ts_local.date], dtype=np.int8)

    df = pd.DataFrame({
        "slot_sin"    : np.sin(two_pi * time_slot / 48),
        "slot_cos"    : np.cos(two_pi * time_slot / 48),
        "hour_sin"    : np.sin(two_pi * hour / 24),
        "hour_cos"    : np.cos(two_pi * hour / 24),
        "weekday_sin" : np.sin(two_pi * weekday / 7),
        "weekday_cos" : np.cos(two_pi * weekday / 7),
        "month_sin"   : np.sin(two_pi * month / 12),
        "month_cos"   : np.cos(two_pi * month / 12),
        "doy_sin"     : np.sin(two_pi * doy / 366),
        "doy_cos"     : np.cos(two_pi * doy / 366),
        "is_weekend"  : is_weekend,
        "is_holiday"  : is_holiday,
    }, index=ts_local)
    df.index.name = "local_time"

    print(f"  {label}: T={len(df)}, holiday比例={is_holiday.mean()*100:.1f}%, "
          f"weekend比例={is_weekend.mean()*100:.1f}%")
    return df


def main():
    print("=" * 55)
    print("  时间特征工程")
    print("=" * 55)

    bj_holidays  = load_bj_holidays()
    nyc_holidays = us_federal_holidays([2014])
    print(f"  北京假日: {len(bj_holidays)} 天")
    print(f"  纽约假日: {len(nyc_holidays)} 天")

    # TaxiBJ — UTC+8
    bj_data = np.load("data/processed/taxibj_clean.npz")
    df_bj   = build_temporal(bj_data["timestamps"], tz_offset=8,
                              holidays=bj_holidays, label="TaxiBJ")
    out_bj  = OUT / "temporal_features_bj.csv"
    df_bj.to_csv(out_bj)
    print(f"  已保存: {out_bj.name}  shape={df_bj.shape}")

    # TaxiNYC — UTC-5
    nyc_data = np.load("data/processed/taxinyc_clean.npz")
    df_nyc   = build_temporal(nyc_data["timestamps"], tz_offset=-5,
                               holidays=nyc_holidays, label="TaxiNYC")
    out_nyc  = OUT / "temporal_features_nyc.csv"
    df_nyc.to_csv(out_nyc)
    print(f"  已保存: {out_nyc.name}  shape={df_nyc.shape}")

    # 样例
    print(f"\n  TaxiBJ 前3行:")
    print(df_bj.head(3).round(4).to_string())


if __name__ == "__main__":
    main()
