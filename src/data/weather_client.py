"""
OpenWeatherMap 数据客户端
- 训练阶段：直接用 data/raw/taxibj/BJ_Meteorology.h5（已覆盖2013-2016）
- 本模块用途：Week 6 实时演示 / 推理时拉取当前及预报天气
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5"

# 北京坐标
BJ_LAT, BJ_LON = 39.9042, 116.4074

WEATHER_CODE_MAP = {
    range(200, 300): "雷暴",
    range(300, 400): "毛毛雨",
    range(500, 600): "雨",
    range(600, 700): "雪",
    range(700, 800): "雾/霾",
    range(800, 801): "晴",
    range(801, 900): "多云",
}

def _weather_desc(code: int) -> str:
    for r, desc in WEATHER_CODE_MAP.items():
        if code in r:
            return desc
    return "未知"


def get_current_weather(city: str = "Beijing") -> dict:
    """获取当前天气，返回标准化字段"""
    resp = requests.get(
        f"{BASE_URL}/weather",
        params={"q": city, "appid": API_KEY, "units": "metric"},
        timeout=10,
    )
    resp.raise_for_status()
    r = resp.json()
    return {
        "timestamp": datetime.fromtimestamp(r["dt"], tz=timezone.utc).isoformat(),
        "temperature": r["main"]["temp"],
        "feels_like": r["main"]["feels_like"],
        "humidity": r["main"]["humidity"],
        "wind_speed": r["wind"]["speed"],
        "weather_code": r["weather"][0]["id"],
        "weather_desc": _weather_desc(r["weather"][0]["id"]),
        "visibility": r.get("visibility", None),
        "city": r["name"],
    }


def get_forecast(city: str = "Beijing", steps: int = 16) -> pd.DataFrame:
    """
    获取未来天气预报（免费版：每3小时一步，最多40步=5天）
    返回 DataFrame，每行一个时间步
    """
    resp = requests.get(
        f"{BASE_URL}/forecast",
        params={"q": city, "appid": API_KEY, "units": "metric", "cnt": steps},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json()["list"]
    rows = []
    for item in items:
        rows.append({
            "timestamp": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "wind_speed": item["wind"]["speed"],
            "weather_code": item["weather"][0]["id"],
            "weather_desc": _weather_desc(item["weather"][0]["id"]),
            "pop": item.get("pop", 0),  # 降水概率
        })
    return pd.DataFrame(rows)


def get_weather_for_inference(hours_ahead: int = 24) -> pd.DataFrame:
    """
    为模型推理拉取未来N小时天气特征，插值到30分钟粒度
    返回与 TaxiBJ 气象格式对齐的 DataFrame
    """
    steps = min(40, (hours_ahead // 3) + 2)
    forecast = get_forecast(steps=steps)
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    forecast = forecast.set_index("timestamp")

    # 插值到 30 分钟粒度
    full_range = pd.date_range(
        start=forecast.index[0],
        periods=hours_ahead * 2,
        freq="30min",
    )
    forecast = forecast.reindex(forecast.index.union(full_range))
    forecast["temperature"] = forecast["temperature"].interpolate("time")
    forecast["wind_speed"] = forecast["wind_speed"].interpolate("time")
    forecast["humidity"] = forecast["humidity"].interpolate("time")
    forecast["weather_code"] = forecast["weather_code"].ffill()
    forecast["weather_desc"] = forecast["weather_desc"].ffill()
    return forecast.loc[full_range].reset_index().rename(columns={"index": "timestamp"})


if __name__ == "__main__":
    print("测试 OpenWeatherMap API...")
    print()
    try:
        current = get_current_weather()
        print("当前北京天气:")
        for k, v in current.items():
            print(f"  {k}: {v}")
        print()
        print("未来12小时预报（每3小时）:")
        fc = get_forecast(steps=4)
        print(fc[["timestamp", "temperature", "wind_speed", "weather_desc", "pop"]].to_string(index=False))
    except requests.exceptions.HTTPError as e:
        if "401" in str(e):
            print("API Key 尚未激活（新注册通常需等待10分钟~2小时）")
            print("Key 激活后重新运行此脚本即可")
        else:
            print(f"请求失败: {e}")
