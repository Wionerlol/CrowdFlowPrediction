"""Streamlit + Kepler.gl interface for the Week 6 city crowd-flow system."""

from __future__ import annotations

import base64
from datetime import date, datetime, time
import io
import math
import struct
import wave

import pandas as pd
import plotly.express as px
import streamlit as st

from src.visualization.api_client import ApiError, Week6ApiClient
from src.visualization.kepler_maps import event_map_html, flow_map_html


st.set_page_config(page_title="城市人流监测与异常预警", page_icon="🌆", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .hero {padding: 1.0rem 1.2rem; border-radius: 14px; background: linear-gradient(120deg,#102a43,#1f5f8b); color:#fff; margin-bottom:1rem;}
      .hero h1 {margin:0; font-size:2rem;}
      .hero p {margin:.35rem 0 0; opacity:.85;}
      .alert-important {background:#fff4e5;border-left:6px solid #f97316;padding:.8rem 1rem;border-radius:8px;margin:.4rem 0;}
      .alert-urgent {background:#fee2e2;border-left:6px solid #dc2626;padding:.8rem 1rem;border-radius:8px;margin:.4rem 0;animation:pulse 1.3s infinite;}
      @keyframes pulse {0%{box-shadow:0 0 0 0 rgba(220,38,38,.35)}70%{box-shadow:0 0 0 12px rgba(220,38,38,0)}100%{box-shadow:0 0 0 0 rgba(220,38,38,0)}}
      div[data-testid="stMetric"] {background:#f7fafc;border:1px solid #dbe4ee;padding:.6rem;border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _beep_data_uri() -> str:
    sample_rate = 16_000
    duration = 0.7
    frequency = 880.0
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        samples = []
        for index in range(int(sample_rate * duration)):
            envelope = min(1.0, index / 500) * min(1.0, (sample_rate * duration - index) / 800)
            value = int(12_000 * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
            samples.append(struct.pack("<h", value))
        audio.writeframes(b"".join(samples))
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@st.dialog("异常预警", width="large")
def alert_dialog(alert: dict) -> None:
    level = alert.get("alert_level_zh", "重要")
    st.error(alert.get("alert_message", f"{level}预警")) if level == "紧急" else st.warning(alert.get("alert_message", f"{level}预警"))
    left, right = st.columns(2)
    left.metric("事件类型", alert.get("event_type_zh", alert.get("event_type", "—")))
    right.metric("融合分数", f"{float(alert.get('max_score', 0)):.3f}")
    st.write(alert.get("alert_action_hint", "请及时核查事件。"))
    if st.button("已知悉", type="primary", width="stretch"):
        st.session_state.acknowledged_alerts.add(int(alert["event_id"]))
        st.rerun()


def render_new_alerts(client: Week6ApiClient, *, sound_enabled: bool) -> None:
    try:
        response = client.alerts(source="realtime", min_alert_level=2, limit=20)
    except ApiError as exc:
        st.caption(str(exc))
        return
    alerts = response.get("events", [])
    if not alerts:
        return
    latest = max(alerts, key=lambda item: (int(item.get("alert_level_id", 0)), str(item.get("start_time", ""))))
    css_class = "alert-urgent" if latest.get("alert_level") == "urgent" else "alert-important"
    st.markdown(f'<div class="{css_class}">{latest.get("alert_message", "检测到异常事件")}</div>', unsafe_allow_html=True)
    event_id = int(latest["event_id"])
    if event_id not in st.session_state.acknowledged_alerts:
        if latest.get("alert_sound") and sound_enabled:
            st.iframe(
                f'<audio autoplay><source src="{_beep_data_uri()}" type="audio/wav"></audio>',
                height=1,
            )
        alert_dialog(latest)


def render_flow_map(payload: dict, *, height: int) -> None:
    st.iframe(flow_map_html(payload["cells"], height=height), height=height + 10)


def render_event_map(events: list[dict], *, height: int) -> None:
    st.iframe(event_map_html(events, height=height), height=height + 10)


def live_tab(client: Week6ApiClient, *, refresh_seconds: int, map_height: int, sound_enabled: bool) -> None:
    st.subheader("城市人流实时态势")
    source_value = st.radio("显示内容", ["observed", "predicted"], format_func=lambda x: "观测人流" if x == "observed" else "h1 预测", horizontal=True)

    @st.fragment(run_every=max(refresh_seconds, 5))
    def live_fragment() -> None:
        try:
            payload = client.heatmap(source="realtime", value_type=source_value)
        except ApiError as exc:
            st.error(str(exc))
            return
        summary = payload["summary"]
        columns = st.columns(5)
        columns[0].metric("数据时间", payload["timestamp"].replace("T", " "))
        columns[1].metric("平均流量", f"{summary['mean']:.1f}")
        columns[2].metric("峰值流量", f"{summary['max']:.0f}")
        columns[3].metric("异常网格", int(summary["anomaly_cells"]))
        columns[4].metric("最高异常分数", f"{summary['max_anomaly_score']:.3f}")
        render_new_alerts(client, sound_enabled=sound_enabled)
        render_flow_map(payload, height=map_height)
        st.caption(f"每 {refresh_seconds} 秒读取一次实时微批产物；地图时间为 TaxiBJ 离线实时模拟时间。")

    live_fragment()


def forecast_tab(client: Week6ApiClient, *, map_height: int) -> None:
    st.subheader("未来 24 小时人流预测动画")
    try:
        payload = client.forecast(horizon_steps=48)
    except ApiError as exc:
        st.error(str(exc))
        st.info("请先运行 realtime pipeline，生成同一起点的 `forecast_24h.npz`。")
        return
    frames = payload["frames"]
    metrics = st.columns(4)
    metrics[0].metric("预测起点", payload["origin_time"].replace("T", " "))
    metrics[1].metric("预测范围", "未来 24 小时")
    metrics[2].metric("时间分辨率", "30 分钟")
    metrics[3].metric("模型成员", len(payload["member_tags"]))

    base = pd.DataFrame(payload["cells"])
    animation_parts = []
    for frame in frames:
        part = base.copy()
        part["predicted_flow"] = frame["values"]
        part["forecast_time"] = frame["timestamp"].replace("T", " ")
        animation_parts.append(part)
    animation = pd.concat(animation_parts, ignore_index=True)
    figure = px.density_map(
        animation,
        lat="center_lat",
        lon="center_lon",
        z="predicted_flow",
        animation_frame="forecast_time",
        radius=18,
        center={"lat": float(base["center_lat"].mean()), "lon": float(base["center_lon"].mean())},
        zoom=8.8,
        map_style="carto-darkmatter",
        color_continuous_scale=["#0B1F3A", "#1D70A2", "#2CA6A4", "#FDE74C", "#FA7921", "#D7263D"],
        range_color=(0, float(animation["predicted_flow"].quantile(0.995))),
        height=map_height,
        title="未来 24 小时网格人流预测（点击播放按钮启动动画）",
    )
    figure.update_layout(margin={"l": 0, "r": 0, "t": 45, "b": 0})
    st.plotly_chart(figure, width="stretch")

    trend = pd.DataFrame({"timestamp": [item["timestamp"] for item in frames], "mean_flow": [item["mean"] for item in frames]})
    trend["timestamp"] = pd.to_datetime(trend["timestamp"])
    st.plotly_chart(px.line(trend, x="timestamp", y="mean_flow", markers=True, title="未来 24 小时全城平均预测流量"), width="stretch")
    st.caption("48 帧来自同一个 forecast origin 的 AGFormer gated+linear 四随机种子 ensemble，不是 48 个独立 h1 预测的拼接。")


def anomaly_tab(client: Week6ApiClient, *, map_height: int) -> None:
    st.subheader("异常事件查询与详情")
    filters = st.columns(5)
    source = filters[0].selectbox("数据源", ["historical", "realtime"], format_func=lambda x: "历史事件" if x == "historical" else "实时事件")
    event_type = filters[1].selectbox("事件类型", [None, "mass_gathering", "traffic_jam", "sudden_incident"], format_func=lambda x: {None: "全部", "mass_gathering": "大型集会", "traffic_jam": "交通拥堵", "sudden_incident": "突发事件"}[x])
    alert_level = filters[2].selectbox("预警等级", [None, "general", "important", "urgent"], format_func=lambda x: {None: "全部", "general": "一般", "important": "重要", "urgent": "紧急"}[x])
    min_score = filters[3].slider("最低融合分数", 0.0, 1.0, 0.0, 0.05)
    use_dates = filters[4].checkbox("按日期筛选", value=False)
    start_time = end_time = None
    if use_dates:
        dates = st.columns(2)
        start_date = dates[0].date_input("开始日期", value=date(2016, 1, 1))
        end_date = dates[1].date_input("结束日期", value=date(2016, 4, 30))
        start_time = datetime.combine(start_date, time.min).isoformat()
        end_time = datetime.combine(end_date, time.max).isoformat()

    try:
        result = client.anomalies(source=source, event_type=event_type, alert_level=alert_level, min_score=min_score or None, start_time=start_time, end_time=end_time, limit=500)
    except ApiError as exc:
        st.error(str(exc))
        return
    events = result["events"]
    st.metric("匹配事件数", result["total"])
    if not events:
        st.info("当前筛选条件下没有异常事件。")
        return

    table = pd.DataFrame(events)
    display_columns = ["event_id", "start_time", "end_time", "event_type_zh", "alert_level_zh", "max_score", "duration_hours", "union_cells"]
    st.dataframe(table[[column for column in display_columns if column in table]], width="stretch", hide_index=True)
    render_event_map(events, height=map_height)

    labels = {
        int(item["event_id"]): f"#{int(item['event_id'])} · {item.get('alert_level_zh', '')} · {item.get('event_type_zh', item.get('event_type'))} · {str(item.get('start_time', '')).replace('T', ' ')}"
        for item in events
    }
    event_id = st.selectbox("查看事件详情", list(labels), format_func=labels.get)
    try:
        detail = client.anomaly_detail(event_id, source=source)
    except ApiError as exc:
        st.error(str(exc))
        return
    st.markdown(f"### {detail.get('alert_level_zh', '')}预警 · {detail.get('event_type_zh', detail.get('event_type', ''))}")
    st.info(detail.get("alert_message", ""))
    metrics = st.columns(6)
    metrics[0].metric("持续时间", f"{float(detail.get('duration_hours', 0)):.1f}h")
    metrics[1].metric("最高分数", f"{float(detail.get('max_score', 0)):.3f}")
    metrics[2].metric("影响网格", int(detail.get("union_cells", 0)))
    metrics[3].metric("最大 Z", f"{float(detail.get('max_z', 0)):.2f}")
    metrics[4].metric("主导方法", detail.get("dominant_method", "—"))
    metrics[5].metric("空间簇数", int(detail.get("cluster_count", 0)))
    with st.expander("方法贡献与处置建议", expanded=True):
        contribution = pd.DataFrame(
            {
                "method": ["Statistical", "Reconstruction", "Prediction"],
                "score": [detail.get("statistical_score", 0), detail.get("reconstruction_score", 0), detail.get("prediction_score", 0)],
            }
        )
        st.plotly_chart(px.bar(contribution, x="method", y="score", color="method", range_y=[0, 1]), width="stretch")
        st.write(detail.get("alert_action_hint", ""))
        st.code(detail.get("top_cells", "暂无 top cells"), language=None)


def statistics_tab(client: Week6ApiClient) -> None:
    st.subheader("历史异常事件统计")
    try:
        stats = client.statistics(source="historical")
    except ApiError as exc:
        st.error(str(exc))
        return
    metrics = st.columns(4)
    metrics[0].metric("历史事件", stats["total_events"])
    metrics[1].metric("平均持续时间", f"{stats['mean_duration_hours']:.2f}h" if stats["mean_duration_hours"] is not None else "—")
    metrics[2].metric("平均最高分数", f"{stats['mean_max_score']:.3f}" if stats["mean_max_score"] is not None else "—")
    metrics[3].metric("紧急预警", stats.get("by_alert_level", {}).get("urgent", 0))
    left, right = st.columns(2)
    type_frame = pd.DataFrame(list(stats["by_event_type"].items()), columns=["event_type", "count"])
    level_frame = pd.DataFrame(list(stats["by_alert_level"].items()), columns=["alert_level", "count"])
    if len(type_frame):
        left.plotly_chart(px.pie(type_frame, names="event_type", values="count", title="异常事件类型分布", hole=0.45), width="stretch")
    if len(level_frame):
        right.plotly_chart(px.bar(level_frame, x="alert_level", y="count", color="alert_level", title="三级预警分布"), width="stretch")
    daily = pd.DataFrame(stats["daily"])
    if len(daily):
        daily["date"] = pd.to_datetime(daily["date"])
        st.plotly_chart(px.bar(daily, x="date", y="count", title="历史异常事件时间分布"), width="stretch")


def main() -> None:
    st.markdown('<div class="hero"><h1>城市人流监测与异常预警平台</h1><p>TaxiBJ · AGFormer 24h Forecast · Week5 Multi-model Anomaly Fusion</p></div>', unsafe_allow_html=True)
    if "acknowledged_alerts" not in st.session_state:
        st.session_state.acknowledged_alerts = set()

    with st.sidebar:
        st.header("服务设置")
        api_url = st.text_input("FastAPI 地址", value="http://127.0.0.1:8000/api/v1")
        refresh_seconds = st.slider("实时刷新间隔（秒）", 5, 120, 30, 5)
        sound_enabled = st.toggle("紧急预警声音", value=True)
        map_height = st.slider("地图高度", 420, 760, 570, 10)
        client = Week6ApiClient(api_url)
        if st.button("检查 API", width="stretch"):
            try:
                status = client.status()
                if status["status"] == "ok":
                    st.success("API 与全部产物就绪")
                else:
                    st.warning("API 可用，但部分产物尚未生成")
                with st.expander("状态详情"):
                    st.json(status)
            except ApiError as exc:
                st.error(str(exc))
        st.caption("浏览器可能阻止首次自动播放声音；点击页面一次后即可正常播放。")

        page = st.radio("功能导航", ["实时热力图", "24h 预测动画", "异常事件", "历史统计"])

    client = Week6ApiClient(api_url)
    if page == "实时热力图":
        live_tab(client, refresh_seconds=refresh_seconds, map_height=map_height, sound_enabled=sound_enabled)
    elif page == "24h 预测动画":
        forecast_tab(client, map_height=map_height)
    elif page == "异常事件":
        anomaly_tab(client, map_height=map_height)
    else:
        statistics_tab(client)


if __name__ == "__main__":
    main()
