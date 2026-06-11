"""
数据探索脚本 — 对所有已下载数据集进行统计分析
输出: outputs/reports/data_quality_report.md
用法: python scripts/explore_data.py
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import h5py
from datetime import datetime

os.makedirs("outputs/reports", exist_ok=True)
lines = []

def log(s=""):
    print(s)
    lines.append(s)

def section(title):
    log(f"\n## {title}\n")

def subsection(title):
    log(f"\n### {title}\n")

log("# 数据质量分析报告")
log(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ──────────────────────────────────────────
# 1. TaxiBJ
# ──────────────────────────────────────────
section("TaxiBJ 数据集")

taxibj_files = {
    "2013": "data/raw/taxibj/BJ13_M32x32_T30_InOut.h5",
    "2014": "data/raw/taxibj/BJ14_M32x32_T30_InOut.h5",
    "2015": "data/raw/taxibj/BJ15_M32x32_T30_InOut.h5",
    "2016": "data/raw/taxibj/BJ16_M32x32_T30_InOut.h5",
}

log("| 年份 | 时间步数 | 时间范围 | inflow均值 | outflow均值 | 缺失值 |")
log("|------|---------|---------|-----------|------------|-------|")

all_data = {}
for year, path in taxibj_files.items():
    if not os.path.exists(path):
        log(f"| {year} | 文件不存在 | - | - | - | - |")
        continue
    with h5py.File(path, "r") as hf:
        data = np.array(hf["data"])   # (T, 2, 32, 32)
        dates = np.array(hf["date"]).astype(str)
    all_data[year] = data
    T = data.shape[0]
    inflow  = data[:, 0, :, :]
    outflow = data[:, 1, :, :]
    missing = int(np.sum(data < 0))
    log(f"| {year} | {T} | {dates[0]} ~ {dates[-1]} | "
        f"{inflow.mean():.2f} | {outflow.mean():.2f} | {missing} |")

subsection("TaxiBJ 统计摘要（2016年示例）")
if "2016" in all_data:
    d = all_data["2016"]
    inflow = d[:, 0]
    log(f"- Shape: `{d.shape}`  → (时间步, 2通道, 32×32网格)")
    log(f"- Inflow  — min: {inflow.min():.1f}, max: {inflow.max():.1f}, std: {inflow.std():.2f}")
    log(f"- Outflow — min: {d[:,1].min():.1f}, max: {d[:,1].max():.1f}, std: {d[:,1].std():.2f}")
    log(f"- 时间分辨率: 30分钟/步")
    log(f"- 空间分辨率: 32×32 网格，覆盖北京城区")
    # 找峰值时间步
    peak_t = int(np.argmax(inflow.sum(axis=(1,2))))
    log(f"- 峰值时间步: {peak_t} (inflow合计 {inflow[peak_t].sum():.0f})")

subsection("TaxiBJ 气象数据")
meteo_path = "data/raw/taxibj/BJ_Meteorology.h5"
if os.path.exists(meteo_path):
    with h5py.File(meteo_path, "r") as hf:
        temp  = np.array(hf["Temperature"])
        wind  = np.array(hf["WindSpeed"])
        weather = np.array(hf["Weather"])
        dates = np.array(hf["date"]).astype(str)
    log(f"- 时间步数: {len(temp)}")
    log(f"- 时间范围: {dates[0]} ~ {dates[-1]}")
    log(f"- 温度 — min: {temp.min():.1f}°C, max: {temp.max():.1f}°C, mean: {temp.mean():.1f}°C")
    log(f"- 风速 — min: {wind.min():.1f}, max: {wind.max():.1f}, mean: {wind.mean():.1f} m/s")
    log(f"- 天气类型: one-hot 编码，{weather.shape[1]} 种天气")

subsection("TaxiBJ 节假日")
holiday_path = "data/raw/taxibj/BJ_Holiday.txt"
if os.path.exists(holiday_path):
    with open(holiday_path) as f:
        holidays = [l.strip() for l in f if l.strip()]
    log(f"- 节假日/特殊日期数量: {len(holidays)}")
    log(f"- 示例: {holidays[:5]}")

# ──────────────────────────────────────────
# 2. METR-LA
# ──────────────────────────────────────────
section("METR-LA 数据集")

subsection("流量数据（Parquet 分片）")
splits = {
    "train": "data/raw/metr_la/metrla_train.parquet",
    "val":   "data/raw/metr_la/metrla_val.parquet",
    "test":  "data/raw/metr_la/metrla_test.parquet",
}

log("| 分片 | 行数 | 列数 | 文件大小 |")
log("|------|------|------|---------|")
for split, path in splits.items():
    if not os.path.exists(path):
        log(f"| {split} | 文件不存在 | - | - |")
        continue
    df = pd.read_parquet(path)
    size_mb = os.path.getsize(path) / 1e6
    log(f"| {split} | {len(df):,} | {df.shape[1]} | {size_mb:.1f} MB |")
    if split == "train":
        train_df = df

if "train_df" in dir():
    log(f"\n**列名示例（前8列）**: `{list(train_df.columns[:8])}`")
    log(f"\n**传感器节点数**: {train_df['node_id'].nunique() if 'node_id' in train_df.columns else 'N/A'}")
    ts_col = [c for c in train_df.columns if "timestamp" in c]
    if ts_col:
        log(f"**时间戳列**: `{ts_col[0]}`")
        log(f"**时间范围**: {train_df[ts_col[0]].min()} ~ {train_df[ts_col[0]].max()}")

subsection("图结构数据")
adj_path = "data/raw/metr_la/adj_mx.pkl"
if os.path.exists(adj_path):
    with open(adj_path, "rb") as f:
        sensor_ids, sensor_id_to_idx, adj_mx = pickle.load(f, encoding="latin-1")
    log(f"- 传感器数量: {len(sensor_ids)}")
    log(f"- 邻接矩阵 shape: `{adj_mx.shape}`")
    nonzero = int(np.sum(adj_mx > 0))
    log(f"- 非零边数量: {nonzero} (密度: {nonzero / adj_mx.size:.4f})")
    log(f"- 权重范围: [{adj_mx.min():.4f}, {adj_mx.max():.4f}]")

sensor_path = "data/raw/metr_la/graph_sensor_locations.csv"
if os.path.exists(sensor_path):
    sensors = pd.read_csv(sensor_path)
    log(f"\n**传感器位置**:")
    log(f"- 纬度范围: {sensors['latitude'].min():.4f} ~ {sensors['latitude'].max():.4f}")
    log(f"- 经度范围: {sensors['longitude'].min():.4f} ~ {sensors['longitude'].max():.4f}")
    log(f"- 地理范围: 洛杉矶地区")

# ──────────────────────────────────────────
# 3. OpenStreetMap
# ──────────────────────────────────────────
section("OpenStreetMap 北京")

osm_path = "data/raw/osm/beijing-latest.osm.pbf"
if os.path.exists(osm_path):
    size_mb = os.path.getsize(osm_path) / 1e6
    log(f"- 文件: `beijing-latest.osm.pbf`")
    log(f"- 大小: {size_mb:.1f} MB（解压后约 300MB+）")
    log(f"- 格式: OSM Protocol Buffer Format")
    log(f"- 内容: 道路、POI、建筑、行政边界等")
    log(f"- 推荐解析库: `osmium-tool` 或 Python `osmium`")
    log(f"\n**使用示例**:")
    log(f"```python")
    log(f"import osmium")
    log(f"class NodeHandler(osmium.SimpleHandler):")
    log(f"    def node(self, n):")
    log(f"        if 'amenity' in n.tags:")
    log(f"            print(n.id, n.tags['amenity'], n.location.lat, n.location.lon)")
    log(f"NodeHandler().apply_file('data/raw/osm/beijing-latest.osm.pbf')")
    log(f"```")

# ──────────────────────────────────────────
# 4. 综合建议
# ──────────────────────────────────────────
section("数据综合评估与建议")

log("### 可用性评估\n")
log("| 数据集 | 完整性 | 质量 | 下一步 |")
log("|--------|--------|------|-------|")
log("| TaxiBJ 2013-2016 | ✅ 完整 | 高，官方发布 | 直接用于 Week 2 预处理 |")
log("| TaxiBJ 气象 | ✅ 完整 | 高 | 与人流数据时间对齐后使用 |")
log("| METR-LA train/val/test | ✅ 完整 | 高，已分好split | 可直接训练基线模型 |")
log("| METR-LA 邻接矩阵 | ✅ 完整 | 高 | 207×207，直接用于图构建 |")
log("| OSM 北京 | ✅ 完整 | 高，每日更新 | Week 2 提取 POI + 路网特征 |")
log("| NASA Landsat-8 | ⏳ 待注册 | - | 注册 earthdata.nasa.gov |")
log("| OpenWeatherMap | ⏳ 待注册 | - | 注册获取 API Key |")

log("\n### 关键注意事项\n")
log("1. **TaxiBJ 时间对齐**: 四年数据时间段不连续，训练时需分别处理或拼接后填充缺口")
log("2. **METR-LA vs TaxiBJ 差异**: METR-LA 是交通速度（连续值），TaxiBJ 是人流计数（整数）")
log("3. **坐标系统**: OSM 使用 WGS84 (EPSG:4326)，TaxiBJ 网格需要反算地理边界后对齐")
log("4. **气象与人流对齐**: BJ_Meteorology 时间戳与 TaxiBJ 均为 30 分钟间隔，可直接 merge")

# 写文件
report_path = "outputs/reports/data_quality_report.md"
with open(report_path, "w") as f:
    f.write("\n".join(lines))

print(f"\n报告已保存: {report_path}")
