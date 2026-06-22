"""
Week 2 报告 PDF 生成脚本
输出：outputs/reports/week2_report.pdf
"""

from pathlib import Path
from weasyprint import HTML, CSS

HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<title>Week 2 进度报告 — 城市人流异常检测系统</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    font-size: 10pt;
    line-height: 1.65;
    color: #1a1a1a;
    background: #fff;
  }

  /* ── Cover page ─────────────────────────────────────── */
  .cover {
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 60px 40px;
    background: linear-gradient(160deg, #0f2650 0%, #1a4a8a 60%, #2563c0 100%);
    color: #fff;
    text-align: center;
  }
  .cover .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 18px;
    font-size: 9pt;
    letter-spacing: 0.08em;
    margin-bottom: 36px;
  }
  .cover h1 {
    font-size: 26pt;
    font-weight: 700;
    line-height: 1.25;
    margin-bottom: 12px;
  }
  .cover h2 {
    font-size: 14pt;
    font-weight: 400;
    opacity: 0.85;
    margin-bottom: 48px;
  }
  .cover .meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 48px;
    text-align: left;
    background: rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 24px 36px;
    font-size: 9.5pt;
  }
  .cover .meta-grid .label { opacity: 0.7; margin-bottom: 2px; }
  .cover .meta-grid .value { font-weight: 600; font-size: 10.5pt; }
  .cover .stats-row {
    display: flex;
    gap: 32px;
    margin-top: 48px;
  }
  .cover .stat-box {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 16px 24px;
    text-align: center;
    min-width: 120px;
  }
  .cover .stat-box .num { font-size: 22pt; font-weight: 700; }
  .cover .stat-box .desc { font-size: 8pt; opacity: 0.75; margin-top: 2px; }

  /* ── Layout ─────────────────────────────────────────── */
  .page { padding: 40px 52px; }

  /* ── Typography ─────────────────────────────────────── */
  h1.section {
    font-size: 16pt;
    font-weight: 700;
    color: #0f2650;
    border-bottom: 3px solid #2563c0;
    padding-bottom: 6px;
    margin: 32px 0 18px;
  }
  h2.sub {
    font-size: 12pt;
    font-weight: 700;
    color: #1a4a8a;
    margin: 22px 0 10px;
    padding-left: 10px;
    border-left: 4px solid #2563c0;
  }
  h3.mini {
    font-size: 10.5pt;
    font-weight: 600;
    color: #1a1a1a;
    margin: 16px 0 6px;
  }
  p { margin: 6px 0; }

  /* ── Tables ─────────────────────────────────────────── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin: 10px 0 16px;
  }
  th {
    background: #0f2650;
    color: #fff;
    padding: 7px 10px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 6px 10px;
    border-bottom: 1px solid #e0e7ef;
    vertical-align: top;
  }
  tr:nth-child(even) td { background: #f4f7fb; }
  tr:hover td { background: #e8eef6; }

  /* ── Callout boxes ──────────────────────────────────── */
  .callout {
    border-radius: 8px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 9.5pt;
  }
  .callout.info  { background: #e8f0fb; border-left: 4px solid #2563c0; }
  .callout.ok    { background: #e6f4ea; border-left: 4px solid #22a052; }
  .callout.warn  { background: #fef9e7; border-left: 4px solid #f0a500; }

  /* ── Stat cards row ─────────────────────────────────── */
  .cards {
    display: flex;
    gap: 12px;
    margin: 14px 0;
    flex-wrap: wrap;
  }
  .card {
    flex: 1;
    min-width: 100px;
    background: #f4f7fb;
    border: 1px solid #cdd9ee;
    border-top: 4px solid #2563c0;
    border-radius: 8px;
    padding: 12px 14px;
    text-align: center;
  }
  .card .num { font-size: 18pt; font-weight: 700; color: #0f2650; }
  .card .unit { font-size: 8pt; color: #666; margin-top: 2px; }
  .card .label { font-size: 8.5pt; color: #444; margin-top: 4px; }

  /* ── Code block ─────────────────────────────────────── */
  pre {
    background: #1e2a3a;
    color: #c9d7e8;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 8pt;
    font-family: "Cascadia Code", "Fira Code", "Courier New", monospace;
    overflow-x: auto;
    margin: 10px 0;
    line-height: 1.5;
  }
  code {
    background: #e8eef6;
    border-radius: 3px;
    padding: 1px 4px;
    font-size: 8.5pt;
    font-family: "Courier New", monospace;
    color: #0f2650;
  }

  /* ── Check list ─────────────────────────────────────── */
  .checklist { list-style: none; padding: 0; margin: 8px 0; }
  .checklist li { padding: 3px 0 3px 24px; position: relative; }
  .checklist li::before {
    content: "✅";
    position: absolute; left: 0;
    font-size: 9pt;
  }

  /* ── Feature index table ─────────────────────────────── */
  .feat-idx th:first-child { width: 80px; }
  .feat-idx td.idx { font-weight: 600; color: #0f2650; font-family: monospace; }

  /* ── Page break ─────────────────────────────────────── */
  .pb { page-break-before: always; }

  /* ── Footer ─────────────────────────────────────────── */
  @page {
    margin: 20mm 15mm 22mm;
    @bottom-center {
      content: "城市人流异常检测系统 · Week 2 报告 · 2026-06-17";
      font-size: 8pt;
      color: #888;
    }
    @bottom-right {
      content: counter(page);
      font-size: 8pt;
      color: #888;
    }
  }
  @page:first { margin: 0; @bottom-center { content: none; } @bottom-right { content: none; } }
</style>
</head>
<body>

<!-- ═══════════════════════════════ COVER ═══════════════════════════════ -->
<div class="cover">
  <div class="badge">WEEK 2 · 数据预处理与特征工程</div>
  <h1>城市人流异常检测系统<br>第二周进度报告</h1>
  <h2>基于多源时空数据融合的城市级人流异常检测与预警系统</h2>

  <div class="meta-grid">
    <div>
      <div class="label">报告日期</div>
      <div class="value">2026-06-17</div>
    </div>
    <div>
      <div class="label">阶段</div>
      <div class="value">Week 2 / 8</div>
    </div>
    <div>
      <div class="label">覆盖城市</div>
      <div class="value">北京（BJ）· 纽约（NYC）</div>
    </div>
    <div>
      <div class="label">数据集</div>
      <div class="value">TaxiBJ · TaxiNYC · 12数据源</div>
    </div>
  </div>

  <div class="stats-row">
    <div class="stat-box">
      <div class="num">41</div>
      <div class="desc">维节点静态特征</div>
    </div>
    <div class="stat-box">
      <div class="num">4</div>
      <div class="desc">类多模态图结构</div>
    </div>
    <div class="stat-box">
      <div class="num">13</div>
      <div class="desc">个预处理脚本</div>
    </div>
    <div class="stat-box">
      <div class="num">3.7<span style="font-size:14pt">GB</span></div>
      <div class="desc">原始+预处理数据</div>
    </div>
  </div>
</div>

<!-- ═══════════════════════════════ PAGE 1 ══════════════════════════════ -->
<div class="page">

<h1 class="section">一、本周完成概览</h1>

<div class="callout ok">
  Week 2 全部四项交付物已完成：数据预处理脚本 13 个、特征工程代码库（src/features/，16个公开符号）、
  预处理数据集（41维节点特征矩阵 + 4类多模态图，共 37 个文件）、数据质量分析报告。
</div>

<div class="cards">
  <div class="card"><div class="num">13</div><div class="unit">个脚本</div><div class="label">数据预处理 Python 脚本</div></div>
  <div class="card"><div class="num">4</div><div class="unit">个模块</div><div class="label">特征工程可导入代码库</div></div>
  <div class="card"><div class="num">37</div><div class="unit">个文件</div><div class="label">预处理产物（100 MB）</div></div>
  <div class="card"><div class="num">12</div><div class="unit">个来源</div><div class="label">数据质量全覆盖分析</div></div>
</div>

<h2 class="sub">1.1 交付物清单</h2>
<ul class="checklist">
  <li>数据预处理 Python 脚本（<code>scripts/</code> 目录，13 个脚本）</li>
  <li>特征工程代码库（<code>src/features/</code>，4 个模块，16 个公开符号，完整可导入）</li>
  <li>预处理数据集：北京 (1024, 41) + 纽约 (75, 41) 节点特征矩阵，NaN=0</li>
  <li>预处理数据集：北京 + 纽约各 4 张多模态图（.pt 格式，PyTorch Geometric 兼容）</li>
  <li>数据质量分析报告（覆盖全部 12 个数据来源）</li>
  <li>特征工程技术报告（<code>outputs/reports/feature_engineering_report.md</code>）</li>
  <li>量化 / TensorRT 加速预研报告</li>
  <li>可视化界面设计文档</li>
</ul>

<!-- ─── 数据规模 ─── -->
<h2 class="sub">1.2 数据规模说明</h2>

<table>
  <tr><th>分类</th><th>大小</th><th>说明</th></tr>
  <tr><td>data/raw/</td><td>3.6 GB</td><td>Landsat-8 2.3GB、MTA GTFS 0.5GB、METR-LA 0.2GB、OSM 0.2GB、taxi 0.4GB 等</td></tr>
  <tr><td>data/processed/</td><td>100 MB</td><td>全部预处理产物（npz / csv / pt / npy，共 37 个文件）</td></tr>
  <tr><td><strong>合计</strong></td><td><strong>~3.7 GB</strong></td><td>—</td></tr>
</table>

<div class="callout warn">
  <strong>关于课题文档估计"约 20 GB"的说明：</strong>该估计基于原始出租车 GPS 轨迹数据（逐条行程记录）。
  本项目使用官方预聚合网格版本（TaxiBJ 32×32 / TaxiNYC 15×5），已将原始轨迹聚合为 inflow/outflow 计数，
  数据量远小于原始轨迹，但信息完整，不影响模型训练与评估。
</div>

<!-- ═══ SECTION 2 ══════════════════════════════════════════════════════ -->
<h1 class="section pb">二、数据预处理脚本</h1>

<h2 class="sub">2.1 脚本清单（scripts/ 目录）</h2>

<table>
  <tr><th>脚本</th><th>功能</th><th>输出</th></tr>
  <tr><td><code>build_spatial_features.py</code></td><td>路网密度（5类）+ 地理距离特征（Haversine）</td><td>spatial_features_{city}.csv (N×16)</td></tr>
  <tr><td><code>build_poi_features.py</code></td><td>OSM POI 语义特征（11类功能性 POI，transport 已剥离）</td><td>poi_features_{city}.csv (N×22)</td></tr>
  <tr><td><code>build_satellite_features.py</code></td><td>Landsat-8 DN→反射率，NDVI/NDBI/MNDWI 指数</td><td>satellite_features_{city}.csv (N×5)</td></tr>
  <tr><td><code>build_transit_features.py</code></td><td>地铁站/入口/公交站密度（OSM PBF）</td><td>transit_features_{city}.csv (N×6)</td></tr>
  <tr><td><code>build_subway_features.py</code></td><td>轨道线路覆盖（政府xlsx + OSM名称匹配）+ G_transit 图</td><td>subway_grid_features_{city}.csv (N×2)<br>graph_transit_{city}.pt</td></tr>
  <tr><td><code>build_bus_features.py</code></td><td>公交线路数（OSM route=bus 关系 + MTA GTFS 5区）</td><td>bus_route_features_{city}.csv (N×1)</td></tr>
  <tr><td><code>build_temporal_features.py</code></td><td>时间周期编码（sin/cos）+ 节假日标志（12维）</td><td>temporal_features_{city}.csv (T×12)</td></tr>
  <tr><td><code>build_graphs.py</code></td><td>G_spatial / G_poi / G_flow 构建 + 41维节点特征合并</td><td>node_features_{city}.npy (N×41)<br>graph_{spatial,poi,flow}_{city}.pt</td></tr>
  <tr><td><code>merge_weather.py</code></td><td>ERA5 + OpenWeatherMap 气象字段合并，插值至30分钟</td><td>weather_{city}_merged.csv</td></tr>
  <tr><td><code>preprocess_all.py</code></td><td>一键顺序执行全部预处理步骤</td><td>—（调度器）</td></tr>
  <tr><td><code>train_stgcn.py</code></td><td>STGCN 基线模型训练（PyTorch Geometric）</td><td>stgcn_best.pt</td></tr>
  <tr><td><code>verify_env.py</code></td><td>依赖与 GPU 环境验证</td><td>终端报告</td></tr>
  <tr><td><code>setup_env.sh</code></td><td>Conda 环境 + 依赖安装（支持 CUDA 118/121/128 / CPU）</td><td>.venv/</td></tr>
</table>

<h2 class="sub">2.2 公共交通特征管道（三步串联）</h2>
<p>公共交通特征的提取分三个脚本串联执行，分别产出不同粒度的特征：</p>

<pre>build_transit_features.py   ─→  地铁站/入口/公交站密度  transit_features (N×6)
        ↓
build_subway_features.py    ─→  轨道线路数/换乘枢纽     subway_grid_features (N×2)
                            ─→  轨道拓扑图              graph_transit.pt
        ↓
build_bus_features.py       ─→  网格内公交线路数         bus_route_features (N×1)
        ↓
build_graphs.py             ─→  合并 41 维节点特征矩阵   node_features (N×41)</pre>

<h3 class="mini">北京公交数据处理</h3>
<p>
OSM 从 <code>route=bus</code> relation 关系直接提取 platform 节点成员，获取 2,098 条关系，
bbox 内覆盖率 88.4%；剩余未覆盖站点通过政府 <code>公交站点信息.xlsx</code>（58,697 条记录，9,416 唯一站名）
模糊匹配（阈值 0.80）补全 890 个站点。
</p>

<h3 class="mini">纽约公交数据处理（MTA GTFS 5区）</h3>
<p>
OSM 数据对纽约 TaxiNYC bbox 覆盖率仅 41.5%，故改用 MTA GTFS 5区公交数据。
5个子目录（Brooklyn / Bronx / Manhattan / Queens / Staten Island）各含完整 routes.txt 和独立 stop_times.txt。
按 bbox 过滤后得 3,168 个唯一站点，255 条 route_type=3 公交线路，覆盖率 100%。
</p>

<!-- ═══ SECTION 3 ═══════════════════════════════════════════════════════ -->
<h1 class="section pb">三、特征工程代码库（src/features/）</h1>

<div class="callout info">
  <code>src/features/</code> 为可导入 Python 包，提供 4 个独立模块和 16 个公开符号，
  用于在训练/评估代码中直接调用特征函数，不依赖任何中间文件路径。
</div>

<h2 class="sub">3.1 模块结构</h2>

<table>
  <tr><th>模块</th><th>文件</th><th>公开函数</th><th>功能</th></tr>
  <tr>
    <td><code>base</code></td><td>base.py</td>
    <td><code>assign_to_grid</code><br><code>haversine_km</code><br><code>haversine_matrix</code><br><code>minmax</code><br><code>zscore</code></td>
    <td>网格坐标分配、球面距离计算（Haversine）、归一化工具</td>
  </tr>
  <tr>
    <td><code>temporal</code></td><td>temporal.py</td>
    <td><code>build_temporal</code><br><code>load_bj_holidays</code><br><code>us_federal_holidays</code></td>
    <td>时间周期编码（sin/cos，12维）、节假日规则（北京/纽约）</td>
  </tr>
  <tr>
    <td><code>graphs</code></td><td>graphs.py</td>
    <td><code>build_spatial_graph</code><br><code>build_poi_graph</code><br><code>build_flow_graph</code><br><code>save_graph</code><br><code>load_graph</code><br><code>graph_stats</code></td>
    <td>G_spatial / G_poi / G_flow 构建；.pt 格式存储/加载；边统计</td>
  </tr>
  <tr>
    <td><code>node_features</code></td><td>node_features.py</td>
    <td><code>build_node_features</code><br><code>hist_stats</code></td>
    <td>组装 (N, 41) 节点特征矩阵；8维历史流量统计</td>
  </tr>
</table>

<h2 class="sub">3.2 典型用法</h2>
<pre>from src.features.base import assign_to_grid, haversine_km
from src.features.temporal import build_temporal, load_bj_holidays
from src.features.graphs import build_spatial_graph, save_graph, load_graph, graph_stats
from src.features.node_features import build_node_features, hist_stats

# 加载图
g = load_graph("data/processed/graph_transit_bj.pt")
print(graph_stats("data/processed/graph_transit_bj.pt"))
# → nodes=1024, edges=3080, avg_deg=3.0

# 组装节点特征
feat = build_node_features(
    grid_csv="data/processed/grid_bj.csv",
    spatial_csv="data/processed/spatial_features_bj.csv",
    poi_csv="data/processed/poi_features_bj.csv",
    satellite_csv="data/processed/satellite_features_bj.csv",
    transit_csv="data/processed/transit_features_bj.csv",
    subway_feat_csv="data/processed/subway_grid_features_bj.csv",
    bus_feat_csv="data/processed/bus_route_features_bj.csv",
    flow_npz="data/raw/taxibj/taxibj.npz",
    tz_offset=8
)
assert feat.shape == (1024, 41) and feat.dtype == "float32"</pre>

<!-- ═══ SECTION 4 ═══════════════════════════════════════════════════════ -->
<h1 class="section pb">四、41 维静态节点特征矩阵</h1>

<h2 class="sub">4.1 特征维度规格</h2>

<table class="feat-idx">
  <tr><th>索引</th><th>分组</th><th>维度</th><th>数据来源</th><th>归一化</th></tr>
  <tr><td class="idx">[0:4]</td><td>地理位置（纬度/经度/城市中心距离/环路ID）</td><td>4</td><td>坐标计算</td><td>lat/lon/dist: min-max；ring: /5</td></tr>
  <tr><td class="idx">[4:6]</td><td>到交通枢纽/商业中心最短距离</td><td>2</td><td>Haversine</td><td>min-max [0,1]</td></tr>
  <tr><td class="idx">[6:11]</td><td>路网密度（总/主干/地方/服务/慢行）</td><td>5</td><td>OSM PBF</td><td>log1p → z-score</td></tr>
  <tr><td class="idx">[11:22]</td><td>POI 类别密度（11类功能性POI）</td><td>11</td><td>OSM PBF</td><td>log1p → z-score</td></tr>
  <tr><td class="idx">[22:27]</td><td>遥感指数（NDVI / NDBI / MNDWI / B4均值 / B4标准差）</td><td>5</td><td>Landsat-8</td><td>z-score</td></tr>
  <tr><td class="idx">[27:35]</td><td>历史流量统计（均值/峰值比/零值率/时段模式等）</td><td>8</td><td>TaxiBJ / TaxiNYC</td><td>见注①</td></tr>
  <tr><td class="idx">[35:38]</td><td>公共交通密度（地铁站/入口/公交站）</td><td>3</td><td>OSM PBF</td><td>log1p → z-score</td></tr>
  <tr><td class="idx">[38:40]</td><td>轨道线路数 / 换乘枢纽标志</td><td>2</td><td>政府数据 + OSM</td><td>log1p / binary</td></tr>
  <tr><td class="idx">[40]</td><td>公交线路数</td><td>1</td><td>MTA GTFS / OSM</td><td>log1p</td></tr>
</table>

<p style="font-size:8.5pt;color:#555;">注①：hist_mean: min-max；p99_ratio: [0,1]；zero_rate: [0,1]；wd_peak_ratio / we_ratio: clip [0,5]；night_ratio: clip [0,1]；flow_dir: min-max；flow_cv: clip [0,5]。</p>

<h2 class="sub">4.2 各分组统计详情</h2>

<h3 class="mini">路网密度（[6:11]）</h3>
<table>
  <tr><th>城市</th><th>提取道路数</th><th>路网密度均值</th><th>说明</th></tr>
  <tr><td>北京</td><td>90,380 条</td><td>10.0 km/km²</td><td>OSM PBF 36.1 MB，UTM 50N 投影</td></tr>
  <tr><td>纽约</td><td>114,938 条</td><td>34.3 km/km²</td><td>OSM PBF 151 MB（BBBike），曼哈顿格网路网密度高</td></tr>
</table>

<h3 class="mini">POI 语义特征（[11:22]）— transport 类完全剥离</h3>
<table>
  <tr><th>类别</th><th>北京数量</th><th>纽约数量</th><th>主要 OSM Tag</th></tr>
  <tr><td>food</td><td>5,326</td><td>10,240</td><td>amenity=restaurant/cafe/fast_food</td></tr>
  <tr><td>shopping</td><td>1,920</td><td>3,809</td><td>shop=supermarket/mall/clothes/...</td></tr>
  <tr><td>entertainment</td><td>249</td><td>407</td><td>leisure=cinema, amenity=theatre</td></tr>
  <tr><td>office</td><td>1,838</td><td>1,811</td><td>office=*, amenity=bank</td></tr>
  <tr><td>residential</td><td>104</td><td>28</td><td>building=apartments/residential</td></tr>
  <tr><td>education</td><td>251</td><td>614</td><td>amenity=school/university/college</td></tr>
  <tr><td>healthcare</td><td>323</td><td>1,199</td><td>amenity=hospital/clinic/pharmacy</td></tr>
  <tr><td>government</td><td>265</td><td>137</td><td>amenity=townhall/police/courthouse</td></tr>
  <tr><td>tourism</td><td>1,878</td><td>1,833</td><td>tourism=*, amenity=place_of_interest</td></tr>
  <tr><td>sports</td><td>342</td><td>729</td><td>leisure=sports_centre/stadium/gym</td></tr>
  <tr><td>religious</td><td>39</td><td>334</td><td>amenity=place_of_worship</td></tr>
  <tr><td><strong>合计</strong></td><td><strong>12,536</strong></td><td><strong>21,144</strong></td><td>—</td></tr>
</table>
<div class="callout info">transport 类已完全从 POI 特征中剥离，改由独立的公共交通密度特征（[35:38]）建模，
避免北京 transport 类 POI（23,549 条）主导余弦相似度导致 G_poi 图退化。</div>

<h3 class="mini">遥感特征（[22:27]）— Landsat-8</h3>
<table>
  <tr><th>特征</th><th>公式</th><th>北京均值</th><th>纽约均值</th></tr>
  <tr><td>ndvi</td><td>(B5−B4)/(B5+B4)</td><td>0.219</td><td>0.034</td></tr>
  <tr><td>ndbi</td><td>(B6−B5)/(B6+B5)</td><td>−0.022</td><td>−0.070</td></tr>
  <tr><td>mndwi</td><td>(B3−B6)/(B3+B6)</td><td>−0.225</td><td>0.009</td></tr>
  <tr><td>b4_mean</td><td>红波段平均反射率</td><td>0.142</td><td>0.115</td></tr>
  <tr><td>b4_std</td><td>红波段标准差</td><td>0.033</td><td>0.035</td></tr>
</table>

<h3 class="mini">公共交通特征（[35:40]）— 详细统计</h3>
<table>
  <tr><th>特征</th><th>北京</th><th>纽约</th></tr>
  <tr><td>subway_station_density</td><td>369站，253/1024格（24.7%）覆盖</td><td>238站，41/75格（54.7%）覆盖</td></tr>
  <tr><td>subway_entrance_density</td><td>983入口，208/1024格覆盖</td><td>1,153入口，41/75格覆盖</td></tr>
  <tr><td>bus_stop_density</td><td>10,831站，787/1024格（76.9%），均值4.56/km²</td><td>3,015站，66/75格（88.0%），均值12.57/km²</td></tr>
  <tr><td>num_subway_lines（[38]）</td><td>208/1024格有线路，55格为换乘枢纽</td><td>41/75格有线路，33格为换乘枢纽</td></tr>
  <tr><td>is_transfer_hub（[39]）</td><td>binary（西直门/国贸/东直门等）</td><td>binary（Times Sq/Grand Central等）</td></tr>
  <tr><td>num_bus_routes（[40]）</td><td>765/1024（74.7%）格，均值7.6，最大64</td><td>48/75（64.0%）格，均值7.6，最大41</td></tr>
</table>

<h2 class="sub">4.3 输出矩阵</h2>
<div class="cards">
  <div class="card"><div class="num">(1024, 41)</div><div class="unit">float32 · 164.1 KB</div><div class="label">node_features_bj.npy · NaN=0 ✅</div></div>
  <div class="card"><div class="num">(75, 41)</div><div class="unit">float32 · 12.1 KB</div><div class="label">node_features_nyc.npy · NaN=0 ✅</div></div>
</div>

<!-- ═══ SECTION 5 ═══════════════════════════════════════════════════════ -->
<h1 class="section pb">五、多模态图结构（方案C 主线）</h1>

<div class="callout info">
  每城市构建 4 张同构图，共享同一节点集合，后续通过门控网络融合。
  图以 PyTorch Geometric 兼容格式存储（edge_index / edge_weight / num_nodes）。
</div>

<h2 class="sub">5.1 四类图构建方法</h2>

<h3 class="mini">G_spatial — 空间邻接图</h3>
<p>Moore 8邻域，边权 = 高斯距离衰减，带宽 σ=1.5格：</p>
<pre>w(i,j) = exp(−d(i,j)² / σ²)
正交邻居 d=1.0 → w=0.641；斜角邻居 d=√2 → w=0.411</pre>

<h3 class="mini">G_poi — POI 功能相似图</h3>
<p>11类功能 POI 密度向量，log1p → L2归一化 → 余弦相似度，取 top-k=10 近邻。</p>

<h3 class="mini">G_flow — 历史流量相关图</h3>
<p>全量 inflow 时序 Pearson 相关矩阵，取 top-k=15 正相关近邻。北京相关均值 0.857，纽约 0.691。</p>

<h3 class="mini">G_transit — 轨道线路拓扑图</h3>
<p>
同一条轨道线路途经的相邻站点之间连边，边权 = exp(−站间距/σ)。
ChebConv 要求对称拉普拉斯，图为无向图（双向有向边实现）。
北京使用政府 xlsx（18条线，342站）与 OSM 坐标匹配；纽约使用 OSM route=subway 关系（31条线，1,601节点）。
</p>

<h2 class="sub">5.2 图统计汇总</h2>

<table>
  <tr><th>城市</th><th>图</th><th>节点数</th><th>边数</th><th>边/节点</th><th>存储文件</th></tr>
  <tr><td>北京</td><td>G_spatial</td><td>1,024</td><td>7,812</td><td>7.6</td><td>graph_spatial_bj.pt</td></tr>
  <tr><td>北京</td><td>G_poi</td><td>1,024</td><td>10,240</td><td>10.0</td><td>graph_poi_bj.pt</td></tr>
  <tr><td>北京</td><td>G_flow</td><td>1,024</td><td>15,255</td><td>14.9</td><td>graph_flow_bj.pt</td></tr>
  <tr><td>北京</td><td>G_transit</td><td>1,024</td><td>3,080</td><td>3.0</td><td>graph_transit_bj.pt</td></tr>
  <tr><td>纽约</td><td>G_spatial</td><td>75</td><td>484</td><td>6.5</td><td>graph_spatial_nyc.pt</td></tr>
  <tr><td>纽约</td><td>G_poi</td><td>75</td><td>750</td><td>10.0</td><td>graph_poi_nyc.pt</td></tr>
  <tr><td>纽约</td><td>G_flow</td><td>75</td><td>1,125</td><td>15.0</td><td>graph_flow_nyc.pt</td></tr>
  <tr><td>纽约</td><td>G_transit</td><td>75</td><td>754</td><td>10.1</td><td>graph_transit_nyc.pt</td></tr>
</table>

<h2 class="sub">5.3 动态时间特征（12维）</h2>

<p>采用正弦余弦编码将周期性时间变量映射到连续向量空间：</p>

<table>
  <tr><th>特征名</th><th>周期</th><th>含义</th></tr>
  <tr><td>slot_sin / slot_cos</td><td>48</td><td>日内第几个 30min 步（0~47）</td></tr>
  <tr><td>hour_sin / hour_cos</td><td>24</td><td>小时（0~23）</td></tr>
  <tr><td>weekday_sin / weekday_cos</td><td>7</td><td>星期（0=周一）</td></tr>
  <tr><td>month_sin / month_cos</td><td>12</td><td>月份（1~12）</td></tr>
  <tr><td>doy_sin / doy_cos</td><td>366</td><td>年内第几天</td></tr>
  <tr><td>is_weekend</td><td>—</td><td>二值：0=工作日，1=周末</td></tr>
  <tr><td>is_holiday</td><td>—</td><td>二值：0=普通日，1=节假日（BJ 8.0%；NYC 2.7%）</td></tr>
</table>

<!-- ═══ SECTION 6 ═══════════════════════════════════════════════════════ -->
<h1 class="section pb">六、数据质量分析</h1>

<h2 class="sub">6.1 总体评估</h2>

<table>
  <tr><th>数据集</th><th>完整性</th><th>可用性</th><th>主要问题</th></tr>
  <tr><td>TaxiBJ 人流</td><td>✅ 无缺失</td><td>高</td><td>四段不连续，非全年</td></tr>
  <tr><td>TaxiNYC 人流</td><td>✅ 无缺失</td><td>高</td><td>零值比例 10.7%（边缘格正常）</td></tr>
  <tr><td>METR-LA 速度</td><td>✅ 无缺失</td><td>高</td><td>速度非人流，定位为辅助基线</td></tr>
  <tr><td>OpenWeatherMap 北京</td><td>⚠️ 部分缺失</td><td>中</td><td>visibility/sea_level/wind_gust 100% 缺失</td></tr>
  <tr><td>OpenWeatherMap 纽约</td><td>⚠️ 部分缺失</td><td>中高</td><td>sea_level/grnd_level 100% 缺失</td></tr>
  <tr><td>Open-Meteo ERA5</td><td>✅ 无缺失</td><td>高</td><td>字段少但全完整，作为核心气象输入</td></tr>
  <tr><td>OSM 北京 / 纽约</td><td>✅ 完整</td><td>高</td><td>纽约初次下载文件 30MB（EOF 损坏），已重新获取 151MB</td></tr>
  <tr><td>Landsat-8 北京 / 纽约</td><td>✅ 完整</td><td>高</td><td>单景静态特征；云覆盖率 &lt;0.2%</td></tr>
  <tr><td>北京政府交通数据</td><td>✅ 完整</td><td>高</td><td>2026年版，新线路在bbox外，无影响</td></tr>
  <tr><td>MTA GTFS（5区公交）</td><td>✅ 完整</td><td>高</td><td>5区分包，100% 站点覆盖</td></tr>
</table>

<h2 class="sub">6.2 TaxiBJ 流量统计</h2>

<table>
  <tr><th>年份</th><th>时间步数</th><th>时间段</th><th>Inflow 均值</th><th>Inflow 最大</th><th>零值占比</th></tr>
  <tr><td>BJ13</td><td>4,888</td><td>2013-07-01 ~ 2013-10-29</td><td>97.3</td><td>1,230</td><td>5.68%</td></tr>
  <tr><td>BJ14</td><td>4,780</td><td>2014-03-01 ~ 2014-06-27</td><td>111.1</td><td>1,285</td><td>5.16%</td></tr>
  <tr><td>BJ15</td><td>5,596</td><td>2015-03-01 ~ 2015-06-30</td><td>115.4</td><td>1,267</td><td>4.71%</td></tr>
  <tr><td>BJ16</td><td>7,220</td><td>2015-11-01 ~ 2016-04-10</td><td>94.5</td><td>1,250</td><td>5.46%</td></tr>
  <tr><td><strong>合计</strong></td><td><strong>22,484</strong></td><td colspan="3">四段（不连续），30分钟/步，(T,2,32,32)</td><td></td></tr>
</table>

<h2 class="sub">6.3 OSM 关键问题记录</h2>
<div class="callout warn">
  <strong>纽约 OSM 文件损坏问题：</strong>初次从 Geofabrik 下载文件仅 30.7MB（完整应为 151MB）。
  OSM PBF 格式中 Node 块先于 Way 块存储，截断文件导致 POI 节点可正常解析但道路几何完全缺失，
  最终路网密度全部为 0。排查后从 BBBike 重新下载完整文件，路网（114,938条）、POI（21,144个）、
  公交站点（3,015个）均恢复正常。
</div>

<h2 class="sub">6.4 MTA GTFS 数据结构</h2>

<table>
  <tr><th>子目录</th><th>覆盖区</th><th>路线数（type=3）</th><th>bbox内站点数</th></tr>
  <tr><td>gtfs_bus/gtfs_b/</td><td>Brooklyn</td><td>255</td><td>648</td></tr>
  <tr><td>gtfs_bus/gtfs_bx/</td><td>Bronx</td><td>255</td><td>656</td></tr>
  <tr><td>gtfs_bus/gtfs_m/</td><td>Manhattan</td><td>255</td><td>1,813</td></tr>
  <tr><td>gtfs_bus/gtfs_q/</td><td>Queens</td><td>255</td><td>27</td></tr>
  <tr><td>gtfs_bus/gtfs_si/</td><td>Staten Island</td><td>255</td><td>156</td></tr>
  <tr><td><strong>合计（去重）</strong></td><td>全市</td><td><strong>255</strong></td><td><strong>3,168</strong></td></tr>
</table>
<p style="font-size:8.5pt;color:#555;">注：routes.txt 在5个区GTFS中内容相同（同一套全市线路），stop_times.txt各区独立（最大 ~141MB），使用 chunksize=200,000 分块读取。</p>

<!-- ═══ SECTION 7 ═══════════════════════════════════════════════════════ -->
<h1 class="section pb">七、预处理产物清单</h1>

<h2 class="sub">7.1 data/processed/ 完整文件列表</h2>

<table>
  <tr><th>文件</th><th>规格</th><th>状态</th></tr>
  <tr><td>temporal_features_bj.csv</td><td>(22484, 12)  北京时间特征</td><td>✅</td></tr>
  <tr><td>temporal_features_nyc.csv</td><td>(17520, 12)  纽约时间特征</td><td>✅</td></tr>
  <tr><td>spatial_features_bj.csv</td><td>(1024, 16)   北京路网+距离</td><td>✅</td></tr>
  <tr><td>spatial_features_nyc.csv</td><td>(75, 16)     纽约路网+距离</td><td>✅</td></tr>
  <tr><td>poi_features_bj.csv</td><td>(1024, 22)   北京 POI（11类×2）</td><td>✅</td></tr>
  <tr><td>poi_features_nyc.csv</td><td>(75, 22)     纽约 POI</td><td>✅</td></tr>
  <tr><td>satellite_features_bj.csv</td><td>(1024, 5)    北京遥感指数</td><td>✅</td></tr>
  <tr><td>satellite_features_nyc.csv</td><td>(75, 5)      纽约遥感指数</td><td>✅</td></tr>
  <tr><td>transit_features_bj.csv</td><td>(1024, 6)    北京公共交通密度</td><td>✅</td></tr>
  <tr><td>transit_features_nyc.csv</td><td>(75, 6)      纽约公共交通密度</td><td>✅</td></tr>
  <tr><td>subway_grid_features_bj.csv</td><td>(1024, 2)    北京轨道线路特征</td><td>✅</td></tr>
  <tr><td>subway_grid_features_nyc.csv</td><td>(75, 2)      纽约轨道线路特征</td><td>✅</td></tr>
  <tr><td>bus_route_features_bj.csv</td><td>(1024, 1)    北京公交线路数</td><td>✅</td></tr>
  <tr><td>bus_route_features_nyc.csv</td><td>(75, 1)      纽约公交线路数</td><td>✅</td></tr>
  <tr><td>node_features_bj.npy</td><td>(1024, 41) float32, 164.1KB, NaN=0</td><td>✅</td></tr>
  <tr><td>node_features_nyc.npy</td><td>(75, 41) float32, 12.1KB, NaN=0</td><td>✅</td></tr>
  <tr><td>graph_spatial_bj.pt</td><td>edges=7,812</td><td>✅</td></tr>
  <tr><td>graph_poi_bj.pt</td><td>edges=10,240</td><td>✅</td></tr>
  <tr><td>graph_flow_bj.pt</td><td>edges=15,255</td><td>✅</td></tr>
  <tr><td>graph_transit_bj.pt</td><td>edges=3,080</td><td>✅</td></tr>
  <tr><td>graph_spatial_nyc.pt</td><td>edges=484</td><td>✅</td></tr>
  <tr><td>graph_poi_nyc.pt</td><td>edges=750</td><td>✅</td></tr>
  <tr><td>graph_flow_nyc.pt</td><td>edges=1,125</td><td>✅</td></tr>
  <tr><td>graph_transit_nyc.pt</td><td>edges=754</td><td>✅</td></tr>
</table>

<!-- ═══ SECTION 8 ═══════════════════════════════════════════════════════ -->
<h1 class="section pb">八、Week 2 完成情况总结</h1>

<h2 class="sub">8.1 任务完成核查</h2>

<table>
  <tr><th>状态</th><th>任务</th><th>涉及数据/文件</th></tr>
  <tr><td>✅</td><td>TaxiBJ 四段时间对齐与拼接</td><td>taxibj/*.h5</td></tr>
  <tr><td>✅</td><td>OpenWeatherMap 时间戳解析、缺失字段处理、插值30分钟</td><td>openweather/*.csv</td></tr>
  <tr><td>✅</td><td>TaxiNYC 气象温度单位转换（°F → °C）</td><td>taxinyc/Meteorology.h5</td></tr>
  <tr><td>✅</td><td>OSM POI 提取与网格聚合（11类功能POI，transport剥离）</td><td>osm/*.pbf</td></tr>
  <tr><td>✅</td><td>Landsat-8 DN → 反射率转换，NDVI/NDBI/MNDWI 计算</td><td>Landsat8_*/B*.TIF</td></tr>
  <tr><td>✅</td><td>ERA5 + OpenWeatherMap 字段合并，30分钟气象特征矩阵</td><td>weather/*.h5 + openweather/*.csv</td></tr>
  <tr><td>✅</td><td>METR-LA parquet 格式解析</td><td>metr_la/*.parquet</td></tr>
  <tr><td>✅</td><td>公共交通特征（地铁站/入口/公交站密度，3维）</td><td>OSM PBF</td></tr>
  <tr><td>✅</td><td>轨道线路特征（num_subway_lines, is_transfer_hub，2维）</td><td>政府xlsx + OSM PBF</td></tr>
  <tr><td>✅</td><td>公交线路特征（num_bus_routes，1维）</td><td>OSM + MTA GTFS 5区</td></tr>
  <tr><td>✅</td><td>多模态图构建（G_spatial / G_poi / G_flow / G_transit）</td><td>各中间特征文件</td></tr>
  <tr><td>✅</td><td>节点特征矩阵合并（N×41，NaN=0）</td><td>build_graphs.py</td></tr>
</table>

<h2 class="sub">8.2 额外完成项</h2>
<ul class="checklist">
  <li>src/features/ 可导入 Python 特征工程代码库（4模块，16公开符号）</li>
  <li>量化 / TensorRT 加速预研报告（INT8 / FP16 方案对比，推理加速评估）</li>
  <li>可视化界面设计文档（Streamlit + Kepler.gl 交互设计）</li>
  <li>完整的 static_features_spec.md v1.4（41维特征完整规格）</li>
  <li>graph_fusion_design.md（多图融合方案C最终设计）</li>
  <li>数据规模差异说明（3.7GB 实际 vs 20GB PDF估计）</li>
</ul>

<h2 class="sub">8.3 Week 3 预告</h2>
<div class="callout info">
  <strong>下周目标：基础时空预测模型</strong><br>
  基于本周完成的 (N,41) 节点特征矩阵和 4 类多模态图，训练并评估：
  ARIMA 时序基线 → LSTM 序列模型 → 标准 GCN（单图）→ STGCN（时空图卷积）。
  评估指标：MAE / RMSE / MAPE，在 TaxiBJ 和 TaxiNYC 两个数据集上对比。
</div>

</div><!-- .page -->
</body>
</html>
"""

OUTPUT_PATH = Path("outputs/reports/week2_report.pdf")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("正在生成 PDF...")
html = HTML(string=HTML_CONTENT, base_url=".")
html.write_pdf(str(OUTPUT_PATH))
print(f"✅ 已生成：{OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
