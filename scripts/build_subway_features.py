"""
轨道站点名称匹配与特征提取

流程：
  1. 从 OSM PBF 提取地铁站（含 name 标签）
  2. 政府数据站名 → OSM 节点 精确/模糊匹配
  3. 生成 data/processed/subway_station_lines_bj.csv
  4. 计算每格 num_subway_lines、is_transfer_hub
  5. 构建 G_transit（线路拓扑无向图）

NYC：通过 OSM route=subway relation 提取，无需政府数据。

输出：
  data/processed/subway_station_lines_bj.csv   — 站名-线路-坐标-grid_id
  data/processed/subway_grid_features_bj.csv   — (1024, 2): num_subway_lines, is_transfer_hub
  data/processed/subway_grid_features_nyc.csv  — (75,   2)
  data/processed/graph_transit_bj.pt
  data/processed/graph_transit_nyc.pt
"""

import math
import numpy as np
import pandas as pd
import osmium
import torch
from difflib import SequenceMatcher
from pathlib import Path

PROC    = Path("data/processed")
GOV_DIR = Path("data/raw/jtw_beijing_gov")

GTRANSIT_K_MAX = 10   # 同一线路最大连通站数
GTRANSIT_SIGMA = 3.0  # 高斯衰减带宽（站跳数）


# ═══════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    """去掉常见前后缀，便于匹配"""
    s = str(name).strip()
    for suffix in ["站", "地铁站"]:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    for prefix in ["地铁"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def assign_to_grid(lon, lat, grid_meta: pd.DataFrame):
    """单点分配到网格，返回 grid_id 或 -1（越界）"""
    H = grid_meta["row"].max() + 1
    W = grid_meta["col"].max() + 1
    glon_min = grid_meta["lon_min"].min()
    glat_max = grid_meta["lat_max"].max()
    cell_lon  = (grid_meta["lon_max"] - grid_meta["lon_min"]).iloc[0]
    cell_lat  = (grid_meta["lat_max"] - grid_meta["lat_min"]).iloc[0]
    col = int((lon - glon_min) / cell_lon)
    row = int((glat_max - lat)  / cell_lat)
    if 0 <= row < H and 0 <= col < W:
        return row * W + col
    return -1


# ═══════════════════════════════════════════════════════════════════
# Step 1 — 提取 OSM 地铁站（含 name）
# ═══════════════════════════════════════════════════════════════════

class SubwayNameExtractor(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = bbox
        self.rows = []

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        if not (self.lon_min <= lon <= self.lon_max
                and self.lat_min <= lat <= self.lat_max):
            return
        tags = dict(n.tags)
        if tags.get("station") == "subway":
            self.rows.append({
                "osm_id"  : n.id,
                "name_osm": tags.get("name", ""),
                "lon"     : lon,
                "lat"     : lat,
            })


def extract_osm_subway(pbf_path, bbox) -> pd.DataFrame:
    ext = SubwayNameExtractor(bbox)
    ext.apply_file(str(pbf_path), locations=True)
    df = pd.DataFrame(ext.rows)
    df["name_norm"] = df["name_osm"].map(normalize_name)
    print(f"    OSM 地铁站提取: {len(df)} 个，唯一名称: {df['name_norm'].nunique()}")
    return df


# ═══════════════════════════════════════════════════════════════════
# Step 2 — 站名匹配
# ═══════════════════════════════════════════════════════════════════

def match_station_names(df_gov: pd.DataFrame,
                        df_osm: pd.DataFrame,
                        threshold: float = 0.80) -> pd.DataFrame:
    """
    df_gov 列: 车站名称, 线路名称, 行车方向, 首班车时间, 末班车时间
    df_osm 列: osm_id, name_osm, lon, lat, name_norm

    返回 merged DataFrame，含 match_score 列。
    """
    # 构建 OSM 名称索引：norm_name → 所有匹配行
    osm_index = {}
    for _, row in df_osm.iterrows():
        key = row["name_norm"]
        osm_index.setdefault(key, []).append(row)

    results = []
    unmatched = []

    for _, gov_row in df_gov.drop_duplicates(["车站名称", "线路名称", "行车方向"]).iterrows():
        gov_norm = normalize_name(gov_row["车站名称"])

        # 精确匹配
        if gov_norm in osm_index:
            # 如果同名有多个 OSM 节点，取第一个（通常是同一站点的不同入口节点，坐标接近）
            osm_row = osm_index[gov_norm][0]
            results.append({**gov_row.to_dict(), **osm_row.to_dict(),
                             "name_norm_gov": gov_norm, "match_score": 1.0})
            continue

        # 模糊匹配
        best_score, best_row = 0.0, None
        for norm_name, osm_rows in osm_index.items():
            score = fuzzy_score(gov_norm, norm_name)
            if score > best_score:
                best_score, best_row = score, osm_rows[0]

        if best_score >= threshold and best_row is not None:
            results.append({**gov_row.to_dict(), **best_row.to_dict(),
                             "name_norm_gov": gov_norm, "match_score": best_score})
        else:
            unmatched.append(gov_norm)

    print(f"    匹配成功: {len(results)}, 未匹配: {len(unmatched)}")
    if unmatched:
        print(f"    未匹配站名（前20）: {unmatched[:20]}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# Step 3 — 计算网格特征
# ═══════════════════════════════════════════════════════════════════

def compute_subway_grid_features(matched_df: pd.DataFrame,
                                  grid_meta: pd.DataFrame) -> pd.DataFrame:
    """返回 (N, 2) DataFrame: num_subway_lines, is_transfer_hub"""
    # 去重：每个(车站名称, 线路名称)唯一一行（去掉双向重复）
    dedup = matched_df.drop_duplicates(["车站名称", "线路名称"]).copy()
    dedup["grid_id"] = dedup.apply(
        lambda r: assign_to_grid(r["lon"], r["lat"], grid_meta), axis=1
    )
    dedup = dedup[dedup["grid_id"] >= 0]

    # 按格子聚合
    agg = dedup.groupby("grid_id").agg(
        num_subway_lines=("线路名称", "nunique"),
        # is_transfer_hub: 同一站名有 ≥2 条线经过
        _max_lines_per_station=("线路名称",
            lambda x: dedup.loc[x.index].groupby("车站名称")["线路名称"].nunique().max())
    )
    agg["is_transfer_hub"] = (agg["_max_lines_per_station"] >= 2).astype(int)
    agg = agg.drop(columns=["_max_lines_per_station"])

    # 补全所有格子（无地铁站的格子 = 0）
    full = pd.DataFrame({"grid_id": grid_meta["grid_id"]}).set_index("grid_id")
    full = full.join(agg, how="left").fillna(0).astype(int)
    return full.reset_index()


# ═══════════════════════════════════════════════════════════════════
# Step 4 — 构建 G_transit（线路拓扑无向图）
# ═══════════════════════════════════════════════════════════════════

def get_line_sequences(matched_df: pd.DataFrame) -> dict:
    """
    为每条线路取一个方向，按首班车时间排序得到站点顺序。
    返回 {line_name: [station_name, ...]}
    """
    import datetime as _dt
    sequences = {}
    for line, grp in matched_df.groupby("线路名称"):
        # 过滤"暂缓开通"/"——"等非 time 值，转为分钟数方便排序
        grp = grp.copy()
        grp["_minutes"] = grp["首班车时间"].apply(
            lambda x: x.hour * 60 + x.minute
            if isinstance(x, _dt.time) else None
        )
        valid = grp.dropna(subset=["_minutes"])
        if valid.empty:
            continue
        # 取首班车时间最早的方向
        dir_min = valid.groupby("行车方向")["_minutes"].min()
        dir0 = dir_min.idxmin()
        ordered = (valid[valid["行车方向"] == dir0]
                   .drop_duplicates("车站名称")
                   .sort_values("_minutes")["车站名称"]
                   .tolist())
        sequences[line] = ordered
    return sequences


def build_gtransit_graph(matched_df: pd.DataFrame,
                          grid_meta: pd.DataFrame,
                          K_max: int = GTRANSIT_K_MAX,
                          sigma: float = GTRANSIT_SIGMA):
    """
    返回 {"edge_index": LongTensor(2,E), "edge_weight": FloatTensor(E), "num_nodes": N}
    """
    N = len(grid_meta)

    # station_name → grid_id（取唯一匹配，多个取第一个）
    dedup = matched_df.drop_duplicates("车站名称").copy()
    dedup["grid_id"] = dedup.apply(
        lambda r: assign_to_grid(r["lon"], r["lat"], grid_meta), axis=1
    )
    station_grid = dict(zip(dedup["车站名称"], dedup["grid_id"]))

    sequences = get_line_sequences(matched_df)

    edge_dict = {}  # (min_gid, max_gid) → min_hops
    for line, stations in sequences.items():
        grids = [(station_grid.get(s, -1), s) for s in stations]
        grids = [(g, s) for g, s in grids if g >= 0]

        for i, (gi, _) in enumerate(grids):
            for j in range(i + 1, min(i + K_max + 1, len(grids))):
                gj = grids[j][0]
                if gi == gj:
                    continue
                hops = j - i
                key = (min(gi, gj), max(gi, gj))
                if key not in edge_dict or edge_dict[key] > hops:
                    edge_dict[key] = hops

    if not edge_dict:
        return {"edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_weight": torch.zeros(0, dtype=torch.float32),
                "num_nodes": N}

    src, dst, weights = [], [], []
    for (gi, gj), hops in edge_dict.items():
        w = math.exp(-hops / sigma)
        src += [gi, gj]
        dst += [gj, gi]
        weights += [w, w]

    return {
        "edge_index" : torch.tensor([src, dst], dtype=torch.long),
        "edge_weight": torch.tensor(weights, dtype=torch.float32),
        "num_nodes"  : N,
    }


# ═══════════════════════════════════════════════════════════════════
# NYC：通过 OSM route=subway relation 提取
# ═══════════════════════════════════════════════════════════════════

class NYCSubwayRelationExtractor(osmium.SimpleHandler):
    """解析 OSM subway route relation，提取站点-线路归属"""
    def __init__(self, bbox):
        super().__init__()
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = bbox
        self.node_coords  = {}  # osm_id → (lon, lat)
        self.node_names   = {}  # osm_id → name
        self.line_members = []  # (line_name, osm_id)

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        if not (self.lon_min <= lon <= self.lon_max
                and self.lat_min <= lat <= self.lat_max):
            return
        tags = dict(n.tags)
        self.node_coords[n.id] = (lon, lat)
        if "name" in tags:
            self.node_names[n.id] = tags["name"]

    def relation(self, r):
        tags = dict(r.tags)
        if tags.get("route") != "subway":
            return
        line_name = tags.get("ref", tags.get("name", "unknown"))
        for m in r.members:
            if m.type == "n":           # node member
                self.line_members.append((line_name, m.ref))


def build_nyc_subway_features(pbf_path, bbox, grid_meta) -> tuple:
    """返回 (feature_df, gtransit_graph_dict)"""
    print(f"    解析 NYC OSM subway relations...")
    ext = NYCSubwayRelationExtractor(bbox)
    try:
        ext.apply_file(str(pbf_path), locations=True)
    except RuntimeError as e:
        print(f"    [!] OSM 警告: {e}")

    # 只保留 bbox 内的站点
    rows = []
    for line_name, osm_id in ext.line_members:
        if osm_id in ext.node_coords:
            lon, lat = ext.node_coords[osm_id]
            name = ext.node_names.get(osm_id, "")
            gid  = assign_to_grid(lon, lat, grid_meta)
            if gid >= 0:
                rows.append({
                    "line_name": line_name,
                    "osm_id"   : osm_id,
                    "name_osm" : name,
                    "lon": lon, "lat": lat,
                    "grid_id": gid,
                })

    if not rows:
        print("    [!] 未提取到 NYC subway relation 成员，features 置 0")
        N = len(grid_meta)
        feat = pd.DataFrame({
            "grid_id"         : grid_meta["grid_id"],
            "num_subway_lines": 0,
            "is_transfer_hub" : 0,
        })
        graph = {"edge_index": torch.zeros(2, 0, dtype=torch.long),
                 "edge_weight": torch.zeros(0), "num_nodes": N}
        return feat, graph

    df = pd.DataFrame(rows)
    print(f"    NYC subway relation 成员: {len(df)}, 线路数: {df['line_name'].nunique()}")

    # 计算网格特征
    agg = df.drop_duplicates(["osm_id", "line_name"]).groupby("grid_id").agg(
        num_subway_lines=("line_name", "nunique"),
        _max_lines=("line_name",
            lambda x: df.loc[x.index].groupby("osm_id")["line_name"].nunique().max())
    )
    agg["is_transfer_hub"] = (agg["_max_lines"] >= 2).astype(int)
    agg = agg.drop(columns=["_max_lines"])

    full = pd.DataFrame({"grid_id": grid_meta["grid_id"]}).set_index("grid_id")
    full = full.join(agg, how="left").fillna(0).astype(int)
    feat = full.reset_index()

    # G_transit：同一站点（osm_id）出现在多条线路即为换乘，按线路连格
    station_grid = df.drop_duplicates("osm_id").set_index("osm_id")["grid_id"].to_dict()

    # NYC relation 不包含顺序，用格子间最近对建边（简化：hops=1 for shared-line neighbors）
    line_grids = df.drop_duplicates(["line_name", "grid_id"]).groupby("line_name")["grid_id"].apply(list)
    edge_dict = {}
    for grids in line_grids:
        grids = sorted(set(grids))
        for i, gi in enumerate(grids):
            for gj in grids[i+1:i+GTRANSIT_K_MAX+1]:
                key = (min(gi, gj), max(gi, gj))
                edge_dict.setdefault(key, 1)

    src, dst, weights = [], [], []
    for (gi, gj), hops in edge_dict.items():
        w = math.exp(-hops / GTRANSIT_SIGMA)
        src += [gi, gj]; dst += [gj, gi]; weights += [w, w]

    graph = {
        "edge_index" : torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long),
        "edge_weight": torch.tensor(weights, dtype=torch.float32) if weights else torch.zeros(0),
        "num_nodes"  : len(grid_meta),
    }
    return feat, graph


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  轨道站点匹配与 G_transit 构建")
    print("=" * 60)

    # ── 北京 ──────────────────────────────────────────────────────
    print("\n[BJ]")
    bj_grid  = pd.read_csv(PROC / "grid_meta_bj.csv")
    bj_pbf   = "data/raw/osm/beijing-latest.osm.pbf"
    bj_bbox  = (116.25, 39.75, 116.75, 40.25)

    # 1. 提取 OSM 站名
    df_osm = extract_osm_subway(bj_pbf, bj_bbox)

    # 2. 载入政府数据
    df_gov = pd.read_excel(GOV_DIR / "轨道站点信息.xlsx")
    print(f"    政府站点记录: {len(df_gov)}, 唯一站名: {df_gov['车站名称'].nunique()}")

    # 3. 匹配
    matched = match_station_names(df_gov, df_osm)

    # 4. 分配网格
    matched["grid_id"] = matched.apply(
        lambda r: assign_to_grid(r["lon"], r["lat"], bj_grid), axis=1
    )
    matched = matched[matched["grid_id"] >= 0].copy()

    # 5. 保存站名-线路表
    out_csv = PROC / "subway_station_lines_bj.csv"
    save_cols = ["车站名称", "线路名称", "行车方向", "首班车时间", "末班车时间",
                 "name_osm", "match_score", "lon", "lat", "grid_id"]
    matched[save_cols].to_csv(out_csv, index=False)
    print(f"    已保存: {out_csv.name}  shape={matched[save_cols].shape}")

    # 6. 网格特征
    feat_bj = compute_subway_grid_features(matched, bj_grid)
    feat_bj.to_csv(PROC / "subway_grid_features_bj.csv", index=False)
    print(f"    num_subway_lines > 0 的格数: {(feat_bj['num_subway_lines'] > 0).sum()}")
    print(f"    is_transfer_hub == 1 的格数: {feat_bj['is_transfer_hub'].sum()}")

    # 7. G_transit
    graph_bj = build_gtransit_graph(matched, bj_grid)
    torch.save(graph_bj, PROC / "graph_transit_bj.pt")
    print(f"    G_transit BJ: {graph_bj['edge_index'].shape[1]//2} 条无向边")

    # ── 纽约 ──────────────────────────────────────────────────────
    print("\n[NYC]")
    nyc_grid = pd.read_csv(PROC / "grid_meta_nyc.csv")
    nyc_pbf  = "data/raw/osm/nyc-latest.osm.pbf"
    nyc_bbox = (-74.0166, 40.6996, -73.9004, 40.9196)

    feat_nyc, graph_nyc = build_nyc_subway_features(nyc_pbf, nyc_bbox, nyc_grid)
    feat_nyc.to_csv(PROC / "subway_grid_features_nyc.csv", index=False)
    torch.save(graph_nyc, PROC / "graph_transit_nyc.pt")
    print(f"    G_transit NYC: {graph_nyc['edge_index'].shape[1]//2} 条无向边")

    print("\n  完成。")


if __name__ == "__main__":
    main()
