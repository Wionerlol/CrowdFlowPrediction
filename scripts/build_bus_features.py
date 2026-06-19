"""
公交线路特征提取

流程：
  北京：
    1. 解析 OSM route=bus 关系 → node_id → route_refs 映射（覆盖率 ~95%）
    2. 提取 bus_stop 节点坐标
    3. 节点 ID 直连，剩余 5% 通过政府站名模糊匹配补全
    4. 将站点分配到网格，统计每格不同线路数 num_bus_routes

  纽约：
    优先使用 MTA GTFS 静态数据（100% 覆盖）。
    未找到 GTFS 时自动回退至 OSM 关系（~41.5% 覆盖）。

    GTFS 数据放置路径（zip 或解压目录均可）：
      data/raw/mta_gtfs/nyc_bus_gtfs.zip
    或
      data/raw/mta_gtfs/（含 stops.txt, routes.txt, trips.txt, stop_times.txt）

    下载方式：搜索 "MTA developer data GTFS" 下载 NYC Transit Bus GTFS，约 20MB。

输出：
  data/processed/bus_route_features_bj.csv  — (1024, 1): num_bus_routes
  data/processed/bus_route_features_nyc.csv — (75,   1): num_bus_routes
"""

import io
import re
import zipfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import osmium
import pandas as pd

PROC     = Path("data/processed")
GOV_DIR  = Path("data/raw/jtw_beijing_gov")
GTFS_DIR     = Path("data/raw/mta_gtfs")
GTFS_BUS_DIR = Path("data/raw/mta_gtfs/gtfs_bus")   # 各区公交子目录的父目录
GTFS_BUS_BOROUGHS = ["gtfs_b", "gtfs_bx", "gtfs_m", "gtfs_q", "gtfs_si"]

FUZZY_THRESHOLD = 0.80   # 政府补全匹配阈值
GTFS_SNAP_M     = 150    # GTFS 站点与 OSM 格网的最大对齐距离（米），用于验证

BJ_BBOX  = (116.25, 39.75, 116.75, 40.25)   # lon_min, lat_min, lon_max, lat_max
NYC_BBOX = (-74.0166, 40.6996, -73.9004, 40.9196)


# ═══════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════

def assign_to_grid(pts_lon: np.ndarray, pts_lat: np.ndarray,
                   grid_meta: pd.DataFrame) -> np.ndarray:
    """将坐标点分配到最近网格，返回 grid_id 数组（越界返回 -1）。"""
    lon_min = grid_meta["lon_min"].values.astype(np.float64)
    lon_max = grid_meta["lon_max"].values.astype(np.float64)
    lat_min = grid_meta["lat_min"].values.astype(np.float64)
    lat_max = grid_meta["lat_max"].values.astype(np.float64)
    gids    = grid_meta["grid_id"].values

    result = np.full(len(pts_lon), -1, dtype=np.int64)
    for i, (lo, la) in enumerate(zip(pts_lon, pts_lat)):
        mask = (lon_min <= lo) & (lo < lon_max) & (lat_min <= la) & (la < lat_max)
        hits = gids[mask]
        if len(hits) > 0:
            result[i] = hits[0]
    return result


def normalize_stop_name(name: str) -> str:
    """去除常见后缀噪声，标准化站名用于模糊匹配。"""
    name = name.strip()
    name = re.sub(r"[（(][^）)]*[）)]", "", name)   # 括号内容
    name = name.replace("公交站", "").replace("站", "").replace("路口", "")
    return name.strip()


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ═══════════════════════════════════════════════════════════════════
# OSM 解析：Pass-1 路线关系
# ═══════════════════════════════════════════════════════════════════

class BusRouteRelationHandler(osmium.SimpleHandler):
    """
    Pass-1：遍历 route=bus 关系，构建 node_id → set(route_ref) 映射。
    同时收集 node_id → set(route_ref) for 'platform*' role 节点。
    """

    def __init__(self):
        super().__init__()
        self.node_routes: dict[int, set] = defaultdict(set)
        self.route_count = 0

    def relation(self, r):
        tags = dict(r.tags)
        if tags.get("route") != "bus":
            return
        ref = tags.get("ref") or tags.get("name", f"route_{r.id}")
        ref = str(ref).strip()
        self.route_count += 1
        for m in r.members:
            if m.type == "n" and ("platform" in m.role or m.role == "stop"):
                self.node_routes[m.ref].add(ref)


# ═══════════════════════════════════════════════════════════════════
# OSM 解析：Pass-2 bus_stop 节点坐标
# ═══════════════════════════════════════════════════════════════════

class BusStopNodeHandler(osmium.SimpleHandler):
    """Pass-2：提取 highway=bus_stop 节点的 id / name / 坐标。"""

    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox   # (lon_min, lat_min, lon_max, lat_max)
        self.stops: list[dict] = []

    def node(self, n):
        if dict(n.tags).get("highway") != "bus_stop":
            return
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        lo_min, la_min, lo_max, la_max = self.bbox
        if not (lo_min <= lon <= lo_max and la_min <= lat <= la_max):
            return
        name = dict(n.tags).get("name", "")
        self.stops.append({"osm_id": n.id, "name_osm": name,
                           "lon": lon, "lat": lat})


# ═══════════════════════════════════════════════════════════════════
# 政府数据补全（北京专用）
# ═══════════════════════════════════════════════════════════════════

def build_gov_stop_routes() -> dict[str, set]:
    """
    从 公交站点信息.xlsx 构建：norm_stop_name → set(route_name)。
    """
    df = pd.read_excel(GOV_DIR / "公交站点信息.xlsx", usecols=[0, 3])
    df.columns = ["route", "stop"]
    df = df.dropna(subset=["stop"])
    mapping: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        norm = normalize_stop_name(str(row["stop"]))
        if norm:
            mapping[norm].add(str(row["route"]).strip())
    return mapping


def supplement_with_gov(stops_df: pd.DataFrame,
                        gov_map: dict[str, set]) -> pd.DataFrame:
    """
    对 routes 为空集的站点，用政府数据模糊匹配补全。
    """
    gov_norms = list(gov_map.keys())
    missing_mask = stops_df["routes"].apply(lambda s: len(s) == 0)
    print(f"    无关系覆盖站点: {missing_mask.sum()}")
    supplemented = 0
    for idx in stops_df.index[missing_mask]:
        name_osm = normalize_stop_name(stops_df.at[idx, "name_osm"])
        if not name_osm:
            continue
        best_score, best_norm = 0.0, None
        for gn in gov_norms:
            s = fuzzy_score(name_osm, gn)
            if s > best_score:
                best_score, best_norm = s, gn
        if best_score >= FUZZY_THRESHOLD and best_norm:
            stops_df.at[idx, "routes"] = gov_map[best_norm]
            supplemented += 1
    print(f"    政府数据补全: {supplemented} 个站点")
    return stops_df


# ═══════════════════════════════════════════════════════════════════
# 核心：每格统计路线数
# ═══════════════════════════════════════════════════════════════════

def compute_grid_bus_routes(stops_df: pd.DataFrame,
                            grid_meta: pd.DataFrame) -> pd.DataFrame:
    """
    对每格：汇总该格内所有站点的路线集合，统计 num_bus_routes。
    返回 DataFrame，index=grid_id，列=num_bus_routes。
    """
    gids = grid_meta["grid_id"].values
    route_sets: dict[int, set] = {g: set() for g in gids}

    for _, row in stops_df.iterrows():
        gid = row["grid_id"]
        if gid == -1:
            continue
        route_sets[gid].update(row["routes"])

    result = pd.DataFrame({
        "grid_id":        gids,
        "num_bus_routes": [len(route_sets[g]) for g in gids],
    }).set_index("grid_id")
    return result


# ═══════════════════════════════════════════════════════════════════
# 北京流程
# ═══════════════════════════════════════════════════════════════════

def build_bj_bus_features():
    print("\n  [BJ] 公交线路特征提取")
    pbf = Path("data/raw/osm/beijing-latest.osm.pbf")
    grid_meta = pd.read_csv(PROC / "grid_meta_bj.csv")

    print("  Pass-1：解析 route=bus 关系...")
    h1 = BusRouteRelationHandler()
    h1.apply_file(str(pbf))
    print(f"    {h1.route_count} 条公交线路，{len(h1.node_routes)} 个节点有关系覆盖")

    print("  Pass-2：提取 bus_stop 节点...")
    h2 = BusStopNodeHandler(BJ_BBOX)
    h2.apply_file(str(pbf), locations=True)
    stops_df = pd.DataFrame(h2.stops)
    print(f"    bbox 内站点: {len(stops_df)}")

    # 关联路线
    stops_df["routes"] = stops_df["osm_id"].map(
        lambda nid: h1.node_routes.get(nid, set())
    )
    covered = (stops_df["routes"].apply(len) > 0).sum()
    print(f"    OSM 关系覆盖率: {covered}/{len(stops_df)} ({covered/len(stops_df)*100:.1f}%)")

    # 政府数据补全
    print("  加载政府公交站点数据...")
    gov_map = build_gov_stop_routes()
    print(f"    政府站名词典: {len(gov_map)} 个规范名")
    stops_df = supplement_with_gov(stops_df, gov_map)

    # 分配到网格
    lons = stops_df["lon"].values
    lats = stops_df["lat"].values
    stops_df["grid_id"] = assign_to_grid(lons, lats, grid_meta)
    in_grid = (stops_df["grid_id"] != -1).sum()
    print(f"    入格站点: {in_grid}/{len(stops_df)}")

    # 统计
    feat_df = compute_grid_bus_routes(stops_df, grid_meta)
    out_path = PROC / "bus_route_features_bj.csv"
    feat_df.to_csv(out_path)

    n_nonzero = (feat_df["num_bus_routes"] > 0).sum()
    print(f"    覆盖网格: {n_nonzero}/{len(feat_df)}")
    print(f"    均值={feat_df['num_bus_routes'].mean():.1f}, "
          f"最大={feat_df['num_bus_routes'].max()}, "
          f"中位数={feat_df['num_bus_routes'].median():.0f}")
    print(f"    → {out_path}")
    return feat_df


# ═══════════════════════════════════════════════════════════════════
# MTA GTFS 解析（NYC 专用）
# ═══════════════════════════════════════════════════════════════════

def _find_gtfs_path() -> Path | None:
    """
    查找可用的单一 GTFS 源（zip 或目录，含 stops.txt）。
    多区分开的公交 GTFS 由 build_gtfs_multi_borough() 专门处理。
    """
    if not GTFS_DIR.exists():
        return None
    zips = sorted(GTFS_DIR.glob("*.zip"))
    if zips:
        return zips[0]
    if (GTFS_DIR / "stops.txt").exists():
        return GTFS_DIR
    return None


def _has_borough_gtfs() -> bool:
    """检查多区公交 GTFS 目录是否存在且完整。"""
    if not GTFS_BUS_DIR.exists():
        return False
    return any((GTFS_BUS_DIR / b / "stops.txt").exists()
               for b in GTFS_BUS_BOROUGHS)


def _read_gtfs_file(gtfs_path: Path, filename: str,
                    usecols: list[str]) -> pd.DataFrame:
    """从 zip 或目录读取 GTFS 文件，只加载需要的列。"""
    if gtfs_path.suffix == ".zip":
        with zipfile.ZipFile(gtfs_path) as zf:
            names = zf.namelist()
            match = next((n for n in names if n.endswith(filename)), None)
            if match is None:
                raise FileNotFoundError(f"{filename} 不在 {gtfs_path} 中")
            with zf.open(match) as f:
                return pd.read_csv(f, usecols=usecols, dtype=str)
    else:
        return pd.read_csv(gtfs_path / filename, usecols=usecols, dtype=str)


def build_gtfs_multi_borough(bbox: tuple) -> pd.DataFrame:
    """
    合并 5 个区的公交 GTFS，解析 bbox 内站点及其所属线路。

    结构：data/raw/mta_gtfs/gtfs_bus/gtfs_{b,bx,m,q,si}/
    返回 DataFrame: stop_id, stop_lat, stop_lon, routes (set)。
    """
    lo_min, la_min, lo_max, la_max = bbox

    # ── 合并 stops（跨区去重）────────────────────────────────────────
    print("  合并各区 stops.txt ...")
    all_stops = []
    for b in GTFS_BUS_BOROUGHS:
        p = GTFS_BUS_DIR / b / "stops.txt"
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["stop_id", "stop_lat", "stop_lon"],
                         dtype=str)
        all_stops.append(df)
    stops = pd.concat(all_stops, ignore_index=True).drop_duplicates("stop_id")
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    in_bbox = (
        (stops["stop_lon"] >= lo_min) & (stops["stop_lon"] <= lo_max) &
        (stops["stop_lat"] >= la_min) & (stops["stop_lat"] <= la_max)
    )
    stops = stops[in_bbox].copy()
    print(f"    去重后 bbox 内站点: {len(stops)}")
    bbox_stop_ids = set(stops["stop_id"])

    # ── 合并 trips（各区 trip_id 可能不同，需全部合并）────────────────
    print("  合并各区 trips.txt ...")
    all_trips = []
    for b in GTFS_BUS_BOROUGHS:
        p = GTFS_BUS_DIR / b / "trips.txt"
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["route_id", "trip_id"], dtype=str)
        all_trips.append(df)
    trips = pd.concat(all_trips, ignore_index=True).drop_duplicates("trip_id")
    trip2route = dict(zip(trips["trip_id"], trips["route_id"]))

    # ── 合并 stop_times（逐区读取，只保留 bbox 内站点）──────────────────
    print("  合并各区 stop_times.txt（仅 bbox 内站点）...")
    all_st = []
    for b in GTFS_BUS_BOROUGHS:
        p = GTFS_BUS_DIR / b / "stop_times.txt"
        if not p.exists():
            continue
        # 分块读取，避免全量载入 stop_times（每区约 20-50MB）
        chunks = pd.read_csv(p, usecols=["trip_id", "stop_id"],
                             dtype=str, chunksize=200_000)
        for chunk in chunks:
            filtered = chunk[chunk["stop_id"].isin(bbox_stop_ids)]
            if len(filtered):
                all_st.append(filtered)
    if not all_st:
        print("    警告: stop_times 中无 bbox 内站点记录")
        stops["routes"] = [set() for _ in range(len(stops))]
        return stops
    stop_times = pd.concat(all_st, ignore_index=True)
    stop_times["route_id"] = stop_times["trip_id"].map(trip2route)
    stop_times = stop_times.dropna(subset=["route_id"])

    # ── 汇总 stop_id → set(route_id) ────────────────────────────────
    stop_routes = (
        stop_times.groupby("stop_id")["route_id"]
        .apply(lambda x: set(x.unique()))
        .reset_index()
        .rename(columns={"route_id": "routes"})
    )
    stops = stops.merge(stop_routes, on="stop_id", how="left")
    stops["routes"] = stops["routes"].apply(
        lambda x: x if isinstance(x, set) else set()
    )

    covered = (stops["routes"].apply(len) > 0).sum()
    print(f"    线路覆盖率: {covered}/{len(stops)} ({covered/len(stops)*100:.1f}%)")
    return stops


def build_gtfs_stop_routes(gtfs_path: Path, bbox: tuple) -> pd.DataFrame:
    """从单一 GTFS zip/目录解析 bbox 内站点线路（单区或合并包）。"""
    print("  读取 GTFS stops.txt ...")
    stops = _read_gtfs_file(gtfs_path, "stops.txt",
                            ["stop_id", "stop_lat", "stop_lon"])
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["stop_lat", "stop_lon"])

    lo_min, la_min, lo_max, la_max = bbox
    stops = stops[
        (stops["stop_lon"] >= lo_min) & (stops["stop_lon"] <= lo_max) &
        (stops["stop_lat"] >= la_min) & (stops["stop_lat"] <= la_max)
    ].copy()
    print(f"    bbox 内站点: {len(stops)}")

    print("  读取 GTFS trips.txt ...")
    trips = _read_gtfs_file(gtfs_path, "trips.txt", ["route_id", "trip_id"])
    trip2route = dict(zip(trips["trip_id"], trips["route_id"]))

    print("  读取 GTFS stop_times.txt ...")
    stop_times = _read_gtfs_file(gtfs_path, "stop_times.txt",
                                 ["trip_id", "stop_id"])
    bbox_stop_ids = set(stops["stop_id"])
    stop_times = stop_times[stop_times["stop_id"].isin(bbox_stop_ids)]
    stop_times["route_id"] = stop_times["trip_id"].map(trip2route)
    stop_times = stop_times.dropna(subset=["route_id"])

    stop_routes = (
        stop_times.groupby("stop_id")["route_id"]
        .apply(lambda x: set(x.unique()))
        .reset_index()
        .rename(columns={"route_id": "routes"})
    )
    stops = stops.merge(stop_routes, on="stop_id", how="left")
    stops["routes"] = stops["routes"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    covered = (stops["routes"].apply(len) > 0).sum()
    print(f"    线路覆盖率: {covered}/{len(stops)} ({covered/len(stops)*100:.1f}%)")
    return stops


# ═══════════════════════════════════════════════════════════════════
# 纽约流程
# ═══════════════════════════════════════════════════════════════════

def build_nyc_bus_features():
    print("\n  [NYC] 公交线路特征提取")
    grid_meta = pd.read_csv(PROC / "grid_meta_nyc.csv")

    # ── 优先使用多区公交 GTFS ──────────────────────────────────────────
    if _has_borough_gtfs():
        print(f"  使用多区公交 GTFS: {GTFS_BUS_DIR}")
        stops_df = build_gtfs_multi_borough(NYC_BBOX)
        stops_df = stops_df.rename(columns={"stop_lat": "lat", "stop_lon": "lon"})
    else:
        gtfs_path = _find_gtfs_path()
        if gtfs_path is not None:
            print(f"  使用单一 GTFS: {gtfs_path}")
            stops_df = build_gtfs_stop_routes(gtfs_path, NYC_BBOX)
            stops_df = stops_df.rename(columns={"stop_lat": "lat", "stop_lon": "lon"})
        else:
            print("  未找到 MTA GTFS，回退至 OSM 关系（覆盖率约 41.5%）")
            print("  建议：将各区 GTFS 放置到 data/raw/mta_gtfs/gtfs_bus/ 以获得 100% 覆盖")
        pbf = Path("data/raw/osm/nyc-latest.osm.pbf")

        print("  Pass-1：解析 route=bus 关系...")
        h1 = BusRouteRelationHandler()
        h1.apply_file(str(pbf))
        print(f"    {h1.route_count} 条公交线路，{len(h1.node_routes)} 个节点有关系覆盖")

        print("  Pass-2：提取 bus_stop 节点...")
        h2 = BusStopNodeHandler(NYC_BBOX)
        h2.apply_file(str(pbf), locations=True)
        stops_df = pd.DataFrame(h2.stops)
        print(f"    bbox 内站点: {len(stops_df)}")

        if len(stops_df) == 0:
            print("    警告: NYC bbox 内无 bus_stop 节点，尝试扩展 bbox...")
            NYC_BBOX_EX = (-74.1, 40.6, -73.85, 40.95)
            h2 = BusStopNodeHandler(NYC_BBOX_EX)
            h2.apply_file(str(pbf), locations=True)
            stops_df = pd.DataFrame(h2.stops)
            print(f"    扩展后站点: {len(stops_df)}")

        stops_df["routes"] = stops_df["osm_id"].map(
            lambda nid: h1.node_routes.get(nid, set())
        )
        covered = (stops_df["routes"].apply(len) > 0).sum()
        if len(stops_df) > 0:
            print(f"    OSM 关系覆盖率: {covered}/{len(stops_df)} "
                  f"({covered/len(stops_df)*100:.1f}%)")

    stops_df["grid_id"] = assign_to_grid(
        stops_df["lon"].values, stops_df["lat"].values, grid_meta
    )
    in_grid = (stops_df["grid_id"] != -1).sum()
    print(f"    入格站点: {in_grid}/{len(stops_df)}")

    feat_df = compute_grid_bus_routes(stops_df, grid_meta)
    out_path = PROC / "bus_route_features_nyc.csv"
    feat_df.to_csv(out_path)

    n_nonzero = (feat_df["num_bus_routes"] > 0).sum()
    print(f"    覆盖网格: {n_nonzero}/{len(feat_df)}")
    if len(feat_df) > 0:
        print(f"    均值={feat_df['num_bus_routes'].mean():.1f}, "
              f"最大={feat_df['num_bus_routes'].max()}")
    print(f"    → {out_path}")
    return feat_df


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  公交线路特征提取")
    print("=" * 55)

    bj_feat  = build_bj_bus_features()
    nyc_feat = build_nyc_bus_features()

    print("\n  完成。输出文件：")
    for city, feat in [("bj", bj_feat), ("nyc", nyc_feat)]:
        p = PROC / f"bus_route_features_{city}.csv"
        print(f"    {p}  shape={feat.shape}")


if __name__ == "__main__":
    main()
