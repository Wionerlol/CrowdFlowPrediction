"""
空间特征工程
2.1 距离特征：每格到城市中心、交通枢纽、商业中心的距离
2.2 路网特征：从 OSM pbf 提取路网密度、道路等级分布

输出：
  data/processed/spatial_features_bj.csv   — (1024, F)
  data/processed/spatial_features_nyc.csv  — (75,   F)
"""

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import osmium
from shapely.geometry import LineString, box
from pathlib import Path

OUT = Path("data/processed")

# ── 地标坐标（WGS84, lon/lat）────────────────────────────────────────────────
LANDMARKS = {
    "bj": {
        "city_center": [(116.3974, 39.9087)],   # 天安门广场
        "transport": [
            (116.4274, 39.9022),   # 北京站
            (116.3221, 39.8942),   # 北京西站
            (116.3784, 39.8648),   # 北京南站
            (116.5883, 40.0799),   # 首都机场T3
        ],
        "commercial": [
            (116.4074, 39.9139),   # 王府井
            (116.4551, 39.9327),   # 三里屯
            (116.3106, 39.9840),   # 中关村
            (116.3664, 39.9133),   # 西单
        ],
    },
    "nyc": {
        "city_center": [(-73.9857, 40.7484)],   # 中城曼哈顿
        "transport": [
            (-73.9937, 40.7506),   # Penn Station
            (-73.9772, 40.7527),   # Grand Central
            (-73.7789, 40.6413),   # JFK Airport
            (-74.0099, 40.7143),   # WTC/Fulton St
        ],
        "commercial": [
            (-73.9855, 40.7580),   # Times Square
            (-74.0094, 40.7069),   # Wall Street
            (-73.9681, 40.7851),   # Columbus Circle
            (-73.9442, 40.7794),   # Upper East Side
        ],
    },
}

# 路网等级分组
ROAD_GROUPS = {
    "highway": ["motorway", "trunk", "primary", "secondary"],
    "local"  : ["tertiary", "unclassified", "residential"],
    "service": ["service"],
    "active" : ["footway", "path", "cycleway", "pedestrian", "steps"],
}
ALL_HIGHWAY_TYPES = [t for g in ROAD_GROUPS.values() for t in g]


# ═══════════════════════════════════════════════════════════════════════════════
# 距离特征
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def build_distance_features(grid_meta: pd.DataFrame,
                             landmarks: dict) -> pd.DataFrame:
    """计算每格到各类地标的距离特征。"""
    rows = []
    for _, g in grid_meta.iterrows():
        clon, clat = g["center_lon"], g["center_lat"]
        rec = {"grid_id": g["grid_id"]}

        # 到城市中心
        dists_center = [haversine_km(clon, clat, *lm)
                        for lm in landmarks["city_center"]]
        rec["dist_center_km"] = min(dists_center)

        # 到最近交通枢纽
        dists_hub = [haversine_km(clon, clat, *lm)
                     for lm in landmarks["transport"]]
        rec["dist_hub_min_km"] = min(dists_hub)
        for i, d in enumerate(dists_hub):
            rec[f"dist_hub{i+1}_km"] = d

        # 到最近商业中心
        dists_com = [haversine_km(clon, clat, *lm)
                     for lm in landmarks["commercial"]]
        rec["dist_commercial_min_km"] = min(dists_com)
        for i, d in enumerate(dists_com):
            rec[f"dist_com{i+1}_km"] = d

        rows.append(rec)
    return pd.DataFrame(rows).set_index("grid_id")


# ═══════════════════════════════════════════════════════════════════════════════
# 路网特征（OSM）
# ═══════════════════════════════════════════════════════════════════════════════

class WayExtractor(osmium.SimpleHandler):
    """从 OSM 提取 highway way 的节点坐标列表。"""
    def __init__(self, bbox, target_types):
        super().__init__()
        self.bbox    = bbox   # (lon_min, lat_min, lon_max, lat_max)
        self.target  = set(target_types)
        self.ways    = []     # list of (highway_type, [(lon, lat), ...])

    def way(self, w):
        tags = dict(w.tags)
        hw   = tags.get("highway")
        if hw not in self.target:
            return
        coords = []
        for n in w.nodes:
            if n.location.valid():
                coords.append((n.location.lon, n.location.lat))
        if len(coords) < 2:
            return
        # 快速 bbox 过滤（任一节点在范围内即保留）
        lo, la_min, la_max_b, la_max = self.bbox
        lon_min, lat_min, lon_max, lat_max = self.bbox
        if any(lon_min <= c[0] <= lon_max and lat_min <= c[1] <= lat_max
               for c in coords):
            self.ways.append((hw, coords))


def build_road_features(grid_meta: pd.DataFrame,
                        pbf_path: Path,
                        bbox: tuple,
                        city_crs: str) -> pd.DataFrame:
    """
    提取路网并计算每格路网密度和道路等级特征。
    city_crs: 用于长度计算的投影坐标系（米）
    """
    print(f"    提取 OSM 道路... ({pbf_path.name})")
    extractor = WayExtractor(bbox, ALL_HIGHWAY_TYPES)
    try:
        extractor.apply_file(str(pbf_path), locations=True)
    except RuntimeError as e:
        print(f"    [!] OSM 解析警告: {e}（使用已读取部分）")
    ways = extractor.ways
    print(f"    提取 way 数: {len(ways)}")

    if not ways:
        n = len(grid_meta)
        return pd.DataFrame(0.0, index=grid_meta["grid_id"],
                            columns=["road_density_km_km2",
                                     "highway_density", "local_density",
                                     "service_density", "active_density"])

    # 构建 GeoDataFrame
    geoms   = [LineString(coords) for _, coords in ways]
    hw_type = [hw for hw, _ in ways]
    ways_gdf = gpd.GeoDataFrame({"highway": hw_type, "geometry": geoms},
                                 crs="EPSG:4326")

    # 网格单元格多边形
    cells = []
    for _, g in grid_meta.iterrows():
        cells.append({
            "grid_id" : g["grid_id"],
            "area_km2": g["area_km2"],
            "geometry": box(g["lon_min"], g["lat_min"],
                            g["lon_max"], g["lat_max"]),
        })
    grid_gdf = gpd.GeoDataFrame(cells, crs="EPSG:4326")

    # 裁剪（intersection）
    print(f"    空间裁剪中...")
    inter = gpd.overlay(ways_gdf, grid_gdf, how="intersection", keep_geom_type=True)

    # 投影到米 → 计算长度
    inter_proj = inter.to_crs(city_crs)
    inter["length_km"] = inter_proj.geometry.length / 1000.0

    # 道路等级分组标签
    type2group = {t: g for g, ts in ROAD_GROUPS.items() for t in ts}
    inter["group"] = inter["highway"].map(type2group)

    # 每格汇总
    total = inter.groupby("grid_id")["length_km"].sum().rename("total_km")
    group_sum = (inter.groupby(["grid_id", "group"])["length_km"]
                 .sum().unstack(fill_value=0.0))
    # 确保所有 group 列都存在
    for g in ROAD_GROUPS:
        if g not in group_sum.columns:
            group_sum[g] = 0.0

    result = grid_meta.set_index("grid_id")[["area_km2"]].join(total, how="left")
    result["total_km"] = result["total_km"].fillna(0.0)
    result = result.join(group_sum, how="left").fillna(0.0)

    result["road_density_km_km2"] = result["total_km"] / result["area_km2"]
    for g in ROAD_GROUPS:
        result[f"{g}_density"] = result[g] / result["area_km2"]

    keep = (["road_density_km_km2"]
            + [f"{g}_density" for g in ROAD_GROUPS])
    return result[keep]


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def process_city(city_key, grid_csv, pbf_path, bbox, city_crs, landmarks, out_name):
    print(f"\n  [{city_key.upper()}]")
    grid_meta = pd.read_csv(grid_csv)

    # 2.1 距离特征
    print(f"  2.1 距离特征...")
    dist_df = build_distance_features(grid_meta, landmarks)
    print(f"      shape={dist_df.shape}, cols={dist_df.columns.tolist()[:4]}...")

    # 2.2 路网特征
    print(f"  2.2 路网特征（OSM）...")
    road_df = build_road_features(grid_meta, pbf_path, bbox, city_crs)
    print(f"      shape={road_df.shape}")
    print(f"      road_density 均值: {road_df['road_density_km_km2'].mean():.3f} km/km²")

    # 合并
    feat = dist_df.join(road_df)
    feat.to_csv(OUT / out_name)
    print(f"  已保存: {out_name}  shape={feat.shape}")
    return feat


def main():
    print("=" * 55)
    print("  空间特征工程（2.1 距离 + 2.2 路网）")
    print("=" * 55)

    process_city(
        city_key  = "bj",
        grid_csv  = "data/processed/grid_meta_bj.csv",
        pbf_path  = Path("data/raw/osm/beijing-latest.osm.pbf"),
        bbox      = (116.25, 39.75, 116.75, 40.25),
        city_crs  = "EPSG:32650",    # UTM Zone 50N
        landmarks = LANDMARKS["bj"],
        out_name  = "spatial_features_bj.csv",
    )

    process_city(
        city_key  = "nyc",
        grid_csv  = "data/processed/grid_meta_nyc.csv",
        pbf_path  = Path("data/raw/osm/nyc-latest.osm.pbf"),
        bbox      = (-74.0166, 40.6996, -73.9004, 40.9196),
        city_crs  = "EPSG:32618",    # UTM Zone 18N
        landmarks = LANDMARKS["nyc"],
        out_name  = "spatial_features_nyc.csv",
    )

    print("\n  完成。")


if __name__ == "__main__":
    main()
