"""
语义特征工程（POI）
从 OSM pbf 提取 POI，聚合到网格，计算各类 POI 数量和密度

12 类 POI（与 static_features_spec.md 对齐）:
  food, shopping, entertainment, office, residential,
  education, healthcare, transport, government, tourism, sports, religious

输出：
  data/processed/poi_features_bj.csv   — (1024, 24)
  data/processed/poi_features_nyc.csv  — (75,   24)
  每类 2 列: {cat}_count, {cat}_density (个/km²)
"""

import numpy as np
import pandas as pd
import osmium
from pathlib import Path

OUT = Path("data/processed")

# POI 分类规则（OSM tag → 类别）
POI_RULES = [
    ("food",          [("amenity", {"restaurant","cafe","fast_food","bar","pub",
                                    "food_court","ice_cream","biergarten"}),
                       ("shop",    {"bakery","butcher","deli","seafood"})]),
    ("shopping",      [("shop",    {"supermarket","mall","department_store","clothes",
                                    "shoes","electronics","furniture","convenience",
                                    "hardware","books","florist","jewelry","gift",
                                    "sports","toy","optician","mobile_phone"})]),
    ("entertainment", [("leisure", {"cinema","theatre","nightclub","arcade",
                                    "amusement_arcade","escape_game","bowling_alley"}),
                       ("amenity", {"cinema","theatre","nightclub","arts_centre",
                                    "community_centre","events_venue"})]),
    ("office",        [("office",  None),   # 任意 office=* 均计入
                       ("amenity", {"bank","atm","bureau_de_change","post_office",
                                    "coworking_space","conference_centre"})]),
    ("residential",   [("landuse", {"residential","apartments"}),
                       ("building",{"apartments","residential","house","detached",
                                    "semidetached_house","terrace"})]),
    ("education",     [("amenity", {"school","university","college","kindergarten",
                                    "language_school","library","research_institute",
                                    "driving_school"})]),
    ("healthcare",    [("amenity", {"hospital","clinic","pharmacy","dentist",
                                    "doctors","veterinary","nursing_home",
                                    "social_facility"})]),
    ("government",    [("amenity", {"townhall","courthouse","police","fire_station",
                                    "embassy","immigration","prison","public_bath"}),
                       ("office",  {"government","administrative"})]),
    ("tourism",       [("tourism", None),   # 任意 tourism=*
                       ("amenity", {"place_of_interest"})]),
    ("sports",        [("leisure", {"sports_centre","stadium","swimming_pool",
                                    "fitness_centre","pitch","track","gym"}),
                       ("sport",   None)]),
    ("religious",     [("amenity", {"place_of_worship"}),
                       ("religion",None)]),
]

CATEGORIES = [r[0] for r in POI_RULES]


def build_tag_lookup() -> dict:
    """构建 (key, value) → category 映射（value=None 表示任意值均匹配）。"""
    lookup = {}  # (key, value_or_None) → cat
    for cat, rules in POI_RULES:
        for key, values in rules:
            if values is None:
                lookup[(key, None)] = cat   # wildcard
            else:
                for v in values:
                    lookup[(key, v)] = cat
    return lookup


TAG_LOOKUP = build_tag_lookup()
WILDCARD_KEYS = {k for (k, v) in TAG_LOOKUP if v is None}


def classify_tags(tags: dict) -> str | None:
    """将 OSM 节点的 tag 映射到 POI 类别，返回第一个匹配。"""
    for key, val in tags.items():
        # 精确匹配
        if (key, val) in TAG_LOOKUP:
            return TAG_LOOKUP[(key, val)]
        # 通配符（只要 key 存在即匹配）
        if key in WILDCARD_KEYS and (key, None) in TAG_LOOKUP:
            return TAG_LOOKUP[(key, None)]
    return None


class POIExtractor(osmium.SimpleHandler):
    """从 OSM 节点中提取 POI，只保留落在 bbox 内的点。"""
    def __init__(self, bbox):
        super().__init__()
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = bbox
        self.pois = []   # list of (lon, lat, category)

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        if not (self.lon_min <= lon <= self.lon_max
                and self.lat_min <= lat <= self.lat_max):
            return
        tags = dict(n.tags)
        cat  = classify_tags(tags)
        if cat:
            self.pois.append((lon, lat, cat))


def assign_to_grid(pois: list, grid_meta: pd.DataFrame) -> pd.DataFrame:
    """将 POI 列表分配到网格，返回 (grid_id → {cat: count}) 的 DataFrame。"""
    if not pois:
        return pd.DataFrame(0, index=grid_meta["grid_id"], columns=CATEGORIES)

    poi_df = pd.DataFrame(pois, columns=["lon", "lat", "category"])

    # 向量化分配：计算每个 POI 所在的 row, col
    lon_min_g = grid_meta["lon_min"].values
    lon_max_g = grid_meta["lon_max"].values
    lat_min_g = grid_meta["lat_min"].values
    lat_max_g = grid_meta["lat_max"].values

    # 从 grid_meta 推算 H, W
    H = grid_meta["row"].max() + 1
    W = grid_meta["col"].max() + 1
    grid_lon_min = lon_min_g.min()
    grid_lat_max = lat_max_g.max()
    cell_lon = (lon_max_g - lon_min_g)[0]
    cell_lat = (lat_max_g - lat_min_g)[0]

    poi_col = ((poi_df["lon"].values - grid_lon_min) / cell_lon).astype(int)
    poi_row = ((grid_lat_max - poi_df["lat"].values) / cell_lat).astype(int)

    # 过滤越界
    valid = ((poi_col >= 0) & (poi_col < W) &
             (poi_row >= 0) & (poi_row < H))
    poi_df = poi_df[valid].copy()
    poi_df["row"] = poi_row[valid]
    poi_df["col"] = poi_col[valid]
    poi_df["grid_id"] = poi_df["row"] * W + poi_df["col"]

    counts = (poi_df.groupby(["grid_id", "category"])
              .size().unstack(fill_value=0))
    # 补全所有类别
    for cat in CATEGORIES:
        if cat not in counts.columns:
            counts[cat] = 0

    # reindex 到全部网格 ID
    counts = counts.reindex(grid_meta["grid_id"], fill_value=0)[CATEGORIES]
    return counts


def build_poi_features(grid_meta: pd.DataFrame,
                       pbf_path: Path,
                       bbox: tuple,
                       label: str) -> pd.DataFrame:
    print(f"    提取 POI ({pbf_path.name})...")
    extractor = POIExtractor(bbox)
    try:
        extractor.apply_file(str(pbf_path), locations=True)
    except RuntimeError as e:
        print(f"    [!] OSM 解析警告: {e}（使用已读取部分）")
    pois = extractor.pois
    print(f"    提取 POI 数: {len(pois)}")

    if pois:
        cat_counts = pd.Series([p[2] for p in pois]).value_counts()
        print(f"    类别分布: {cat_counts.to_dict()}")

    counts = assign_to_grid(pois, grid_meta)

    # 计算密度（个/km²）
    area = grid_meta.set_index("grid_id")["area_km2"]
    density = counts.div(area, axis=0)

    # 合并 count + density 列
    counts.columns  = [f"{c}_count"   for c in counts.columns]
    density.columns = [f"{c}_density" for c in density.columns]
    feat = pd.concat([counts, density], axis=1)

    # 按 count → density 交叉排列（更直观）
    ordered_cols = []
    for cat in CATEGORIES:
        ordered_cols += [f"{cat}_count", f"{cat}_density"]
    feat = feat[ordered_cols]

    print(f"    shape={feat.shape}")
    print(f"    每格平均 POI 总数: {counts.sum(axis=1).mean():.1f}")
    return feat


def main():
    print("=" * 55)
    print("  POI 语义特征工程")
    print("=" * 55)

    for city, grid_csv, pbf_path, bbox, out_name in [
        ("北京", "data/processed/grid_meta_bj.csv",
         Path("data/raw/osm/beijing-latest.osm.pbf"),
         (116.25, 39.75, 116.75, 40.25),
         "poi_features_bj.csv"),
        ("纽约", "data/processed/grid_meta_nyc.csv",
         Path("data/raw/osm/nyc-latest.osm.pbf"),
         (-74.0166, 40.6996, -73.9004, 40.9196),
         "poi_features_nyc.csv"),
    ]:
        print(f"\n  [{city}]")
        grid_meta = pd.read_csv(grid_csv)
        feat = build_poi_features(grid_meta, pbf_path, bbox, city)
        feat.to_csv(OUT / out_name)
        print(f"  已保存: {out_name}  shape={feat.shape}")

    print("\n  完成。")


if __name__ == "__main__":
    main()
