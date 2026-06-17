"""
公共交通特征提取
从 OSM PBF 提取地铁站、地铁入口和公交站，聚合到网格

输出：
  data/processed/transit_features_bj.csv   — (1024, 6)
  data/processed/transit_features_nyc.csv  — (75,   6)

特征列：
  subway_station_count   : 格内地铁站数量  (station=subway)
  subway_station_density : 地铁站数 / 格面积 (个/km²)
  subway_entrance_count  : 格内地铁入口数量 (railway=subway_entrance)
  subway_entrance_density: 地铁入口数 / 格面积 (个/km²)
  bus_stop_count         : 格内公交站数量  (highway=bus_stop)
  bus_stop_density       : 公交站数 / 格面积 (个/km²)
"""

import numpy as np
import pandas as pd
import osmium
from pathlib import Path

OUT = Path("data/processed")

CITIES = [
    ("bj",  "data/raw/osm/beijing-latest.osm.pbf",
             (116.25, 39.75, 116.75, 40.25),
             "data/processed/grid_meta_bj.csv",
             "transit_features_bj.csv"),
    ("nyc", "data/raw/osm/nyc-latest.osm.pbf",
             (-74.0166, 40.6996, -73.9004, 40.9196),
             "data/processed/grid_meta_nyc.csv",
             "transit_features_nyc.csv"),
]


class TransitExtractor(osmium.SimpleHandler):
    """提取 bbox 内的地铁站和地铁入口节点。"""
    def __init__(self, bbox):
        super().__init__()
        self.lon_min, self.lat_min, self.lon_max, self.lat_max = bbox
        self.stations  = []   # (lon, lat)  station=subway
        self.entrances = []   # (lon, lat)  railway=subway_entrance
        self.bus_stops = []   # (lon, lat)  highway=bus_stop

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        if not (self.lon_min <= lon <= self.lon_max
                and self.lat_min <= lat <= self.lat_max):
            return
        tags = dict(n.tags)
        if tags.get("station") == "subway":
            self.stations.append((lon, lat))
        elif tags.get("railway") == "subway_entrance":
            self.entrances.append((lon, lat))
        elif tags.get("highway") == "bus_stop":
            self.bus_stops.append((lon, lat))


def assign_to_grid(points, grid_meta):
    """将点列表分配到网格，返回每格计数 Series（indexed by grid_id）。"""
    if not points:
        return pd.Series(0, index=grid_meta["grid_id"])

    H = grid_meta["row"].max() + 1
    W = grid_meta["col"].max() + 1
    grid_lon_min = grid_meta["lon_min"].min()
    grid_lat_max = grid_meta["lat_max"].max()
    cell_lon = (grid_meta["lon_max"] - grid_meta["lon_min"]).iloc[0]
    cell_lat = (grid_meta["lat_max"] - grid_meta["lat_min"]).iloc[0]

    lons = np.array([p[0] for p in points])
    lats = np.array([p[1] for p in points])

    cols = ((lons - grid_lon_min) / cell_lon).astype(int)
    rows = ((grid_lat_max - lats) / cell_lat).astype(int)

    valid = (cols >= 0) & (cols < W) & (rows >= 0) & (rows < H)
    grid_ids = rows[valid] * W + cols[valid]

    counts = pd.Series(grid_ids).value_counts()
    return counts.reindex(grid_meta["grid_id"], fill_value=0)


def process_city(city, pbf_path, bbox, grid_csv, out_name):
    print(f"\n  [{city.upper()}]")
    grid_meta = pd.read_csv(grid_csv)

    print(f"    提取地铁数据 ({Path(pbf_path).name})...")
    ext = TransitExtractor(bbox)
    try:
        ext.apply_file(pbf_path, locations=True)
    except RuntimeError as e:
        print(f"    [!] OSM 解析警告: {e}")

    print(f"    地铁站 (station=subway): {len(ext.stations)}")
    print(f"    地铁入口 (subway_entrance): {len(ext.entrances)}")
    print(f"    公交站 (highway=bus_stop): {len(ext.bus_stops)}")

    area = grid_meta.set_index("grid_id")["area_km2"]

    st_count  = assign_to_grid(ext.stations,  grid_meta).rename("subway_station_count")
    ent_count = assign_to_grid(ext.entrances, grid_meta).rename("subway_entrance_count")
    bus_count = assign_to_grid(ext.bus_stops, grid_meta).rename("bus_stop_count")

    feat = pd.DataFrame({
        "subway_station_count"   : st_count,
        "subway_station_density" : st_count  / area,
        "subway_entrance_count"  : ent_count,
        "subway_entrance_density": ent_count / area,
        "bus_stop_count"         : bus_count,
        "bus_stop_density"       : bus_count / area,
    })

    out = OUT / out_name
    feat.to_csv(out)
    print(f"    已保存: {out_name}  shape={feat.shape}")
    print(f"    格内有地铁站的格数: {(feat.subway_station_count > 0).sum()} / {len(feat)}")
    print(f"    格内有地铁入口的格数: {(feat.subway_entrance_count > 0).sum()} / {len(feat)}")
    print(f"    格内有公交站的格数: {(feat.bus_stop_count > 0).sum()} / {len(feat)}")
    print(f"    公交站密度均值: {feat.bus_stop_density.mean():.3f} 个/km²")
    return feat


def main():
    print("=" * 55)
    print("  公共交通特征提取（地铁站 + 地铁入口）")
    print("=" * 55)
    for args in CITIES:
        process_city(*args)
    print("\n  完成。")


if __name__ == "__main__":
    main()
