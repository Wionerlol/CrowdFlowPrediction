"""Build the display-only TaxiBJ grid and Beijing ring-road reference layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import osmium
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.geo_calibration import (  # noqa: E402
    DEFAULT_COL_OFFSET,
    DEFAULT_ROW_OFFSET,
    build_display_grid_frame,
)


DEFAULT_SOURCE_GRID = "data/processed/grid_meta_bj.csv"
DEFAULT_DISPLAY_GRID = "data/processed/grid_display_meta_bj.csv"
DEFAULT_OSM = "data/raw/osm/beijing-latest.osm.pbf"
DEFAULT_RING_ROADS = "src/visualization/assets/beijing_ring_roads.geojson"
RING_RELATIONS = {
    295094: "二环路",
    295097: "北京三环路",
    18649348: "北京四环路",
    295957: "五环路",
}


class _RingRelationCollector(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.ways: dict[int, tuple[int, str]] = {}

    def relation(self, relation: osmium.osm.Relation) -> None:
        relation_id = int(relation.id)
        if relation_id not in RING_RELATIONS:
            return
        name = RING_RELATIONS[relation_id]
        for member in relation.members:
            if member.type == "w":
                self.ways[int(member.ref)] = (relation_id, name)


class _RingGeometryCollector(osmium.SimpleHandler):
    def __init__(self, ways: dict[int, tuple[int, str]]) -> None:
        super().__init__()
        self.ways = ways
        self.features: list[dict[str, Any]] = []

    def way(self, way: osmium.osm.Way) -> None:
        way_id = int(way.id)
        relation = self.ways.get(way_id)
        if relation is None:
            return
        coordinates = [
            [float(node.lon), float(node.lat)]
            for node in way.nodes
            if node.location.valid()
        ]
        if len(coordinates) < 2:
            return
        relation_id, name = relation
        self.features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "relation_id": relation_id,
                    "way_id": way_id,
                },
                "geometry": {"type": "LineString", "coordinates": coordinates},
            }
        )


def build_display_grid(source_path: Path, output_path: Path) -> None:
    source = pd.read_csv(source_path)
    display = build_display_grid_frame(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display.to_csv(output_path, index=False)


def build_ring_roads(source_path: Path, output_path: Path) -> int:
    relations = _RingRelationCollector()
    relations.apply_file(str(source_path), locations=False)
    missing = set(RING_RELATIONS).difference(
        relation_id for relation_id, _ in relations.ways.values()
    )
    if missing:
        raise RuntimeError(f"OSM extract is missing ring-road relations: {sorted(missing)}")

    geometries = _RingGeometryCollector(relations.ways)
    geometries.apply_file(str(source_path), locations=True)
    collection = {
        "type": "FeatureCollection",
        "properties": {
            "source": str(source_path),
            "purpose": "display-only spatial reference",
        },
        "features": geometries.features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(collection, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return len(geometries.features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-grid", default=DEFAULT_SOURCE_GRID)
    parser.add_argument("--display-grid", default=DEFAULT_DISPLAY_GRID)
    parser.add_argument("--osm", default=DEFAULT_OSM)
    parser.add_argument("--ring-roads", default=DEFAULT_RING_ROADS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_display_grid(Path(args.source_grid), Path(args.display_grid))
    feature_count = build_ring_roads(Path(args.osm), Path(args.ring_roads))
    print(
        json.dumps(
            {
                "display_grid": args.display_grid,
                "row_offset": DEFAULT_ROW_OFFSET,
                "col_offset": DEFAULT_COL_OFFSET,
                "ring_roads": args.ring_roads,
                "ring_segment_count": feature_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
