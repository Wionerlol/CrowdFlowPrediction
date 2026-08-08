"""Build a lightweight, georeferenced Beijing Landsat basemap for Week6 UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds


DEFAULT_SOURCE = (
    "data/raw/Landsat8_beijing/"
    "LC08_L1TP_123032_20150416_20200909_02_T1_refl.tif"
)
DEFAULT_OUTPUT = "src/visualization/assets/beijing_landsat_20150416.webp"
BEIJING_BOUNDS = (116.25, 39.75, 116.75, 40.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=88)
    return parser.parse_args()


def contrast_stretch(rgb: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    """Apply a robust per-band stretch while preserving nodata pixels."""
    output = np.zeros_like(rgb, dtype=np.uint8)
    percentiles: list[list[float]] = []
    valid_mask = np.any(rgb > 0, axis=0)
    for band_index in range(3):
        band = rgb[band_index].astype(np.float32)
        valid = band[valid_mask]
        low, high = np.percentile(valid, [2.0, 98.0])
        scaled = np.clip((band - low) / max(high - low, 1.0), 0.0, 1.0)
        output[band_index] = np.round(np.power(scaled, 0.92) * 255.0).astype(np.uint8)
        percentiles.append([round(float(low), 3), round(float(high), 3)])
    output[:, ~valid_mask] = 0
    return output, percentiles


def build_basemap(source_path: Path, output_path: Path, *, width: int, quality: int) -> dict:
    if width < 320:
        raise ValueError("width must be at least 320 pixels")
    west, south, east, north = BEIJING_BOUNDS
    with rasterio.open(source_path) as source:
        if source.count < 3:
            raise ValueError(f"expected at least three RGB bands in {source_path}")
        source_bounds = transform_bounds(
            CRS.from_epsg(4326), source.crs, west, south, east, north, densify_pts=21
        )
        source_aspect = (source_bounds[3] - source_bounds[1]) / (
            source_bounds[2] - source_bounds[0]
        )
        height = max(320, round(width * source_aspect))
        destination = np.zeros((3, height, width), dtype=np.uint8)
        destination_transform = from_bounds(west, south, east, north, width, height)
        for band_index in range(1, 4):
            reproject(
                source=rasterio.band(source, band_index),
                destination=destination[band_index - 1],
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=destination_transform,
                dst_crs=CRS.from_epsg(4326),
                dst_nodata=0,
                resampling=Resampling.bilinear,
            )
        source_crs = str(source.crs)

    stretched, stretch_percentiles = contrast_stretch(destination)
    image = Image.fromarray(np.moveaxis(stretched, 0, -1), mode="RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="WEBP", quality=quality, method=6)

    metadata = {
        "source": str(source_path),
        "product": "Landsat 8 Collection 2 Level-1 true-colour browse composite",
        "scene_id": "LC08_L1TP_123032_20150416_20200909_02_T1",
        "acquired_on": "2015-04-16",
        "native_resolution_m": 30,
        "source_crs": source_crs,
        "output_crs": "EPSG:4326",
        "bounds": {"west": west, "south": south, "east": east, "north": north},
        "size": {"width": width, "height": height},
        "stretch_percentiles": stretch_percentiles,
        "output": str(output_path),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = build_basemap(
        Path(args.source), Path(args.output), width=args.width, quality=args.quality
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
