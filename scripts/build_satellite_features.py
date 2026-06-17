"""
Landsat-8 遥感特征工程
输入：data/raw/Landsat8_beijing/  + data/raw/Landsat8_NewYorkCity/
      （L1TP 级 DN，使用 MTL.txt 校正参数）

处理流程：
  1. DN → TOA 反射率（REFLECTANCE_MULT/ADD + 太阳高度角修正）
  2. 裁剪到研究区域 bbox（重投影至 WGS84）
  3. 计算遥感指数 NDVI / NDBI / MNDWI
  4. 聚合到网格（每格像素均值/标准差）

输出：
  data/processed/satellite_features_bj.csv   — (1024, 5)
  data/processed/satellite_features_nyc.csv  — (75,   5)

特征列：ndvi, ndbi, mndwi, b4_mean, b4_std
"""

import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from pathlib import Path

OUT = Path("data/processed")

CITIES = {
    "bj": {
        "raw_dir"  : Path("data/raw/Landsat8_beijing"),
        "prefix"   : "LC08_L1TP_123032_20150416_20200909_02_T1",
        "bbox_wgs" : (116.25, 39.75, 116.75, 40.25),   # lon_min,lat_min,lon_max,lat_max
        "city_crs" : "EPSG:32650",
        "grid_csv" : "data/processed/grid_meta_bj.csv",
        "out_name" : "satellite_features_bj.csv",
    },
    "nyc": {
        "raw_dir"  : Path("data/raw/Landsat8_NewYorkCity"),
        "prefix"   : "LC08_L1TP_013032_20140410_20200911_02_T1",
        "bbox_wgs" : (-74.0166, 40.6996, -73.9004, 40.9196),
        "city_crs" : "EPSG:32618",
        "grid_csv" : "data/processed/grid_meta_nyc.csv",
        "out_name" : "satellite_features_nyc.csv",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MTL 校正参数
# ═══════════════════════════════════════════════════════════════════════════════

def read_mtl(mtl_path: Path) -> dict:
    params = {}
    with open(mtl_path) as f:
        for line in f:
            s = line.strip()
            for key in ["REFLECTANCE_MULT_BAND_3", "REFLECTANCE_MULT_BAND_4",
                        "REFLECTANCE_MULT_BAND_5", "REFLECTANCE_MULT_BAND_6",
                        "REFLECTANCE_ADD_BAND_3",  "REFLECTANCE_ADD_BAND_4",
                        "REFLECTANCE_ADD_BAND_5",  "REFLECTANCE_ADD_BAND_6",
                        "SUN_ELEVATION"]:
                if s.startswith(key + " ="):
                    params[key] = float(s.split("=")[1].strip())
    return params


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 读取 + 裁剪 + 重投影到 WGS84
# ═══════════════════════════════════════════════════════════════════════════════

def load_band_wgs84(tif_path: Path, bbox_wgs: tuple,
                    mult: float, add: float, sun_elev: float,
                    res: float = 0.00025) -> tuple:
    """
    读取一个波段，裁剪到 bbox_wgs，重投影至 WGS84，
    返回 (reflectance_2d: float32, transform)。
    nodata（DN=0）设为 NaN。
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs
    sin_elev = math.sin(math.radians(sun_elev))

    with rasterio.open(tif_path) as src:
        # bbox → 原始 CRS → 读取窗口
        src_bbox = transform_bounds(CRS.from_epsg(4326), src.crs,
                                    lon_min, lat_min, lon_max, lat_max)
        win = from_bounds(*src_bbox, transform=src.transform)
        dn  = src.read(1, window=win).astype(np.float32)
        win_tf = src.window_transform(win)
        src_crs = src.crs

    # DN=0 → NaN（nodata）
    nodata_mask = (dn == 0)
    dn[nodata_mask] = np.nan

    # TOA 反射率
    rho = (mult * dn + add) / sin_elev
    rho = np.clip(rho, 0.0, 1.0)

    # 重投影至 WGS84
    dst_crs = CRS.from_epsg(4326)
    dst_tf, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs,
        dn.shape[1], dn.shape[0],
        left=src_bbox[0], bottom=src_bbox[1],
        right=src_bbox[2], top=src_bbox[3],
        resolution=res,
    )
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        rho, dst,
        src_transform=win_tf, src_crs=src_crs,
        dst_transform=dst_tf, dst_crs=dst_crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return dst, dst_tf


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 遥感指数
# ═══════════════════════════════════════════════════════════════════════════════

def safe_norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b)，分母近 0 时返回 NaN。"""
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(denom) > 1e-6, (a - b) / denom, np.nan)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 聚合到网格
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_to_grid(raster: np.ndarray, transform,
                      grid_meta: pd.DataFrame) -> pd.Series:
    """
    将重投影后的 WGS84 栅格聚合到网格，返回每格均值 (NaN 填充 0)。
    """
    tf = transform
    # affine: x = col*tf.a + tf.c, y = row*tf.e + tf.f  (tf.e < 0)
    H, W_r = raster.shape
    results = {}

    for _, g in grid_meta.iterrows():
        lon_min, lon_max = g["lon_min"], g["lon_max"]
        lat_min, lat_max = g["lat_min"], g["lat_max"]

        # 格子 bbox → 像素范围
        col_l = int((lon_min - tf.c) / tf.a)
        col_r = int(math.ceil((lon_max - tf.c) / tf.a))
        row_t = int((lat_max - tf.f) / tf.e)
        row_b = int(math.ceil((lat_min - tf.f) / tf.e))

        col_l = max(0, col_l); col_r = min(W_r, col_r)
        row_t = max(0, row_t); row_b = min(H,   row_b)

        if row_t >= row_b or col_l >= col_r:
            results[g["grid_id"]] = np.nan
            continue

        patch = raster[row_t:row_b, col_l:col_r]
        valid = patch[np.isfinite(patch)]
        results[g["grid_id"]] = float(valid.mean()) if len(valid) > 0 else np.nan

    return pd.Series(results)


def aggregate_std(raster: np.ndarray, transform,
                  grid_meta: pd.DataFrame) -> pd.Series:
    tf = transform
    H, W_r = raster.shape
    results = {}
    for _, g in grid_meta.iterrows():
        col_l = int((g["lon_min"] - tf.c) / tf.a)
        col_r = int(math.ceil((g["lon_max"] - tf.c) / tf.a))
        row_t = int((g["lat_max"] - tf.f) / tf.e)
        row_b = int(math.ceil((g["lat_min"] - tf.f) / tf.e))
        col_l = max(0, col_l); col_r = min(W_r, col_r)
        row_t = max(0, row_t); row_b = min(H,   row_b)
        if row_t >= row_b or col_l >= col_r:
            results[g["grid_id"]] = np.nan
            continue
        patch = raster[row_t:row_b, col_l:col_r]
        valid = patch[np.isfinite(patch)]
        results[g["grid_id"]] = float(valid.std()) if len(valid) > 1 else 0.0
    return pd.Series(results)


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def process_city(city_key: str, cfg: dict):
    print(f"\n  [{city_key.upper()}]")
    raw_dir  = cfg["raw_dir"]
    prefix   = cfg["prefix"]
    bbox_wgs = cfg["bbox_wgs"]

    mtl  = read_mtl(raw_dir / f"{prefix}_MTL.txt")
    elev = mtl["SUN_ELEVATION"]
    print(f"  太阳高度角: {elev:.2f}°")

    def load(band_num):
        tif  = raw_dir / f"{prefix}_B{band_num}.TIF"
        mult = mtl[f"REFLECTANCE_MULT_BAND_{band_num}"]
        add  = mtl[f"REFLECTANCE_ADD_BAND_{band_num}"]
        print(f"  读取 B{band_num}: mult={mult:.2e}, add={add:.3f}")
        data, tf = load_band_wgs84(tif, bbox_wgs, mult, add, elev)
        valid_frac = np.isfinite(data).mean()
        print(f"    shape={data.shape}, 有效像素={valid_frac*100:.1f}%")
        return data, tf

    b3, tf = load(3)
    b4, _  = load(4)
    b5, _  = load(5)
    b6, _  = load(6)

    # 遥感指数
    ndvi  = safe_norm_diff(b5, b4)
    ndbi  = safe_norm_diff(b6, b5)
    mndwi = safe_norm_diff(b3, b6)
    print(f"  NDVI  均值={np.nanmean(ndvi):.3f}  [{np.nanmin(ndvi):.3f}, {np.nanmax(ndvi):.3f}]")
    print(f"  NDBI  均值={np.nanmean(ndbi):.3f}")
    print(f"  MNDWI 均值={np.nanmean(mndwi):.3f}")

    # 聚合到网格
    print(f"  聚合到网格...")
    grid_meta = pd.read_csv(cfg["grid_csv"])

    feat = pd.DataFrame({
        "ndvi"   : aggregate_to_grid(ndvi,  tf, grid_meta),
        "ndbi"   : aggregate_to_grid(ndbi,  tf, grid_meta),
        "mndwi"  : aggregate_to_grid(mndwi, tf, grid_meta),
        "b4_mean": aggregate_to_grid(b4,    tf, grid_meta),
        "b4_std" : aggregate_std(b4,        tf, grid_meta),
    })

    # 统计 NaN
    nan_count = feat.isna().sum().sum()
    if nan_count:
        print(f"  [!] NaN 格数: {nan_count}（填 0）")
        feat = feat.fillna(0.0)

    out = OUT / cfg["out_name"]
    feat.to_csv(out)
    print(f"  已保存: {cfg['out_name']}  shape={feat.shape}")

    print(f"  特征统计:")
    print(feat.describe().round(4).to_string())
    return feat


def main():
    print("=" * 55)
    print("  Landsat-8 遥感特征工程")
    print("=" * 55)
    for key, cfg in CITIES.items():
        process_city(key, cfg)
    print("\n  完成。")


if __name__ == "__main__":
    main()
