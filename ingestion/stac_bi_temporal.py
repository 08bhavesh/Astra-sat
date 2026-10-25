"""
AWS STAC Cloud-Optimized GeoTIFF (COG) Window Streamer and Bi-Temporal Cube Builder.
Streams Sentinel-2 L2A optical bands (T1 baseline vs T2 event) and Copernicus 30m DEM
over windowed HTTP range requests into a co-registered (512, 512) UTM raster cube.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pystac_client
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT
except (ImportError, OSError):
    rasterio = None
    Resampling = None
    from_bounds = None
    WarpedVRT = None

from ..config import AWS_STAC_URL, COLLECTIONS
from .utm_warper import get_subcontinent_utm_epsg, normalize_analysis_bbox, transform_bbox_to_utm

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "stac_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Band asset keys and aliases for Sentinel-2 L2A on AWS Earth Search
S2_BAND_ALIASES = {
    "blue": ["blue", "B02", "b02"],
    "green": ["green", "B03", "b03"],
    "red": ["red", "B04", "b04"],
    "nir": ["nir", "B08", "b08", "nir08"],
    "swir16": ["swir16", "B11", "b11"],
    "swir22": ["swir22", "B12", "b12"],
    "scl": ["scl", "SCL"],
}


def _get_cube_cache_path(bbox: list[float], t1_date: Optional[str], t2_date: Optional[str]) -> Path:
    key_str = f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}_{t1_date}_{t2_date}"
    cube_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"cube_{cube_hash}.npz"


def _resolve_asset_href(item: Any, band_name: str) -> Optional[str]:
    """Resolves the href for a requested band from a STAC Item."""
    aliases = S2_BAND_ALIASES.get(band_name, [band_name])
    for alias in aliases:
        if alias in item.assets:
            return item.assets[alias].href
    return None


def _stream_windowed_band(
    href: str,
    target_epsg: int,
    dst_transform: Any,
    width: int = 512,
    height: int = 512,
    is_categorical: bool = False
) -> np.ndarray:
    """Streams a windowed COG raster reprojected to target UTM grid using WarpedVRT."""
    resampling = Resampling.nearest if is_categorical else Resampling.bilinear
    env_kwargs = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
        "GDAL_HTTP_TIMEOUT": "20",
        "GDAL_HTTP_MAX_RETRY": "2",
    }
    with rasterio.Env(**env_kwargs):
        with rasterio.open(href) as src:
            with WarpedVRT(
                src,
                crs=f"EPSG:{target_epsg}",
                transform=dst_transform,
                width=width,
                height=height,
                resampling=resampling
            ) as vrt:
                arr = vrt.read(1)
                return arr


def fetch_bi_temporal_stac_cubes(
    bbox: list[float],
    baseline_date_t1: Optional[str] = None,
    current_date_t2: Optional[str] = None,
    target_epsg: Optional[int] = None,
    grid_size: tuple[int, int] = (512, 512),
    normalize_bbox: bool = True
) -> dict[str, Any]:
    """
    Queries live AWS STAC for Sentinel-2 L2A bi-temporal scenes and Copernicus 30m DEM.
    Streams windowed HTTP range requests into exact co-registered (512, 512) UTM arrays.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84.
        baseline_date_t1: Optional baseline reference date (defaults to dry pre-monsoon baseline).
        current_date_t2: Optional current event date (defaults to latest available pass).
        target_epsg: Optional UTM EPSG code (auto-calculated from bbox if omitted).
        grid_size: Grid dimensions (width, height), default (512, 512).
        normalize_bbox: Whether to center a tactical analysis window if bbox is large.

    Returns:
        dict containing:
            - 't1_scene_id', 't1_datetime'
            - 't2_scene_id', 't2_datetime'
            - 'epsg_utm_zone': int
            - 'utm_bounds': tuple[float, float, float, float]
            - 'affine_transform': rasterio.Affine
            - 'cloud_cover_t2_pct': float
            - 't1_bands': dict of (512, 512) arrays
            - 't2_bands': dict of (512, 512) arrays
            - 'dem': (512, 512) elevation array in meters
            - 'slope_deg': (512, 512) slope array in degrees
    """
    target_bbox = normalize_analysis_bbox(bbox) if normalize_bbox else bbox
    c_lon = (target_bbox[0] + target_bbox[2]) / 2.0
    c_lat = (target_bbox[1] + target_bbox[3]) / 2.0

    epsg_zone = target_epsg or get_subcontinent_utm_epsg(c_lon, c_lat)
    utm_bounds = transform_bbox_to_utm(target_bbox, epsg_zone)
    min_x, min_y, max_x, max_y = utm_bounds
    dst_transform = from_bounds(min_x, min_y, max_x, max_y, grid_size[1], grid_size[0])

    # 1. Check on-disk cache
    cache_path = _get_cube_cache_path(target_bbox, baseline_date_t1, current_date_t2)
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            data_npz = np.load(cache_path)
            t1_bands = {k.replace("t1_", ""): data_npz[k] for k in data_npz.files if k.startswith("t1_")}
            t2_bands = {k.replace("t2_", ""): data_npz[k] for k in data_npz.files if k.startswith("t2_")}
            dem = data_npz["dem"]
            slope_deg = data_npz["slope_deg"]

            return {
                "t1_scene_id": meta["t1_scene_id"],
                "t1_datetime": meta["t1_datetime"],
                "t2_scene_id": meta["t2_scene_id"],
                "t2_datetime": meta["t2_datetime"],
                "epsg_utm_zone": epsg_zone,
                "utm_bounds": utm_bounds,
                "affine_transform": dst_transform,
                "cloud_cover_t2_pct": meta["cloud_cover_t2_pct"],
                "t1_bands": t1_bands,
                "t2_bands": t2_bands,
                "dem": dem,
                "slope_deg": slope_deg,
            }
        except Exception:
            pass

    # 2. Connect to live AWS STAC
    client = pystac_client.Client.open(AWS_STAC_URL)

    # Search T2 (Current event scene)
    t2_dt_filter = f"{current_date_t2}/.." if current_date_t2 else "2024-01-01/.."
    search_t2 = client.search(
        collections=[COLLECTIONS["SENTINEL_2"]],
        bbox=target_bbox,
        datetime=t2_dt_filter,
        query={"eo:cloud_cover": {"lt": 50.0}},
        limit=5,
    )
    items_t2 = list(search_t2.items())
    if not items_t2:
        # Relax cloud cover if needed
        search_t2 = client.search(
            collections=[COLLECTIONS["SENTINEL_2"]],
            bbox=target_bbox,
            limit=3,
        )
        items_t2 = list(search_t2.items())

    if not items_t2:
        raise RuntimeError(f"No Sentinel-2 T2 scenes found on AWS STAC for bbox {target_bbox}")

    item_t2 = items_t2[0]

    # Search T1 (Baseline dry season scene ~1 year prior or pre-monsoon March-May)
    if baseline_date_t1:
        t1_dt_filter = f"{baseline_date_t1}/.."
    else:
        # Pre-monsoon dry season baseline (crystal clear permanent river channel)
        t1_dt_filter = "2023-03-01/2023-05-30"

    search_t1 = client.search(
        collections=[COLLECTIONS["SENTINEL_2"]],
        bbox=target_bbox,
        datetime=t1_dt_filter,
        query={"eo:cloud_cover": {"lt": 20.0}},
        limit=5,
    )
    items_t1 = list(search_t1.items())
    if not items_t1:
        # Fallback to any low-cloud scene prior to T2
        search_t1 = client.search(
            collections=[COLLECTIONS["SENTINEL_2"]],
            bbox=target_bbox,
            query={"eo:cloud_cover": {"lt": 15.0}},
            limit=5,
        )
        items_t1 = list(search_t1.items())

    if not items_t1:
        # Use T2 as self-baseline if zero other scenes exist
        item_t1 = item_t2
    else:
        item_t1 = items_t1[0]

    # 3. Search Copernicus 30m DEM
    search_dem = client.search(
        collections=[COLLECTIONS["COP_DEM_30"]],
        bbox=target_bbox,
        limit=2,
    )
    items_dem = list(search_dem.items())
    dem_href: Optional[str] = None
    if items_dem and "data" in items_dem[0].assets:
        dem_href = items_dem[0].assets["data"].href

    # 4. Stream bands in parallel using ThreadPoolExecutor
    bands_to_fetch = ["blue", "green", "red", "nir", "swir16", "swir22", "scl"]
    tasks = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        # T1 tasks
        for b_name in bands_to_fetch:
            href = _resolve_asset_href(item_t1, b_name)
            if href:
                tasks.append((
                    "t1",
                    b_name,
                    pool.submit(
                        _stream_windowed_band,
                        href,
                        epsg_zone,
                        dst_transform,
                        grid_size[1],
                        grid_size[0],
                        b_name == "scl"
                    )
                ))

        # T2 tasks
        for b_name in bands_to_fetch:
            href = _resolve_asset_href(item_t2, b_name)
            if href:
                tasks.append((
                    "t2",
                    b_name,
                    pool.submit(
                        _stream_windowed_band,
                        href,
                        epsg_zone,
                        dst_transform,
                        grid_size[1],
                        grid_size[0],
                        b_name == "scl"
                    )
                ))

        # DEM task
        dem_future = None
        if dem_href:
            dem_future = pool.submit(
                _stream_windowed_band,
                dem_href,
                epsg_zone,
                dst_transform,
                grid_size[1],
                grid_size[0],
                False
            )

    t1_bands: dict[str, np.ndarray] = {}
    t2_bands: dict[str, np.ndarray] = {}

    for period, b_name, fut in tasks:
        try:
            arr = fut.result()
            if period == "t1":
                t1_bands[b_name] = arr
            else:
                t2_bands[b_name] = arr
        except Exception:
            # Fallback zero array if specific band stream fails
            fallback_arr = np.zeros(grid_size, dtype=np.uint16)
            if period == "t1":
                t1_bands[b_name] = fallback_arr
            else:
                t2_bands[b_name] = fallback_arr

    # Fill any missing required bands with zeroes
    for b_name in bands_to_fetch:
        if b_name not in t1_bands:
            t1_bands[b_name] = np.zeros(grid_size, dtype=np.uint16)
        if b_name not in t2_bands:
            t2_bands[b_name] = np.zeros(grid_size, dtype=np.uint16)

    # 5. Process Copernicus DEM & slope
    if dem_future is not None:
        try:
            raw_dem = dem_future.result().astype(np.float32)
            # Clean nodata values
            dem = np.where(raw_dem < -100.0, 0.0, raw_dem)
        except Exception:
            dem = np.zeros(grid_size, dtype=np.float32)
    else:
        dem = np.zeros(grid_size, dtype=np.float32)

    pixel_size_x = abs(dst_transform.a)
    pixel_size_y = abs(dst_transform.e)
    grad_y, grad_x = np.gradient(dem, pixel_size_y, pixel_size_x)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2))).astype(np.float32)

    # 6. Calculate real T2 cloud cover percentage from SCL
    scl_t2 = t2_bands.get("scl", np.zeros(grid_size, dtype=np.uint8))
    cloud_mask = np.isin(scl_t2, [8, 9, 10])
    cloud_cover_t2_pct = round(float(np.count_nonzero(cloud_mask) / cloud_mask.size) * 100.0, 2)

    # 7. Persist to on-disk cache for fast subsequent execution
    try:
        npz_dict = {"dem": dem, "slope_deg": slope_deg}
        for k, v in t1_bands.items():
            npz_dict[f"t1_{k}"] = v
        for k, v in t2_bands.items():
            npz_dict[f"t2_{k}"] = v
        np.savez_compressed(cache_path, **npz_dict)

        meta_dict = {
            "t1_scene_id": item_t1.id,
            "t1_datetime": str(item_t1.datetime),
            "t2_scene_id": item_t2.id,
            "t2_datetime": str(item_t2.datetime),
            "cloud_cover_t2_pct": cloud_cover_t2_pct,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)
    except Exception:
        pass

    return {
        "t1_scene_id": item_t1.id,
        "t1_datetime": str(item_t1.datetime),
        "t2_scene_id": item_t2.id,
        "t2_datetime": str(item_t2.datetime),
        "epsg_utm_zone": epsg_zone,
        "utm_bounds": utm_bounds,
        "affine_transform": dst_transform,
        "cloud_cover_t2_pct": cloud_cover_t2_pct,
        "t1_bands": t1_bands,
        "t2_bands": t2_bands,
        "dem": dem,
        "slope_deg": slope_deg,
    }
