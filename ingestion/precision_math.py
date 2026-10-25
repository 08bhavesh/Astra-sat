"""
Zero-Hallucination Bi-Temporal Anomaly Math and 3-Layer Precision Filter Engine.
Calculates spectral indices (NDWI, MNDWI, NDVI, NBR, BSI), enforces multi-layer physical guardrails
(SCL cloud-shadow rejection, Copernicus DEM slope gravity filtering, and morphological sieve denoising),
and computes deterministic anomaly areas and vector geometries.
"""

from typing import Any, Union
import numpy as np
import pyproj
from rasterio.features import shapes, sieve
from shapely.geometry import mapping, shape
import shapely

from ..config import MAX_SLOPE_DEGREES_FOR_WATER, MIN_SIEVE_PIXEL_CLUSTER
from ..schemas import BiTemporalComparison, DisasterType, PrecisionFilterStats


def _normalized_diff(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Safe normalized difference calculation with zero-division guard."""
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)
    denom = a + b
    diff = np.where(denom != 0.0, (a - b) / (denom + 1e-6), 0.0)
    return diff.astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters): (Green - NIR) / (Green + NIR)."""
    return _normalized_diff(green, nir)


def compute_mndwi(green: np.ndarray, swir16: np.ndarray) -> np.ndarray:
    """
    Modified Normalized Difference Water Index (Xu): (Green - SWIR16) / (Green + SWIR16).
    Superior separation of turbid flood water from built-up soil in Indian river basins.
    """
    return _normalized_diff(green, swir16)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    return _normalized_diff(nir, red)


def compute_nbr(nir: np.ndarray, swir22: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio: (NIR - SWIR22) / (NIR + SWIR22)."""
    return _normalized_diff(nir, swir22)


def compute_bsi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir16: np.ndarray
) -> np.ndarray:
    """
    Bare Soil Index: ((Red + SWIR16) - (NIR + Blue)) / ((Red + SWIR16) + (NIR + Blue)).
    Identifies active scars, rockfall, and bare earth in landslide zones.
    """
    b = blue.astype(np.float32)
    r = red.astype(np.float32)
    n = nir.astype(np.float32)
    s = swir16.astype(np.float32)
    num = (r + s) - (n + b)
    denom = (r + s) + (n + b) + 1e-6
    return np.where(denom != 0.0, num / denom, 0.0).astype(np.float32)


def vectorize_mask_to_geojson(
    binary_mask: np.ndarray,
    affine_transform: Any,
    source_epsg: int
) -> dict[str, Any]:
    """
    Vectorizes a binary NumPy mask into an EPSG:4326 GeoJSON FeatureCollection dictionary.

    Args:
        binary_mask: 2D boolean or uint8 mask array.
        affine_transform: Rasterio Affine transform for pixel-to-coordinate mapping.
        source_epsg: Source UTM EPSG zone (e.g. 32645).

    Returns:
        Valid GeoJSON FeatureCollection dictionary.
    """
    transformer = pyproj.Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)
    features = []

    mask_uint8 = (binary_mask > 0).astype(np.uint8)
    if np.count_nonzero(mask_uint8) == 0:
        return {"type": "FeatureCollection", "features": []}

    for geom, val in shapes(mask_uint8, mask=(mask_uint8 == 1), transform=affine_transform):
        if val == 1:
            poly = shape(geom)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                # Transform UTM geometry to WGS84
                wgs84_poly = shapely.transform(poly, transformer.transform, interleaved=False)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(wgs84_poly),
                    "properties": {
                        "value": 1,
                        "area_m2": round(float(poly.area), 2),
                    },
                })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def run_bi_temporal_anomaly_pipeline(
    cube: dict[str, Any],
    disaster_type: Union[DisasterType, str] = DisasterType.FLOOD,
    max_slope_threshold_deg: float = MAX_SLOPE_DEGREES_FOR_WATER,
    sieve_size: int = MIN_SIEVE_PIXEL_CLUSTER
) -> dict[str, Any]:
    """
    Executes the 3-Layer Zero-Hallucination Bi-Temporal Anomaly Pipeline.

    Step 1: Compute Raw Bi-temporal Detection (T2 - T1 delta separation).
    Step 2: Filter 1 - SCL Cloud & Cloud-Shadow Rejection (removes false shadow lakes).
    Step 3: Filter 2 - Real Copernicus 30m DEM Slope Guardrail (gravity check: water cannot pool on >12° slope).
    Step 4: Filter 3 - Morphological Sieve Noise Filter (cleans isolated <15px pixel noise).
    Step 5: Deterministic Area Accounting & EPSG:4326 GeoJSON Vectorization.

    Args:
        cube: Bi-temporal raster cube dictionary returned by fetch_bi_temporal_stac_cubes.
        disaster_type: Classified disaster type enum or string.
        max_slope_threshold_deg: Maximum allowable slope for pooled flood water in degrees (default 12.0).
        sieve_size: Minimum pixel cluster size for morphological sieve (default 15).

    Returns:
        dict containing:
            - 'precision_filters': PrecisionFilterStats
            - 'bi_temporal': BiTemporalComparison
            - 'hazard_anomaly_geojson': GeoJSON FeatureCollection
            - 'baseline_permanent_geojson': GeoJSON FeatureCollection
            - 'anomaly_mask': 2D boolean array
            - 'baseline_mask': 2D boolean array
    """
    t1 = cube["t1_bands"]
    t2 = cube["t2_bands"]
    slope_deg = cube["slope_deg"]
    transform = cube["affine_transform"]
    epsg_zone = cube["epsg_utm_zone"]

    d_type_val = disaster_type.value if hasattr(disaster_type, "value") else str(disaster_type).upper()

    # ---------------------------------------------------------------------------
    # Step 1: Raw Bi-Temporal Detection
    # ---------------------------------------------------------------------------
    if d_type_val in ["FLOOD", "CYCLONE"]:
        # T1 Permanent river channel detection
        ndwi_t1 = compute_ndwi(t1["green"], t1["nir"])
        mndwi_t1 = compute_mndwi(t1["green"], t1["swir16"])
        scl_t1 = t1.get("scl", np.zeros_like(ndwi_t1))
        mask_t1 = (mndwi_t1 > 0.08) | (ndwi_t1 > 0.10) | (scl_t1 == 6)

        # T2 Current event water detection
        ndwi_t2 = compute_ndwi(t2["green"], t2["nir"])
        mndwi_t2 = compute_mndwi(t2["green"], t2["swir16"])
        scl_t2 = t2.get("scl", np.zeros_like(ndwi_t2))
        mask_t2 = (mndwi_t2 > 0.05) | (ndwi_t2 > 0.08) | (scl_t2 == 6)

        # True Newly Inundated Anomaly = T2 water NOT present in T1 baseline permanent channel
        raw_new_hazard = (mask_t2 == 1) & (mask_t1 == 0)

    elif d_type_val == "FOREST_FIRE":
        nbr_t1 = compute_nbr(t1["nir"], t1["swir22"])
        nbr_t2 = compute_nbr(t2["nir"], t2["swir22"])
        ndvi_t1 = compute_ndvi(t1["nir"], t1["red"])
        dnbr = nbr_t1 - nbr_t2

        mask_t1 = ndvi_t1 > 0.30
        mask_t2 = dnbr > 0.22
        raw_new_hazard = (dnbr > 0.22) & (ndvi_t1 > 0.25)

    elif d_type_val == "LANDSLIDE":
        ndvi_t1 = compute_ndvi(t1["nir"], t1["red"])
        ndvi_t2 = compute_ndvi(t2["nir"], t2["red"])
        bsi_t2 = compute_bsi(t2["blue"], t2["red"], t2["nir"], t2["swir16"])
        veg_loss = (ndvi_t1 - ndvi_t2) > 0.20

        mask_t1 = ndvi_t1 > 0.30
        mask_t2 = veg_loss & (bsi_t2 > 0.05)
        raw_new_hazard = veg_loss & (bsi_t2 > 0.05) & (slope_deg > 15.0)

    else:
        # Default hydrographic anomaly
        ndwi_t1 = compute_ndwi(t1["green"], t1["nir"])
        ndwi_t2 = compute_ndwi(t2["green"], t2["nir"])
        mask_t1 = ndwi_t1 > 0.10
        mask_t2 = ndwi_t2 > 0.08
        raw_new_hazard = (mask_t2 == 1) & (mask_t1 == 0)

    # ---------------------------------------------------------------------------
    # Step 2: Filter 1 - SCL Cloud & Cloud-Shadow Rejection
    # ---------------------------------------------------------------------------
    scl_t2_arr = t2.get("scl", np.zeros(mask_t1.shape, dtype=np.uint8))
    # SCL 3: Cloud Shadows, SCL 8: Cloud Med Prob, SCL 9: Cloud High Prob, SCL 10: Cirrus, SCL 11: Snow/Ice
    is_cloud_or_shadow = np.isin(scl_t2_arr, [3, 8, 9, 10, 11])
    scl_cloud_shadow_removed = int(np.count_nonzero(raw_new_hazard & is_cloud_or_shadow))
    hazard_f1 = raw_new_hazard & (~is_cloud_or_shadow)

    # ---------------------------------------------------------------------------
    # Step 3: Filter 2 - Real Copernicus 30m DEM Slope Physics Guardrail
    # ---------------------------------------------------------------------------
    if d_type_val in ["FLOOD", "CYCLONE"]:
        steep_slope_mask = slope_deg > max_slope_threshold_deg
        dem_steep_slope_removed = int(np.count_nonzero(hazard_f1 & steep_slope_mask))
        hazard_f2 = hazard_f1 & (~steep_slope_mask)
    else:
        dem_steep_slope_removed = 0
        hazard_f2 = hazard_f1

    # ---------------------------------------------------------------------------
    # Step 4: Filter 3 - Morphological Sieve Noise Removal
    # ---------------------------------------------------------------------------
    sieved_mask = sieve(hazard_f2.astype(np.uint8), size=sieve_size)
    hazard_final = sieved_mask > 0
    morphological_sieve_removed = int(np.count_nonzero(hazard_f2) - np.count_nonzero(hazard_final))

    # Also clean baseline T1 permanent channel
    sieved_baseline = sieve(mask_t1.astype(np.uint8), size=sieve_size)
    baseline_final = sieved_baseline > 0

    # ---------------------------------------------------------------------------
    # Step 5: Area Accounting & GeoJSON Vectorization
    # ---------------------------------------------------------------------------
    pixel_area_m2 = abs(float(transform.a) * float(transform.e))
    baseline_t1_area_km2 = round(float(np.count_nonzero(baseline_final) * pixel_area_m2 / 1e6), 4)
    net_anomaly_area_km2 = round(float(np.count_nonzero(hazard_final) * pixel_area_m2 / 1e6), 4)
    current_t2_total_area_km2 = round(float(baseline_t1_area_km2 + net_anomaly_area_km2), 4)
    net_anomaly_hectares = round(net_anomaly_area_km2 * 100.0, 2)

    if baseline_t1_area_km2 > 0:
        growth_pct = round((net_anomaly_area_km2 / baseline_t1_area_km2) * 100.0, 2)
    else:
        growth_pct = 100.0 if net_anomaly_area_km2 > 0 else 0.0

    precision_filters = PrecisionFilterStats(
        scl_cloud_shadow_pixels_removed=scl_cloud_shadow_removed,
        dem_steep_slope_pixels_removed=dem_steep_slope_removed,
        morphological_sieve_noise_removed=morphological_sieve_removed,
        max_slope_threshold_deg=max_slope_threshold_deg,
    )

    bi_temporal = BiTemporalComparison(
        t1_scene_id=cube["t1_scene_id"],
        t1_datetime=cube["t1_datetime"],
        t2_scene_id=cube["t2_scene_id"],
        t2_datetime=cube["t2_datetime"],
        baseline_t1_area_km2=baseline_t1_area_km2,
        current_t2_total_area_km2=current_t2_total_area_km2,
        net_anomaly_area_km2=net_anomaly_area_km2,
        net_anomaly_hectares=net_anomaly_hectares,
        anomaly_growth_pct=growth_pct,
    )

    hazard_anomaly_geojson = vectorize_mask_to_geojson(hazard_final, transform, epsg_zone)
    baseline_permanent_geojson = vectorize_mask_to_geojson(baseline_final, transform, epsg_zone)

    return {
        "precision_filters": precision_filters,
        "bi_temporal": bi_temporal,
        "hazard_anomaly_geojson": hazard_anomaly_geojson,
        "baseline_permanent_geojson": baseline_permanent_geojson,
        "anomaly_mask": hazard_final,
        "baseline_mask": baseline_final,
        "pixel_area_m2": pixel_area_m2,
    }
