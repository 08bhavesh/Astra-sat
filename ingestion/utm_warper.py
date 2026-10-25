"""
UTM projection transformation and bounding-box normalization module for Sat Query.
Computes deterministic UTM EPSG projections for the Indian Subcontinent and centers tactical analysis windows.
"""

from typing import Union
import pyproj
from ..config import SUBCONTINENT_BOUNDS


def get_subcontinent_utm_epsg(lon: float, lat: float) -> int:
    """
    Deterministically calculates the UTM Zone EPSG code for any point in the Indian Subcontinent.

    Formula: zone = int((lon + 180.0) / 6.0) + 1
    Indian Subcontinent falls in WGS84 Northern Hemisphere (EPSG 32642 to 32647).

    Args:
        lon: Longitude in WGS84 degrees.
        lat: Latitude in WGS84 degrees.

    Returns:
        EPSG integer (e.g. 32645 for Bihar/Kosi/Patna/West Bengal).
    """
    zone = int((float(lon) + 180.0) / 6.0) + 1
    # Base 32600 for WGS84 UTM Northern Hemisphere
    base_epsg = 32600 if lat >= 0 else 32700
    return base_epsg + zone


def transform_bbox_to_utm(
    bbox_wgs84: list[float],
    target_epsg: int
) -> tuple[float, float, float, float]:
    """
    Transforms [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 to UTM planar meters (min_x, min_y, max_x, max_y).

    Evaluates all 4 corner coordinates to prevent any bounding box deformation from projection curvature.

    Args:
        bbox_wgs84: [min_lon, min_lat, max_lon, max_lat] in WGS84.
        target_epsg: Target UTM EPSG code (e.g. 32645).

    Returns:
        (min_x, min_y, max_x, max_y) in meters.
    """
    min_lon, min_lat, max_lon, max_lat = (float(coord) for coord in bbox_wgs84)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)

    # Transform all 4 corners
    lons = [min_lon, max_lon, min_lon, max_lon]
    lats = [min_lat, min_lat, max_lat, max_lat]
    xs, ys = transformer.transform(lons, lats)

    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def normalize_analysis_bbox(
    bbox: list[float],
    target_span_deg: float = 0.08
) -> list[float]:
    """
    Normalizes a potentially large geographic bounding box (e.g. entire river length or state)
    to a crisp tactical analysis window of ~target_span_deg (~8.8 km x 8.8 km) centered on the centroid.

    Keeps STAC streaming crisp and fast (<5s) at ~17m/pixel over 512x512 resolution,
    while ensuring the window remains strictly within Indian Subcontinent bounds.

    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84.
        target_span_deg: Desired window span in degrees (default 0.08 deg ~ 8.8 km).

    Returns:
        Normalized [min_lon, min_lat, max_lon, max_lat].
    """
    min_lon, min_lat, max_lon, max_lat = (float(coord) for coord in bbox)
    span_lon = max_lon - min_lon
    span_lat = max_lat - min_lat

    # Only shrink if larger than target span
    if span_lon > target_span_deg or span_lat > target_span_deg:
        c_lon = (min_lon + max_lon) / 2.0
        c_lat = (min_lat + max_lat) / 2.0
        half_span = target_span_deg / 2.0
        n_min_lon = c_lon - half_span
        n_max_lon = c_lon + half_span
        n_min_lat = c_lat - half_span
        n_max_lat = c_lat + half_span
    else:
        n_min_lon, n_min_lat, n_max_lon, n_max_lat = min_lon, min_lat, max_lon, max_lat

    # Clamp strictly within Indian Subcontinent Bounds
    b_min_lon = SUBCONTINENT_BOUNDS["min_lon"]
    b_max_lon = SUBCONTINENT_BOUNDS["max_lon"]
    b_min_lat = SUBCONTINENT_BOUNDS["min_lat"]
    b_max_lat = SUBCONTINENT_BOUNDS["max_lat"]

    clamped_min_lon = max(b_min_lon, min(n_min_lon, b_max_lon - 0.01))
    clamped_max_lon = min(b_max_lon, max(n_max_lon, b_min_lon + 0.01))
    clamped_min_lat = max(b_min_lat, min(n_min_lat, b_max_lat - 0.01))
    clamped_max_lat = min(b_max_lat, max(n_max_lat, b_min_lat + 0.01))

    return [
        round(clamped_min_lon, 6),
        round(clamped_min_lat, 6),
        round(clamped_max_lon, 6),
        round(clamped_max_lat, 6),
    ]
