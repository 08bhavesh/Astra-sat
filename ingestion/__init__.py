"""
Sat Query Geospatial Ingestion and Analytics Engine.
Provides UTM projection warping, live OSM river corridor resolution,
AWS STAC bi-temporal window streaming, and zero-hallucination precision math.
"""

from .utm_warper import (
    get_subcontinent_utm_epsg,
    transform_bbox_to_utm,
    normalize_analysis_bbox,
)
from .river_locator import resolve_location_and_river_geometry
from .stac_bi_temporal import fetch_bi_temporal_stac_cubes
from .precision_math import (
    compute_ndwi,
    compute_mndwi,
    compute_ndvi,
    compute_nbr,
    compute_bsi,
    vectorize_mask_to_geojson,
    run_bi_temporal_anomaly_pipeline,
)

__all__ = [
    "get_subcontinent_utm_epsg",
    "transform_bbox_to_utm",
    "normalize_analysis_bbox",
    "resolve_location_and_river_geometry",
    "fetch_bi_temporal_stac_cubes",
    "compute_ndwi",
    "compute_mndwi",
    "compute_ndvi",
    "compute_nbr",
    "compute_bsi",
    "vectorize_mask_to_geojson",
    "run_bi_temporal_anomaly_pipeline",
]
