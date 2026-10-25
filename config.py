"""
Sat Query Production Configuration.
Provides verified live endpoints for AWS STAC, OpenStreetMap Overpass,
Open-Meteo GloFAS Flood discharge, TinyFish SSE telemetry, and Indian Subcontinent bounds.
"""

from pathlib import Path
from typing import TypedDict
from pydantic_settings import BaseSettings, SettingsConfigDict


# Live Geospatial & Hydrographic Production Endpoints
AWS_STAC_URL: str = "https://earth-search.aws.element84.com/v1"


class CollectionsDict(dict):
    """
    STAC Collection identifiers for live satellite assets.
    Supports both dict indexing and attribute access.
    """
    def __getattr__(self, name: str) -> str:
        if name in self:
            return self[name]
        raise AttributeError(f"'CollectionsDict' object has no attribute '{name}'")


COLLECTIONS = CollectionsDict({
    "SENTINEL_2": "sentinel-2-l2a",
    "SENTINEL_1": "sentinel-1-grd",
    "COP_DEM_30": "cop-dem-glo-30",
})

OVERPASS_API_URL: str = "https://overpass-api.de/api/interpreter"
NOMINATIM_API_URL: str = "https://nominatim.openstreetmap.org"
OPEN_METEO_FLOOD_API: str = "https://flood-api.open-meteo.com/v1/flood"
TINYFISH_SSE_URL: str = "https://agent.tinyfish.ai/v1/automation/run-sse"


class SubcontinentBoundsDict(TypedDict):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


# Indian Subcontinent Bounding Box Guardrail
SUBCONTINENT_BOUNDS: SubcontinentBoundsDict = {
    "min_lon": 60.0,
    "min_lat": 5.0,
    "max_lon": 100.0,
    "max_lat": 38.0,
}

# Physics & Remote Sensing Guardrails
MAX_SLOPE_DEGREES_FOR_WATER: float = 12.0  # Physics guardrail: water cannot pool on slopes > 12°
CLOUD_COVER_SAR_SWITCH_PCT: float = 30.0   # Switch to Sentinel-1 SAR if Sentinel-2 cloud cover exceeds 30%
MIN_SIEVE_PIXEL_CLUSTER: int = 15          # Morphological noise removal cluster size

# Database Persistence Path
SQLITE_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "trinetra_live_audit.db"
SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    AWS_STAC_URL: str = AWS_STAC_URL
    OVERPASS_API_URL: str = OVERPASS_API_URL
    NOMINATIM_API_URL: str = NOMINATIM_API_URL
    OPEN_METEO_FLOOD_API: str = OPEN_METEO_FLOOD_API
    TINYFISH_SSE_URL: str = TINYFISH_SSE_URL
    MAX_SLOPE_DEGREES_FOR_WATER: float = MAX_SLOPE_DEGREES_FOR_WATER
    CLOUD_COVER_SAR_SWITCH_PCT: float = CLOUD_COVER_SAR_SWITCH_PCT
    MIN_SIEVE_PIXEL_CLUSTER: int = MIN_SIEVE_PIXEL_CLUSTER
    SQLITE_DB_PATH: str = str(SQLITE_DB_PATH)

    model_config = SettingsConfigDict(env_prefix="SAT_QUERY_", extra="ignore")


settings = Settings()
