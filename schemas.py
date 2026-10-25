"""
Sat Query Pydantic v2 Data Contracts.
Strict schema definitions for STAC telemetry, bi-temporal satellite comparisons,
precision filter pixel accounting, ground truth verification, and full-river contracts.
"""

from datetime import datetime
from enum import Enum
import re
from typing import Any, Optional
from typing_extensions import Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .config import (
    MAX_SLOPE_DEGREES_FOR_WATER,
    SUBCONTINENT_BOUNDS,
)


class DisasterType(str, Enum):
    FLOOD = "FLOOD"
    FOREST_FIRE = "FOREST_FIRE"
    LANDSLIDE = "LANDSLIDE"
    CYCLONE = "CYCLONE"
    DROUGHT = "DROUGHT"
    GENERAL_VQA = "GENERAL_VQA"


class TaskType(str, Enum):
    BI_TEMPORAL_ANOMALY = "BI_TEMPORAL_ANOMALY"
    OPT_SAR_FUSION_SEGMENTATION = "OPT_SAR_FUSION_SEGMENTATION"
    BURN_SEVERITY_DNBR = "BURN_SEVERITY_DNBR"
    LANDSLIDE_SLOPE_CHANGE = "LANDSLIDE_SLOPE_CHANGE"
    VISUAL_GROUNDING_VQA = "VISUAL_GROUNDING_VQA"


class AnalyzeRequest(BaseModel):
    """
    Inbound user query and spatial-temporal constraints contract.
    Enforces subcontinent bounds and requires at least one spatial anchor.
    """
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=5, description="Natural language disaster query")
    river_or_location_name: Optional[str] = Field(
        default=None,
        description="Geographic anchor (e.g. 'Kosi River', 'Ganga Patna', 'Nainital')"
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat] in WGS84"
    )
    baseline_date_t1: Optional[str] = Field(
        default=None,
        description="Baseline reference date (ISO YYYY-MM-DD)"
    )
    current_date_t2: Optional[str] = Field(
        default=None,
        description="Current disaster event date (ISO YYYY-MM-DD)"
    )

    @field_validator("bbox", mode="after")
    @classmethod
    def validate_subcontinent_bbox(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return v
        if len(v) != 4:
            raise ValueError(f"bbox must contain exactly 4 coordinates [min_lon, min_lat, max_lon, max_lat], got {len(v)}")
        
        min_lon, min_lat, max_lon, max_lat = (float(coord) for coord in v)
        if min_lon >= max_lon:
            raise ValueError(f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})")
        if min_lat >= max_lat:
            raise ValueError(f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})")

        b_min_lon = SUBCONTINENT_BOUNDS["min_lon"]
        b_max_lon = SUBCONTINENT_BOUNDS["max_lon"]
        b_min_lat = SUBCONTINENT_BOUNDS["min_lat"]
        b_max_lat = SUBCONTINENT_BOUNDS["max_lat"]

        if min_lon < b_min_lon or max_lon > b_max_lon or min_lat < b_min_lat or max_lat > b_max_lat:
            raise ValueError(
                f"bbox [{min_lon}, {min_lat}, {max_lon}, {max_lat}] is outside Indian Subcontinent bounds: "
                f"lon: [{b_min_lon}, {b_max_lon}], lat: [{b_min_lat}, {b_max_lat}]"
            )
        return [min_lon, min_lat, max_lon, max_lat]

    @field_validator("baseline_date_t1", "current_date_t2", mode="after")
    @classmethod
    def validate_iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            try:
                datetime.strptime(v, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Date '{v}' must be formatted as ISO YYYY-MM-DD")
        return v

    @model_validator(mode="after")
    def validate_spatial_anchor_provided(self) -> Self:
        has_name = bool(self.river_or_location_name and self.river_or_location_name.strip())
        has_bbox = self.bbox is not None and len(self.bbox) == 4
        if not has_name and not has_bbox:
            raise ValueError("At least one of 'bbox' or 'river_or_location_name' must be provided.")
        return self


class PrecisionFilterStats(BaseModel):
    """
    Rigorous accounting of anti-hallucination and physics filter pixel rejections.
    """
    model_config = ConfigDict(extra="forbid")

    scl_cloud_shadow_pixels_removed: int = Field(
        ...,
        ge=0,
        description="Sentinel-2 SCL cloud and shadow pixels filtered"
    )
    dem_steep_slope_pixels_removed: int = Field(
        ...,
        ge=0,
        description="Pixels filtered by DEM slope > threshold (gravity physics rule)"
    )
    morphological_sieve_noise_removed: int = Field(
        ...,
        ge=0,
        description="Isolated pixel clusters filtered via morphological sieve"
    )
    max_slope_threshold_deg: float = Field(
        default=MAX_SLOPE_DEGREES_FOR_WATER,
        ge=0.0,
        description="Maximum slope threshold in degrees"
    )

    @property
    def total_false_positives_removed(self) -> int:
        return (
            self.scl_cloud_shadow_pixels_removed
            + self.dem_steep_slope_pixels_removed
            + self.morphological_sieve_noise_removed
        )


class BiTemporalComparison(BaseModel):
    """
    Bi-temporal satellite comparison contract (Baseline T1 vs Event T2).
    Enforces mathematical consistency of net anomaly calculation.
    """
    model_config = ConfigDict(extra="forbid")

    t1_scene_id: str = Field(..., min_length=1, description="STAC Item ID for Baseline scene")
    t1_datetime: str = Field(..., description="Timestamp of baseline pass")
    t2_scene_id: str = Field(..., min_length=1, description="STAC Item ID for Event scene")
    t2_datetime: str = Field(..., description="Timestamp of event pass")
    baseline_t1_area_km2: float = Field(..., ge=0.0, description="Permanent baseline area in km²")
    current_t2_total_area_km2: float = Field(..., ge=0.0, description="Total detected area at T2 in km²")
    net_anomaly_area_km2: float = Field(..., ge=0.0, description="Net newly inundated/burned area = max(0, T2 - T1)")
    net_anomaly_hectares: float = Field(..., ge=0.0, description="Net anomaly in hectares (km² * 100)")
    anomaly_growth_pct: float = Field(..., description="Percentage growth relative to T1 baseline")

    @model_validator(mode="before")
    @classmethod
    def auto_populate_anomaly_math(cls, data: Any) -> Any:
        if isinstance(data, dict):
            t1 = data.get("baseline_t1_area_km2")
            t2 = data.get("current_t2_total_area_km2")
            if t1 is not None and t2 is not None:
                t1_f = float(t1)
                t2_f = float(t2)
                net = max(0.0, t2_f - t1_f)
                if "net_anomaly_area_km2" not in data or data["net_anomaly_area_km2"] is None:
                    data["net_anomaly_area_km2"] = round(net, 4)
                if "net_anomaly_hectares" not in data or data["net_anomaly_hectares"] is None:
                    data["net_anomaly_hectares"] = round(data["net_anomaly_area_km2"] * 100.0, 2)
                if "anomaly_growth_pct" not in data or data["anomaly_growth_pct"] is None:
                    if t1_f > 0:
                        data["anomaly_growth_pct"] = round((net / t1_f) * 100.0, 2)
                    else:
                        data["anomaly_growth_pct"] = 100.0 if net > 0 else 0.0
        return data

    @model_validator(mode="after")
    def validate_anomaly_math_consistency(self) -> Self:
        expected_net = max(0.0, self.current_t2_total_area_km2 - self.baseline_t1_area_km2)
        if abs(self.net_anomaly_area_km2 - expected_net) > 0.05:
            raise ValueError(
                f"net_anomaly_area_km2 ({self.net_anomaly_area_km2}) is mathematically inconsistent with "
                f"max(0, current_t2 ({self.current_t2_total_area_km2}) - baseline_t1 ({self.baseline_t1_area_km2})) = {expected_net}"
            )
        expected_hectares = self.net_anomaly_area_km2 * 100.0
        if abs(self.net_anomaly_hectares - expected_hectares) > 5.0:
            raise ValueError(
                f"net_anomaly_hectares ({self.net_anomaly_hectares}) is mathematically inconsistent with "
                f"net_anomaly_area_km2 * 100 ({expected_hectares})"
            )
        return self


class LiveGroundTruthTelemetry(BaseModel):
    """
    Independent ground truth telemetry contract for cross-verification.
    """
    model_config = ConfigDict(extra="forbid")

    portal_source: str = Field(..., min_length=2, description="Telemetry source (e.g. CWC_INDIA_TINYFISH, OPEN_METEO_GLOFAS)")
    station_or_basin: str = Field(..., min_length=1, description="Hydrological station or basin identifier")
    observed_metric: str = Field(..., description="Observed metric value (e.g. '12,450 m³/s discharge')")
    reference_threshold: str = Field(..., description="Warning/Danger reference mark (e.g. 'Danger Level: 71.50m')")
    is_anomaly_confirmed: bool = Field(..., description="Whether ground sensor confirms satellite anomaly")
    verification_timestamp_utc: str = Field(..., description="UTC verification timestamp")


class AnalyzeResponse(BaseModel):
    """
    Primary output contract representing the verified geospatial intelligence payload.
    Includes full-river geometry, bitemporal anomaly polygons, epistemic confidence, and SHA-256 digest.
    """
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, description="Unique trace identifier")
    query: str = Field(..., min_length=5, description="Original user query")
    disaster_type: DisasterType = Field(..., description="Classified hazard type")
    routed_task: TaskType = Field(..., description="Specialized model execution pipeline")
    resolved_bbox: list[float] = Field(..., description="Final resolved WGS84 bounding box")
    epsg_utm_zone: int = Field(..., description="Projected coordinate system EPSG code for planar calculations")
    sensor_pipeline: str = Field(..., description="Satellite sensor used (e.g. 'Sentinel-2 L2A Optical', 'Sentinel-1 GRD SAR')")
    cloud_cover_t2_pct: float = Field(..., ge=0.0, le=100.0, description="Cloud cover percentage at T2")
    bi_temporal: BiTemporalComparison = Field(..., description="Baseline vs Event delta accounting")
    precision_filters: PrecisionFilterStats = Field(..., description="Precision filter rejection metrics")
    full_river_geojson: Optional[dict] = Field(
        default=None,
        description="GeoJSON FeatureCollection representing the full river corridor from OSM Overpass"
    )
    hazard_anomaly_geojson: dict = Field(
        ...,
        description="GeoJSON FeatureCollection representing verified net hazard anomaly polygons"
    )
    baseline_permanent_geojson: Optional[dict] = Field(
        default=None,
        description="GeoJSON FeatureCollection representing permanent baseline water/vegetation"
    )
    ground_truth: LiveGroundTruthTelemetry = Field(..., description="Independent sensor ground truth telemetry")
    epistemic_confidence_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Epistemic confidence score from multi-sensor and ground-truth validation (0-100)"
    )
    agentic_execution_trace: list[str] = Field(
        default_factory=list,
        description="Sequential list of agent execution steps"
    )
    vlm_tactical_summary: str = Field(..., description="Synthesized tactical military/civil defense situation report")
    sha256_audit_hash: str = Field(..., min_length=64, max_length=64, description="Cryptographic SHA-256 audit fingerprint")
    processing_latency_ms: float = Field(..., ge=0.0, description="Total pipeline latency in milliseconds")

    @field_validator("resolved_bbox", mode="after")
    @classmethod
    def validate_resolved_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError(f"resolved_bbox must contain 4 coordinates, got {len(v)}")
        min_lon, min_lat, max_lon, max_lat = (float(coord) for coord in v)
        if min_lon >= max_lon:
            raise ValueError(f"min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})")
        if min_lat >= max_lat:
            raise ValueError(f"min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})")
        b_min_lon = SUBCONTINENT_BOUNDS["min_lon"]
        b_max_lon = SUBCONTINENT_BOUNDS["max_lon"]
        b_min_lat = SUBCONTINENT_BOUNDS["min_lat"]
        b_max_lat = SUBCONTINENT_BOUNDS["max_lat"]
        if min_lon < b_min_lon or max_lon > b_max_lon or min_lat < b_min_lat or max_lat > b_max_lat:
            raise ValueError(f"resolved_bbox [{min_lon}, {min_lat}, {max_lon}, {max_lat}] outside subcontinent bounds")
        return [min_lon, min_lat, max_lon, max_lat]

    @field_validator("sha256_audit_hash", mode="after")
    @classmethod
    def validate_sha256_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not re.fullmatch(r"^[0-9a-f]{64}$", clean):
            raise ValueError("sha256_audit_hash must be a 64-character lowercase hexadecimal string")
        return clean

    @field_validator("hazard_anomaly_geojson", "full_river_geojson", "baseline_permanent_geojson", mode="after")
    @classmethod
    def validate_geojson_structure(cls, v: Optional[dict]) -> Optional[dict]:
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("GeoJSON must be a dictionary")
        if v.get("type") != "FeatureCollection":
            raise ValueError(f"GeoJSON must be a 'FeatureCollection', got '{v.get('type')}'")
        if "features" not in v or not isinstance(v["features"], list):
            raise ValueError("GeoJSON FeatureCollection must contain a 'features' array")
        return v
