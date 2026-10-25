"""
Cryptographic SHA-256 Audit Engine for Sat Query.
Produces deterministic, tamper-evident cryptographic digests of disaster analysis payloads
for multi-agency validation and military/civil defense auditability.
"""

import hashlib
import json
from typing import Any, Optional, Union


def _to_clean_dict(val: Any) -> Any:
    """Recursively converts Pydantic models or dicts to plain Python dicts/primitives."""
    if hasattr(val, "model_dump"):
        return _to_clean_dict(val.model_dump())
    if isinstance(val, dict):
        return {k: _to_clean_dict(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_clean_dict(x) for x in val]
    return val


def compute_cryptographic_audit_hash(
    resolved_bbox: list[float],
    t1_scene_id: str,
    t2_scene_id: str,
    net_anomaly_area_km2: float,
    precision_filters_dict: Optional[Union[dict, Any]] = None,
    ground_truth_dict: Optional[Union[dict, Any]] = None,
    *,
    precision_filters: Optional[Union[dict, Any]] = None,
    ground_truth: Optional[Union[dict, Any]] = None,
) -> str:
    """
    Computes a canonical SHA-256 cryptographic digest over the core disaster assessment metrics.

    Args:
        resolved_bbox: [min_lon, min_lat, max_lon, max_lat] in WGS84.
        t1_scene_id: Real STAC Item ID for baseline pass.
        t2_scene_id: Real STAC Item ID for event pass.
        net_anomaly_area_km2: Net newly inundated or burned area in km².
        precision_filters_dict: Pixel accounting breakdown for false-positive rejections.
        ground_truth_dict: Independent physical ground-truth sensor telemetry.

    Returns:
        64-character lowercase hexadecimal SHA-256 string.
    """
    pf_target = precision_filters_dict if precision_filters_dict is not None else precision_filters
    gt_target = ground_truth_dict if ground_truth_dict is not None else ground_truth

    if pf_target is None:
        raise ValueError("precision_filters_dict must be provided.")
    if gt_target is None:
        raise ValueError("ground_truth_dict must be provided.")

    clean_pf = _to_clean_dict(pf_target)
    clean_gt = _to_clean_dict(gt_target)
    clean_bbox = [float(x) for x in resolved_bbox]

    canonical_payload = {
        "ground_truth_dict": clean_gt,
        "net_anomaly_area_km2": float(net_anomaly_area_km2),
        "precision_filters_dict": clean_pf,
        "resolved_bbox": clean_bbox,
        "t1_scene_id": str(t1_scene_id),
        "t2_scene_id": str(t2_scene_id),
    }

    # Canonical sorted serialization with no whitespace
    canonical_json = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )

    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest().lower()
    return digest


def compute_audit_hash_from_response(response: Any) -> str:
    """Convenience helper to compute audit hash directly from an AnalyzeResponse instance."""
    pf = response.precision_filters
    gt = response.ground_truth
    bi = response.bi_temporal

    return compute_cryptographic_audit_hash(
        resolved_bbox=response.resolved_bbox,
        t1_scene_id=bi.t1_scene_id,
        t2_scene_id=bi.t2_scene_id,
        net_anomaly_area_km2=bi.net_anomaly_area_km2,
        precision_filters_dict=pf,
        ground_truth_dict=gt,
    )
