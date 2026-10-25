"""
Thread-safe SQLite persistence engine for Sat Query live disaster audits.
Provides full persistence, retrieval, and telemetry indexing for audit compliance.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional, Union

from .config import SQLITE_DB_PATH
from .schemas import AnalyzeResponse

# Process-level thread lock for safe concurrent SQLite execution
_DB_LOCK = threading.Lock()


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Creates a configured sqlite3 connection with Row factory and WAL mode."""
    target_path = db_path or SQLITE_DB_PATH
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_audit_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initializes the SQLite schema for disaster audit logs.
    Idempotent operation ensuring required table and indexes exist.
    """
    with _DB_LOCK:
        conn = get_connection(db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS live_disaster_audits (
                        request_id TEXT PRIMARY KEY,
                        timestamp_utc TEXT NOT NULL,
                        query TEXT NOT NULL,
                        disaster_type TEXT NOT NULL,
                        resolved_bbox_json TEXT NOT NULL,
                        t1_scene_id TEXT NOT NULL,
                        t2_scene_id TEXT NOT NULL,
                        baseline_t1_km2 REAL NOT NULL,
                        current_t2_km2 REAL NOT NULL,
                        net_anomaly_km2 REAL NOT NULL,
                        false_positives_removed_total INTEGER NOT NULL,
                        confidence_score REAL NOT NULL,
                        sha256_audit_hash TEXT NOT NULL,
                        full_response_json TEXT NOT NULL
                    );
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audits_timestamp 
                    ON live_disaster_audits (timestamp_utc DESC);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audits_disaster_type 
                    ON live_disaster_audits (disaster_type);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audits_sha256 
                    ON live_disaster_audits (sha256_audit_hash);
                """)
        finally:
            conn.close()


def save_audit_record(
    response: Union[AnalyzeResponse, dict[str, Any]],
    db_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Persists an AnalyzeResponse audit record to SQLite.
    Performs an idempotent upsert (INSERT OR REPLACE).
    """
    if isinstance(response, dict):
        resp_obj = AnalyzeResponse.model_validate(response)
    else:
        resp_obj = response

    init_audit_db(db_path)

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    disaster_type_val = (
        resp_obj.disaster_type.value
        if hasattr(resp_obj.disaster_type, "value")
        else str(resp_obj.disaster_type)
    )
    resolved_bbox_json = json.dumps(resp_obj.resolved_bbox)
    full_response_json = resp_obj.model_dump_json()

    fp_total = (
        resp_obj.precision_filters.scl_cloud_shadow_pixels_removed
        + resp_obj.precision_filters.dem_steep_slope_pixels_removed
        + resp_obj.precision_filters.morphological_sieve_noise_removed
    )

    with _DB_LOCK:
        conn = get_connection(db_path)
        try:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO live_disaster_audits (
                        request_id,
                        timestamp_utc,
                        query,
                        disaster_type,
                        resolved_bbox_json,
                        t1_scene_id,
                        t2_scene_id,
                        baseline_t1_km2,
                        current_t2_km2,
                        net_anomaly_km2,
                        false_positives_removed_total,
                        confidence_score,
                        sha256_audit_hash,
                        full_response_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    resp_obj.request_id,
                    timestamp_utc,
                    resp_obj.query,
                    disaster_type_val,
                    resolved_bbox_json,
                    resp_obj.bi_temporal.t1_scene_id,
                    resp_obj.bi_temporal.t2_scene_id,
                    float(resp_obj.bi_temporal.baseline_t1_area_km2),
                    float(resp_obj.bi_temporal.current_t2_total_area_km2),
                    float(resp_obj.bi_temporal.net_anomaly_area_km2),
                    int(fp_total),
                    float(resp_obj.epistemic_confidence_score),
                    resp_obj.sha256_audit_hash,
                    full_response_json,
                ))
        finally:
            conn.close()


def get_audit_record_by_id(
    request_id: str,
    db_path: Optional[Union[str, Path]] = None
) -> Optional[dict[str, Any]]:
    """
    Retrieves an audit record by its primary request_id.
    Returns a dictionary of columns or None if not found.
    """
    init_audit_db(db_path)
    with _DB_LOCK:
        conn = get_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM live_disaster_audits WHERE request_id = ?",
                (request_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()


def list_recent_audits(
    limit: int = 20,
    db_path: Optional[Union[str, Path]] = None
) -> list[dict[str, Any]]:
    """
    Returns the most recent audit records ordered by timestamp descending.
    """
    init_audit_db(db_path)
    with _DB_LOCK:
        conn = get_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM live_disaster_audits ORDER BY timestamp_utc DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
