"""
FastAPI Production Server for Sat Query Tactical Disaster Command Center.
Exposes endpoints for health checks, live bi-temporal disaster intelligence analysis,
multilingual free-text disaster queries with real sensor data & TinyFish,
and cryptographic SQLite audit persistence retrieval.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .agents.langgraph_agent import run_sat_query_agent, run_trinetra_agent
from .database import get_audit_record_by_id, init_audit_db, list_recent_audits
from .query_service import process_disaster_query
from .schemas import AnalyzeRequest, AnalyzeResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sat_query.api")


class FreeTextQueryRequest(BaseModel):
    query: str
    location: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite persistence schemas on server startup."""
    init_audit_db()
    logger.info("Sat Query SQLite audit persistence engine initialized.")
    yield


app = FastAPI(
    title="Sat Query Tactical Disaster Command Center API",
    description=(
        "Mission-critical autonomous geospatial disaster intelligence engine "
        "for the Indian Subcontinent. Integrates live AWS STAC (Sentinel-2 & Sentinel-1), "
        "Copernicus 30m DEM slope physics filtering, OSM Overpass river geometry, "
        "live Copernicus GloFAS river discharge telemetry, TinyFish official alerts, "
        "and cryptographic SHA-256 auditing."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS Middleware for seamless local and remote frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Live system health and operational readiness check."""
    return {
        "status": "healthy",
        "service": "Sat Query Subcontinent Disaster Engine",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/api/v1/query",
    tags=["Disaster Intelligence"],
    status_code=status.HTTP_200_OK,
)
def handle_disaster_query(request: FreeTextQueryRequest) -> dict[str, Any]:
    """
    Executes free-text natural language disaster query analysis (Fix 2 & Fix 3):
    1. Multilingual disaster entity extraction (English, Hindi, Hinglish)
    2. Nominatim location geocoding and bounding box calculation
    3. Real sensor data: Copernicus GloFAS river discharge, Open-Meteo precipitation, USGS seismic
    4. TinyFish official emergency alerts and news integration
    5. Real GeoJSON affected-area polygon with severity styling
    6. Executive briefing with risk assessment, affected areas, and recommended actions
    """
    try:
        result = process_disaster_query(request.query, request.location)
        return result
    except Exception as e:
        logger.error(f"Error handling disaster query: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "message": f"Query processing error: {str(e)}",
            "error_detail": str(e),
        }


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Disaster Intelligence"],
)
def analyze_disaster(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Executes the 5-Node LangGraph autonomous agent pipeline for a disaster query:
    1. Intent & Temporal Routing
    2. OSM Geography & River Corridor Resolution
    3. AWS STAC Window Streaming & 3-Layer Precision Math
    4. Copernicus GloFAS & Open-Meteo Physical Telemetry Verification
    5. Epistemic Synthesis, SHA-256 Fingerprinting, and SQLite Logging
    """
    try:
        response = run_sat_query_agent(request)
        return response
    except (ValueError, ValidationError) as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Error during agentic execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}"
        )


@app.get("/api/v1/audits", tags=["Audit & Persistence"])
def get_recent_audits(
    limit: int = Query(default=20, ge=1, le=100, description="Max audit logs to return")
) -> list[dict[str, Any]]:
    """Retrieves chronologically sorted recent disaster analysis audit logs from SQLite."""
    records = list_recent_audits(limit=limit)
    return records


@app.get("/api/v1/audit/{request_id}", tags=["Audit & Persistence"])
def get_audit_by_id(request_id: str) -> dict[str, Any]:
    """Retrieves a single cryptographic disaster audit record by unique request_id."""
    record = get_audit_record_by_id(request_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit record '{request_id}' not found"
        )
    return record


# ---------------------------------------------------------------------------
# Static Assets & Single-Page Application (SPA) Serving on Port 8000
# ---------------------------------------------------------------------------

FRONTEND_DIST_DIR = (
    Path(__file__).resolve().parent.parent.parent / "dist"
    if (Path(__file__).resolve().parent.parent.parent / "dist").exists()
    else Path(__file__).resolve().parent.parent.parent.parent / "frontend-astra" / "dist"
)
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS_DIR)), name="assets")


@app.get("/", include_in_schema=False)
def serve_spa_root():
    """Serves the Sat Query tactical command center SPA."""
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "online",
        "service": "Sat Query Tactical API",
        "message": "Frontend distribution not built. Run 'npm.cmd run build' in frontend/.",
    }


@app.get("/{full_path:path}", include_in_schema=False)
def serve_spa_fallback(full_path: str):
    """
    Catch-all route to support client-side routing, static assets, and favicon.
    Passes through API and documentation routes without capturing them.
    """
    if full_path.startswith(("api/", "health", "docs", "openapi.json", "redoc")):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")

    target_file = FRONTEND_DIST_DIR / full_path
    if target_file.is_file():
        return FileResponse(str(target_file))

    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
