from __future__ import annotations

import math
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import text
from sqlalchemy.orm import Session

from industrial_pm import __version__
from industrial_pm.api.artifacts import ArtifactRepository, get_artifacts
from industrial_pm.api.database import RetrainingAudit, create_tables, get_session
from industrial_pm.api.schemas import (
    EngineDetail,
    FleetAsset,
    FleetSummary,
    HealthResponse,
    ModelMetric,
    Page,
    RetrainingRequest,
    RetrainingResponse,
    TelemetryPoint,
)
from industrial_pm.config import get_settings

settings = get_settings()
REQUESTS = Counter("industrial_pm_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("industrial_pm_http_duration_seconds", "HTTP latency", ["path"])


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    create_tables()
    yield


app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Fleet RUL, risk, anomaly, explanation, and model evidence API.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", settings.api_key_header, "X-Request-ID"],
)


@app.middleware("http")
async def observability(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        detail = "Unexpected server error" if settings.app_env == "production" else str(exc)
        response = JSONResponse(
            status_code=500,
            content={"error": "internal_error", "detail": detail, "request_id": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    REQUESTS.labels(request.method, request.url.path, response.status_code).inc()
    LATENCY.labels(request.url.path).observe(time.perf_counter() - started)
    return response


async def artifact_dependency() -> ArtifactRepository:
    """Resolve cached artifacts without dispatching through AnyIO's thread pool."""

    return get_artifacts()


async def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
async def liveness() -> HealthResponse:
    try:
        get_artifacts()
        loaded = True
    except FileNotFoundError:
        loaded = False
    return HealthResponse(status="ok", version=__version__, model_loaded=loaded, database="unchecked")


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def readiness(session: Annotated[Session, Depends(get_session)]) -> HealthResponse:
    session.execute(text("SELECT 1"))
    get_artifacts()
    return HealthResponse(status="ready", version=__version__, model_loaded=True, database="ready")


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/fleet/summary", response_model=FleetSummary, tags=["fleet"])
async def fleet_summary(
    repo: Annotated[ArtifactRepository, Depends(artifact_dependency)],
) -> FleetSummary:
    states = repo.predictions.apply(repo.state, axis=1)
    return FleetSummary(
        total_assets=len(states),
        critical_assets=int((states == "critical").sum()),
        elevated_assets=int((states == "elevated").sum()),
        anomalous_assets=int(repo.predictions["is_anomaly"].sum()),
        median_predicted_rul=float(repo.predictions["prediction_selected"].median()),
        model_run_id=repo.metrics["run_id"],
        selected_model=repo.metrics["selection"]["selected_model"],
    )


@app.get("/api/v1/fleet", response_model=Page[FleetAsset], tags=["fleet"])
async def fleet(
    repo: Annotated[ArtifactRepository, Depends(artifact_dependency)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
    health_state: Annotated[str | None, Query(pattern="^(critical|elevated|nominal)$")] = None,
) -> dict[str, Any]:
    return repo.fleet_page(page, page_size, health_state)


@app.get("/api/v1/engines/{unit_id}", response_model=EngineDetail, tags=["engines"])
async def engine_detail(
    unit_id: int,
    repo: Annotated[ArtifactRepository, Depends(artifact_dependency)],
) -> dict[str, Any]:
    matches = repo.predictions.loc[repo.predictions["unit_id"] == unit_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Engine not found")
    row = matches.iloc[0]
    asset = repo.asset(row)
    asset.update(
        {
            "absolute_error": float(row["absolute_error"]),
            "model_run_id": repo.metrics["run_id"],
            "top_explanations": repo.importance.head(8).to_dict(orient="records"),
        }
    )
    return asset


@app.get("/api/v1/engines/{unit_id}/telemetry", response_model=Page[TelemetryPoint], tags=["engines"])
async def telemetry(
    unit_id: int,
    repo: Annotated[ArtifactRepository, Depends(artifact_dependency)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    rows = repo.telemetry.loc[repo.telemetry["unit_id"] == unit_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Engine not found")
    start = (page - 1) * page_size
    items = [
        {
            "cycle": int(row["cycle"]),
            "actual_rul": float(row["rul_uncapped"]),
            "sensors": {f"sensor_{index}": float(row[f"sensor_{index}"]) for index in range(1, 22)},
        }
        for _, row in rows.iloc[start : start + page_size].iterrows()
    ]
    return {"items": items, "page": page, "page_size": page_size, "total": len(rows), "pages": math.ceil(len(rows) / page_size)}


@app.get("/api/v1/models", response_model=list[ModelMetric], tags=["models"])
async def models(
    repo: Annotated[ArtifactRepository, Depends(artifact_dependency)],
) -> list[ModelMetric]:
    selected = repo.metrics["selection"]["selected_model"]
    return [
        ModelMetric(
            model=name,
            validation_rmse=repo.metrics["validation"][name]["rmse"],
            test_rmse_capped=values["rmse"],
            test_rmse_uncapped=values["rmse_uncapped"],
            test_mae_capped=values["mae"],
            nasa_score_uncapped=values["nasa_score_uncapped"],
            selected=name == selected,
        )
        for name, values in repo.metrics["test"].items()
    ]


@app.post("/api/v1/operations/retrain", response_model=RetrainingResponse, tags=["operations"], dependencies=[Depends(require_api_key)])
async def retrain(
    request: RetrainingRequest,
    session: Annotated[Session, Depends(get_session)],
) -> RetrainingResponse:
    safeguards = {"dry_run_required_via_api": request.dry_run, "manual_promotion_required": True, "test_isolation_required": True}
    if not request.dry_run:
        raise HTTPException(status_code=409, detail="API retraining is dry-run only; use the audited CLI pipeline")
    session.add(RetrainingAudit(status="dry_run", reason=request.reason))
    session.commit()
    return RetrainingResponse(status="validated", dry_run=True, safeguards=safeguards, command="make data && make train")
