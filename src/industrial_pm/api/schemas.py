from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    items: list[T]
    page: int
    page_size: int
    total: int
    pages: int


class FleetAsset(BaseModel):
    unit_id: int
    predicted_rul: float
    actual_rul: float | None = None
    interval_lower: float
    interval_upper: float
    risk_15: float = Field(ge=0, le=1)
    risk_30: float = Field(ge=0, le=1)
    risk_60: float = Field(ge=0, le=1)
    anomaly_score: float
    is_anomaly: bool
    health_state: str
    data_source: str = "NASA C-MAPSS FD001 official test"


class FleetSummary(BaseModel):
    total_assets: int
    critical_assets: int
    elevated_assets: int
    anomalous_assets: int
    median_predicted_rul: float
    model_run_id: str
    selected_model: str


class TelemetryPoint(BaseModel):
    cycle: int
    actual_rul: float
    sensors: dict[str, float]
    source: str = "measured_simulator_output"


class EngineDetail(FleetAsset):
    absolute_error: float | None = None
    model_run_id: str
    top_explanations: list[dict[str, Any]]


class ModelMetric(BaseModel):
    model: str
    validation_rmse: float
    test_rmse_capped: float
    test_rmse_uncapped: float
    test_mae_capped: float
    nasa_score_uncapped: float
    selected: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    database: str


class RetrainingRequest(BaseModel):
    dry_run: bool = True
    reason: str = Field(min_length=5, max_length=500)


class RetrainingResponse(BaseModel):
    status: str
    dry_run: bool
    safeguards: dict[str, bool]
    command: str | None = None
