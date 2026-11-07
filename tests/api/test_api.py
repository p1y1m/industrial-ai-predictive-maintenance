from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from industrial_pm.api import main as api_main
from industrial_pm.api.artifacts import ArtifactRepository

app = api_main.app


@pytest.fixture(autouse=True)
def deterministic_artifact_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep API contracts hermetic; real artifacts are covered by the Compose smoke test."""

    repo = ArtifactRepository.__new__(ArtifactRepository)
    units = np.arange(1, 101)
    predictions = np.linspace(10, 120, 100)
    repo.predictions = pd.DataFrame(
        {
            "unit_id": units,
            "prediction_selected": predictions,
            "actual_rul_uncapped": predictions + 2,
            "interval_lower": np.maximum(0, predictions - 10),
            "interval_upper": predictions + 10,
            "risk_15_cycles": np.clip(1 - predictions / 80, 0, 1),
            "risk_30_cycles": np.clip(1 - predictions / 100, 0, 1),
            "risk_60_cycles": np.clip(1 - predictions / 140, 0, 1),
            "anomaly_score": np.linspace(0.9, 0.1, 100),
            "is_anomaly": units <= 8,
            "absolute_error": np.full(100, 2.0),
        }
    )
    repo.importance = pd.DataFrame(
        [{"feature": "sensor_11__rolling_mean_5", "importance_mean": 0.42}]
    )
    repo.telemetry = pd.DataFrame(
        [
            {
                "unit_id": unit,
                "cycle": cycle,
                "rul_uncapped": float(20 - cycle),
                **{f"sensor_{index}": float(index + cycle) for index in range(1, 22)},
            }
            for unit in units
            for cycle in range(1, 6)
        ]
    )
    repo.metrics = {
        "run_id": "fd001-test-fixture",
        "selection": {"selected_model": "lstm"},
        "validation": {"lstm": {"rmse": 15.4}},
        "test": {
            "lstm": {
                "rmse": 15.1,
                "rmse_uncapped": 16.0,
                "mae": 10.7,
                "nasa_score_uncapped": 602.1,
            }
        },
    }
    monkeypatch.setattr(api_main, "get_artifacts", lambda: repo)


@pytest.mark.asyncio
async def test_health_fleet_engine_and_models_contracts() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            ready = await client.get("/health/ready")
            assert ready.status_code == 200
            assert ready.json()["model_loaded"] is True
            assert (await client.get("/health/live")).status_code == 200
            metrics_response = await client.get("/metrics")
            assert metrics_response.status_code == 200
            assert "industrial_pm_http_requests_total" in metrics_response.text

            summary = await client.get("/api/v1/fleet/summary")
            assert summary.status_code == 200
            assert summary.json()["total_assets"] == 100
            assert summary.json()["selected_model"] == "lstm"

            fleet = await client.get("/api/v1/fleet", params={"page": 1, "page_size": 7})
            assert fleet.status_code == 200
            assert len(fleet.json()["items"]) == 7
            assert fleet.json()["total"] == 100

            unit_id = fleet.json()["items"][0]["unit_id"]
            detail = await client.get(f"/api/v1/engines/{unit_id}")
            assert detail.status_code == 200
            assert detail.json()["data_source"].startswith("NASA C-MAPSS")
            assert detail.json()["top_explanations"]

            telemetry = await client.get(
                f"/api/v1/engines/{unit_id}/telemetry", params={"page_size": 5}
            )
            assert telemetry.status_code == 200
            assert len(telemetry.json()["items"]) == 5
            assert len(telemetry.json()["items"][0]["sensors"]) == 21

            critical = await client.get(
                "/api/v1/fleet", params={"health_state": "critical", "page_size": 100}
            )
            assert critical.status_code == 200
            assert all(item["health_state"] == "critical" for item in critical.json()["items"])

            models = await client.get("/api/v1/models")
            assert models.status_code == 200
            assert sum(item["selected"] for item in models.json()) == 1


@pytest.mark.asyncio
async def test_errors_and_retraining_safeguard() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/v1/engines/9999")).status_code == 404
            assert (await client.get("/api/v1/engines/9999/telemetry")).status_code == 404
            invalid = await client.get("/api/v1/fleet", params={"page_size": 1000})
            assert invalid.status_code == 422
            blocked = await client.post(
                "/api/v1/operations/retrain",
                json={"dry_run": False, "reason": "requested test retraining"},
            )
            assert blocked.status_code == 409
            dry_run = await client.post(
                "/api/v1/operations/retrain",
                json={"dry_run": True, "reason": "detected monitored drift"},
            )
            assert dry_run.status_code == 200
            assert dry_run.json()["command"] == "make data && make train"
