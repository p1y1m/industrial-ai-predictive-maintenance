from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_pm.config import get_settings


class ArtifactRepository:
    def __init__(self, registry_path: Path | None = None) -> None:
        settings = get_settings()
        registry = registry_path or settings.model_registry_root / "champion.json"
        if not registry.exists():
            raise FileNotFoundError("No champion model. Run `make data && make train` first.")
        self.registry = json.loads(registry.read_text(encoding="utf-8"))
        configured_path = Path(self.registry["artifact_path"])
        portable_path = settings.artifact_root / str(self.registry["run_id"])
        self.artifact_dir = configured_path if configured_path.exists() else portable_path
        if not self.artifact_dir.exists():
            raise FileNotFoundError(
                f"Champion artifacts not found at {configured_path} or {portable_path}"
            )
        self.metrics = json.loads((self.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        self.predictions = pd.read_csv(self.artifact_dir / "test_predictions.csv")
        self.importance = pd.read_csv(self.artifact_dir / "feature_importance.csv")
        subset = self.metrics["dataset"]["subset"].lower()
        self.telemetry = pd.read_csv(settings.data_root / "processed" / subset / "test.csv.gz")

    @staticmethod
    def state(row: pd.Series) -> str:
        if float(row["risk_30_cycles"]) >= 0.7 or float(row["prediction_selected"]) <= 20:
            return "critical"
        if float(row["risk_60_cycles"]) >= 0.55 or bool(row["is_anomaly"]):
            return "elevated"
        return "nominal"

    def asset(self, row: pd.Series) -> dict[str, Any]:
        return {
            "unit_id": int(row["unit_id"]),
            "predicted_rul": round(float(row["prediction_selected"]), 2),
            "actual_rul": round(float(row["actual_rul_uncapped"]), 2),
            "interval_lower": round(float(row["interval_lower"]), 2),
            "interval_upper": round(float(row["interval_upper"]), 2),
            "risk_15": float(row["risk_15_cycles"]),
            "risk_30": float(row["risk_30_cycles"]),
            "risk_60": float(row["risk_60_cycles"]),
            "anomaly_score": float(row["anomaly_score"]),
            "is_anomaly": bool(row["is_anomaly"]),
            "health_state": self.state(row),
        }

    def fleet_page(self, page: int, page_size: int, state: str | None) -> dict[str, Any]:
        records = self.predictions.copy()
        records["health_state"] = records.apply(self.state, axis=1)
        if state:
            records = records.loc[records["health_state"] == state]
        total = len(records)
        start = (page - 1) * page_size
        items = [self.asset(row) for _, row in records.iloc[start : start + page_size].iterrows()]
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": math.ceil(total / page_size) if total else 0,
        }


@lru_cache(maxsize=1)
def get_artifacts() -> ArtifactRepository:
    return ArtifactRepository()
