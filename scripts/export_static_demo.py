"""Export a deterministic backend-free payload from verified artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from industrial_pm.api.artifacts import ArtifactRepository


def build_payload(repository: ArtifactRepository) -> dict[str, Any]:
    assets = repository.fleet_page(page=1, page_size=100, state=None)["items"]
    states = [asset["health_state"] for asset in assets]
    predictions = sorted(float(asset["predicted_rul"]) for asset in assets)
    middle = len(predictions) // 2
    selected = repository.metrics["selection"]["selected_model"]
    models = [
        {
            "model": name,
            "validation_rmse": repository.metrics["validation"][name]["rmse"],
            "test_rmse_capped": values["rmse"],
            "test_rmse_uncapped": values["rmse_uncapped"],
            "test_mae_capped": values["mae"],
            "nasa_score_uncapped": values["nasa_score_uncapped"],
            "selected": name == selected,
        }
        for name, values in repository.metrics["test"].items()
    ]
    return {
        "metadata": {
            "mode": "offline_static_demo",
            "source": "Precomputed official NASA C-MAPSS FD001 benchmark artifacts",
            "run_id": repository.metrics["run_id"],
            "generated_from_artifacts": True,
            "not_live_telemetry": True,
        },
        "summary": {
            "total_assets": len(assets),
            "critical_assets": states.count("critical"),
            "elevated_assets": states.count("elevated"),
            "anomalous_assets": sum(bool(asset["is_anomaly"]) for asset in assets),
            "median_predicted_rul": (predictions[middle - 1] + predictions[middle]) / 2,
            "model_run_id": repository.metrics["run_id"],
            "selected_model": selected,
        },
        "fleet": {"items": assets, "total": len(assets)},
        "models": models,
        "risk_metrics": repository.metrics["risk"],
        "uncertainty": repository.metrics["uncertainty"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("frontend/public/data/demo.json"))
    args = parser.parse_args()
    payload = build_payload(ArtifactRepository())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(payload['fleet']['items'])} verified assets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
