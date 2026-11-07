from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from industrial_pm.api import artifacts as artifact_module
from industrial_pm.api.artifacts import ArtifactRepository


def test_repository_resolves_portable_run_id_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "portable-run"
    registry_root = tmp_path / "registry"
    artifact_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    bundle = artifact_root / run_id
    telemetry_dir = data_root / "processed" / "fd001"
    registry_root.mkdir()
    bundle.mkdir(parents=True)
    telemetry_dir.mkdir(parents=True)

    (registry_root / "champion.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "artifact_path": "/a/host-only/path/that/does/not/exist",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "metrics.json").write_text(
        json.dumps({"run_id": run_id, "dataset": {"subset": "FD001"}}), encoding="utf-8"
    )
    pd.DataFrame([{"unit_id": 1}]).to_csv(bundle / "test_predictions.csv", index=False)
    pd.DataFrame([{"feature": "sensor_1", "importance_mean": 1.0}]).to_csv(
        bundle / "feature_importance.csv", index=False
    )
    pd.DataFrame([{"unit_id": 1, "cycle": 1}]).to_csv(
        telemetry_dir / "test.csv.gz", index=False, compression="gzip"
    )
    settings = SimpleNamespace(
        model_registry_root=registry_root,
        artifact_root=artifact_root,
        data_root=data_root,
    )
    monkeypatch.setattr(artifact_module, "get_settings", lambda: settings)

    repository = ArtifactRepository()

    assert repository.artifact_dir == bundle
    assert repository.metrics["run_id"] == run_id
