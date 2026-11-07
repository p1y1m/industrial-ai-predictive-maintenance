from __future__ import annotations

import numpy as np
import pytest

from industrial_pm.modeling.anomaly import HealthyIsolationForest
from industrial_pm.modeling.risk import CalibratedHorizonRisk


def test_healthy_isolation_forest_requires_fit_and_flags_outlier() -> None:
    rng = np.random.default_rng(42)
    healthy = rng.normal(0.0, 0.25, size=(120, 3))
    detector = HealthyIsolationForest(seed=42)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        detector.is_anomaly(healthy[:1])

    detector.fit(healthy)
    scores = detector.score(np.vstack([healthy[:1], np.full((1, 3), 8.0)]))
    flags = detector.is_anomaly(np.vstack([healthy[:1], np.full((1, 3), 8.0)]))
    assert scores[1] > scores[0]
    assert bool(flags[1]) is True


def test_calibrated_horizon_risk_fit_predict_and_evaluate() -> None:
    rng = np.random.default_rng(7)
    train_rul = np.linspace(1, 80, 160)
    calibration_rul = np.linspace(1, 80, 80)
    x_train = np.column_stack(
        [train_rul + rng.normal(0, 2, len(train_rul)), rng.normal(size=len(train_rul))]
    )
    x_calibration = np.column_stack(
        [
            calibration_rul + rng.normal(0, 2, len(calibration_rul)),
            rng.normal(size=len(calibration_rul)),
        ]
    )

    model = CalibratedHorizonRisk(horizon=30, seed=42).fit(
        x_train, train_rul, x_calibration, calibration_rul
    )
    probabilities = model.predict_proba(x_calibration)
    metrics = model.evaluate(x_calibration, calibration_rul)

    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert metrics["brier"] < 0.15
    assert metrics["roc_auc"] > 0.9
    assert 0 < metrics["positive_rate"] < 1


def test_calibrated_horizon_risk_rejects_single_class() -> None:
    features = np.ones((50, 2))
    with pytest.raises(ValueError, match="needs both classes"):
        CalibratedHorizonRisk(horizon=30).fit(
            features,
            np.full(50, 100.0),
            features,
            np.full(50, 100.0),
        )
