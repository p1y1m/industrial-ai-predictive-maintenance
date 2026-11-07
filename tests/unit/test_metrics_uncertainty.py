from __future__ import annotations

import numpy as np

from industrial_pm.modeling.metrics import nasa_asymmetric_score, regression_metrics
from industrial_pm.modeling.uncertainty import ConformalInterval


def test_nasa_score_penalizes_late_predictions_more() -> None:
    truth = np.array([50.0])
    early = nasa_asymmetric_score(truth, np.array([40.0]))
    late = nasa_asymmetric_score(truth, np.array([60.0]))
    assert late > early > 0


def test_regression_metrics_are_exact_for_perfect_predictions() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    metrics = regression_metrics(truth, truth)
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["r2"] == 1
    assert metrics["nasa_score"] == 0


def test_conformal_interval_uses_finite_sample_quantile() -> None:
    truth = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    prediction = np.array([8.0, 18.0, 27.0, 43.0, 54.0])
    interval = ConformalInterval.fit(truth, prediction, coverage=0.8, upper_bound=None)
    lower, upper = interval.predict(np.array([25.0]))
    assert interval.residual_quantile == 4.0
    assert lower[0] == 21.0
    assert upper[0] == 29.0
