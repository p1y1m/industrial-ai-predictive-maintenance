from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def nasa_asymmetric_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 asymmetric score: late predictions are penalized more heavily."""

    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalties = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    return float(penalties.sum())


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)),
        "nasa_score": nasa_asymmetric_score(truth, prediction),
        "bias": float(np.mean(prediction - truth)),
    }


def expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    truth = np.asarray(y_true, dtype=float)
    proba = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    result = 0.0
    for index in range(bins):
        right_closed = index == bins - 1
        mask = (proba >= edges[index]) & (
            (proba <= edges[index + 1]) if right_closed else (proba < edges[index + 1])
        )
        if mask.any():
            result += float(mask.mean()) * abs(float(truth[mask].mean() - proba[mask].mean()))
    return result if total else float("nan")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
