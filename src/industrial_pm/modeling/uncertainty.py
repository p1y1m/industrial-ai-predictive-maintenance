from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalInterval:
    coverage: float
    residual_quantile: float
    lower_bound: float = 0.0
    upper_bound: float | None = 125.0

    @classmethod
    def fit(
        cls,
        y_calibration: np.ndarray,
        prediction_calibration: np.ndarray,
        coverage: float = 0.9,
        lower_bound: float = 0.0,
        upper_bound: float | None = 125.0,
    ) -> ConformalInterval:
        if not 0 < coverage < 1:
            raise ValueError("coverage must be between 0 and 1")
        residuals = np.abs(np.asarray(y_calibration) - np.asarray(prediction_calibration))
        if not len(residuals):
            raise ValueError("Calibration data must not be empty")
        level = min(1.0, np.ceil((len(residuals) + 1) * coverage) / len(residuals))
        quantile = float(np.quantile(residuals, level, method="higher"))
        return cls(coverage, quantile, lower_bound, upper_bound)

    def predict(self, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(prediction, dtype=float)
        lower = np.maximum(self.lower_bound, values - self.residual_quantile)
        upper = values + self.residual_quantile
        if self.upper_bound is not None:
            upper = np.minimum(self.upper_bound, upper)
        return lower, upper

    def evaluate(
        self, y_true: np.ndarray, prediction: np.ndarray
    ) -> dict[str, float]:
        lower, upper = self.predict(prediction)
        truth = np.asarray(y_true, dtype=float)
        covered = (truth >= lower) & (truth <= upper)
        return {
            "target_coverage": self.coverage,
            "empirical_coverage": float(covered.mean()),
            "mean_interval_width": float(np.mean(upper - lower)),
            "residual_quantile": self.residual_quantile,
        }
