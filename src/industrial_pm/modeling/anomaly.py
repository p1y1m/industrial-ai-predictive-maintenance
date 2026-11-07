from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class HealthyIsolationForest:
    """Isolation Forest fitted only on early-life, healthy development samples."""

    def __init__(self, seed: int = 42) -> None:
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            max_samples="auto",
            random_state=seed,
            n_jobs=-1,
        )
        self.threshold_: float | None = None

    def fit(self, features: np.ndarray) -> HealthyIsolationForest:
        self.model.fit(features)
        scores = -self.model.score_samples(features)
        self.threshold_ = float(np.quantile(scores, 0.95))
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(-self.model.score_samples(features), dtype=float)

    def is_anomaly(self, features: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Anomaly detector has not been fitted")
        return self.score(features) >= self.threshold_
