from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from industrial_pm.modeling.metrics import expected_calibration_error


@dataclass
class CalibratedHorizonRisk:
    horizon: int
    seed: int = 42

    def __post_init__(self) -> None:
        self.classifier = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=self.seed,
        )
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(
        self,
        x_train: np.ndarray,
        rul_train: np.ndarray,
        x_calibration: np.ndarray,
        rul_calibration: np.ndarray,
    ) -> CalibratedHorizonRisk:
        y_train = (np.asarray(rul_train) <= self.horizon).astype(int)
        y_calibration = (np.asarray(rul_calibration) <= self.horizon).astype(int)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_calibration)) < 2:
            raise ValueError(f"Risk horizon {self.horizon} needs both classes")
        self.classifier.fit(x_train, y_train)
        raw = self.classifier.predict_proba(x_calibration)[:, 1]
        self.calibrator.fit(raw, y_calibration)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raw = self.classifier.predict_proba(features)[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def evaluate(self, features: np.ndarray, rul: np.ndarray) -> dict[str, float]:
        truth = (np.asarray(rul) <= self.horizon).astype(int)
        proba = self.predict_proba(features)
        auc = float(roc_auc_score(truth, proba)) if len(np.unique(truth)) > 1 else float("nan")
        return {
            "brier": float(brier_score_loss(truth, proba)),
            "roc_auc": auc,
            "ece": expected_calibration_error(truth, proba),
            "positive_rate": float(truth.mean()),
        }
