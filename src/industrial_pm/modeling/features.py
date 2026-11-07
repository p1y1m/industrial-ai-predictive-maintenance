from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from industrial_pm.data.constants import SENSOR_COLUMNS, SETTING_COLUMNS


@dataclass
class FeatureBuilder:
    """Leakage-safe causal feature engineering fitted on development engines only."""

    rolling_windows: tuple[int, ...] = (5, 15)
    variance_threshold: float = 1e-12
    base_columns_: list[str] = field(default_factory=list, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)

    def fit(self, frame: pd.DataFrame) -> FeatureBuilder:
        candidates = SETTING_COLUMNS + SENSOR_COLUMNS
        variance = frame[candidates].var(ddof=0)
        self.base_columns_ = [
            column for column in candidates if float(variance[column]) > self.variance_threshold
        ]
        if not self.base_columns_:
            raise ValueError("No nonconstant model features found")
        self.feature_columns_ = list(self.base_columns_)
        for window in self.rolling_windows:
            self.feature_columns_.extend(
                f"{column}_mean_{window}" for column in self.base_columns_
            )
            self.feature_columns_.extend(
                f"{column}_trend_{window}" for column in self.base_columns_
            )
        self.feature_columns_.append("cycle_scaled")
        return self

    def _transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.base_columns_:
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        output = frame[self.base_columns_].copy()
        if frame.empty:
            for window in self.rolling_windows:
                for column in self.base_columns_:
                    output[f"{column}_mean_{window}"] = pd.Series(dtype=float)
                for column in self.base_columns_:
                    output[f"{column}_trend_{window}"] = pd.Series(dtype=float)
            output["cycle_scaled"] = pd.Series(dtype=float)
            return output

        grouped = frame.groupby("unit_id", sort=False)
        for window in self.rolling_windows:
            rolling_mean = grouped[self.base_columns_].rolling(window, min_periods=1).mean()
            rolling_mean = rolling_mean.reset_index(level=0, drop=True)
            lagged = grouped[self.base_columns_].shift(window - 1)
            trend = (frame[self.base_columns_] - lagged) / max(1, window - 1)
            trend = trend.fillna(0.0)
            output = output.join(rolling_mean.add_suffix(f"_mean_{window}"))
            output = output.join(trend.add_suffix(f"_trend_{window}"))
        # Fixed scaling is causal; normalizing by an engine's eventual lifetime leaks the future.
        output["cycle_scaled"] = frame["cycle"].clip(upper=400) / 400.0
        return output.astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = self._transform_features(frame)
        if self.feature_columns_ and list(features.columns) != self.feature_columns_:
            raise AssertionError("Feature order changed between fit and transform")
        return features

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


def final_cycle_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["cycle"].eq(frame.groupby("unit_id")["cycle"].transform("max"))


def pseudo_test_endpoint_mask(
    frame: pd.DataFrame,
    remaining_life_cutoffs: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120),
) -> pd.Series:
    """Select causal truncation points that simulate official-test endpoints.

    The cutoffs are fixed before evaluation and use only run-to-failure development engines.
    Each engine contributes several horizons so model selection cannot collapse to predicting zero.
    """

    selected = pd.Series(False, index=frame.index)
    for _, engine in frame.groupby("unit_id", sort=False):
        lifetime = int(engine["cycle"].max())
        cycles = {lifetime - cutoff for cutoff in remaining_life_cutoffs if lifetime - cutoff >= 1}
        selected.loc[engine.index] = engine["cycle"].isin(cycles)
    if not selected.any():
        raise ValueError("No pseudo-test endpoints were available")
    return selected
