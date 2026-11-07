from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineSplit:
    train_engine_ids: tuple[int, ...]
    validation_engine_ids: tuple[int, ...]

    def assert_disjoint(self) -> None:
        overlap = set(self.train_engine_ids) & set(self.validation_engine_ids)
        if overlap:
            raise AssertionError(f"Engine leakage detected: {sorted(overlap)}")


def split_development_engines(
    frame: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42
) -> EngineSplit:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    engines = np.array(sorted(frame["unit_id"].astype(int).unique()))
    if len(engines) < 5:
        raise ValueError("At least five engines are required for a stable development split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(engines)
    validation_count = max(1, round(len(engines) * validation_fraction))
    split = EngineSplit(
        train_engine_ids=tuple(sorted(int(value) for value in shuffled[validation_count:])),
        validation_engine_ids=tuple(sorted(int(value) for value in shuffled[:validation_count])),
    )
    split.assert_disjoint()
    return split


def select_engines(frame: pd.DataFrame, engine_ids: tuple[int, ...]) -> pd.DataFrame:
    return frame.loc[frame["unit_id"].isin(engine_ids)].copy()
