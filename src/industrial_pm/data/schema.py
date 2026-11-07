from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from industrial_pm.data.constants import CMAPSS_COLUMNS


class DataValidationError(ValueError):
    """Raised when raw C-MAPSS data violates an invariant."""


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    engines: int
    min_cycles: int
    max_cycles: int
    duplicate_rows: int
    missing_values: int
    non_finite_values: int
    constant_columns: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_cmapss_table(path: Path) -> pd.DataFrame:
    """Read one whitespace-delimited C-MAPSS trajectory table."""

    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS, engine="python")
    return frame


def validate_cmapss_frame(frame: pd.DataFrame) -> DataQualityReport:
    """Validate structural, temporal, and numeric invariants."""

    if list(frame.columns) != CMAPSS_COLUMNS:
        raise DataValidationError(
            f"Expected {len(CMAPSS_COLUMNS)} ordered columns, got {list(frame.columns)}"
        )
    if frame.empty:
        raise DataValidationError("C-MAPSS frame is empty")

    numeric = frame.to_numpy(dtype=float)
    missing = int(frame.isna().sum().sum())
    non_finite = int((~np.isfinite(numeric)).sum())
    duplicates = int(frame.duplicated(subset=["unit_id", "cycle"]).sum())
    if missing:
        raise DataValidationError(f"Found {missing} missing values")
    if non_finite:
        raise DataValidationError(f"Found {non_finite} non-finite values")
    if duplicates:
        raise DataValidationError(f"Found {duplicates} duplicate engine-cycle rows")
    if (frame[["unit_id", "cycle"]].to_numpy() <= 0).any():
        raise DataValidationError("unit_id and cycle must be positive")
    if not np.allclose(frame["unit_id"], frame["unit_id"].astype(int)):
        raise DataValidationError("unit_id values must be integers")
    if not np.allclose(frame["cycle"], frame["cycle"].astype(int)):
        raise DataValidationError("cycle values must be integers")

    warnings: list[str] = []
    for unit_id, engine in frame.groupby("unit_id", sort=False):
        cycles = engine["cycle"].to_numpy(dtype=int)
        expected = np.arange(1, len(cycles) + 1)
        if not np.array_equal(cycles, expected):
            raise DataValidationError(
                f"Engine {int(cast(float, unit_id))} cycles must be ordered and contiguous from 1"
            )

    cycle_counts = frame.groupby("unit_id")["cycle"].max()
    constant_columns = [
        column for column in frame.columns if frame[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        warnings.append(
            "Constant columns are expected in some subsets and must be learned from training only: "
            + ", ".join(constant_columns)
        )

    return DataQualityReport(
        rows=len(frame),
        engines=int(frame["unit_id"].nunique()),
        min_cycles=int(cycle_counts.min()),
        max_cycles=int(cycle_counts.max()),
        duplicate_rows=duplicates,
        missing_values=missing,
        non_finite_values=non_finite,
        constant_columns=constant_columns,
        warnings=warnings,
    )


def add_train_rul(frame: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    result = frame.copy()
    maximum = result.groupby("unit_id")["cycle"].transform("max")
    result["rul_uncapped"] = maximum - result["cycle"]
    result["rul"] = result["rul_uncapped"].clip(upper=cap) if cap else result["rul_uncapped"]
    return result


def add_test_rul(
    frame: pd.DataFrame, final_rul_path: Path, cap: int | None = 125
) -> pd.DataFrame:
    final_rul = pd.read_csv(final_rul_path, sep=r"\s+", header=None, names=["final_rul"])
    engines = sorted(int(value) for value in frame["unit_id"].unique())
    if len(final_rul) != len(engines):
        raise DataValidationError(
            f"Expected {len(engines)} final RUL labels, found {len(final_rul)}"
        )
    mapping = dict(zip(engines, final_rul["final_rul"].astype(int), strict=True))
    result = frame.copy()
    max_observed = result.groupby("unit_id")["cycle"].transform("max")
    end_rul = result["unit_id"].map(mapping)
    result["rul_uncapped"] = max_observed - result["cycle"] + end_rul
    result["rul"] = result["rul_uncapped"].clip(upper=cap) if cap else result["rul_uncapped"]
    return result
