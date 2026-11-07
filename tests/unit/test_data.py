from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from industrial_pm.data.ingest import safe_extract_archive
from industrial_pm.data.schema import DataValidationError, add_train_rul, validate_cmapss_frame
from industrial_pm.data.split import select_engines, split_development_engines
from industrial_pm.data.synthetic import generate_synthetic_cmapss


def test_synthetic_data_is_deterministic_and_schema_valid() -> None:
    first = generate_synthetic_cmapss(engines=8, min_cycles=20, max_cycles=30, seed=7)
    second = generate_synthetic_cmapss(engines=8, min_cycles=20, max_cycles=30, seed=7)
    pd.testing.assert_frame_equal(first, second)
    report = validate_cmapss_frame(first)
    assert report.engines == 8
    assert report.missing_values == 0


def test_rul_counts_down_to_zero_and_caps() -> None:
    frame = generate_synthetic_cmapss(engines=5, min_cycles=20, max_cycles=20)
    labeled = add_train_rul(frame, cap=12)
    first = labeled.loc[labeled.unit_id == 1]
    assert first.iloc[0].rul == 12
    assert first.iloc[-1].rul == 0
    assert np.array_equal(first.rul_uncapped.to_numpy(), np.arange(19, -1, -1))


def test_duplicate_engine_cycle_fails_validation() -> None:
    frame = generate_synthetic_cmapss(engines=1, min_cycles=20, max_cycles=20)
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_cmapss_frame(duplicate)


def test_engine_split_is_deterministic_and_disjoint() -> None:
    frame = generate_synthetic_cmapss(engines=20, min_cycles=20, max_cycles=20)
    first = split_development_engines(frame, validation_fraction=0.25, seed=8)
    second = split_development_engines(frame, validation_fraction=0.25, seed=8)
    assert first == second
    assert len(first.validation_engine_ids) == 5
    assert set(first.train_engine_ids).isdisjoint(first.validation_engine_ids)
    selected = select_engines(frame, first.validation_engine_ids)
    assert set(selected.unit_id.unique()) == set(first.validation_engine_ids)


def test_safe_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract_archive(archive, tmp_path / "output")
