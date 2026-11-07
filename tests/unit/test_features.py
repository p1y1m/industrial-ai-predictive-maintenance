from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_pm.data.schema import add_train_rul
from industrial_pm.data.synthetic import generate_synthetic_cmapss
from industrial_pm.modeling.features import FeatureBuilder, final_cycle_mask


def test_causal_features_do_not_change_when_future_changes() -> None:
    frame = add_train_rul(generate_synthetic_cmapss(engines=5, min_cycles=30, max_cycles=30))
    builder = FeatureBuilder((5, 10)).fit(frame)
    before = builder.transform(frame)
    modified = frame.copy()
    future_index = modified.index[(modified.unit_id == 1) & (modified.cycle == 30)][0]
    modified.loc[future_index, "sensor_2"] += 10_000
    after = builder.transform(modified)
    past_mask = (frame.unit_id == 1) & (frame.cycle < 30)
    pd.testing.assert_frame_equal(before.loc[past_mask], after.loc[past_mask])


def test_fit_drops_constant_features_and_preserves_order() -> None:
    frame = add_train_rul(generate_synthetic_cmapss(engines=5, min_cycles=20, max_cycles=20))
    builder = FeatureBuilder((5,)).fit(frame)
    transformed = builder.transform(frame)
    assert "setting_3" not in builder.base_columns_
    assert list(transformed.columns) == builder.feature_columns_
    assert np.isfinite(transformed.to_numpy()).all()
    assert int(final_cycle_mask(frame).sum()) == 5
