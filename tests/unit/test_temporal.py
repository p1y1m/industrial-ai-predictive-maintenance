from __future__ import annotations

import numpy as np

from industrial_pm.data.schema import add_train_rul
from industrial_pm.data.synthetic import generate_synthetic_cmapss
from industrial_pm.modeling.temporal import build_sequences


def test_sequences_are_left_padded_and_do_not_cross_engines() -> None:
    frame = add_train_rul(generate_synthetic_cmapss(engines=5, min_cycles=12, max_cycles=12))
    feature = np.column_stack([frame.unit_id.to_numpy(), frame.cycle.to_numpy()]).astype(np.float32)
    batch = build_sequences(frame, feature, sequence_length=5, final_only=False)
    first = batch.features[0]
    assert np.all(first[:, 0] == 1)
    assert np.all(first[:, 1] == 1)
    assert np.all(batch.features[:, :, 0] == batch.unit_ids[:, None])


def test_final_only_returns_one_sequence_per_engine() -> None:
    frame = add_train_rul(generate_synthetic_cmapss(engines=6, min_cycles=12, max_cycles=15))
    features = frame[["sensor_2", "sensor_3"]].to_numpy(dtype=np.float32)
    batch = build_sequences(frame, features, sequence_length=8, final_only=True)
    assert len(batch.targets) == 6
    assert np.all(batch.targets == 0)
