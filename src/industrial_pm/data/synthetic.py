from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_pm.data.constants import CMAPSS_COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS


def generate_synthetic_cmapss(
    engines: int = 24, min_cycles: int = 80, max_cycles: int = 180, seed: int = 42
) -> pd.DataFrame:
    """Generate deterministic, clearly synthetic trajectories for tests/fallback demos."""

    if engines < 1 or min_cycles < 10 or max_cycles < min_cycles:
        raise ValueError("Invalid synthetic dataset dimensions")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    direction = np.array(
        [0.0, 0.8, 1.3, 1.8, 0.0, 0.2, -1.2, 0.5, 1.2, 0.0, 0.9,
         -0.7, 0.4, 1.0, -0.6, 0.0, 0.7, 0.0, 0.0, -0.5, -0.8]
    )
    baseline = np.linspace(100.0, 500.0, len(SENSOR_COLUMNS))
    for unit_id in range(1, engines + 1):
        lifetime = int(rng.integers(min_cycles, max_cycles + 1))
        wear_offset = rng.normal(0, 0.03)
        unit_bias = rng.normal(0, 0.8, len(SENSOR_COLUMNS))
        for cycle in range(1, lifetime + 1):
            age = cycle / lifetime
            degradation = max(0.0, (age - 0.25 + wear_offset) / 0.75) ** 1.7
            row: dict[str, float | int] = {"unit_id": unit_id, "cycle": cycle}
            row.update(
                {
                    SETTING_COLUMNS[0]: float(rng.normal(0, 0.004)),
                    SETTING_COLUMNS[1]: float(rng.normal(0, 0.0003)),
                    SETTING_COLUMNS[2]: 100.0,
                }
            )
            seasonal = np.sin(cycle / 11.0) * 0.08
            values = baseline + unit_bias + direction * degradation + seasonal
            values += rng.normal(0, 0.12, len(SENSOR_COLUMNS))
            row.update(dict(zip(SENSOR_COLUMNS, values.astype(float), strict=True)))
            rows.append(row)
    return pd.DataFrame(rows, columns=CMAPSS_COLUMNS)
