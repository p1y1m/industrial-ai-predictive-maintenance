from __future__ import annotations

import numpy as np

from industrial_pm.monitoring.drift import drift_level, population_stability_index


def test_population_stability_index_detects_shift() -> None:
    reference = np.linspace(0, 1, 1000)
    stable = population_stability_index(reference, reference.copy())
    shifted = population_stability_index(reference, reference + 0.7)
    assert stable == 0
    assert shifted > 0.25
    assert drift_level(shifted) == "critical"
