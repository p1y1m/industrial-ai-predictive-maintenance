from __future__ import annotations

import numpy as np


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    reference_hist, _ = np.histogram(reference, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)
    reference_rate = np.clip(reference_hist / max(1, reference_hist.sum()), 1e-6, None)
    current_rate = np.clip(current_hist / max(1, current_hist.sum()), 1e-6, None)
    return float(np.sum((current_rate - reference_rate) * np.log(current_rate / reference_rate)))


def drift_level(psi: float) -> str:
    return "critical" if psi >= 0.25 else "warning" if psi >= 0.1 else "stable"
