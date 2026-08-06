from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> NDArray[np.bool_]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    rejected = np.zeros(len(values), dtype=np.bool_)
    for rank, index in enumerate(order):
        threshold = alpha / (len(values) - rank)
        if values[index] <= threshold:
            rejected[index] = True
        else:
            break
    return rejected


def benjamini_hochberg(p_values: Sequence[float], q: float = 0.10) -> NDArray[np.bool_]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    thresholds = q * np.arange(1, len(values) + 1) / len(values)
    passing = np.flatnonzero(sorted_values <= thresholds)
    rejected = np.zeros(len(values), dtype=np.bool_)
    if passing.size:
        rejected[order[: passing[-1] + 1]] = True
    return rejected
