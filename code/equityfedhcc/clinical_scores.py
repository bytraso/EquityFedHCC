from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fib4(
    age: ArrayLike,
    ast: ArrayLike,
    alt: ArrayLike,
    platelets: ArrayLike,
) -> NDArray[np.float64]:
    age_values = np.asarray(age, dtype=np.float64)
    ast_values = np.asarray(ast, dtype=np.float64)
    alt_values = np.asarray(alt, dtype=np.float64)
    platelet_values = np.asarray(platelets, dtype=np.float64)
    denominator = platelet_values * np.sqrt(np.maximum(alt_values, 1e-12))
    return age_values * ast_values / np.maximum(denominator, 1e-12)


def amap(
    age: ArrayLike,
    sex_male: ArrayLike,
    albumin: ArrayLike,
    bilirubin: ArrayLike,
    platelets: ArrayLike,
) -> NDArray[np.float64]:
    age_values = np.asarray(age, dtype=np.float64)
    male_values = np.asarray(sex_male, dtype=np.float64)
    albumin_values = np.asarray(albumin, dtype=np.float64)
    bilirubin_values = np.asarray(bilirubin, dtype=np.float64)
    platelet_values = np.asarray(platelets, dtype=np.float64)
    bilirubin_term = 0.66 * np.log10(np.maximum(bilirubin_values, 1e-12))
    score = (
        0.06 * age_values
        + 0.89 * male_values
        - 0.11 * albumin_values
        + bilirubin_term
        - 0.01 * platelet_values
        + 7.4
    )
    return np.clip((score - 0) / 15 * 100, 0, 100)


def galad_proxy(
    age: ArrayLike,
    sex_male: ArrayLike,
    afp: ArrayLike,
) -> NDArray[np.float64]:
    age_values = np.asarray(age, dtype=np.float64)
    male_values = np.asarray(sex_male, dtype=np.float64)
    afp_values = np.asarray(afp, dtype=np.float64)
    linear = -10.08 + 0.09 * age_values + 1.67 * male_values + 2.34 * np.log10(afp_values + 1)
    return 1 / (1 + np.exp(-linear))
