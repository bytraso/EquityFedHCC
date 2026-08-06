from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class CalibrationReport:
    brier: float
    expected_calibration_error: float
    slope: float
    intercept: float


def brier(labels: Sequence[float], scores: Sequence[float]) -> float:
    targets = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(scores, dtype=np.float64)
    return float(np.mean(np.square(probabilities - targets)))


def expected_calibration_error(
    labels: Sequence[float],
    scores: Sequence[float],
    bins: int = 10,
) -> float:
    targets = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(scores, dtype=np.float64)
    boundaries = np.linspace(0, 1, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        inclusive = upper == 1
        mask = (probabilities >= lower) & (
            probabilities <= upper if inclusive else probabilities < upper
        )
        if mask.any():
            error += mask.mean() * abs(targets[mask].mean() - probabilities[mask].mean())
    return float(error)


def calibration_slope_intercept(
    labels: Sequence[float],
    scores: Sequence[float],
) -> tuple[float, float]:
    targets = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs")
    model.fit(logits, targets)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def reliability_curve(
    labels: Sequence[float],
    scores: Sequence[float],
    bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    targets = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(scores, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(targets[mask].mean()))
            counts.append(int(mask.sum()))
    return (
        np.asarray(predicted),
        np.asarray(observed),
        np.asarray(counts, dtype=np.int64),
    )


def calibration_report(
    labels: Sequence[float],
    scores: Sequence[float],
    bins: int = 10,
) -> CalibrationReport:
    slope, intercept = calibration_slope_intercept(labels, scores)
    return CalibrationReport(
        brier=brier(labels, scores),
        expected_calibration_error=expected_calibration_error(labels, scores, bins),
        slope=slope,
        intercept=intercept,
    )
