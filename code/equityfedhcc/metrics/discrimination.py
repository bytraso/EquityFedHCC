from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class DiscriminationReport:
    auroc: ConfidenceInterval
    auprc: ConfidenceInterval
    sample_count: int
    positive_count: int


def _as_arrays(
    labels: Sequence[float],
    scores: Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    label_array = np.asarray(labels, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape != score_array.shape:
        raise ValueError("labels and scores must have equal shape")
    if label_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    return label_array, score_array


def auroc(labels: Sequence[float], scores: Sequence[float]) -> float:
    label_array, score_array = _as_arrays(labels, scores)
    if np.unique(label_array).size < 2:
        return float("nan")
    return float(roc_auc_score(label_array, score_array))


def auprc(labels: Sequence[float], scores: Sequence[float]) -> float:
    label_array, score_array = _as_arrays(labels, scores)
    if label_array.sum() == 0:
        return float("nan")
    return float(average_precision_score(label_array, score_array))


def stratified_bootstrap_indices(
    labels: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    parts: list[NDArray[np.int64]] = []
    for outcome in np.unique(labels):
        indices = np.flatnonzero(labels == outcome)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    result = np.concatenate(parts)
    rng.shuffle(result)
    return result


def bootstrap_interval(
    labels: Sequence[float],
    scores: Sequence[float],
    statistic: str,
    resamples: int = 2000,
    seed: int = 1234,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    label_array, score_array = _as_arrays(labels, scores)
    metric = auroc if statistic == "auroc" else auprc
    estimate = metric(label_array, score_array)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = stratified_bootstrap_indices(label_array, rng)
        values[index] = metric(label_array[sampled], score_array[sampled])
    alpha = 1 - confidence
    lower, upper = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return ConfidenceInterval(estimate, float(lower), float(upper))


def discrimination_report(
    labels: Sequence[float],
    scores: Sequence[float],
    resamples: int = 2000,
    seed: int = 1234,
) -> DiscriminationReport:
    return DiscriminationReport(
        auroc=bootstrap_interval(labels, scores, "auroc", resamples, seed),
        auprc=bootstrap_interval(labels, scores, "auprc", resamples, seed + 1),
        sample_count=len(labels),
        positive_count=int(np.asarray(labels).sum()),
    )
