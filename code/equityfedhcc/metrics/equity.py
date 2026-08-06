from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class StratumResult:
    stratum: str
    sample_count: int
    positive_count: int
    auroc: float
    selection_rate: float
    true_positive_rate: float
    false_positive_rate: float
    exploratory: bool


@dataclass(frozen=True)
class EquityReport:
    strata: tuple[StratumResult, ...]
    demographic_parity_difference: float
    equalized_odds_difference: float
    maximum_auroc_gap: float


def _safe_rate(numerator: np.ndarray, denominator: np.ndarray) -> float:
    count = int(denominator.sum())
    return float(numerator[denominator].mean()) if count else float("nan")


def stratum_result(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    group: str,
    minimum_confirmatory_count: int = 30,
) -> StratumResult:
    mask = groups == group
    group_labels = labels[mask]
    group_scores = scores[mask]
    group_predictions = predictions[mask]
    area = (
        float(roc_auc_score(group_labels, group_scores))
        if np.unique(group_labels).size == 2
        else float("nan")
    )
    positives = group_labels == 1
    negatives = group_labels == 0
    return StratumResult(
        stratum=group,
        sample_count=int(mask.sum()),
        positive_count=int(positives.sum()),
        auroc=area,
        selection_rate=float(group_predictions.mean()),
        true_positive_rate=_safe_rate(group_predictions, positives),
        false_positive_rate=_safe_rate(group_predictions, negatives),
        exploratory=int(mask.sum()) < minimum_confirmatory_count,
    )


def equity_report(
    labels: Sequence[float],
    scores: Sequence[float],
    groups: Sequence[str],
    threshold: float = 0.5,
    minimum_confirmatory_count: int = 30,
) -> EquityReport:
    targets = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(scores, dtype=np.float64)
    group_array = np.asarray(groups, dtype=str)
    predictions = probabilities >= threshold
    results = tuple(
        stratum_result(
            targets,
            probabilities,
            predictions,
            group_array,
            group,
            minimum_confirmatory_count,
        )
        for group in sorted(np.unique(group_array))
    )
    selection = np.array([result.selection_rate for result in results])
    tpr = np.array([result.true_positive_rate for result in results])
    fpr = np.array([result.false_positive_rate for result in results])
    areas = np.array([result.auroc for result in results])
    dpd = float(np.nanmax(selection) - np.nanmin(selection))
    tpr_gap = float(np.nanmax(tpr) - np.nanmin(tpr))
    fpr_gap = float(np.nanmax(fpr) - np.nanmin(fpr))
    area_gap = float(np.nanmax(areas) - np.nanmin(areas))
    return EquityReport(
        strata=results,
        demographic_parity_difference=dpd,
        equalized_odds_difference=max(tpr_gap, fpr_gap),
        maximum_auroc_gap=area_gap,
    )


def cross_site_concordance(site_aurocs: dict[str, float]) -> float:
    if not site_aurocs:
        raise ValueError("site AUROCs cannot be empty")
    values = np.asarray(list(site_aurocs.values()), dtype=np.float64)
    return float(np.nanmax(values) - np.nanmin(values))
