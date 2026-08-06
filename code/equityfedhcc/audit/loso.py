from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class FoldAudit:
    held_out_site: str
    stratum_aurocs: dict[str, float]
    sample_counts: dict[str, int]


@dataclass(frozen=True)
class ConcordanceAudit:
    folds: tuple[FoldAudit, ...]
    stratum_gaps: dict[str, float]
    maximum_gap: float


ScoreFunction = Callable[
    [tuple[str, ...], str],
    tuple[Sequence[float], Sequence[float], Sequence[str]],
]


def leave_one_site_out_audit(
    sites: Sequence[str],
    scorer: ScoreFunction,
    minimum_count: int = 30,
) -> ConcordanceAudit:
    unique_sites = tuple(sorted(set(sites)))
    folds: list[FoldAudit] = []
    by_stratum: dict[str, list[float]] = {}
    for held_out in unique_sites:
        training_sites = tuple(site for site in unique_sites if site != held_out)
        labels, scores, strata = scorer(training_sites, held_out)
        label_array = np.asarray(labels)
        score_array = np.asarray(scores)
        stratum_array = np.asarray(strata)
        stratum_aurocs: dict[str, float] = {}
        counts: dict[str, int] = {}
        for stratum in sorted(set(strata)):
            mask = stratum_array == stratum
            counts[stratum] = int(mask.sum())
            if counts[stratum] >= minimum_count and np.unique(label_array[mask]).size == 2:
                area = float(roc_auc_score(label_array[mask], score_array[mask]))
                stratum_aurocs[stratum] = area
                by_stratum.setdefault(stratum, []).append(area)
        folds.append(FoldAudit(held_out, stratum_aurocs, counts))
    gaps = {
        stratum: max(values) - min(values)
        for stratum, values in by_stratum.items()
        if len(values) >= 2
    }
    maximum = max(gaps.values()) if gaps else float("nan")
    return ConcordanceAudit(tuple(folds), gaps, maximum)


def bootstrap_gap_interval(
    site_scores: Mapping[str, Sequence[float]],
    resamples: int = 1000,
    seed: int = 1234,
) -> tuple[float, float, float]:
    if len(site_scores) < 2:
        raise ValueError("at least two sites are required")
    arrays = {site: np.asarray(scores) for site, scores in site_scores.items()}
    point_means = [float(values.mean()) for values in arrays.values()]
    estimate = max(point_means) - min(point_means)
    rng = np.random.default_rng(seed)
    gaps = np.empty(resamples)
    for index in range(resamples):
        means = [
            float(rng.choice(values, size=len(values), replace=True).mean())
            for values in arrays.values()
        ]
        gaps[index] = max(means) - min(means)
    lower, upper = np.quantile(gaps, [0.025, 0.975])
    return estimate, float(lower), float(upper)
