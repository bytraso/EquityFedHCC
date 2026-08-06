from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from equityfedhcc.data.contracts import MultimodalBatch
from equityfedhcc.metrics import (
    CalibrationReport,
    DiscriminationReport,
    EquityReport,
    calibration_report,
    discrimination_report,
    equity_report,
)
from equityfedhcc.models.system import EquityFedHCC


@dataclass(frozen=True)
class EvaluationResult:
    discrimination: DiscriminationReport
    calibration: CalibrationReport
    equity: EquityReport
    labels: tuple[float, ...]
    scores: tuple[float, ...]
    sites: tuple[str, ...]
    strata: tuple[str, ...]


def evaluate(
    model: EquityFedHCC,
    batches: Iterable[MultimodalBatch],
    device: torch.device,
    resamples: int,
    seed: int,
    ece_bins: int,
) -> EvaluationResult:
    model.eval()
    labels: list[float] = []
    scores: list[float] = []
    sites: list[str] = []
    strata: list[str] = []
    with torch.no_grad():
        for raw_batch in batches:
            batch = raw_batch.to(device)
            output = model(batch)
            labels.extend(batch.labels.detach().cpu().tolist())
            scores.extend(output.probabilities.detach().cpu().tolist())
            sites.extend(batch.site_ids)
            strata.extend(batch.stratum_ids)
    if not labels:
        raise ValueError("evaluation received no samples")
    return EvaluationResult(
        discrimination=discrimination_report(labels, scores, resamples, seed),
        calibration=calibration_report(labels, scores, ece_bins),
        equity=equity_report(labels, scores, strata),
        labels=tuple(labels),
        scores=tuple(scores),
        sites=tuple(sites),
        strata=tuple(strata),
    )


def per_site_aurocs(result: EvaluationResult) -> dict[str, float]:
    values: dict[str, float] = {}
    labels = np.asarray(result.labels)
    scores = np.asarray(result.scores)
    sites = np.asarray(result.sites)
    for site in sorted(set(result.sites)):
        mask = sites == site
        if np.unique(labels[mask]).size == 2:
            from sklearn.metrics import roc_auc_score

            values[site] = float(roc_auc_score(labels[mask], scores[mask]))
    return values
