from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from equityfedhcc.federation.state import ClientUpdate, ParameterState, weighted_state_sum


@dataclass(frozen=True)
class AggregationWeights:
    site: dict[str, float]
    stratum: dict[str, dict[str, float]]


def qffl_site_weights(updates: tuple[ClientUpdate, ...], q: float) -> dict[str, float]:
    if not updates:
        raise ValueError("client updates cannot be empty")
    raw = np.asarray([max(update.task_loss, 1e-12) ** q for update in updates], dtype=np.float64)
    raw /= raw.sum()
    return {update.site_id: float(weight) for update, weight in zip(updates, raw, strict=True)}


def prevalence_ratios(
    observed: Mapping[str, float],
    target: Mapping[str, float],
) -> dict[str, float]:
    keys = sorted(set(observed) & set(target))
    if not keys:
        raise ValueError("observed and target prevalence do not overlap")
    raw = {key: target[key] / max(observed[key], 1e-12) for key in keys}
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def audit_adjusted_ratios(
    ratios: Mapping[str, float],
    audit_gaps: Mapping[str, float],
    strength: float = 1.0,
) -> dict[str, float]:
    adjusted = {
        stratum: weight * np.exp(strength * max(audit_gaps.get(stratum, 0.0), 0.0))
        for stratum, weight in ratios.items()
    }
    total = sum(adjusted.values())
    return {key: float(value / total) for key, value in adjusted.items()}


class DualAxisAggregator:
    def __init__(
        self,
        q: float,
        target_prevalence: Mapping[str, float],
        audit_strength: float = 1.0,
    ) -> None:
        self.q = q
        self.target_prevalence = dict(target_prevalence)
        self.audit_strength = audit_strength

    def weights(
        self,
        updates: tuple[ClientUpdate, ...],
        audit_gaps: Mapping[str, float] | None = None,
    ) -> AggregationWeights:
        site_weights = qffl_site_weights(updates, self.q)
        stratum_weights: dict[str, dict[str, float]] = {}
        for update in updates:
            total = sum(item.sample_count for item in update.strata)
            observed = {
                item.stratum: item.sample_count / total
                for item in update.strata
                if item.sample_count > 0
            }
            ratios = prevalence_ratios(observed, self.target_prevalence)
            if audit_gaps:
                ratios = audit_adjusted_ratios(ratios, audit_gaps, self.audit_strength)
            stratum_weights[update.site_id] = ratios
        return AggregationWeights(site=site_weights, stratum=stratum_weights)

    def aggregate(
        self,
        updates: tuple[ClientUpdate, ...],
        audit_gaps: Mapping[str, float] | None = None,
    ) -> tuple[ParameterState, AggregationWeights]:
        weights = self.weights(updates, audit_gaps)
        site_states: list[ParameterState] = []
        site_weight_values: list[float] = []
        for update in updates:
            available = [
                item for item in update.strata if item.stratum in weights.stratum[update.site_id]
            ]
            state = weighted_state_sum(
                [item.parameters for item in available],
                [weights.stratum[update.site_id][item.stratum] for item in available],
            )
            site_states.append(state)
            site_weight_values.append(weights.site[update.site_id])
        return weighted_state_sum(site_states, site_weight_values), weights


def stratum_loss_summary(updates: tuple[ClientUpdate, ...]) -> dict[str, float]:
    weighted: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for update in updates:
        for item in update.strata:
            weighted[item.stratum] += item.loss * item.sample_count
            counts[item.stratum] += item.sample_count
    return {key: weighted[key] / counts[key] for key in counts if counts[key]}
