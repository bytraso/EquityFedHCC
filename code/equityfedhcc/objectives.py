from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


def binary_task_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))


def brier_score(probabilities: Tensor, labels: Tensor) -> Tensor:
    return (probabilities - labels.to(probabilities.dtype)).square().mean()


def grouped_brier_scores(
    probabilities: Tensor,
    labels: Tensor,
    groups: Sequence[str],
) -> dict[str, Tensor]:
    if probabilities.numel() != len(groups):
        raise ValueError("probabilities and groups must have equal length")
    result: dict[str, Tensor] = {}
    for group in sorted(set(groups)):
        mask = torch.tensor([value == group for value in groups], device=probabilities.device)
        result[group] = brier_score(probabilities[mask], labels[mask])
    return result


def cross_modal_calibration_loss(
    modality_probabilities: Tensor,
    fusion_probabilities: Tensor,
    labels: Tensor,
    modality_mask: Tensor,
    strata: Sequence[str],
) -> Tensor:
    if modality_probabilities.shape != modality_mask.shape:
        raise ValueError("modality probabilities and mask must have equal shape")
    if modality_probabilities.shape[0] != len(strata):
        raise ValueError("batch and strata must have equal length")
    terms: list[Tensor] = []
    for stratum in sorted(set(strata)):
        stratum_mask = torch.tensor(
            [value == stratum for value in strata],
            device=fusion_probabilities.device,
        )
        fusion_brier = brier_score(fusion_probabilities[stratum_mask], labels[stratum_mask])
        for modality in range(modality_probabilities.shape[1]):
            present = stratum_mask & modality_mask[:, modality]
            if present.any():
                modality_brier = brier_score(
                    modality_probabilities[present, modality],
                    labels[present],
                )
                terms.append(modality_brier - fusion_brier)
    if not terms:
        return fusion_probabilities.sum() * 0
    return torch.stack(terms).mean()


def hard_demographic_parity_difference(
    predictions: Tensor,
    strata: Sequence[str],
) -> Tensor:
    rates: list[Tensor] = []
    for stratum in sorted(set(strata)):
        mask = torch.tensor([value == stratum for value in strata], device=predictions.device)
        rates.append(predictions[mask].to(torch.float32).mean())
    if len(rates) < 2:
        return predictions.sum() * 0
    stacked = torch.stack(rates)
    return stacked.max() - stacked.min()


def smooth_demographic_parity_loss(
    probabilities: Tensor,
    strata: Sequence[str],
    temperature: float = 5.0,
    margin: float = 0.05,
) -> Tensor:
    rates: list[Tensor] = []
    for stratum in sorted(set(strata)):
        mask = torch.tensor([value == stratum for value in strata], device=probabilities.device)
        rates.append(probabilities[mask].mean())
    if len(rates) < 2:
        return probabilities.sum() * 0
    stacked = torch.stack(rates)
    differences = torch.abs(stacked[:, None] - stacked[None, :])
    active = functional.relu(differences - margin)
    return torch.logsumexp(temperature * active.flatten(), dim=0) / temperature


def equalized_odds_difference(
    predictions: Tensor,
    labels: Tensor,
    strata: Sequence[str],
) -> Tensor:
    gaps: list[Tensor] = []
    for outcome in (0, 1):
        rates: list[Tensor] = []
        outcome_mask = labels == outcome
        for stratum in sorted(set(strata)):
            stratum_mask = torch.tensor(
                [value == stratum for value in strata],
                device=predictions.device,
            )
            mask = outcome_mask & stratum_mask
            if mask.any():
                rates.append(predictions[mask].to(torch.float32).mean())
        if len(rates) >= 2:
            values = torch.stack(rates)
            gaps.append(values.max() - values.min())
    if not gaps:
        return predictions.sum() * 0
    return torch.stack(gaps).max()


@dataclass(frozen=True)
class ObjectiveOutput:
    total: Tensor
    task: Tensor
    calibration: Tensor
    equity: Tensor


class EquityFederatedObjective(nn.Module):
    def __init__(
        self,
        calibration_weight: float = 0.5,
        equity_weight: float = 1.0,
        temperature: float = 5.0,
        margin: float = 0.05,
    ) -> None:
        super().__init__()
        self.calibration_weight = calibration_weight
        self.equity_weight = equity_weight
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        logits: Tensor,
        fusion_probabilities: Tensor,
        modality_probabilities: Tensor,
        labels: Tensor,
        modality_mask: Tensor,
        strata: Sequence[str],
    ) -> ObjectiveOutput:
        task = binary_task_loss(logits, labels)
        calibration = cross_modal_calibration_loss(
            modality_probabilities,
            fusion_probabilities,
            labels,
            modality_mask,
            strata,
        )
        equity = smooth_demographic_parity_loss(
            fusion_probabilities,
            strata,
            self.temperature,
            self.margin,
        )
        total = task + self.calibration_weight * calibration + self.equity_weight * equity
        return ObjectiveOutput(total=total, task=task, calibration=calibration, equity=equity)
