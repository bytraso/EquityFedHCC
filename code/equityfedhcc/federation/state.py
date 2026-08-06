from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

ParameterState = OrderedDict[str, Tensor]


@dataclass(frozen=True)
class StratumUpdate:
    stratum: str
    parameters: ParameterState
    sample_count: int
    loss: float


@dataclass(frozen=True)
class ClientUpdate:
    site_id: str
    strata: tuple[StratumUpdate, ...]
    task_loss: float
    sample_count: int
    round_index: int


def clone_state(module: nn.Module) -> ParameterState:
    return OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in module.state_dict().items()
    )


def subtract_states(current: Mapping[str, Tensor], origin: Mapping[str, Tensor]) -> ParameterState:
    if current.keys() != origin.keys():
        raise ValueError("state keys differ")
    return OrderedDict((name, current[name] - origin[name]) for name in current)


def add_states(origin: Mapping[str, Tensor], update: Mapping[str, Tensor]) -> ParameterState:
    if origin.keys() != update.keys():
        raise ValueError("state keys differ")
    return OrderedDict((name, origin[name] + update[name]) for name in origin)


def scale_state(state: Mapping[str, Tensor], weight: float) -> ParameterState:
    return OrderedDict((name, value * weight) for name, value in state.items())


def sum_states(states: list[Mapping[str, Tensor]]) -> ParameterState:
    if not states:
        raise ValueError("states cannot be empty")
    keys = states[0].keys()
    if any(state.keys() != keys for state in states[1:]):
        raise ValueError("state keys differ")
    return OrderedDict(
        (name, torch.stack([state[name] for state in states]).sum(dim=0)) for name in keys
    )


def weighted_state_sum(
    states: list[Mapping[str, Tensor]],
    weights: list[float],
) -> ParameterState:
    if len(states) != len(weights):
        raise ValueError("states and weights must have equal length")
    if not states:
        raise ValueError("states cannot be empty")
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    normalized = [weight / total for weight in weights]
    pairs = zip(states, normalized, strict=True)
    return sum_states([scale_state(state, weight) for state, weight in pairs])
