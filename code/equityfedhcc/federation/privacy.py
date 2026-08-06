from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import Tensor

from equityfedhcc.federation.state import ParameterState


def state_l2_norm(state: Mapping[str, Tensor]) -> Tensor:
    values = [value.to(torch.float64).square().sum() for value in state.values()]
    return torch.stack(values).sum().sqrt()


def clip_update(state: Mapping[str, Tensor], maximum_norm: float) -> ParameterState:
    if maximum_norm <= 0:
        raise ValueError("maximum norm must be positive")
    norm = state_l2_norm(state)
    factor = min(1.0, maximum_norm / max(float(norm), 1e-12))
    return OrderedDict((name, value * factor) for name, value in state.items())


def gaussian_update(
    state: Mapping[str, Tensor],
    noise_multiplier: float,
    maximum_norm: float,
    generator: torch.Generator,
) -> ParameterState:
    clipped = clip_update(state, maximum_norm)
    standard_deviation = noise_multiplier * maximum_norm
    return OrderedDict(
        (
            name,
            value
            + torch.randn(
                value.shape,
                dtype=value.dtype,
                device=value.device,
                generator=generator,
            )
            * standard_deviation,
        )
        for name, value in clipped.items()
    )


def approximate_gaussian_noise_multiplier(
    epsilon: float,
    delta: float,
    rounds: int,
) -> float:
    if epsilon <= 0 or not 0 < delta < 1 or rounds <= 0:
        raise ValueError("invalid privacy parameters")
    return float((2 * rounds * torch.log(torch.tensor(1.25 / delta))).sqrt() / epsilon)
