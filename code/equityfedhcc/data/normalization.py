from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class StandardizationState:
    mean: Tensor
    scale: Tensor
    observed: Tensor


class MaskedStandardizer:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.state: StandardizationState | None = None

    def fit(self, values: Tensor, observed: Tensor | None = None) -> MaskedStandardizer:
        if values.ndim != 2:
            raise ValueError("values must have shape [samples, features]")
        mask = observed if observed is not None else torch.isfinite(values)
        safe = torch.where(mask, values, torch.zeros_like(values))
        counts = mask.sum(dim=0).clamp_min(1)
        mean = safe.sum(dim=0) / counts
        centered = torch.where(mask, values - mean, torch.zeros_like(values))
        variance = centered.square().sum(dim=0) / counts
        scale = variance.sqrt().clamp_min(self.epsilon)
        self.state = StandardizationState(mean=mean, scale=scale, observed=counts)
        return self

    def transform(self, values: Tensor, observed: Tensor | None = None) -> Tensor:
        if self.state is None:
            raise RuntimeError("standardizer must be fitted")
        mask = observed if observed is not None else torch.isfinite(values)
        normalized = (values - self.state.mean) / self.state.scale
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    def fit_transform(self, values: Tensor, observed: Tensor | None = None) -> Tensor:
        return self.fit(values, observed).transform(values, observed)


def winsorize(values: Tensor, lower: float = 0.01, upper: float = 0.99) -> Tensor:
    if not 0 <= lower < upper <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    low = torch.quantile(values, lower, dim=0)
    high = torch.quantile(values, upper, dim=0)
    return values.clamp(min=low, max=high)


def resample_volume(
    volume: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> Tensor:
    if volume.ndim != 3:
        raise ValueError("volume must be three-dimensional")
    source = torch.from_numpy(volume).float()[None, None]
    output_shape = tuple(
        max(1, int(round(size * old / new)))
        for size, old, new in zip(volume.shape, current_spacing, target_spacing, strict=True)
    )
    return torch.nn.functional.interpolate(
        source,
        size=output_shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]


def clip_hounsfield(volume: Tensor, lower: float = -200.0, upper: float = 300.0) -> Tensor:
    clipped = volume.clamp(lower, upper)
    return (clipped - lower) / (upper - lower)
