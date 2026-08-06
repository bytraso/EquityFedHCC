from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = width * expansion
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.layers(self.norm(values))


class StructuredEHREncoder(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.missing_token = nn.Parameter(torch.zeros(input_dimension))
        self.input = nn.Sequential(
            nn.Linear(input_dimension * 2, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(output_dimension, 4, dropout) for _ in range(3)]
        )

    def forward(self, values: Tensor) -> Tensor:
        observed = torch.isfinite(values)
        filled = torch.where(observed, values, self.missing_token.expand_as(values))
        encoded = self.input(torch.cat([filled, observed.to(values.dtype)], dim=-1))
        for block in self.blocks:
            encoded = block(encoded)
        return encoded


class BiomarkerEncoder(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, output_dimension),
            nn.LayerNorm(output_dimension),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(torch.nan_to_num(values))


class VolumeEncoder(nn.Module):
    def __init__(self, output_dimension: int = 768) -> None:
        super().__init__()
        channels = (1, 32, 64, 128, 256)
        blocks: list[nn.Module] = []
        for source, target in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend(
                [
                    nn.Conv3d(source, target, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(target),
                    nn.GELU(),
                    nn.Conv3d(target, target, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(target),
                    nn.GELU(),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(channels[-1], output_dimension)

    def forward(self, volume: Tensor) -> Tensor:
        features = self.features(volume)
        pooled = self.pool(features).flatten(1)
        return self.projection(pooled)


class AttentionPool(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        super().__init__()
        self.value = nn.Linear(input_dimension, output_dimension)
        self.score = nn.Sequential(
            nn.Linear(input_dimension, output_dimension // 2),
            nn.Tanh(),
            nn.Linear(output_dimension // 2, 1),
        )

    def forward(self, tiles: Tensor, mask: Tensor | None = None) -> Tensor:
        scores = self.score(tiles).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * self.value(tiles), dim=1)


class OmicsEncoder(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int = 256) -> None:
        super().__init__()
        self.input = nn.Linear(input_dimension, 512)
        self.blocks = nn.ModuleList([ResidualMLPBlock(512, 2, 0.1) for _ in range(6)])
        self.output = nn.Linear(512, output_dimension)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.input(torch.log1p(values.clamp_min(0)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class FrozenAdapter(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dimension: int,
        output_dimension: int,
        extraction: Callable[[object], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.extraction = extraction
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.projection = nn.Linear(backbone_dimension, output_dimension)

    def train(self, mode: bool = True) -> FrozenAdapter:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, values: Tensor) -> Tensor:
        with torch.no_grad():
            result = self.backbone(values)
            encoded = self.extraction(result) if self.extraction is not None else result
        if not isinstance(encoded, Tensor):
            raise TypeError("backbone extraction must return a tensor")
        return self.projection(encoded)


@dataclass(frozen=True)
class EncoderDimensions:
    ehr: int = 768
    ct: int = 768
    pathology: int = 1536
    omics: int = 256
    biomarkers: int = 64

    def as_tuple(self) -> tuple[int, ...]:
        return self.ehr, self.ct, self.pathology, self.omics, self.biomarkers
