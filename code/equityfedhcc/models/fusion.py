from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class ModalityProjection(nn.Module):
    def __init__(self, dimensions: tuple[int, ...], output_dimension: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, output_dimension),
                    nn.LayerNorm(output_dimension),
                    nn.GELU(),
                )
                for dimension in dimensions
            ]
        )
        self.missing_tokens = nn.Parameter(torch.empty(len(dimensions), output_dimension))
        nn.init.normal_(self.missing_tokens, std=0.02)

    def forward(self, embeddings: tuple[Tensor | None, ...], mask: Tensor) -> Tensor:
        if len(embeddings) != len(self.projections):
            raise ValueError("embedding count differs from configured modalities")
        batch_size = mask.shape[0]
        projected: list[Tensor] = []
        pairs = zip(embeddings, self.projections, strict=True)
        for index, (embedding, projection) in enumerate(pairs):
            missing = self.missing_tokens[index].expand(batch_size, -1)
            if embedding is None:
                projected.append(missing)
            else:
                present = projection(embedding)
                projected.append(torch.where(mask[:, index, None], present, missing))
        return torch.stack(projected, dim=1)


class MaskedCrossAttentionLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.pre_attention = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pre_mlp = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, unavailable: Tensor) -> Tensor:
        normalized = self.pre_attention(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=unavailable,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.pre_mlp(tokens))


@dataclass(frozen=True)
class FusionOutput:
    logits: Tensor
    probabilities: Tensor
    modality_logits: Tensor
    modality_probabilities: Tensor
    attention_weights: Tensor
    representation: Tensor


class LateFusionHead(nn.Module):
    def __init__(
        self,
        input_dimensions: tuple[int, ...],
        dimension: int = 768,
        heads: int = 8,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = ModalityProjection(input_dimensions, dimension)
        self.layers = nn.ModuleList(
            [MaskedCrossAttentionLayer(dimension, heads, dropout) for _ in range(layers)]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 1),
        )
        self.modality_heads = nn.ModuleList([nn.Linear(dimension, 1) for _ in input_dimensions])
        self.risk_head = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension // 2, 1),
        )

    def forward(self, embeddings: tuple[Tensor | None, ...], mask: Tensor) -> FusionOutput:
        if mask.ndim != 2:
            raise ValueError("modality mask must have shape [batch, modalities]")
        if (~mask).all(dim=1).any():
            raise ValueError("each sample must contain at least one modality")
        tokens = self.projection(embeddings, mask)
        unavailable = ~mask
        for layer in self.layers:
            tokens = layer(tokens, unavailable)
        gate_logits = self.gate(tokens).squeeze(-1)
        gate_logits = gate_logits.masked_fill(unavailable, torch.finfo(gate_logits.dtype).min)
        weights = torch.softmax(gate_logits, dim=-1)
        pooled = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        logits = self.risk_head(pooled).squeeze(-1)
        modality_logits = torch.stack(
            [head(tokens[:, index]).squeeze(-1) for index, head in enumerate(self.modality_heads)],
            dim=1,
        )
        modality_logits = modality_logits.masked_fill(unavailable, 0)
        return FusionOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            modality_logits=modality_logits,
            modality_probabilities=torch.sigmoid(modality_logits),
            attention_weights=weights,
            representation=pooled,
        )
