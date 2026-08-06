from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from equityfedhcc.data.contracts import MultimodalBatch
from equityfedhcc.models.encoders import (
    AttentionPool,
    BiomarkerEncoder,
    EncoderDimensions,
    OmicsEncoder,
    StructuredEHREncoder,
    VolumeEncoder,
)
from equityfedhcc.models.fusion import FusionOutput, LateFusionHead


@dataclass(frozen=True)
class ModelInputs:
    ehr_features: int
    pathology_features: int
    omics_features: int
    biomarker_features: int


class EquityFedHCC(nn.Module):
    def __init__(
        self,
        inputs: ModelInputs,
        dimensions: EncoderDimensions | None = None,
        fusion_dimension: int = 768,
        fusion_heads: int = 8,
        fusion_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = dimensions or EncoderDimensions()
        self.ehr = StructuredEHREncoder(inputs.ehr_features, dimensions.ehr, dropout)
        self.ct = VolumeEncoder(dimensions.ct)
        self.pathology = AttentionPool(inputs.pathology_features, dimensions.pathology)
        self.omics = OmicsEncoder(inputs.omics_features, dimensions.omics)
        self.biomarkers = BiomarkerEncoder(inputs.biomarker_features, dimensions.biomarkers)
        self.fusion = LateFusionHead(
            dimensions.as_tuple(),
            fusion_dimension,
            fusion_heads,
            fusion_layers,
            dropout,
        )

    def _encode_ehr(self, values: Tensor | None) -> Tensor | None:
        return self.ehr(values) if values is not None else None

    def _encode_ct(self, values: Tensor | None) -> Tensor | None:
        return self.ct(values) if values is not None else None

    def _encode_pathology(self, values: Tensor | None) -> Tensor | None:
        return self.pathology(values) if values is not None else None

    def _encode_omics(self, values: Tensor | None) -> Tensor | None:
        return self.omics(values) if values is not None else None

    def _encode_biomarkers(self, values: Tensor | None) -> Tensor | None:
        return self.biomarkers(values) if values is not None else None

    def forward(self, batch: MultimodalBatch) -> FusionOutput:
        embeddings = (
            self._encode_ehr(batch.ehr),
            self._encode_ct(batch.ct),
            self._encode_pathology(batch.pathology),
            self._encode_omics(batch.omics),
            self._encode_biomarkers(batch.biomarkers),
        )
        return self.fusion(embeddings, batch.modality_mask)
