from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ModalityIndex(IntEnum):
    EHR = 0
    CT = 1
    PATHOLOGY = 2
    OMICS = 3
    BIOMARKERS = 4


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    site_id: str
    stratum_id: str
    label: float
    ehr: Tensor | None = None
    ct: Tensor | None = None
    pathology: Tensor | None = None
    omics: Tensor | None = None
    biomarkers: Tensor | None = None

    def modality_mask(self) -> Tensor:
        return torch.tensor(
            [
                self.ehr is not None,
                self.ct is not None,
                self.pathology is not None,
                self.omics is not None,
                self.biomarkers is not None,
            ],
            dtype=torch.bool,
        )


@dataclass(frozen=True)
class MultimodalBatch:
    patient_ids: tuple[str, ...]
    site_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    labels: Tensor
    modality_mask: Tensor
    ehr: Tensor | None
    ct: Tensor | None
    pathology: Tensor | None
    omics: Tensor | None
    biomarkers: Tensor | None

    def to(self, device: torch.device) -> MultimodalBatch:
        def move(value: Tensor | None) -> Tensor | None:
            return value.to(device) if value is not None else None

        return MultimodalBatch(
            patient_ids=self.patient_ids,
            site_ids=self.site_ids,
            stratum_ids=self.stratum_ids,
            labels=self.labels.to(device),
            modality_mask=self.modality_mask.to(device),
            ehr=move(self.ehr),
            ct=move(self.ct),
            pathology=move(self.pathology),
            omics=move(self.omics),
            biomarkers=move(self.biomarkers),
        )


class RecordDataset(Dataset[PatientRecord]):
    def __init__(self, records: Sequence[PatientRecord]) -> None:
        self._records = tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> PatientRecord:
        return self._records[index]

    def strata(self) -> tuple[str, ...]:
        return tuple(sorted({record.stratum_id for record in self._records}))

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted({record.site_id for record in self._records}))

    def subset(self, indices: Sequence[int]) -> RecordDataset:
        return RecordDataset([self._records[index] for index in indices])


def _stack_optional(values: Sequence[Tensor | None]) -> Tensor | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    prototype = present[0]
    filled = [value if value is not None else torch.zeros_like(prototype) for value in values]
    return torch.stack(filled)


def collate_records(records: Sequence[PatientRecord]) -> MultimodalBatch:
    if not records:
        raise ValueError("records cannot be empty")
    return MultimodalBatch(
        patient_ids=tuple(record.patient_id for record in records),
        site_ids=tuple(record.site_id for record in records),
        stratum_ids=tuple(record.stratum_id for record in records),
        labels=torch.tensor([record.label for record in records], dtype=torch.float32),
        modality_mask=torch.stack([record.modality_mask() for record in records]),
        ehr=_stack_optional([record.ehr for record in records]),
        ct=_stack_optional([record.ct for record in records]),
        pathology=_stack_optional([record.pathology for record in records]),
        omics=_stack_optional([record.omics for record in records]),
        biomarkers=_stack_optional([record.biomarkers for record in records]),
    )


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def ensure_disjoint(self) -> None:
        train = set(self.train)
        validation = set(self.validation)
        test = set(self.test)
        if train & validation or train & test or validation & test:
            raise ValueError("patient identifiers overlap across splits")


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str
    version: str
    license_name: str
    source_url: str
    restricted: bool
    modalities: tuple[ModalityIndex, ...]
    partition_field: str


class RecordReader:
    def read(self, location: Path) -> Iterator[PatientRecord]:
        raise NotImplementedError


def prevalence_by_stratum(records: Sequence[PatientRecord]) -> Mapping[str, float]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.stratum_id] = counts.get(record.stratum_id, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}
