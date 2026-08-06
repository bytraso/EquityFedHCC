from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from equityfedhcc.data.contracts import PatientRecord, RecordDataset, SplitManifest

T = TypeVar("T")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.7
    validation: float = 0.1
    test: float = 0.2

    def validate(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios must be nonnegative")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("split ratios must sum to one")


def grouped_stratified_split(
    records: Sequence[PatientRecord],
    ratios: SplitRatios,
    seed: int,
) -> tuple[RecordDataset, RecordDataset, RecordDataset, SplitManifest]:
    ratios.validate()
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record.stratum_id, int(record.label >= 0.5))].append(index)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for indices in grouped.values():
        shuffled = np.array(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        train_end = int(round(len(shuffled) * ratios.train))
        validation_end = train_end + int(round(len(shuffled) * ratios.validation))
        train_indices.extend(shuffled[:train_end].tolist())
        validation_indices.extend(shuffled[train_end:validation_end].tolist())
        test_indices.extend(shuffled[validation_end:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)
    train = RecordDataset([records[index] for index in train_indices])
    validation = RecordDataset([records[index] for index in validation_indices])
    test = RecordDataset([records[index] for index in test_indices])
    manifest = SplitManifest(
        train=tuple(records[index].patient_id for index in train_indices),
        validation=tuple(records[index].patient_id for index in validation_indices),
        test=tuple(records[index].patient_id for index in test_indices),
    )
    manifest.ensure_disjoint()
    return train, validation, test, manifest


def temporal_split(
    records: Sequence[PatientRecord],
    years: Sequence[int],
    test_start: int,
) -> tuple[RecordDataset, RecordDataset]:
    if len(records) != len(years):
        raise ValueError("records and years must have equal length")
    train = [record for record, year in zip(records, years, strict=True) if year < test_start]
    test = [record for record, year in zip(records, years, strict=True) if year >= test_start]
    return RecordDataset(train), RecordDataset(test)


def leave_one_group_out(
    values: Sequence[T],
    groups: Sequence[Hashable],
) -> list[tuple[list[T], list[T]]]:
    if len(values) != len(groups):
        raise ValueError("values and groups must have equal length")
    unique_groups = sorted(set(groups), key=str)
    folds: list[tuple[list[T], list[T]]] = []
    for held_out in unique_groups:
        pairs = zip(values, groups, strict=True)
        train = [value for value, group in pairs if group != held_out]
        pairs = zip(values, groups, strict=True)
        test = [value for value, group in pairs if group == held_out]
        folds.append((train, test))
    return folds
