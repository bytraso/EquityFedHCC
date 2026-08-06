from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentSettings:
    seed: int = 1234
    seeds: tuple[int, ...] = (1234, 2345, 3456, 4567, 5678)
    device: str = "cuda"
    precision: str = "fp32"
    rounds: int = 19
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_rounds: int = 0
    gradient_clip: float = 1.0
    client_fraction: float = 1.0
    q: float = 2.0
    lambda_calibration: float = 0.5
    lambda_equity: float = 1.0
    equity_temperature: float = 5.0
    fairness_margin: float = 0.05
    patience: int = 3
    modalities: tuple[str, ...] = ("ehr", "ct", "pathology", "omics", "biomarkers")
    fusion_dimension: int = 768
    fusion_heads: int = 8
    fusion_layers: int = 2
    dropout: float = 0.1
    bootstrap_resamples: int = 2000
    cross_site_resamples: int = 1000
    ece_bins: int = 10
    output_dir: Path = field(default_factory=lambda: Path("runs"))

    def validate(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be positive")
        if self.local_epochs < 1:
            raise ValueError("local_epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.q < 0:
            raise ValueError("q must be nonnegative")
        if not 0 < self.client_fraction <= 1:
            raise ValueError("client_fraction must be in (0, 1]")
        if self.fusion_dimension % self.fusion_heads:
            raise ValueError("fusion_dimension must be divisible by fusion_heads")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be positive")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ExperimentSettings:
        known = {item.name for item in cls.__dataclass_fields__.values()}
        cleaned = {key: value for key, value in values.items() if key in known}
        if isinstance(cleaned.get("seeds"), list):
            cleaned["seeds"] = tuple(cleaned["seeds"])
        if isinstance(cleaned.get("modalities"), list):
            cleaned["modalities"] = tuple(cleaned["modalities"])
        if "output_dir" in cleaned:
            cleaned["output_dir"] = Path(cleaned["output_dir"])
        settings = cls(**cleaned)
        settings.validate()
        return settings

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentSettings:
        with path.open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("configuration root must be a mapping")
        return cls.from_mapping(values)

    def with_seed(self, seed: int) -> ExperimentSettings:
        values = dict(self.__dict__)
        values["seed"] = seed
        return ExperimentSettings.from_mapping(values)
