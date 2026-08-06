from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class RunState:
    round_index: int
    seed: int
    best_validation_loss: float
    stale_rounds: int


def atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_training_state(
    destination: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: object,
    state: RunState,
) -> None:
    scheduler_state = scheduler.state_dict() if hasattr(scheduler, "state_dict") else {}
    atomic_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
            "run_state": asdict(state),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        destination,
    )


def restore_training_state(
    source: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: object,
    map_location: str | torch.device,
) -> RunState:
    payload = torch.load(source, map_location=map_location)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return RunState(**payload["run_state"])
