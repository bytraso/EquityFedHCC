from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from equityfedhcc.data.contracts import MultimodalBatch
from equityfedhcc.federation.state import ClientUpdate, StratumUpdate, clone_state
from equityfedhcc.models.system import EquityFedHCC
from equityfedhcc.objectives import EquityFederatedObjective


@dataclass(frozen=True)
class EpochLoss:
    total: float
    task: float
    calibration: float
    equity: float
    batches: int


class LocalTrainer:
    def __init__(
        self,
        model: EquityFedHCC,
        objective: EquityFederatedObjective,
        optimizer: Optimizer,
        device: torch.device,
        gradient_clip: float,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> None:
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.device = device
        self.gradient_clip = gradient_clip
        self.scaler = scaler

    def _forward_loss(self, batch: MultimodalBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        output = self.model(batch)
        losses = self.objective(
            output.logits,
            output.probabilities,
            output.modality_probabilities,
            batch.labels,
            batch.modality_mask,
            batch.stratum_ids,
        )
        return losses.total, losses.task, losses.calibration, losses.equity

    def train_epoch(self, batches: Iterable[MultimodalBatch]) -> EpochLoss:
        self.model.train()
        totals = torch.zeros(4, dtype=torch.float64)
        count = 0
        for raw_batch in batches:
            batch = raw_batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.scaler is not None,
                dtype=torch.float16,
            ):
                total, task, calibration, equity = self._forward_loss(batch)
            if self.scaler is None:
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            else:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            totals += torch.tensor(
                [total.item(), task.item(), calibration.item(), equity.item()],
                dtype=torch.float64,
            )
            count += 1
        if count == 0:
            raise ValueError("local epoch received no batches")
        values = totals / count
        return EpochLoss(
            total=float(values[0]),
            task=float(values[1]),
            calibration=float(values[2]),
            equity=float(values[3]),
            batches=count,
        )

    def train_round(
        self,
        site_id: str,
        batches: Iterable[MultimodalBatch],
        local_epochs: int,
        round_index: int,
    ) -> ClientUpdate:
        cached = tuple(batches)
        if not cached:
            raise ValueError("site has no local batches")
        strata_counts: dict[str, int] = defaultdict(int)
        for batch in cached:
            for stratum in batch.stratum_ids:
                strata_counts[stratum] += 1
        histories = [self.train_epoch(cached) for _ in range(local_epochs)]
        final_state = clone_state(self.model)
        mean_task = sum(item.task for item in histories) / len(histories)
        strata = tuple(
            StratumUpdate(
                stratum=stratum,
                parameters=final_state,
                sample_count=count,
                loss=mean_task,
            )
            for stratum, count in sorted(strata_counts.items())
        )
        return ClientUpdate(
            site_id=site_id,
            strata=strata,
            task_loss=mean_task,
            sample_count=sum(strata_counts.values()),
            round_index=round_index,
        )
