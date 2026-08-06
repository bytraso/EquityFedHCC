from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from equityfedhcc.data.contracts import MultimodalBatch
from equityfedhcc.federation.aggregation import DualAxisAggregator
from equityfedhcc.federation.state import ClientUpdate
from equityfedhcc.models.system import EquityFedHCC
from equityfedhcc.objectives import EquityFederatedObjective
from equityfedhcc.runtime.checkpoint import RunState, save_training_state
from equityfedhcc.settings import ExperimentSettings
from equityfedhcc.training.local import LocalTrainer

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiteLoaders:
    train: tuple[MultimodalBatch, ...]
    validation: tuple[MultimodalBatch, ...]


@dataclass(frozen=True)
class RoundSummary:
    index: int
    site_losses: dict[str, float]
    site_weights: dict[str, float]
    validation_loss: float
    learning_rate: float


ValidationFunction = Callable[[EquityFedHCC, Iterable[MultimodalBatch], torch.device], float]


class FederatedCoordinator:
    def __init__(
        self,
        model_factory: Callable[[], EquityFedHCC],
        sites: Mapping[str, SiteLoaders],
        target_prevalence: Mapping[str, float],
        settings: ExperimentSettings,
        validation: ValidationFunction,
    ) -> None:
        self.model_factory = model_factory
        self.sites = dict(sites)
        self.target_prevalence = dict(target_prevalence)
        self.settings = settings
        self.validation = validation
        self.device = torch.device(settings.device)
        self.global_model = model_factory().to(self.device)
        self.aggregator = DualAxisAggregator(settings.q, target_prevalence)

    def _site_model(self) -> EquityFedHCC:
        model = self.model_factory().to(self.device)
        model.load_state_dict(self.global_model.state_dict())
        return model

    def _train_site(self, site_id: str, round_index: int) -> ClientUpdate:
        model = self._site_model()
        optimizer = AdamW(
            model.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        objective = EquityFederatedObjective(
            self.settings.lambda_calibration,
            self.settings.lambda_equity,
            self.settings.equity_temperature,
            self.settings.fairness_margin,
        )
        scaler = (
            torch.cuda.amp.GradScaler()
            if self.settings.precision == "fp16" and self.device.type == "cuda"
            else None
        )
        trainer = LocalTrainer(
            model,
            objective,
            optimizer,
            self.device,
            self.settings.gradient_clip,
            scaler,
        )
        return trainer.train_round(
            site_id,
            self.sites[site_id].train,
            self.settings.local_epochs,
            round_index,
        )

    def _apply_state(self, state: Mapping[str, Tensor]) -> None:
        current = self.global_model.state_dict()
        merged = OrderedDict()
        for name, value in state.items():
            target = current[name]
            merged[name] = value.to(device=target.device, dtype=target.dtype)
        self.global_model.load_state_dict(merged)

    def run(self) -> tuple[RoundSummary, ...]:
        optimizer = AdamW(
            self.global_model.parameters(),
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.settings.rounds)
        best = float("inf")
        stale = 0
        summaries: list[RoundSummary] = []
        all_validation = tuple(batch for site in self.sites.values() for batch in site.validation)
        for round_index in range(self.settings.rounds):
            updates = tuple(
                self._train_site(site_id, round_index) for site_id in sorted(self.sites)
            )
            aggregate, weights = self.aggregator.aggregate(updates)
            self._apply_state(aggregate)
            validation_loss = self.validation(self.global_model, all_validation, self.device)
            scheduler.step()
            site_losses = {update.site_id: update.task_loss for update in updates}
            summaries.append(
                RoundSummary(
                    index=round_index,
                    site_losses=site_losses,
                    site_weights=weights.site,
                    validation_loss=validation_loss,
                    learning_rate=float(scheduler.get_last_lr()[0]),
                )
            )
            LOGGER.info(
                "round=%d validation_loss=%.6f site_losses=%s",
                round_index,
                validation_loss,
                site_losses,
            )
            if validation_loss < best:
                best = validation_loss
                stale = 0
                save_training_state(
                    self.settings.output_dir / f"seed_{self.settings.seed}" / "best.pt",
                    self.global_model,
                    optimizer,
                    scheduler,
                    RunState(round_index, self.settings.seed, best, stale),
                )
            else:
                stale += 1
            if stale >= self.settings.patience:
                break
        return tuple(summaries)
