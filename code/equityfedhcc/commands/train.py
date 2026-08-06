from __future__ import annotations

import argparse
import logging
from pathlib import Path

from equityfedhcc.settings import ExperimentSettings

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    instance = argparse.ArgumentParser(prog="equityfedhcc-train")
    instance.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    instance.add_argument("--manifest", type=Path, required=True)
    instance.add_argument("--output", type=Path)
    return instance


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    arguments = parser().parse_args()
    settings = ExperimentSettings.from_yaml(arguments.config)
    if arguments.output is not None:
        values = dict(settings.__dict__)
        values["output_dir"] = arguments.output
        settings = ExperimentSettings.from_mapping(values)
    if not arguments.manifest.exists():
        raise FileNotFoundError(arguments.manifest)
    LOGGER.info(
        "configuration loaded rounds=%d local_epochs=%d batch_size=%d seed=%d",
        settings.rounds,
        settings.local_epochs,
        settings.batch_size,
        settings.seed,
    )
    message = "dataset-specific manifest adapter must be selected during cohort preparation"
    raise RuntimeError(message)


if __name__ == "__main__":
    main()
