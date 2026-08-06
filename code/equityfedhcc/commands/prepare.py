from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from equityfedhcc.data.cohorts import is_cirrhosis_code

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    instance = argparse.ArgumentParser(prog="equityfedhcc-prepare")
    instance.add_argument(
        "--cohort",
        required=True,
        choices=("eicu", "tcga_lihc", "nhanes", "lits_msd"),
    )
    instance.add_argument("--input", type=Path, required=True)
    instance.add_argument("--output", type=Path, required=True)
    return instance


def prepare_eicu(source: Path, destination: Path) -> None:
    frame = pd.read_csv(source)
    if "diagnosisstring" not in frame:
        raise ValueError("diagnosisstring is required")
    selected = frame[frame["diagnosisstring"].fillna("").map(is_cirrhosis_code)].copy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(destination, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    arguments = parser().parse_args()
    if arguments.cohort == "eicu":
        prepare_eicu(arguments.input, arguments.output)
    else:
        raise ValueError(f"preparation adapter is not configured for {arguments.cohort}")
    LOGGER.info("prepared cohort=%s destination=%s", arguments.cohort, arguments.output)


if __name__ == "__main__":
    main()
