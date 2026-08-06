from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from equityfedhcc.metrics import calibration_report, discrimination_report, equity_report
from equityfedhcc.reporting import write_json

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    instance = argparse.ArgumentParser(prog="equityfedhcc-evaluate")
    instance.add_argument("--predictions", type=Path, required=True)
    instance.add_argument("--output", type=Path, required=True)
    instance.add_argument("--resamples", type=int, default=2000)
    instance.add_argument("--seed", type=int, default=1234)
    return instance


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    arguments = parser().parse_args()
    frame = pd.read_csv(arguments.predictions)
    required = {"label", "score", "stratum"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing prediction columns: {sorted(missing)}")
    labels = frame["label"].tolist()
    scores = frame["score"].tolist()
    strata = frame["stratum"].astype(str).tolist()
    report = {
        "discrimination": discrimination_report(
            labels,
            scores,
            arguments.resamples,
            arguments.seed,
        ),
        "calibration": calibration_report(labels, scores),
        "equity": equity_report(labels, scores, strata),
    }
    write_json(report, arguments.output)
    LOGGER.info("evaluation written to %s", arguments.output)


if __name__ == "__main__":
    main()
