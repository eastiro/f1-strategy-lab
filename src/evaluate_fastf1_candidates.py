from __future__ import annotations

import argparse

import pandas as pd

from .config import RESULTS_DIR
from .evaluate_holdouts import evaluate_v24_state_machine_race, summarise_signal_coverage

FASTF1_2021_CANDIDATES = ["Emilia Romagna", "Hungary"]


def evaluate_fastf1_2021() -> pd.DataFrame:
    frames = [evaluate_v24_state_machine_race(2021, country) for country in FASTF1_2021_CANDIDATES]
    report = pd.concat([frame.dropna(axis=1, how="all") for frame in frames], ignore_index=True, sort=False)
    report.insert(0, "source", "FastF1")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pre-registered FastF1 V2.4 candidates.")
    parser.add_argument(
        "--output",
        default=RESULTS_DIR / "v2_fastf1_evaluation_2021.csv",
    )
    args = parser.parse_args()
    output = pd.io.common.stringify_path(args.output)
    report = evaluate_fastf1_2021()
    report.to_csv(output, index=False)
    print(f"Saved {len(report)} transition rows to {output}")
    print(summarise_signal_coverage(report))


if __name__ == "__main__":
    main()
