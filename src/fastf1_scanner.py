from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, RESULTS_DIR
from .v2_validation import SLICK_COMPOUNDS, WET_COMPOUNDS


def count_wet_to_slick_transitions(laps: pd.DataFrame) -> int:
    """Count observed compound changes from wet tyres to slicks in FastF1 laps."""
    required = {"Driver", "LapNumber", "Stint", "Compound"}
    if laps.empty or not required.issubset(laps.columns):
        return 0

    count = 0
    for _, driver_laps in laps.dropna(subset=["Driver", "LapNumber", "Stint"]).groupby("Driver"):
        stints = (
            driver_laps.sort_values("LapNumber")
            .drop_duplicates(subset=["Stint"], keep="first")
            [["Stint", "Compound"]]
            .dropna(subset=["Compound"])
        )
        compounds = [str(value).upper() for value in stints["Compound"]]
        for previous, current in zip(compounds, compounds[1:]):
            if previous in WET_COMPOUNDS and current in SLICK_COMPOUNDS:
                count += 1
    return count


def scan_year(
    year: int,
    cache_dir: Path | None = None,
    round_start: int | None = None,
    round_end: int | None = None,
) -> pd.DataFrame:
    """Audit every historical FastF1 Race session for weather/tyre eligibility."""
    try:
        import fastf1
    except ImportError as error:  # pragma: no cover - dependency is declared in requirements.
        raise RuntimeError("FastF1 is required; install dependencies from requirements.txt.") from error

    cache = cache_dir or DATA_DIR / "fastf1_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache)
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if round_start is not None:
        schedule = schedule[schedule["RoundNumber"] >= round_start]
    if round_end is not None:
        schedule = schedule[schedule["RoundNumber"] <= round_end]
    rows: list[dict[str, object]] = []

    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        event_name = str(event["EventName"])
        base_row = {"year": year, "round": round_number, "event_name": event_name}
        try:
            session = fastf1.get_session(year, round_number, "R")
            session.load(telemetry=False, weather=True, messages=False)
        except Exception as error:  # Public historical sources can lack individual sessions.
            rows.append({**base_row, "selection_status": "session_unavailable", "detail": type(error).__name__})
            continue

        try:
            weather = session.weather_data
        except Exception as error:
            rows.append({**base_row, "selection_status": "weather_unavailable", "detail": type(error).__name__})
            continue
        rainfall_records = int(pd.to_numeric(weather.get("Rainfall"), errors="coerce").fillna(False).astype(bool).sum())
        if rainfall_records == 0:
            rows.append({**base_row, "rainfall_records": 0, "eligible_transitions": 0, "selection_status": "no_rainfall_record"})
            continue

        try:
            laps = session.laps
        except Exception as error:
            rows.append(
                {
                    **base_row,
                    "rainfall_records": rainfall_records,
                    "eligible_transitions": 0,
                    "selection_status": "timing_laps_unavailable",
                    "detail": type(error).__name__,
                }
            )
            continue
        transitions = count_wet_to_slick_transitions(laps)
        status = "candidate" if transitions else "no_wet_to_slick_transition"
        rows.append(
            {
                **base_row,
                "rainfall_records": rainfall_records,
                "eligible_transitions": transitions,
                "selection_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a historical FastF1 season for V2.4 candidates.")
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--round-start", type=int, default=None)
    parser.add_argument("--round-end", type=int, default=None)
    args = parser.parse_args()

    audit = scan_year(args.year, round_start=args.round_start, round_end=args.round_end)
    output = args.output or RESULTS_DIR / f"v2_fastf1_audit_{args.year}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    print(f"Saved {len(audit)} audit rows to {output}")
    print(audit["selection_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
