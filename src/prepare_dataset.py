from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DEFAULT_COUNTRY, DEFAULT_YEAR, PROCESSED_DIR, RAW_DIR
from .openf1_client import make_race_slug


def load_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src.download_openf1 first.")
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def prepare_race_dataset(year: int, country: str) -> dict[str, Path]:
    slug = make_race_slug(year, country)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    drivers = load_json(RAW_DIR / f"{slug}_drivers.json")
    laps = load_json(RAW_DIR / f"{slug}_laps.json")
    pit = load_json(RAW_DIR / f"{slug}_pit.json")
    stints = load_json(RAW_DIR / f"{slug}_stints.json")
    weather = load_json(RAW_DIR / f"{slug}_weather.json")

    driver_cols = [
        "driver_number",
        "broadcast_name",
        "full_name",
        "name_acronym",
        "team_name",
        "team_colour",
    ]
    drivers = drivers[[col for col in driver_cols if col in drivers.columns]].drop_duplicates()

    lap_cols = [
        "session_key",
        "driver_number",
        "lap_number",
        "lap_duration",
        "duration_sector_1",
        "duration_sector_2",
        "duration_sector_3",
        "is_pit_out_lap",
        "date_start",
    ]
    laps = laps[[col for col in lap_cols if col in laps.columns]]
    laps = laps.dropna(subset=["lap_number", "driver_number"])
    laps["lap_number"] = laps["lap_number"].astype(int)

    if not stints.empty:
        stints = stints.rename(columns={"lap_start": "stint_lap_start", "lap_end": "stint_lap_end"})
        lap_rows = []
        for _, lap in laps.iterrows():
            driver_stints = stints[stints["driver_number"] == lap["driver_number"]]
            match = driver_stints[
                (driver_stints["stint_lap_start"] <= lap["lap_number"])
                & (driver_stints["stint_lap_end"] >= lap["lap_number"])
            ]
            row = lap.to_dict()
            if not match.empty:
                stint = match.iloc[0]
                row["compound"] = stint.get("compound")
                row["stint_number"] = stint.get("stint_number")
                row["tyre_age_at_lap_start"] = (
                    stint.get("tyre_age_at_start", 0)
                    + lap["lap_number"]
                    - stint.get("stint_lap_start", lap["lap_number"])
                )
                row["stint_lap_start"] = stint.get("stint_lap_start")
                row["stint_lap_end"] = stint.get("stint_lap_end")
            lap_rows.append(row)
        laps = pd.DataFrame(lap_rows)

    laps = laps.merge(drivers, on="driver_number", how="left")

    if not pit.empty:
        pit_summary = pit[["driver_number", "lap_number", "pit_duration"]].copy()
        pit_summary["pit_lap"] = True
        laps = laps.merge(pit_summary, on=["driver_number", "lap_number"], how="left")
    else:
        laps["pit_lap"] = False
        laps["pit_duration"] = pd.NA

    laps["pit_lap"] = laps["pit_lap"].where(laps["pit_lap"].notna(), False).astype(bool)
    laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    laps["clean_lap"] = (
        laps["lap_duration"].notna()
        & ~laps["pit_lap"]
        & ~laps.get("is_pit_out_lap", False).fillna(False)
    )

    lap_path = PROCESSED_DIR / f"{slug}_laps_clean.csv"
    pit_path = PROCESSED_DIR / f"{slug}_pit.csv"
    stint_path = PROCESSED_DIR / f"{slug}_stints.csv"
    weather_path = PROCESSED_DIR / f"{slug}_weather.csv"

    laps.to_csv(lap_path, index=False)
    pit.to_csv(pit_path, index=False)
    stints.to_csv(stint_path, index=False)
    weather.to_csv(weather_path, index=False)

    return {
        "laps": lap_path,
        "pit": pit_path,
        "stints": stint_path,
        "weather": weather_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare clean race datasets.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    args = parser.parse_args()

    saved = prepare_race_dataset(args.year, args.country)
    for name, path in saved.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
