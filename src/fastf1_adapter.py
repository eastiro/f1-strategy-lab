from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, PROCESSED_DIR, RAW_DIR
from .openf1_client import make_race_slug


def build_stints_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Convert FastF1 lap rows into the small stint schema used by V2."""
    rows: list[dict[str, object]] = []
    valid = laps.dropna(subset=["Driver", "DriverNumber", "LapNumber", "Stint", "Compound"])
    for driver, driver_laps in valid.groupby("Driver"):
        for stint_number, stint_laps in driver_laps.groupby("Stint"):
            ordered = stint_laps.sort_values("LapNumber")
            rows.append(
                {
                    "driver_number": int(ordered["DriverNumber"].iloc[0]),
                    "stint_number": int(stint_number),
                    "compound": str(ordered["Compound"].iloc[0]).upper(),
                    "stint_lap_start": int(ordered["LapNumber"].min()),
                    "stint_lap_end": int(ordered["LapNumber"].max()),
                    "driver": str(driver),
                }
            )
    return pd.DataFrame(rows).sort_values(["driver_number", "stint_number"]).reset_index(drop=True)


def _session_timestamp(session: object, laps: pd.DataFrame) -> pd.Timestamp:
    """Find a timestamp origin even when legacy FastF1 sessions lack ``t0_date``."""
    try:
        value = getattr(session, "t0_date")
    except Exception:
        value = None
    if value is None or pd.isna(value):
        dated_laps = laps.dropna(subset=["LapStartDate", "LapStartTime"])
        if not dated_laps.empty:
            value = pd.Timestamp(dated_laps["LapStartDate"].iloc[0]) - pd.Timedelta(
                dated_laps["LapStartTime"].iloc[0]
            )
    if value is None or pd.isna(value):
        value = getattr(session, "date")
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def save_fastf1_bundle(year: int, event: str, country_slug: str | None = None) -> dict[str, Path]:
    """Download a FastF1 Race session into V2's normalized public-data schema."""
    try:
        import fastf1
    except ImportError as error:  # pragma: no cover - dependency is declared in requirements.
        raise RuntimeError("FastF1 is required; install dependencies from requirements.txt.") from error

    cache_dir = DATA_DIR / "fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, event, "R")
    session.load(telemetry=False, weather=True, messages=True)
    slug = make_race_slug(year, country_slug or event)
    source_laps = session.laps.copy()
    origin = _session_timestamp(session, source_laps)
    lap_start_dates = pd.to_datetime(source_laps["LapStartDate"], utc=True, errors="coerce")
    fallback_start = origin + pd.to_timedelta(source_laps["LapStartTime"], errors="coerce")
    laps = pd.DataFrame(
        {
            "name_acronym": source_laps["Driver"],
            "driver_number": pd.to_numeric(source_laps["DriverNumber"], errors="coerce"),
            "lap_number": pd.to_numeric(source_laps["LapNumber"], errors="coerce"),
            "date_start": lap_start_dates.fillna(fallback_start).astype("string"),
            "lap_duration": pd.to_timedelta(source_laps["LapTime"], errors="coerce").dt.total_seconds(),
        }
    ).dropna(subset=["name_acronym", "driver_number", "lap_number", "date_start"])
    laps["driver_number"] = laps["driver_number"].astype(int)
    laps["lap_number"] = laps["lap_number"].astype(int)

    weather_source = session.weather_data.copy()
    weather_dates = origin + pd.to_timedelta(weather_source["Time"], errors="coerce")
    weather = pd.DataFrame(
        {
            "date": weather_dates.astype("string"),
            "rainfall": weather_source["Rainfall"].fillna(False).astype(bool).astype(int),
            "track_temperature": pd.to_numeric(weather_source.get("TrackTemp"), errors="coerce"),
            "humidity": pd.to_numeric(weather_source.get("Humidity"), errors="coerce"),
        }
    ).dropna(subset=["date"])

    messages = session.race_control_messages.copy()
    message_time = messages.get("Time", pd.Series(dtype="timedelta64[ns]"))
    if pd.api.types.is_datetime64_any_dtype(message_time):
        message_dates = pd.to_datetime(message_time, utc=True, errors="coerce")
    else:
        message_dates = origin + pd.to_timedelta(message_time, errors="coerce")
    race_control = pd.DataFrame(
        {
            "lap_number": pd.to_numeric(messages.get("Lap"), errors="coerce"),
            "message": messages.get("Message", pd.Series(dtype=str)),
            "category": messages.get("Category", pd.Series(dtype=str)),
            "date": message_dates.astype("string"),
        }
    ).dropna(subset=["lap_number"])
    race_control["lap_number"] = race_control["lap_number"].astype(int)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "laps": PROCESSED_DIR / f"{slug}_laps_clean.csv",
        "stints": PROCESSED_DIR / f"{slug}_stints.csv",
        "weather": PROCESSED_DIR / f"{slug}_weather.csv",
        "race_control": RAW_DIR / f"{slug}_race_control.json",
        "metadata": RAW_DIR / f"{slug}_fastf1_metadata.json",
    }
    laps.to_csv(paths["laps"], index=False)
    build_stints_from_laps(source_laps).to_csv(paths["stints"], index=False)
    weather.to_csv(paths["weather"], index=False)
    paths["race_control"].write_text(race_control.to_json(orient="records", force_ascii=False), encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps({"source": "FastF1", "year": year, "event": event, "slug": slug}, indent=2),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a FastF1 Race session for V2 evaluation.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--country-slug", default=None)
    args = parser.parse_args()
    paths = save_fastf1_bundle(args.year, args.event, args.country_slug)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
