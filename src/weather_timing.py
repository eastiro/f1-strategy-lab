from __future__ import annotations

import pandas as pd


def prepare_driver_lap_timing(
    laps: pd.DataFrame,
    driver: str,
    default_lap_duration_seconds: float = 90.0,
) -> pd.DataFrame:
    """Return one driver's valid laps with normalized start and end timestamps."""
    required_columns = {"name_acronym", "lap_number", "date_start", "lap_duration"}
    if not required_columns.issubset(laps.columns):
        return pd.DataFrame(columns=["lap_number", "lap_start_time", "lap_duration", "lap_end_time"])

    driver_laps = laps[
        (laps["name_acronym"] == driver)
        & (laps["lap_number"].notna())
        & (laps["date_start"].notna())
    ].copy()
    if driver_laps.empty:
        return pd.DataFrame(columns=["lap_number", "lap_start_time", "lap_duration", "lap_end_time"])

    driver_laps["lap_number"] = pd.to_numeric(driver_laps["lap_number"], errors="coerce")
    driver_laps["lap_start_time"] = pd.to_datetime(
        driver_laps["date_start"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    driver_laps["lap_duration"] = pd.to_numeric(driver_laps["lap_duration"], errors="coerce").fillna(
        default_lap_duration_seconds
    )
    driver_laps = driver_laps.dropna(subset=["lap_number", "lap_start_time"]).copy()
    if driver_laps.empty:
        return pd.DataFrame(columns=["lap_number", "lap_start_time", "lap_duration", "lap_end_time"])

    driver_laps["lap_number"] = driver_laps["lap_number"].astype(int)
    driver_laps["lap_end_time"] = driver_laps["lap_start_time"] + pd.to_timedelta(
        driver_laps["lap_duration"],
        unit="s",
    )
    return driver_laps.sort_values("lap_number").reset_index(drop=True)


def map_timestamps_to_laps(
    lap_timing: pd.DataFrame,
    timestamps: pd.Series,
    total_laps: int,
) -> list[int]:
    """Map event timestamps to containing laps, falling back to the nearest lap end."""
    required_columns = {"lap_number", "lap_start_time", "lap_end_time"}
    if lap_timing.empty or not required_columns.issubset(lap_timing.columns):
        return []

    normalized_timestamps = pd.to_datetime(timestamps, utc=True, errors="coerce", format="mixed").dropna()
    mapped_laps: set[int] = set()
    for timestamp in normalized_timestamps:
        matching_laps = lap_timing[
            (lap_timing["lap_start_time"] <= timestamp) & (lap_timing["lap_end_time"] >= timestamp)
        ]
        if matching_laps.empty:
            nearest_index = (lap_timing["lap_end_time"] - timestamp).abs().idxmin()
            lap_number = int(lap_timing.loc[nearest_index, "lap_number"])
        else:
            lap_number = int(matching_laps["lap_number"].iloc[0])
        if 1 <= lap_number <= total_laps:
            mapped_laps.add(lap_number)
    return sorted(mapped_laps)
