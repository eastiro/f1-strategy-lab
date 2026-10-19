from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from .openf1_client import make_race_slug
from .strategy_model import build_wetness_from_weather
from .v2_validation import (
    RaceCase,
    dry_track_confirmation_details,
    find_recovery_slick_window,
    focal_stints,
    fresh_crossover_wetness,
    observed_wet_to_slick_transitions,
)
from .weather_timing import map_timestamps_to_laps, prepare_driver_lap_timing


def rainfall_event_laps(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    driver: str,
    total_laps: int,
) -> list[int]:
    """Map raw rainfall observations to the affected laps for one driver."""
    if weather.empty or "rainfall" not in weather.columns or "date" not in weather.columns:
        return []
    driver_laps = prepare_driver_lap_timing(laps, driver)
    if driver_laps.empty:
        return []
    weather_rows = weather.copy()
    weather_rows["rainfall"] = pd.to_numeric(weather_rows["rainfall"], errors="coerce").fillna(0)
    return map_timestamps_to_laps(
        driver_laps,
        weather_rows.loc[weather_rows["rainfall"] > 0, "date"],
        total_laps,
    )


def count_rain_episodes(rain_laps: list[int], end_lap: int) -> int:
    relevant = [lap for lap in rain_laps if lap <= end_lap]
    if not relevant:
        return 0
    episodes = 1
    for previous, current in zip(relevant, relevant[1:]):
        if current > previous + 1:
            episodes += 1
    return episodes


def nearby_race_control_phases(year: int, country: str, lap: int, radius: int = 3) -> str:
    slug = make_race_slug(year, country)
    path = RAW_DIR / f"{slug}_race_control.json"
    if not path.exists():
        return ""
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if rows.empty or "lap_number" not in rows.columns:
        return ""
    nearby = rows[pd.to_numeric(rows["lap_number"], errors="coerce").between(lap - radius, lap + radius)]
    messages = nearby.get("message", pd.Series("", index=nearby.index)).fillna("").astype(str).str.upper()
    phases: list[str] = []
    if messages.str.contains("RED FLAG").any():
        phases.append("red_flag")
    if messages.str.contains("VIRTUAL SAFETY").any():
        phases.append("virtual_safety_car")
    if messages.str.contains("SAFETY CAR").any():
        phases.append("safety_car")
    return "|".join(phases)


def evaluate_holdout_race(year: int, country: str) -> pd.DataFrame:
    slug = make_race_slug(year, country)
    laps = pd.read_csv(PROCESSED_DIR / f"{slug}_laps_clean.csv")
    stints = pd.read_csv(PROCESSED_DIR / f"{slug}_stints.csv")
    weather = pd.read_csv(PROCESSED_DIR / f"{slug}_weather.csv")
    total_laps = int(laps["lap_number"].max())
    race_case = RaceCase(
        case_id=f"{slug}_holdout",
        year=year,
        country=country,
        title=f"{year} {country}",
        focal_driver="",
        purpose="Holdout evaluation",
        review_note="Frozen V2.3 rules.",
    )

    rows: list[dict[str, object]] = []
    for driver in sorted(laps["name_acronym"].dropna().unique()):
        driver_stints = focal_stints(laps, stints, driver)
        transitions = observed_wet_to_slick_transitions(driver_stints)
        if not transitions:
            continue

        rainfall_wetness = build_wetness_from_weather(laps, weather, driver, total_laps)
        rain_laps = rainfall_event_laps(laps, weather, driver, total_laps)
        for transition in transitions:
            actual_lap = int(transition["lap"])
            wet_stint_start = int(transition["wet_stint_start"])
            wet_stint_laps = int(transition["wet_stint_laps"])
            compound = str(transition["to_compound"])
            crossover = fresh_crossover_wetness(compound)
            rainfall_window = find_recovery_slick_window(
                rainfall_wetness,
                crossover,
                total_laps,
                start_lap=wet_stint_start,
            )
            confirmation_lap, confirmation_source = dry_track_confirmation_details(
                race_case,
                wet_stint_start,
                total_laps,
            )
            candidates = [value for value in [rainfall_window, confirmation_lap] if value is not None]
            public_latest = max(candidates) if candidates else None
            public_earliest = min(candidates) if candidates else None
            rain_episodes = count_rain_episodes(rain_laps, actual_lap)
            race_control_phases = nearby_race_control_phases(year, country, actual_lap)
            if wet_stint_laps <= 2:
                event_class = "short_wet_stint"
            elif actual_lap <= 5:
                event_class = "early_race_transition"
            else:
                event_class = "sustained_wet_to_dry"
            if event_class != "sustained_wet_to_dry":
                event_quality = "start_or_restart"
            elif race_control_phases:
                event_quality = "race_control_adjacent"
            elif rain_episodes >= 2:
                event_quality = "multi_rain_phase"
            else:
                event_quality = "single_rain_phase"
            rows.append(
                {
                    "year": year,
                    "country": country,
                    "driver": driver,
                    "actual_transition_lap": actual_lap,
                    "wet_stint_laps": wet_stint_laps,
                    "rain_episodes_before_transition": rain_episodes,
                    "nearby_race_control_phases": race_control_phases,
                    "to_compound": compound,
                    "rainfall_only_window_start": rainfall_window,
                    "dry_track_confirmation_lap": confirmation_lap,
                    "dry_track_confirmation_source": confirmation_source,
                    "public_signal_window_start": public_earliest,
                    "public_signal_window_end": public_latest,
                    "public_signal_error_laps": None if public_latest is None else public_latest - actual_lap,
                    "event_class": event_class,
                    "event_quality": event_quality,
                }
            )
    return pd.DataFrame(rows)


def evaluate_v24_state_machine_race(year: int, country: str) -> pd.DataFrame:
    """Evaluate V2.4's state-machine rule for one normalized public-data race."""
    slug = make_race_slug(year, country)
    laps = pd.read_csv(PROCESSED_DIR / f"{slug}_laps_clean.csv")
    stints = pd.read_csv(PROCESSED_DIR / f"{slug}_stints.csv")
    weather = pd.read_csv(PROCESSED_DIR / f"{slug}_weather.csv")
    total_laps = int(laps["lap_number"].max())
    race_case = RaceCase(
        case_id=f"{slug}_v24_diagnostic",
        year=year,
        country=country,
        title=f"{year} {country}",
        focal_driver="",
        purpose="V2.4 state-machine evaluation",
        review_note="Source and validation split must be documented by the caller.",
    )
    rows: list[dict[str, object]] = []
    for driver in sorted(laps["name_acronym"].dropna().unique()):
        driver_stints = focal_stints(laps, stints, driver)
        transitions = observed_wet_to_slick_transitions(driver_stints)
        if not transitions:
            continue
        rainfall_wetness = build_wetness_from_weather(laps, weather, driver, total_laps)
        rain_laps = rainfall_event_laps(laps, weather, driver, total_laps)
        for transition in transitions:
            actual_lap = int(transition["lap"])
            wet_stint_start = int(transition["wet_stint_start"])
            wet_stint_laps = int(transition["wet_stint_laps"])
            compound = str(transition["to_compound"])
            crossover = fresh_crossover_wetness(compound)
            rainfall_window = find_recovery_slick_window(
                rainfall_wetness,
                crossover,
                total_laps,
                start_lap=wet_stint_start,
            )
            recent_rain_laps = [lap for lap in rain_laps if wet_stint_start <= lap <= actual_lap]
            confirmation_reset_lap = max(recent_rain_laps, default=wet_stint_start)
            rain_episodes = count_rain_episodes(rain_laps, actual_lap)
            race_control_phases = nearby_race_control_phases(year, country, actual_lap)
            if wet_stint_laps <= 2 or actual_lap <= 5:
                event_quality = "start_or_restart"
            elif race_control_phases:
                event_quality = "race_control_adjacent"
            elif rain_episodes >= 2:
                event_quality = "multi_rain_phase"
            else:
                event_quality = "single_rain_phase"
            confirmation_lap, confirmation_source = dry_track_confirmation_details(
                race_case,
                confirmation_reset_lap,
                total_laps,
            )
            candidates = [value for value in [rainfall_window, confirmation_lap] if value is not None]
            public_latest = max(candidates) if candidates else None
            # V2.4's first state-machine gate does not force a signal through
            # situations where a race-level dry message cannot identify a driver's
            # individual decision window.
            signal_eligible = event_quality == "single_rain_phase"
            rows.append(
                {
                    "year": year,
                    "country": country,
                    "driver": driver,
                    "actual_transition_lap": actual_lap,
                    "wet_stint_laps": wet_stint_laps,
                    "recent_rain_reset_lap": confirmation_reset_lap,
                    "rain_episodes_before_transition": rain_episodes,
                    "nearby_race_control_phases": race_control_phases,
                    "event_quality": event_quality,
                    "state_machine_outcome": "signal" if signal_eligible else "abstain",
                    "signal_eligible": signal_eligible,
                    "rainfall_only_window_start": rainfall_window,
                    "dry_track_confirmation_lap": confirmation_lap,
                    "dry_track_confirmation_source": confirmation_source,
                    "public_signal_error_laps": (
                        None if public_latest is None or not signal_eligible else public_latest - actual_lap
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_holdout_race_v24_diagnostic(year: int, country: str) -> pd.DataFrame:
    """Backward-compatible name for the already-seen 2025 V2.4 diagnostic."""
    return evaluate_v24_state_machine_race(year, country)


def summarise_holdout_results(results: pd.DataFrame) -> dict[str, float | int]:
    errors = pd.to_numeric(results.get("public_signal_error_laps"), errors="coerce").dropna().abs()
    sustained = (
        results[results["event_class"] == "sustained_wet_to_dry"]
        if "event_class" in results.columns
        else results
    )
    sustained_errors = pd.to_numeric(sustained.get("public_signal_error_laps"), errors="coerce").dropna().abs()
    return {
        "observed_transitions": int(len(results)),
        "public_signal_available": int(len(errors)),
        "within_two_laps": int((errors <= 2).sum()),
        "mean_absolute_error": round(float(errors.mean()), 2) if not errors.empty else 0.0,
        "sustained_transitions": int(len(sustained)),
        "sustained_mean_absolute_error": round(float(sustained_errors.mean()), 2)
        if not sustained_errors.empty
        else 0.0,
    }


def summarise_signal_coverage(results: pd.DataFrame) -> dict[str, float | int]:
    """Summarise an abstaining decision rule without treating abstention as error."""
    if "signal_eligible" not in results.columns:
        return {"events": int(len(results)), "signals_issued": int(len(results)), "abstentions": 0}
    eligible = results[results["signal_eligible"].fillna(False)].copy()
    errors = pd.to_numeric(eligible.get("public_signal_error_laps"), errors="coerce").dropna().abs()
    return {
        "events": int(len(results)),
        "signals_issued": int(len(eligible)),
        "abstentions": int(len(results) - len(eligible)),
        "issued_within_two_laps": int((errors <= 2).sum()),
        "issued_mean_absolute_error": round(float(errors.mean()), 2) if not errors.empty else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen V2.3 rules on holdout races.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--countries", nargs="+", default=["Australia", "United Kingdom", "Belgium"])
    parser.add_argument("--mode", choices=["v23", "v24_diagnostic"], default="v23")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    evaluator = evaluate_holdout_race if args.mode == "v23" else evaluate_holdout_race_v24_diagnostic
    results = [evaluator(args.year, country) for country in args.countries]
    report = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    output = args.output or RESULTS_DIR / f"v2_holdout_{args.mode}_{args.year}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    summary = summarise_holdout_results(report)
    print(f"Saved {len(report)} transition rows to {output}")
    print(summary)
    if args.mode == "v24_diagnostic":
        print(summarise_signal_coverage(report))


if __name__ == "__main__":
    main()
