from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from .config import RESULTS_DIR
from .openf1_client import _get
from .v2_validation import SLICK_COMPOUNDS, WET_COMPOUNDS


def count_wet_to_slick_transitions(stints: list[dict[str, object]]) -> int:
    frame = pd.DataFrame(stints)
    if frame.empty or not {"driver_number", "stint_number", "compound"}.issubset(frame.columns):
        return 0

    count = 0
    for _, driver_stints in frame.groupby("driver_number"):
        ordered = driver_stints.sort_values("stint_number").reset_index(drop=True)
        for index in range(1, len(ordered)):
            previous = str(ordered.loc[index - 1, "compound"]).upper()
            current = str(ordered.loc[index, "compound"]).upper()
            if previous in WET_COMPOUNDS and current in SLICK_COMPOUNDS:
                count += 1
    return count


def scan_holdout_audit(year: int) -> pd.DataFrame:
    """Audit every listed Race session before selecting unseen candidates."""
    sessions = _get("sessions", year=year, session_name="Race")
    rows: list[dict[str, object]] = []
    for session in sessions:
        session_key = session["session_key"]
        base_row = {
            "year": year,
            "country": session.get("country_name"),
            "meeting_name": session.get("meeting_name"),
            "date_start": session.get("date_start"),
            "session_key": session_key,
            "rainfall_records": 0,
            "eligible_transitions": 0,
        }
        try:
            weather = _get("weather", session_key=session_key)
        except requests.RequestException:
            rows.append({**base_row, "selection_status": "missing_weather_data"})
            continue
        rainfall_records = int(
            pd.to_numeric(pd.DataFrame(weather).get("rainfall", pd.Series(dtype=float)), errors="coerce")
            .fillna(0)
            .sum()
        )
        if rainfall_records == 0:
            rows.append({**base_row, "selection_status": "no_rainfall_record"})
            continue

        try:
            stints = _get("stints", session_key=session_key)
        except requests.RequestException:
            rows.append({**base_row, "rainfall_records": rainfall_records, "selection_status": "missing_stint_data"})
            continue
        transitions = count_wet_to_slick_transitions(stints)
        if transitions == 0:
            rows.append(
                {
                    **base_row,
                    "rainfall_records": rainfall_records,
                    "selection_status": "no_wet_to_slick_transition",
                }
            )
            continue

        rows.append(
            {
                **base_row,
                "rainfall_records": rainfall_records,
                "eligible_transitions": transitions,
                "selection_status": "candidate",
            }
        )
    return pd.DataFrame(rows).sort_values(["date_start", "country"]).reset_index(drop=True)


def scan_holdout_candidates(year: int) -> pd.DataFrame:
    audit = scan_holdout_audit(year)
    candidates = audit[audit["selection_status"] == "candidate"].copy()
    if candidates.empty:
        return candidates.assign(validation_split=pd.Series(dtype=str))
    candidates["validation_split"] = "holdout_candidate"
    return candidates.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan OpenF1 races for pre-defined holdout candidates.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "v2_holdout_candidates_2025.csv",
    )
    parser.add_argument("--audit-output", type=Path, default=None)
    args = parser.parse_args()

    audit = scan_holdout_audit(args.year)
    candidates = audit[audit["selection_status"] == "candidate"].copy()
    if not candidates.empty:
        candidates["validation_split"] = "holdout_candidate"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    audit_output = args.audit_output or args.output.with_name(f"{args.output.stem}_audit.csv")
    audit.to_csv(audit_output, index=False)
    print(f"Saved {len(candidates)} candidates to {args.output}")
    print(f"Saved {len(audit)} audit rows to {audit_output}")


if __name__ == "__main__":
    main()
