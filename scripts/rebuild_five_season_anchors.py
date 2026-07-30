"""Rebuild the five-season Task-11 outputs with honest timing anchors (Task 12).

Reads the *existing* per-year audit/event CSVs (produced by the frozen V2.4
rule), enriches every event row with an independent FastF1 pit-service anchor
(reused from the cached FastF1 sessions — no re-scan, no rule change), and
recombines the combined outputs. The frozen rule is never re-run, so candidate
selection and the V2.4 transition/signal classification are byte-for-byte
unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fastf1_five_season_runner import SEASONS, combine
from src.fastf1_timing_anchor import derive_race_anchors

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

NEW_COLUMNS = [
    "pit_service_lap",
    "timing_anchor_lap",
    "timing_anchor_source",
    "reporting_signed_error_laps",
    "reporting_absolute_error_laps",
]


def _enrich_year(year: int) -> dict:
    audit_path = RESULTS_DIR / f"v3_fastf1_audit_{year}.csv"
    events_path = RESULTS_DIR / f"v3_fastf1_events_{year}.csv"
    if not audit_path.exists() or not events_path.exists():
        print(f"[{year}] missing CSVs; skip")
        return {}

    audit = pd.read_csv(audit_path)
    events = pd.read_csv(events_path)
    if events.empty:
        print(f"[{year}] no events; skip")
        return {}

    # Round lookup per race (events have event_name; audit has round+event_name).
    round_lookup = {
        (int(r.season), r.event_name): int(r.round)
        for r in audit.itertuples()
    }

    # Group events by race so we load each FastF1 session once.
    for (season, event_name), group in events.groupby(["season", "event_name"]):
        rnd = round_lookup.get((int(season), event_name))
        if rnd is None:
            # No matching audit row: keep first-slick fallback for every row.
            for idx in group.index:
                events.at[idx, "timing_anchor_source"] = "first_slick_fallback"
                events.at[idx, "pit_service_lap"] = None
                events.at[idx, "timing_anchor_lap"] = int(events.at[idx, "first_slick_lap"])
            continue
        ev_list = [
            {
                "driver": r.driver,
                "first_slick_lap": int(r.first_slick_lap),
                "wet_stint_laps": int(r.wet_stint_laps),
            }
            for r in group.itertuples()
        ]
        anchors = derive_race_anchors(int(season), rnd, ev_list)
        for idx in group.index:
            driver = events.at[idx, "driver"]
            fsl = int(events.at[idx, "first_slick_lap"])
            a = anchors.get(
                (driver, fsl),
                {
                    "pit_service_lap": None,
                    "timing_anchor_lap": fsl,
                    "timing_anchor_source": "first_slick_fallback",
                },
            )
            events.at[idx, "pit_service_lap"] = a["pit_service_lap"]
            events.at[idx, "timing_anchor_lap"] = a["timing_anchor_lap"]
            events.at[idx, "timing_anchor_source"] = a["timing_anchor_source"]
            sig = events.at[idx, "signal_lap"]
            signed = events.at[idx, "signed_error_laps"]
            if pd.notna(sig) and pd.notna(signed):
                rep = float(sig) - float(a["timing_anchor_lap"])
                events.at[idx, "reporting_signed_error_laps"] = rep
                events.at[idx, "reporting_absolute_error_laps"] = abs(rep)
            else:
                events.at[idx, "reporting_signed_error_laps"] = None
                events.at[idx, "reporting_absolute_error_laps"] = None

    # Ensure the new columns exist even if a season produced no anchors.
    for col in NEW_COLUMNS:
        if col not in events.columns:
            events[col] = None

    events.to_csv(events_path, index=False)
    return {"rows": len(events), "path": str(events_path)}


def main() -> None:
    for year in SEASONS:
        info = _enrich_year(year)
        print(f"[{year}] enriched {info.get('rows', 0)} event rows")
    combine()
    print("Rebuild complete.")


if __name__ == "__main__":
    main()
