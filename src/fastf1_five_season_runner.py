"""Resumable five-season (2021-2025) FastF1 robustness audit runner.

This module is a *narrow* orchestration layer. It reuses the existing frozen
rule chain unchanged:

- ``fastf1_scanner.scan_year``        (candidate / audit-status scan)
- ``fastf1_adapter.save_fastf1_bundle`` (normalize a session into V2 schema)
- ``evaluate_holdouts.evaluate_v24_state_machine_race`` (frozen V2.4 rule)

It adds NO evaluator or threshold logic. It exists only to make a uniform,
resumable, year/round-batched scan across 2021-2025 and to write the audit and
event CSVs the task requires.

Run per year (background-friendly)::

    python -m src.fastf1_five_season_runner --year 2022
    python -m src.fastf1_five_season_runner --year 2023 --round-start 1 --round-end 11

Then combine all per-year outputs::

    python -m src.fastf1_five_season_runner --combine
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

# --- Network fix (Task 11) -------------------------------------------------
# Execution-time finding (2026-07-23): the F1 livetiming *primary* host
# (livetiming.formula1.com) returned HTTP 200 for 2021-2025 at scan time, so it
# is the working source. The official mirror (livetiming-mirror.fastf1.dev) only
# serves <=2022 and 404s for 2023-2025, so it must NOT be forced. We therefore
# keep FastF1's DEFAULT primary host (no base_url override). FastF1's own
# built-in mirror fallback (on a raised exception) still applies if the primary
# host ever 403s again. This is a runtime redirect only -- it does NOT edit the
# frozen scanner/adapter/evaluator source files or change any normalization /
# V2.4 semantics (protocol SHA-256 intact).
#
# Earlier in the project (2026-07-22) the primary host WAS 403-blocked to this
# sandbox's egress IP; that block had lifted by execution time, which is why the
# mirror was briefly needed and then reverted.
import fastf1
import fastf1._api as _fastf1_api
import fastf1.req as _fastf1_req

# (No base_url override -- use the default primary host that works at scan time.)

# --- Anti-hang fix (Task 11, 2026-07-23) ------------------------------------
# FastF1 performs *every* HTTP request (GET and POST, cached or not) through
# ``fastf1.req.Cache._cached_request`` (fastf1/req.py). That method forwards
# ``**kwargs`` straight to the underlying ``requests.Session`` and NEVER sets a
# ``timeout``. Under burst load the F1 livetiming host can *silently hang* a
# connection (socket open, no bytes) -- with no timeout this freezes the whole
# scan forever. OBSERVED: on 2026-07-23 all three concurrent year-scans stalled
# >50 min mid-fetch (logs frozen at "Fetching weather/session status/driver
# list..."). Patching ``requests.Session.get/post`` does NOT help, because the
# cached-session class overrides those and funnels through ``_cached_request``.
# So we wrap ``Cache._cached_request`` itself -- FastF1's own single chokepoint,
# which requests-cache does not override -- to inject a timeout on every request.
# A hung fetch then becomes a fast ``requests.exceptions.Timeout``; FastF1 retries
# and/or our pre-warm loop records the session as unavailable -- never a stall.
# Runtime-only; no frozen file is touched.
import requests as _requests

_REQUEST_TIMEOUT = 90


def _patch_fastf1_request_timeout():
    # Primary: wrap FastF1's single HTTP chokepoint (classmethod).
    _orig_cached_request = _fastf1_req.Cache._cached_request.__func__

    def _cached_request_with_timeout(cls, method, url, **kwargs):
        kwargs.setdefault("timeout", _REQUEST_TIMEOUT)
        return _orig_cached_request(cls, method, url, **kwargs)

    _fastf1_req.Cache._cached_request = classmethod(_cached_request_with_timeout)

    # Backstop: also patch the lowest-level session entry point, in case any path
    # (e.g. the non-cached fallback session) bypasses _cached_request.
    _orig_request = _requests.Session.request

    def _request_with_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", _REQUEST_TIMEOUT)
        return _orig_request(self, method, url, **kwargs)

    _requests.Session.request = _request_with_timeout


_patch_fastf1_request_timeout()

from .config import DATA_DIR, PROCESSED_DIR, RAW_DIR, RESULTS_DIR

# Use the project's persistent cache (not FastF1's default TEMP cache) so the
# audit is reproducible and reuses already-cached seasons (e.g. 2021). The
# frozen scanner/adapter/evaluator read from this same cache at runtime.
fastf1.Cache.enable_cache(DATA_DIR / "fastf1_cache")

from .evaluate_holdouts import evaluate_v24_state_machine_race, summarise_signal_coverage
from .fastf1_adapter import save_fastf1_bundle
from .fastf1_scanner import scan_year
from .fastf1_timing_anchor import derive_race_anchors
from .openf1_client import make_race_slug

SEASONS = [2021, 2022, 2023, 2024, 2025]

EXPOSURE_LABELS = {
    2021: "previously audited",
    2022: "new pre-registered extension",
    2023: "partially development-exposed",
    2024: "partially development-exposed",
    2025: "development-exposed",
}

# FastF1 EventName -> short slug-name, so bundle filenames match the existing
# 2021_hungary / 2024_united_kingdom / 2023_netherlands / ... bundles and the
# make_race_slug contract used by the frozen evaluator.
_EVENT_SUFFIX_OVERRIDES = {
    "Hungarian": "Hungary",
    "British": "United Kingdom",
    "Dutch": "Netherlands",
    "Canadian": "Canada",
    "Brazilian": "Brazil",
    "Australian": "Australia",
    "Belgian": "Belgium",
    "Austrian": "Austria",
    "Russian": "Russia",
    "Spanish": "Spain",
    "Portuguese": "Portugal",
    "French": "France",
    "Styrian": "Styria",
    "Italian": "Italy",
    "Sao Paulo": "Sao Paulo",
    "São Paulo": "Sao Paulo",
    "Saudi Arabian": "Saudi Arabia",
    "Mexico City": "Mexico City",
    "United States": "United States",
    "Abu Dhabi": "Abu Dhabi",
    "Qatar": "Qatar",
    "Singapore": "Singapore",
    "Japanese": "Japan",
    "Chinese": "China",
    "Miami": "Miami",
    "Las Vegas": "Las Vegas",
    "Bahrain": "Bahrain",
    "Azerbaijan": "Azerbaijan",
    "Monaco": "Monaco",
    "Emilia Romagna": "Emilia Romagna",
}


def short_slug_name(event_name: str) -> str:
    """Map a FastF1 EventName to the short slug-name used by the frozen bundles."""
    stripped = event_name.replace(" Grand Prix", "").strip()
    return _EVENT_SUFFIX_OVERRIDES.get(stripped, stripped)


def _bundle_exists(slug: str) -> bool:
    return (PROCESSED_DIR / f"{slug}_laps_clean.csv").exists()


def _prewarm_session(year: int, rnd: int, attempts: int = 4) -> bool:
    """Load a session once via the mirror so the frozen logic reads from cache.

    The mirror can rate-limit a burst of uncached streams, dropping the last
    ones (weather / race-control). Retry a few times -- each retry only refetches
    the still-missing streams (the rest are now cached), so it succeeds quickly.
    """
    import fastf1

    for attempt in range(1, attempts + 1):
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(telemetry=False, weather=True, messages=True)
        except Exception as error:  # session may be unavailable; frozen logic records it
            print(f"  [prewarm {year} R{rnd}] load error (attempt {attempt}): {type(error).__name__}", flush=True)
            time.sleep(3 * attempt)
            continue
        # NOTE: FastF1 exposes `weather_data` / `race_control_messages` as
        # properties that RAISE DataNotLoadedError when the stream failed to
        # load (mirror rate-limit). Access the private backing attributes
        # directly to test presence without triggering that exception.
        weather = getattr(session, "_weather_data", None)
        rc = getattr(session, "_race_control_messages", None)
        weather_len = 0 if weather is None else len(weather)
        rc_len = 0 if rc is None else len(rc)
        if weather_len > 0 and rc_len > 0:
            return True
        print(f"  [prewarm {year} R{rnd}] weather={weather_len} race_control={rc_len} (retry {attempt}/{attempts})", flush=True)
        time.sleep(4 + 3 * attempt)
    print(f"  [prewarm {year} R{rnd}] gave up after {attempts} attempts; frozen logic will record status", flush=True)
    return False


def _prewarm_year(year: int, round_start: int | None, round_end: int | None) -> None:
    """Pre-load every Race session of a season into the FastF1 cache."""
    import fastf1

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as error:
        print(f"[prewarm {year}] schedule fetch failed: {type(error).__name__}; scan_year will handle", flush=True)
        return
    if round_start is not None:
        schedule = schedule[schedule["RoundNumber"] >= round_start]
    if round_end is not None:
        schedule = schedule[schedule["RoundNumber"] <= round_end]
    print(f"[prewarm {year}] {len(schedule)} sessions via primary host (timeout={_REQUEST_TIMEOUT}s)", flush=True)
    for _, event in schedule.iterrows():
        # Small gap between sessions to avoid triggering mirror burst throttling.
        time.sleep(2)
        _prewarm_session(year, int(event["RoundNumber"]))


def _safe_int(v, default=0):
    """NaN/None/inf-safe int cast. FastF1 weather stats can be NaN."""
    if pd.isna(v):
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError, OverflowError):
        return default


def run_year(
    year: int,
    round_start: int | None = None,
    round_end: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan one season and evaluate every candidate. Returns (audit_df, events_df)."""
    exposure = EXPOSURE_LABELS[year]
    # Pre-warm every session through the mirror so scan_year / save_fastf1_bundle /
    # evaluate_v24_state_machine_race read from cache (no silent weather loss).
    _prewarm_year(year, round_start, round_end)
    audit = scan_year(year, round_start=round_start, round_end=round_end)

    audit_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for _, row in audit.iterrows():
        event_name = str(row["event_name"])
        status = str(row["selection_status"])
        print(f"  [{year}] R{int(row['round'])} {event_name}: {status}", flush=True)
        short = short_slug_name(event_name)
        slug = make_race_slug(year, short)
        audit_rows.append(
            {
                "season": year,
                "round": int(row["round"]),
                "event_name": event_name,
                "source": "FastF1",
                "exposure_label": exposure,
                "rainfall_records": _safe_int(row.get("rainfall_records", 0)),
                "eligible_transitions": _safe_int(row.get("eligible_transitions", 0)),
                "selection_status": status,
                "detail": "" if pd.isna(row.get("detail")) else str(row["detail"]),
                "caveat": "",
            }
        )
        if status != "candidate":
            continue

        # Reuse an existing normalized bundle when present; otherwise build it.
        if not _bundle_exists(slug):
            try:
                save_fastf1_bundle(year, event_name, country_slug=short)
            except Exception as error:  # network/build failure: record, never crash the year
                audit_rows[-1]["selection_status"] = "bundle_error"
                audit_rows[-1]["detail"] = type(error).__name__
                audit_rows[-1]["caveat"] = "bundle build failed; recorded, not dropped"
                continue
        try:
            events = evaluate_v24_state_machine_race(year, short)
        except Exception as error:  # surface as a non-evaluable audit note, never silent
            audit_rows[-1]["selection_status"] = "evaluation_error"
            audit_rows[-1]["detail"] = type(error).__name__
            audit_rows[-1]["caveat"] = "evaluator raised; recorded, not dropped"
            continue
        if events.empty:
            continue
        anchor_inputs: list[dict] = []
        for _, ev in events.iterrows():
            signal = bool(ev.get("signal_eligible", False))
            signed = ev.get("public_signal_error_laps", None)
            fsl = int(ev["actual_transition_lap"])
            event_rows.append(
                {
                    "season": year,
                    "race": short,
                    "event_name": event_name,
                    "driver": ev["driver"],
                    "source": "FastF1",
                    "exposure_label": exposure,
                    "transition_lap": fsl,
                    "first_slick_lap": fsl,
                    "wet_stint_laps": int(ev["wet_stint_laps"]),
                    "rain_episodes_before_transition": int(ev["rain_episodes_before_transition"]),
                    "nearby_race_control_phases": ev.get("nearby_race_control_phases", ""),
                    "event_quality": ev["event_quality"],
                    "state_machine_outcome": ev["state_machine_outcome"],
                    "signal_eligible": signal,
                    "public_signal_error_laps": ev["public_signal_error_laps"],
                    "signal_lap": (
                        None
                        if not signal or ev["public_signal_error_laps"] is None
                        else int(ev["actual_transition_lap"] + ev["public_signal_error_laps"])
                    ),
                    "signed_error_laps": (None if signed is None else float(signed)) if signal else None,
                    "absolute_error_laps": (
                        None if signed is None else float(abs(signed))
                    )
                    if signal
                    else None,
                    "rainfall_only_window_start": ev.get("rainfall_only_window_start", None),
                    "dry_track_confirmation_lap": ev.get("dry_track_confirmation_lap", None),
                    "dry_track_confirmation_source": ev.get("dry_track_confirmation_source", None),
                    "bundle_slug": slug,
                    "caveat": "",
                }
            )
            anchor_inputs.append(
                {"driver": ev["driver"], "first_slick_lap": fsl, "wet_stint_laps": int(ev["wet_stint_laps"])}
            )

        # Task 12 honest timing anchor: derive an independent pit-service lap from
        # the cached FastF1 session when FastF1 yields one, otherwise keep the
        # first-slick lap as an explicit fallback. The frozen V2.4 rule output is
        # untouched; only these reporting-only columns are appended.
        if anchor_inputs:
            anchor_map = derive_race_anchors(year, int(row["round"]), anchor_inputs)
        else:
            anchor_map = {}
        for er in event_rows[-len(anchor_inputs):]:
            fsl = er["first_slick_lap"]
            a = anchor_map.get(
                (er["driver"], fsl),
                {
                    "pit_service_lap": None,
                    "timing_anchor_lap": fsl,
                    "timing_anchor_source": "first_slick_fallback",
                },
            )
            er["pit_service_lap"] = a["pit_service_lap"]
            er["timing_anchor_lap"] = a["timing_anchor_lap"]
            er["timing_anchor_source"] = a["timing_anchor_source"]
            sig = er.get("signal_lap")
            if sig is not None and er.get("signed_error_laps") is not None:
                er["reporting_signed_error_laps"] = float(sig) - float(a["timing_anchor_lap"])
                er["reporting_absolute_error_laps"] = abs(er["reporting_signed_error_laps"])
            else:
                er["reporting_signed_error_laps"] = None
                er["reporting_absolute_error_laps"] = None

    audit_df = pd.DataFrame(audit_rows)
    events_df = pd.DataFrame(event_rows)
    return audit_df, events_df


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # A previous run was killed mid-write, leaving a 0-byte file. Treat as
        # no existing data so the year is regenerated cleanly (overwrite).
        return pd.DataFrame()


def _completed_rounds(existing: pd.DataFrame, year: int) -> set[int]:
    if existing.empty:
        return set()
    mask = (existing["season"] == year) & (existing["selection_status"] != "evaluation_error")
    return set(existing.loc[mask, "round"].dropna().astype(int).tolist())


def write_year(year: int, round_start: int | None = None, round_end: int | None = None) -> None:
    """Run one year and idempotently append new rounds to the per-year CSVs."""
    audit_path = RESULTS_DIR / f"v3_fastf1_audit_{year}.csv"
    events_path = RESULTS_DIR / f"v3_fastf1_events_{year}.csv"

    existing_audit = _read_existing(audit_path)
    existing_events = _read_existing(events_path)
    done = _completed_rounds(existing_audit, year)

    audit_df, events_df = run_year(year, round_start=round_start, round_end=round_end)

    # Skip rounds already recorded (idempotent resume). The events frame from
    # this call already covers only this run's scope; dedup on combine handles
    # any overlap with previously written rows.
    if not audit_df.empty and done:
        audit_df = audit_df[~audit_df["round"].isin(done)]

    combined_audit = pd.concat([existing_audit, audit_df], ignore_index=True)
    combined_audit = combined_audit.drop_duplicates(subset=["season", "round", "event_name"], keep="last")
    combined_audit = combined_audit.sort_values(["round"]).reset_index(drop=True)
    combined_audit.to_csv(audit_path, index=False)

    # Events: keep only rows whose round was freshly scanned (avoid dup on resume).
    fresh_rounds = set(audit_df["round"].tolist()) if not audit_df.empty else set()
    if events_df is not None and not events_df.empty and fresh_rounds:
        new_events = events_df  # run_year already scopes to this call's rounds
    else:
        new_events = events_df if events_df is not None else pd.DataFrame()
    combined_events = pd.concat([existing_events, new_events], ignore_index=True)
    combined_events = combined_events.drop_duplicates(
        subset=["season", "race", "driver", "transition_lap"], keep="last"
    )
    combined_events.to_csv(events_path, index=False)

    print(f"[{year}] audit rows={len(combined_audit)} events={len(combined_events)}")
    print(summarise_signal_coverage(combined_events) if not combined_events.empty else "no events")


def combine() -> None:
    """Union all per-year CSVs into the two required combined outputs."""
    audit_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for year in SEASONS:
        a = RESULTS_DIR / f"v3_fastf1_audit_{year}.csv"
        e = RESULTS_DIR / f"v3_fastf1_events_{year}.csv"
        audit_frames.append(_read_existing(a))
        event_frames.append(_read_existing(e))

    if not audit_frames:
        raise RuntimeError("No per-year audit CSVs found; run the years first.")
    audit_all = pd.concat(audit_frames, ignore_index=True).sort_values(
        ["season", "round"]
    ).reset_index(drop=True)
    audit_all.to_csv(RESULTS_DIR / "v3_fastf1_audit_2021_2025.csv", index=False)

    events_all = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )
    events_all.to_csv(RESULTS_DIR / "v3_fastf1_events_2021_2025.csv", index=False)
    print(f"Combined audit={len(audit_all)} events={len(events_all)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable five-season FastF1 audit runner.")
    parser.add_argument("--year", type=int, choices=SEASONS, default=None)
    parser.add_argument("--round-start", type=int, default=None)
    parser.add_argument("--round-end", type=int, default=None)
    parser.add_argument("--combine", action="store_true", help="Union per-year CSVs into combined outputs.")
    args = parser.parse_args()

    if args.combine:
        combine()
        return
    if args.year is None:
        parser.error("either --year Y or --combine is required")
    write_year(args.year, round_start=args.round_start, round_end=args.round_end)


if __name__ == "__main__":
    main()
