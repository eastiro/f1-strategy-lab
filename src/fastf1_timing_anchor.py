"""Reporting-only FastF1 timing-anchor helper (Task 12).

This module derives an *independent* pit-service anchor for each observed
wet-to-slick transition. It reuses the **cached** FastF1 session's own lap/pit
data: FastF1 exposes a per-lap ``PitInTime`` (and ``PitOutTime``), but the
project's FastF1 adapter does **not** persist a ``*_pit.csv`` file, so we read
the session directly rather than from a bundle.

It reuses the *frozen* ``v2_validation.match_pit_service_lap`` for the actual
pit-window selection (read-only import; no frozen file is edited and no rule
logic is added). It adds nothing to the V2.4 decision rule — it only affects
the post-race *comparison* anchor and its honest labelling.

Deterministic matching rule (also documented in the protocol / report):
  For a transition event (driver D, first_slick_lap F, wet_stint_laps W):
    1. wet_stint_start = F - W  (exact: first_slick = wet_stint_end + 1).
    2. Build the pit set: every FastF1 lap of D with PitInTime not null,
       carrying its lap_number L.
    3. Candidate pit laps: L in [wet_stint_start, F], extended to F+1 to absorb
       a one-lap FastF1 timing mismatch.
    4. Choose the latest candidate at or before F (the stop that fitted the new
       slick compound). If found -> pit_service_lap = L.
    5. If a pit_service_lap exists  -> timing_anchor_lap = pit_service_lap,
                                        timing_anchor_source = "pit_service".
    6. Otherwise                   -> timing_anchor_lap = F (first_slick_lap),
                                        timing_anchor_source = "first_slick_fallback",
                                        pit_service_lap = None.

A fallback is **never** labelled as a pit-service anchor. If FastF1 cannot yield
independent pit evidence for an event, the event stays in the data with
``first_slick_fallback`` — it is never dropped or converted to a pit row.
"""

from __future__ import annotations

import pandas as pd

import fastf1
import fastf1.req as _fastf1_req
import requests as _requests

from .config import DATA_DIR
from .v2_validation import match_pit_service_lap

# --- Same anti-hang timeout patch as the runner (runtime-only, idempotent) ---
_REQUEST_TIMEOUT = 90


def _ensure_patched() -> None:
    _orig_cached_request = _fastf1_req.Cache._cached_request.__func__

    def _cached_request_with_timeout(cls, method, url, **kwargs):
        kwargs.setdefault("timeout", _REQUEST_TIMEOUT)
        return _orig_cached_request(cls, method, url, **kwargs)

    _fastf1_req.Cache._cached_request = classmethod(_cached_request_with_timeout)

    _orig_request = _requests.Session.request

    def _request_with_timeout(self, method, url, **kwargs):
        kwargs.setdefault("timeout", _REQUEST_TIMEOUT)
        return _orig_request(self, method, url, **kwargs)

    _requests.Session.request = _request_with_timeout


_ensure_patched()
fastf1.Cache.enable_cache(DATA_DIR / "fastf1_cache")


def _load_session_laps(year: int, rnd: int) -> pd.DataFrame:
    """Return a race session's laps DataFrame (cached; laps only)."""
    session = fastf1.get_session(year, rnd, "R")
    session.load(telemetry=False, weather=False, messages=False)
    return session.laps.copy()


def _build_pits(laps: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real pit stop: driver_number + lap_number (PitInTime not null)."""
    if "PitInTime" not in laps.columns:
        return pd.DataFrame(columns=["driver_number", "lap_number"])
    pit_rows = laps[laps["PitInTime"].notna()].copy()
    if pit_rows.empty:
        return pd.DataFrame(columns=["driver_number", "lap_number"])
    out = pd.DataFrame(
        {
            "driver_number": pd.to_numeric(pit_rows["DriverNumber"], errors="coerce"),
            "lap_number": pd.to_numeric(pit_rows["LapNumber"], errors="coerce"),
        }
    )
    out = out.dropna()
    if out.empty:
        return pd.DataFrame(columns=["driver_number", "lap_number"])
    return out.astype(int)


def derive_race_anchors(
    year: int,
    rnd: int,
    events: list[dict],
) -> dict:
    """Derive timing anchors for all transitions of one race.

    ``events`` is a list of dicts with keys ``driver``, ``first_slick_lap``,
    and ``wet_stint_laps``. Returns a dict keyed by ``(driver, first_slick_lap)``
    with ``pit_service_lap`` / ``timing_anchor_lap`` / ``timing_anchor_source``.
    The FastF1 session is loaded once per race call.
    """
    result: dict = {}
    if not events:
        return result

    laps = _load_session_laps(year, rnd)
    pits = _build_pits(laps)

    driver_number = (
        laps.dropna(subset=["Driver", "DriverNumber"])
        .drop_duplicates("Driver")
        .set_index("Driver")["DriverNumber"]
        .astype(int)
        .to_dict()
    )

    for ev in events:
        driver = ev["driver"]
        first_slick = int(ev["first_slick_lap"])
        wet_stint_laps = int(ev.get("wet_stint_laps", 0) or 0)
        # Exact: first slick lap = wet_stint_end + 1 = wet_stint_start + wet_stint_laps.
        wet_stint_start = first_slick - wet_stint_laps if wet_stint_laps > 0 else max(1, first_slick - 1)
        dnum = driver_number.get(driver)

        pit_lap = None
        if dnum is not None and not pits.empty:
            candidate, _source_note = match_pit_service_lap(
                pits, int(dnum), first_slick, int(wet_stint_start)
            )
            pit_lap = candidate

        if pit_lap is not None:
            result[(driver, first_slick)] = {
                "pit_service_lap": int(pit_lap),
                "timing_anchor_lap": int(pit_lap),
                "timing_anchor_source": "pit_service",
            }
        else:
            result[(driver, first_slick)] = {
                "pit_service_lap": None,
                "timing_anchor_lap": first_slick,
                "timing_anchor_source": "first_slick_fallback",
            }
    return result
