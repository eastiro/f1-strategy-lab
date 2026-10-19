from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from .config import OPENF1_BASE_URL, RAW_DIR

COUNTRY_ALIASES = {
    "great britain": "United Kingdom",
    "britain": "United Kingdom",
    "uk": "United Kingdom",
}


def normalize_country(country: str) -> str:
    return COUNTRY_ALIASES.get(country.strip().lower(), country.strip())


def _get(endpoint: str, **params: Any) -> list[dict[str, Any]]:
    last_error: requests.RequestException | None = None
    for attempt in range(4):
        try:
            response = requests.get(f"{OPENF1_BASE_URL}/{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            status_code = getattr(error.response, "status_code", None)
            # A missing OpenF1 endpoint is a data-availability outcome, not a
            # transient network failure. Let the caller record it immediately.
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                raise
            if attempt < 3:
                time.sleep(1.5 * (2**attempt))

    assert last_error is not None
    raise last_error


def get_race_session(year: int, country: str, session_name: str = "Race") -> dict[str, Any]:
    country = normalize_country(country)
    sessions = _get("sessions", year=year, country_name=country, session_name=session_name)
    if not sessions:
        raise ValueError(f"No OpenF1 session found for {year} {country} {session_name}.")
    if len(sessions) > 1:
        sessions = sorted(sessions, key=lambda item: item.get("date_start", ""))
    return sessions[-1]


def fetch_session_bundle(
    year: int,
    country: str,
    session_name: str = "Race",
    output_dir: Path = RAW_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    country = normalize_country(country)
    session = get_race_session(year, country, session_name)
    session_key = session["session_key"]

    endpoints = {
        "session": [session],
        "drivers": _get("drivers", session_key=session_key),
        "laps": _get("laps", session_key=session_key),
        "pit": _get("pit", session_key=session_key),
        "stints": _get("stints", session_key=session_key),
        "weather": _get("weather", session_key=session_key),
        "race_control": _get("race_control", session_key=session_key),
    }

    slug = make_race_slug(year, country)
    saved: dict[str, Path] = {}
    for name, payload in endpoints.items():
        path = output_dir / f"{slug}_{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        saved[name] = path
    return saved


def make_race_slug(year: int, country: str) -> str:
    country = normalize_country(country)
    return f"{year}_{country.lower().replace(' ', '_')}"
