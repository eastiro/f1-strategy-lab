"""V3 reliability views built from frozen V2 evaluation outputs."""

from __future__ import annotations

import pandas as pd

RELIABILITY_COLUMNS = [
    "source",
    "year",
    "country",
    "driver",
    "actual_transition_lap",
    "public_signal_error_laps",
    "event_quality",
    "state_machine_outcome",
    "outcome",
    "reason_group",
]


def _as_number(value: object) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(number) else float(number)


def _reason_group(row: pd.Series) -> str:
    quality = str(row.get("event_quality", "")).strip().lower()
    nearby_control = str(row.get("nearby_race_control_phases", "")).strip().lower()
    if quality == "race_control_adjacent" or nearby_control not in {"", "nan", "none"}:
        return "race_control_adjacent"
    if quality == "multi_rain_phase":
        return "multi_rain_phase"
    if quality == "start_or_restart":
        return "start_or_restart"
    return "clean_or_other"


def _outcome(row: pd.Series) -> str:
    error = _as_number(row.get("public_signal_error_laps"))
    state_machine_outcome = str(row.get("state_machine_outcome", "")).strip().lower()
    signal_available = state_machine_outcome == "signal" or pd.notna(row.get("public_signal_window_start"))
    if not signal_available or error is None or state_machine_outcome == "abstain":
        return "abstain"
    return "signal_accurate" if abs(error) <= 2 else "signal_inaccurate"


def _normalise_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)

    result = frame.copy()
    result["source"] = source
    for column in RELIABILITY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["state_machine_outcome"] = result.apply(
        lambda row: str(row.get("state_machine_outcome", "")).strip().lower()
        or ("signal" if pd.notna(row.get("public_signal_window_start")) else "abstain"),
        axis=1,
    )
    result["outcome"] = result.apply(_outcome, axis=1)
    result["reason_group"] = result.apply(_reason_group, axis=1)
    result["actual_transition_lap"] = pd.to_numeric(result["actual_transition_lap"], errors="coerce")
    result["public_signal_error_laps"] = pd.to_numeric(
        result["public_signal_error_laps"],
        errors="coerce",
    )
    return result[RELIABILITY_COLUMNS].reset_index(drop=True)


def build_reliability_events(
    challenge_set: pd.DataFrame | None = None,
    fastf1_evaluation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine frozen challenge and historical evaluations for V3 visualisations."""
    frames: list[pd.DataFrame] = []
    if challenge_set is not None and not challenge_set.empty:
        frames.append(_normalise_frame(challenge_set, "2025 挑战集"))
    if fastf1_evaluation is not None and not fastf1_evaluation.empty:
        frames.append(_normalise_frame(fastf1_evaluation, "2021 FastF1"))
    if not frames:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def summarize_reliability_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return stable counts for the V3 headline metrics."""
    if events.empty:
        return pd.DataFrame(columns=["source", "outcome", "reason_group", "事件数"])
    return (
        events.groupby(["source", "outcome", "reason_group"], dropna=False)
        .size()
        .reset_index(name="事件数")
        .sort_values(["source", "outcome", "reason_group"])
        .reset_index(drop=True)
    )
