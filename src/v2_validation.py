from __future__ import annotations

"""Cross-race validation helpers for the V2 research build.

V1 is a retrospective Silverstone case study.  V2 deliberately distinguishes
between a completed-race reconstruction (which may use shared pace residuals)
and a decision-time proxy (rainfall evidence available by that lap only).
"""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, RESULTS_DIR
from .errors import CaseDataUnavailableError
from .strategy_model import (
    COMPOUND_BASE_DELTA,
    COMPOUND_WET_PENALTY,
    build_reconstructed_track_state,
    build_track_state_diagnostics,
    build_wetness_from_weather,
    estimate_pace_residuals,
)


SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

# actual_transition_lap is retained for compatibility. For wet-to-slick events it is
# equal to first_slick_lap (the first lap whose compound record shows the new slick).
# Strategy timing is scored against actual_pit_lap, not actual_transition_lap.

CASE_DATA_REQUIRED_COLUMNS = {
    "laps": {
        "name_acronym",
        "lap_number",
        "lap_duration",
        "date_start",
        "compound",
        "clean_lap",
        "tyre_age_at_lap_start",
    },
    "stints": {"driver_number", "stint_number", "compound", "stint_lap_start", "stint_lap_end"},
    "weather": {"date", "rainfall"},
}


@dataclass(frozen=True)
class RaceCase:
    case_id: str
    year: int
    country: str
    title: str
    focal_driver: str
    purpose: str
    review_note: str

    @property
    def slug(self) -> str:
        return f"{self.year}_{self.country.lower().replace(' ', '_')}"


V2_CASES = (
    RaceCase(
        case_id="silverstone_2024",
        year=2024,
        country="United Kingdom",
        title="2024 英国大奖赛",
        focal_driver="NOR",
        purpose="主案例：后段湿转干、回到光头胎",
        review_note="V1 的银石案例，用来检验方法能否复原真实的最终回光头胎窗口。",
    ),
    RaceCase(
        case_id="montreal_2024",
        year=2024,
        country="Canada",
        title="2024 加拿大大奖赛",
        focal_driver="VER",
        purpose="多次早期换胎的湿转干压力测试",
        review_note="多次轮胎阶段使赛况更复杂，任何结论都需要结合赛会背景复核。",
    ),
    RaceCase(
        case_id="zandvoort_2023",
        year=2023,
        country="Netherlands",
        title="2023 荷兰大奖赛",
        focal_driver="VER",
        purpose="早段湿转干，后段再次降雨",
        review_note="只验证早段回光头胎，不把后段第二次降雨混进同一个窗口。",
    ),
    RaceCase(
        case_id="zandvoort_2024_control",
        year=2024,
        country="Netherlands",
        title="2024 荷兰大奖赛",
        focal_driver="NOR",
        purpose="干地对照比赛",
        review_note="OpenF1 没有降雨记录时，合理的方法不应生成湿转干窗口。",
    ),
    RaceCase(
        case_id="sao_paulo_2024_control",
        year=2024,
        country="Brazil",
        title="2024 圣保罗大奖赛",
        focal_driver="VER",
        purpose="复杂湿地对照：整场没有回到光头胎",
        review_note="前排车手的真实轮胎记录没有出现湿胎转光头胎，模型不应强行生成回干窗口。",
    ),
)


def load_case_data(case: RaceCase) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "laps": PROCESSED_DIR / f"{case.slug}_laps_clean.csv",
        "stints": PROCESSED_DIR / f"{case.slug}_stints.csv",
        "weather": PROCESSED_DIR / f"{case.slug}_weather.csv",
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        filenames = ", ".join(missing)
        raise CaseDataUnavailableError(
            f"Missing processed data for {case.year} {case.country}: {filenames}. "
            "Run the OpenF1 download and dataset preparation commands first."
        )
    try:
        laps = pd.read_csv(paths["laps"])
        stints = pd.read_csv(paths["stints"])
        weather = pd.read_csv(paths["weather"])
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise CaseDataUnavailableError(
            f"Could not read processed data for {case.year} {case.country}: {error}"
        ) from error

    for label, frame in {"laps": laps, "stints": stints, "weather": weather}.items():
        missing_columns = sorted(CASE_DATA_REQUIRED_COLUMNS[label] - set(frame.columns))
        if missing_columns:
            raise CaseDataUnavailableError(
                f"Processed {label} data for {case.year} {case.country} is missing columns: "
                f"{', '.join(missing_columns)}. Regenerate the dataset before running validation."
            )
    return laps, stints, weather


def nearby_race_control_evidence(case: RaceCase, lap: int, radius: int = 5) -> str:
    """Summarise public race-control messages near an observed tyre change."""
    path = RAW_DIR / f"{case.slug}_race_control.json"
    if not path.exists():
        return "未下载赛会控制记录"

    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if rows.empty or "lap_number" not in rows.columns:
        return "没有带圈数标记的赛会控制记录"

    nearby = rows[
        pd.to_numeric(rows["lap_number"], errors="coerce").between(lap - radius, lap + radius)
    ].copy()
    if nearby.empty:
        return "附近没有关键赛会控制提示"

    message = nearby.get("message", pd.Series("", index=nearby.index)).fillna("").astype(str)
    category = nearby.get("category", pd.Series("", index=nearby.index)).fillna("").astype(str)
    important = nearby[
        category.isin(["SafetyCar", "Drs"])
        | message.str.contains("NORMAL GRIP|SLIPPERY|SAFETY CAR|VIRTUAL SAFETY|RED FLAG", case=False, regex=True)
    ]
    if important.empty:
        return "附近没有安全车、DRS、抓地或红旗提示"

    snippets = []
    for _, event in important.head(3).iterrows():
        event_lap = int(event["lap_number"])
        text = str(event.get("message", "")).upper()
        if "NORMAL GRIP" in text:
            summary = "赛道恢复正常抓地"
        elif "SLIPPERY" in text:
            summary = "赛道湿滑提示"
        elif "DRS ENABLED" in text:
            summary = "DRS 重新启用"
        elif "VIRTUAL SAFETY" in text:
            summary = "虚拟安全车"
        elif "SAFETY CAR" in text:
            summary = "安全车出动"
        elif "RED FLAG" in text:
            summary = "红旗"
        else:
            continue
        snippets.append(f"第 {event_lap} 圈：{summary}")
    if not snippets:
        return "附近没有可归类的关键赛会提示"
    return " | ".join(snippets)


def first_dry_track_confirmation(case: RaceCase, start_lap: int, total_laps: int) -> int | None:
    """Find the first public dry-track confirmation after a wet-tyre phase.

    ``NORMAL GRIP CONDITIONS`` is preferred because it is a direct condition
    message. If absent, DRS re-enabled is retained as a weaker public signal.
    """
    path = RAW_DIR / f"{case.slug}_race_control.json"
    if not path.exists():
        return None
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if rows.empty or "lap_number" not in rows.columns:
        return None

    rows = rows.copy()
    rows["lap_number"] = pd.to_numeric(rows["lap_number"], errors="coerce")
    rows = rows[rows["lap_number"].between(start_lap, total_laps)]
    messages = rows.get("message", pd.Series("", index=rows.index)).fillna("").astype(str).str.upper()

    normal_grip = rows[messages.str.contains("NORMAL GRIP")]
    if not normal_grip.empty:
        return int(normal_grip["lap_number"].min())
    drs_enabled = rows[messages.str.contains("DRS ENABLED")]
    if not drs_enabled.empty:
        return int(drs_enabled["lap_number"].min())
    return None


def dry_track_confirmation_details(
    case: RaceCase,
    start_lap: int,
    total_laps: int,
) -> tuple[int | None, str | None]:
    """Return the first dry confirmation and its public message source."""
    path = RAW_DIR / f"{case.slug}_race_control.json"
    if not path.exists():
        return None, None
    rows = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if rows.empty or "lap_number" not in rows.columns:
        return None, None
    rows = rows.copy()
    rows["lap_number"] = pd.to_numeric(rows["lap_number"], errors="coerce")
    rows = rows[rows["lap_number"].between(start_lap, total_laps)]
    messages = rows.get("message", pd.Series("", index=rows.index)).fillna("").astype(str).str.upper()
    normal_grip = rows[messages.str.contains("NORMAL GRIP")]
    if not normal_grip.empty:
        return int(normal_grip["lap_number"].min()), "正常抓地提示"
    drs_enabled = rows[messages.str.contains("DRS ENABLED")]
    if not drs_enabled.empty:
        return int(drs_enabled["lap_number"].min()), "DRS 重新启用"
    return None, None


def focal_stints(laps: pd.DataFrame, stints: pd.DataFrame, driver: str) -> pd.DataFrame:
    driver_rows = laps[laps["name_acronym"] == driver]
    if driver_rows.empty:
        raise ValueError(f"{driver} is not present in this case.")
    driver_number = driver_rows["driver_number"].iloc[0]
    result = stints[stints["driver_number"] == driver_number].copy()
    return result.dropna(subset=["stint_lap_start", "stint_lap_end"]).sort_values("stint_number")


def load_pit_data(case: RaceCase) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{case.slug}_pit.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        pits = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame()
    if pits.empty or "lap_number" not in pits.columns or "driver_number" not in pits.columns:
        return pd.DataFrame()
    pits = pits.copy()
    pits["lap_number"] = pd.to_numeric(pits["lap_number"], errors="coerce")
    pits["driver_number"] = pd.to_numeric(pits["driver_number"], errors="coerce")
    return pits.dropna(subset=["lap_number", "driver_number"])


def driver_number_for(laps: pd.DataFrame, driver: str) -> int:
    driver_rows = laps[laps["name_acronym"] == driver]
    if driver_rows.empty:
        raise ValueError(f"{driver} is not present in this case.")
    return int(driver_rows["driver_number"].iloc[0])


def first_slick_lap_from_laps(
    laps: pd.DataFrame,
    driver: str,
    wet_stint_start: int,
    to_compound: str | None = None,
) -> int | None:
    """First lap where the driver's compound record shows the new slick tyre."""
    driver_laps = laps[laps["name_acronym"] == driver].copy()
    if driver_laps.empty:
        return None
    driver_laps["lap_number"] = pd.to_numeric(driver_laps["lap_number"], errors="coerce")
    driver_laps = driver_laps.dropna(subset=["lap_number"]).sort_values("lap_number")
    compounds = driver_laps["compound"].astype(str).str.upper()
    target = None if to_compound is None else str(to_compound).upper()
    previous_wet = False
    for _, row in driver_laps.iterrows():
        lap = int(row["lap_number"])
        compound = str(row["compound"]).upper()
        if lap < wet_stint_start:
            previous_wet = compound in WET_COMPOUNDS
            continue
        if compound in WET_COMPOUNDS:
            previous_wet = True
            continue
        if compound in SLICK_COMPOUNDS and previous_wet:
            if target is None or compound == target:
                return lap
            # Different slick compound still counts once we left wet rubber.
            return lap
        previous_wet = False
    # Fallback: first target slick lap after the wet stint starts.
    if target is not None:
        matches = driver_laps[
            (driver_laps["lap_number"] >= wet_stint_start) & (compounds == target)
        ]
        if not matches.empty:
            return int(matches["lap_number"].iloc[0])
    return None


def match_pit_service_lap(
    pits: pd.DataFrame,
    driver_number: int,
    first_slick_lap: int | None,
    wet_stint_start: int,
    stint_boundary_lap: int | None = None,
) -> tuple[int | None, str]:
    """Match a public pit service lap near a wet-to-slick transition.

    Preference order:
    1. Pit lap in [wet_stint_start, first_slick_lap] closest to first_slick_lap
    2. Pit lap equal to stint boundary or one lap before first slick
    Returns (lap, source_note). source_note is 'pit' or explains fallback.
    """
    if pits.empty:
        return None, "没有独立 pit 锚点（缺少 pit 文件）"

    driver_pits = pits[pits["driver_number"] == driver_number].copy()
    if driver_pits.empty:
        return None, "没有独立 pit 锚点（该车手无 pit 记录）"

    upper = first_slick_lap if first_slick_lap is not None else stint_boundary_lap
    if upper is None:
        return None, "没有独立 pit 锚点（缺少 first slick 参考）"

    window = driver_pits[
        (driver_pits["lap_number"] >= wet_stint_start)
        & (driver_pits["lap_number"] <= upper)
    ]
    if window.empty:
        # Allow one-lap slack after first slick for timing mismatches.
        window = driver_pits[
            (driver_pits["lap_number"] >= wet_stint_start)
            & (driver_pits["lap_number"] <= upper + 1)
        ]
    if window.empty:
        return None, "没有独立 pit 锚点（窗口内无 pit）"

    # Prefer the latest pit at or before first slick (service before new compound appears).
    on_or_before = window[window["lap_number"] <= upper]
    chosen = on_or_before if not on_or_before.empty else window
    pit_lap = int(chosen["lap_number"].max())
    return pit_lap, "pit"


def observed_wet_to_slick_transitions(stints: pd.DataFrame) -> list[dict[str, object]]:
    transitions: list[dict[str, object]] = []
    rows = stints.reset_index(drop=True)
    for index in range(1, len(rows)):
        previous = rows.iloc[index - 1]
        current = rows.iloc[index]
        from_compound = str(previous["compound"]).upper()
        to_compound = str(current["compound"]).upper()
        if from_compound in WET_COMPOUNDS and to_compound in SLICK_COMPOUNDS:
            stint_boundary = int(current["stint_lap_start"])
            transitions.append(
                {
                    # Historical field: stint boundary. Prefer first_slick_lap / actual_pit_lap.
                    "lap": stint_boundary,
                    "stint_boundary_lap": stint_boundary,
                    "wet_stint_start": int(previous["stint_lap_start"]),
                    "wet_stint_end": int(previous["stint_lap_end"]),
                    "wet_stint_laps": int(previous["stint_lap_end"] - previous["stint_lap_start"] + 1),
                    "from_compound": from_compound,
                    "to_compound": to_compound,
                }
            )
    return transitions


def enrich_transitions_with_timing(
    transitions: list[dict[str, object]],
    laps: pd.DataFrame,
    pits: pd.DataFrame,
    driver: str,
) -> list[dict[str, object]]:
    """Attach actual_pit_lap and first_slick_lap to stint-derived transitions."""
    if not transitions:
        return transitions
    number = driver_number_for(laps, driver)
    enriched: list[dict[str, object]] = []
    for transition in transitions:
        wet_start = int(transition["wet_stint_start"])
        to_compound = str(transition["to_compound"])
        stint_boundary = int(transition.get("stint_boundary_lap", transition["lap"]))
        first_slick = first_slick_lap_from_laps(laps, driver, wet_start, to_compound)
        if first_slick is None:
            first_slick = stint_boundary
        pit_lap, pit_source = match_pit_service_lap(
            pits,
            number,
            first_slick,
            wet_start,
            stint_boundary_lap=stint_boundary,
        )
        item = dict(transition)
        item["first_slick_lap"] = first_slick
        item["actual_pit_lap"] = pit_lap
        item["pit_anchor_source"] = pit_source
        # Scoring / chart primary anchor: pit service when available.
        item["lap"] = int(pit_lap) if pit_lap is not None else int(first_slick)
        item["timing_anchor"] = "actual_pit_lap" if pit_lap is not None else "first_slick_lap"
        enriched.append(item)
    return enriched


def field_wet_to_slick_wave(
    case: RaceCase,
    phase_start: int,
    phase_end: int,
) -> dict[str, object]:
    """Compact field-wide wet-to-slick wave for the validated weather phase."""
    laps, stints, _weather = load_case_data(case)
    pits = load_pit_data(case)
    driver_map = (
        laps.dropna(subset=["name_acronym", "driver_number"])
        .drop_duplicates("driver_number")
        .set_index("driver_number")["name_acronym"]
        .astype(str)
        .to_dict()
    )
    events: list[dict[str, object]] = []
    for driver_number, driver_stints in stints.groupby("driver_number"):
        ordered = (
            driver_stints.dropna(subset=["stint_lap_start", "stint_lap_end"])
            .sort_values("stint_number")
        )
        base = observed_wet_to_slick_transitions(ordered)
        if not base:
            continue
        driver_name = str(driver_map.get(int(driver_number), f"#{int(driver_number)}"))
        driver_laps = laps[pd.to_numeric(laps["driver_number"], errors="coerce") == int(driver_number)].copy()
        for transition in base:
            wet_start = int(transition["wet_stint_start"])
            stint_boundary = int(transition["stint_boundary_lap"])
            if stint_boundary < phase_start or wet_start > phase_end:
                continue
            first_slick = None
            if not driver_laps.empty:
                driver_laps["lap_number"] = pd.to_numeric(driver_laps["lap_number"], errors="coerce")
                ordered_laps = driver_laps.dropna(subset=["lap_number"]).sort_values("lap_number")
                prev_wet = False
                for _, row in ordered_laps.iterrows():
                    lap = int(row["lap_number"])
                    compound = str(row["compound"]).upper()
                    if lap < wet_start:
                        prev_wet = compound in WET_COMPOUNDS
                        continue
                    if compound in WET_COMPOUNDS:
                        prev_wet = True
                    elif compound in SLICK_COMPOUNDS and prev_wet:
                        first_slick = lap
                        break
            if first_slick is None:
                first_slick = stint_boundary
            pit_lap, pit_source = match_pit_service_lap(
                pits,
                int(driver_number),
                first_slick,
                wet_start,
                stint_boundary_lap=stint_boundary,
            )
            anchor = pit_lap if pit_lap is not None else first_slick
            if anchor < phase_start or anchor > phase_end + 2:
                continue
            events.append(
                {
                    "driver": driver_name,
                    "driver_number": int(driver_number),
                    "actual_pit_lap": pit_lap,
                    "first_slick_lap": first_slick,
                    "to_compound": str(transition["to_compound"]),
                    "pit_anchor_source": pit_source,
                }
            )
    pit_laps = sorted(
        int(event["actual_pit_lap"])
        for event in events
        if event.get("actual_pit_lap") is not None
    )
    return {
        "phase_start": phase_start,
        "phase_end": phase_end,
        "events": events,
        "pit_laps": pit_laps,
        "pit_lap_min": min(pit_laps) if pit_laps else None,
        "pit_lap_max": max(pit_laps) if pit_laps else None,
        "event_count": len(events),
    }


def later_rainfall_phase_start(
    case: RaceCase,
    after_lap: int,
    total_laps: int,
) -> int | None:
    """Detect a later independent rainfall phase after the validated switch."""
    _laps, _stints, weather = load_case_data(case)
    if weather.empty or "date" not in weather.columns:
        return None
    # Use lap-mapped rainfall via weather timestamps when possible; fall back to
    # race-control / stint evidence is handled by the caller. Here scan weather
    # rows already prepared as lap wetness from the focal driver curve.
    laps, _, _ = load_case_data(case)
    wetness = build_wetness_from_weather(laps, weather, case.focal_driver, total_laps)
    # Look for a fresh rise after a dry gap following the validated event.
    saw_dry = False
    for lap in range(after_lap + 1, total_laps + 1):
        value = float(wetness.get(lap, 0.0))
        if value <= 0.05:
            saw_dry = True
            continue
        if saw_dry and value >= 0.15:
            return lap
    return None


def fresh_crossover_wetness(slick_compound: str) -> float:
    """Wetness where a fresh slick and fresh intermediate tie in the V1 model."""
    compound = slick_compound.upper()
    if compound not in SLICK_COMPOUNDS:
        raise ValueError(f"Expected a slick compound, got {slick_compound}.")
    numerator = COMPOUND_BASE_DELTA["INTERMEDIATE"] - COMPOUND_BASE_DELTA[compound]
    denominator = COMPOUND_WET_PENALTY[compound] - COMPOUND_WET_PENALTY["INTERMEDIATE"]
    return numerator / denominator


def find_recovery_slick_window(
    wetness_by_lap: dict[int, float],
    crossover: float,
    total_laps: int,
    minimum_laps: int = 2,
    start_lap: int = 1,
) -> int | None:
    """Return the first sustained slick window after a modeled wet phase.

    A wet-to-dry question is not answered by the first dry lap in a race. The
    state must first cross into the model's "slick unsuitable" region, then
    recover below the crossover threshold.
    """
    wet_laps = [
        lap
        for lap in range(max(1, start_lap), total_laps + 1)
        if wetness_by_lap.get(lap, 0.0) > crossover
    ]
    if not wet_laps:
        return None

    search_start = max(wet_laps) + 1
    run_start: int | None = None
    for lap in range(search_start, total_laps + 1):
        if wetness_by_lap.get(lap, 0.0) <= crossover:
            run_start = lap if run_start is None else run_start
            if lap - run_start + 1 >= minimum_laps:
                return run_start
        else:
            run_start = None
    return None


def select_reference_drivers(laps: pd.DataFrame, focal_driver: str, limit: int = 4) -> list[str]:
    finish_order = (
        laps.groupby("name_acronym")["lap_number"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    selected = [focal_driver]
    selected.extend(driver for driver in finish_order if driver != focal_driver)
    return selected[:limit]


def validate_case(case: RaceCase) -> pd.DataFrame:
    laps, stints, weather = load_case_data(case)
    pits = load_pit_data(case)
    total_laps = int(laps["lap_number"].max())
    focal = focal_stints(laps, stints, case.focal_driver)
    transitions = enrich_transitions_with_timing(
        observed_wet_to_slick_transitions(focal),
        laps,
        pits,
        case.focal_driver,
    )
    rainfall_records = int(pd.to_numeric(weather.get("rainfall", 0), errors="coerce").fillna(0).sum())

    rainfall_wetness = build_wetness_from_weather(laps, weather, case.focal_driver, total_laps)
    reference_drivers = select_reference_drivers(laps, case.focal_driver)
    residuals = estimate_pace_residuals(laps, reference_drivers)
    diagnostics = build_track_state_diagnostics(rainfall_wetness, residuals, total_laps)
    retrospective_wetness = build_reconstructed_track_state(rainfall_wetness, diagnostics, total_laps)

    rows: list[dict[str, object]] = []
    if not transitions:
        wet_control = rainfall_records > 0
        rows.append(
            {
                "case_id": case.case_id,
                "race": case.title,
                "driver": case.focal_driver,
                "purpose": case.purpose,
                "rainfall_records": rainfall_records,
                "actual_transition_lap": pd.NA,
                "actual_pit_lap": pd.NA,
                "first_slick_lap": pd.NA,
                "stint_boundary_lap": pd.NA,
                "timing_anchor": pd.NA,
                "pit_anchor_source": pd.NA,
                "to_compound": pd.NA,
                "decision_time_peak_wetness": 0.0,
                "retrospective_peak_wetness": 0.0,
                "decision_time_window_start": pd.NA,
                "decision_time_window_label": "无公开信号窗口",
                "retrospective_window_start": pd.NA,
                "decision_time_wetness_at_transition": pd.NA,
                "retrospective_wetness_at_transition": pd.NA,
                "classification": "湿地对照：没有湿胎转光头胎" if wet_control else "干地对照：没有湿胎转光头胎",
                "failure_type": "湿地对照：无回光头胎事件" if wet_control else "干地对照：无回光头胎事件",
                "evidence_note": (
                    "有降雨记录，但真实轮胎记录没有回到光头胎；模型应保持沉默。"
                    if wet_control
                    else "没有降雨记录，也没有湿胎转光头胎事件；模型应保持沉默。"
                ),
                "nearby_race_control": "不适用：没有真实湿转干事件",
                "review_note": case.review_note,
            }
        )
        return pd.DataFrame(rows)

    for transition in transitions:
        # Strategy timing anchor: pit service lap when available.
        pit_lap = transition.get("actual_pit_lap")
        first_slick = transition.get("first_slick_lap")
        stint_boundary = transition.get("stint_boundary_lap")
        anchor_lap = int(pit_lap) if pit_lap is not None else int(first_slick)
        # Compatibility: actual_transition_lap remains first_slick_lap semantics.
        transition_lap = int(first_slick) if first_slick is not None else anchor_lap
        wet_stint_start = int(transition["wet_stint_start"])
        compound = str(transition["to_compound"])
        crossover = fresh_crossover_wetness(compound)
        decision_peak = max(
            (
                rainfall_wetness.get(candidate_lap, 0.0)
                for candidate_lap in range(wet_stint_start, anchor_lap + 1)
            ),
            default=0.0,
        )
        retrospective_peak = max(
            (
                retrospective_wetness.get(candidate_lap, 0.0)
                for candidate_lap in range(wet_stint_start, anchor_lap + 1)
            ),
            default=0.0,
        )
        decision_start = find_recovery_slick_window(
            rainfall_wetness,
            crossover,
            total_laps,
            start_lap=wet_stint_start,
        )
        dry_track_confirmation, dry_track_source = dry_track_confirmation_details(
            case,
            wet_stint_start,
            total_laps,
        )
        public_signal_candidates = [
            candidate for candidate in [decision_start, dry_track_confirmation] if candidate is not None
        ]
        public_signal_start = max(public_signal_candidates) if public_signal_candidates else None
        public_signal_end = min(public_signal_candidates) if public_signal_candidates else None
        public_signal_label = (
            f"第 {public_signal_end}-{public_signal_start} 圈"
            if public_signal_candidates and public_signal_end != public_signal_start
            else f"第 {public_signal_start} 圈"
            if public_signal_start is not None
            else "无公开信号窗口"
        )
        retrospective_start = find_recovery_slick_window(
            retrospective_wetness,
            crossover,
            total_laps,
            start_lap=wet_stint_start,
        )
        later_wet_phase = bool(
            stints[
                (pd.to_numeric(stints["stint_lap_start"], errors="coerce") > anchor_lap)
                & (stints["compound"].astype(str).str.upper().isin(WET_COMPOUNDS))
            ].shape[0]
        )
        if public_signal_start is not None and abs(public_signal_start - anchor_lap) <= 2:
            classification = "决策时公共信号接近真实进站动作"
        elif decision_start is None and retrospective_start is None:
            classification = "湿地信号不足，无法判断恢复窗口"
        elif retrospective_start is not None and abs(retrospective_start - anchor_lap) <= 2:
            classification = "赛后重建接近真实进站动作"
        else:
            classification = "与真实进站动作有偏差，需结合赛况复核"

        if later_wet_phase:
            failure_type = "多阶段天气：当前只验证本次湿转干"
        elif decision_start is None and dry_track_confirmation is None:
            failure_type = "天气证据不足"
        elif public_signal_start is not None and abs(public_signal_start - anchor_lap) <= 2:
            failure_type = "公开信号覆盖充分"
        else:
            failure_type = "公开信号与真实进站动作偏差较大"

        if decision_peak < crossover and dry_track_confirmation is not None:
            evidence_note = (
                "降雨代理不足，但赛会控制在第 "
                f"{dry_track_confirmation} 圈给出干地确认，补足决策时公共信号。"
            )
        elif decision_peak < crossover:
            evidence_note = "降雨代理在真实湿胎阶段从未达到“光头胎不适用”阈值，输入证据不足。"
        elif retrospective_start is not None and abs(retrospective_start - anchor_lap) > 2:
            evidence_note = "赛后重建识别到湿地阶段，但恢复窗口与真实进站动作相差较大。"
        else:
            evidence_note = "降雨代理与赛后重建都识别到清晰的湿转干过程。"

        rows.append(
            {
                "case_id": case.case_id,
                "race": case.title,
                "driver": case.focal_driver,
                "purpose": case.purpose,
                "rainfall_records": rainfall_records,
                # Compatibility: first lap showing the new slick compound.
                "actual_transition_lap": transition_lap,
                "actual_pit_lap": pd.NA if pit_lap is None else int(pit_lap),
                "first_slick_lap": pd.NA if first_slick is None else int(first_slick),
                "stint_boundary_lap": pd.NA if stint_boundary is None else int(stint_boundary),
                "timing_anchor": transition.get("timing_anchor", "actual_pit_lap"),
                "pit_anchor_source": transition.get("pit_anchor_source", "pit"),
                "wet_stint_start": wet_stint_start,
                "to_compound": compound,
                "fresh_crossover_wetness": round(crossover, 3),
                "decision_time_peak_wetness": round(decision_peak, 3),
                "retrospective_peak_wetness": round(retrospective_peak, 3),
                "rainfall_only_window_start": decision_start,
                "dry_track_confirmation_lap": dry_track_confirmation,
                "dry_track_confirmation_source": dry_track_source,
                "decision_time_window_start": public_signal_start,
                "decision_time_window_end": public_signal_end,
                "decision_time_window_label": public_signal_label,
                "retrospective_window_start": retrospective_start,
                "decision_time_wetness_at_transition": round(
                    rainfall_wetness.get(anchor_lap, 0.0), 3
                ),
                "retrospective_wetness_at_transition": round(
                    retrospective_wetness.get(anchor_lap, 0.0), 3
                ),
                # Errors are scored against the pit service anchor.
                "decision_time_error_laps": (
                    None if public_signal_start is None else public_signal_start - anchor_lap
                ),
                "retrospective_error_laps": (
                    None if retrospective_start is None else retrospective_start - anchor_lap
                ),
                "classification": classification,
                "failure_type": failure_type,
                "evidence_note": evidence_note,
                "nearby_race_control": nearby_race_control_evidence(case, anchor_lap),
                "review_note": case.review_note,
            }
        )
    return pd.DataFrame(rows)


def build_case_track_state(case: RaceCase) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Return both V2 state estimates for one race and its observed transitions."""
    laps, stints, weather = load_case_data(case)
    pits = load_pit_data(case)
    total_laps = int(laps["lap_number"].max())
    focal = focal_stints(laps, stints, case.focal_driver)
    rainfall_wetness = build_wetness_from_weather(laps, weather, case.focal_driver, total_laps)
    reference_drivers = select_reference_drivers(laps, case.focal_driver)
    residuals = estimate_pace_residuals(laps, reference_drivers)
    diagnostics = build_track_state_diagnostics(rainfall_wetness, residuals, total_laps)
    retrospective_wetness = build_reconstructed_track_state(rainfall_wetness, diagnostics, total_laps)
    state = pd.DataFrame({"lap_number": range(1, total_laps + 1)})
    state["decision_time_wetness"] = state["lap_number"].map(rainfall_wetness).fillna(0.0)
    state["retrospective_wetness"] = state["lap_number"].map(retrospective_wetness).fillna(0.0)
    transitions = enrich_transitions_with_timing(
        observed_wet_to_slick_transitions(focal),
        laps,
        pits,
        case.focal_driver,
    )
    return state, transitions


def build_validation_report(cases: tuple[RaceCase, ...] = V2_CASES) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in cases:
        rows.extend(validate_case(case).to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_validation_metrics(report: pd.DataFrame | None = None) -> dict[str, float | int]:
    """Summarise cross-race validation without hiding the control cases.

    Transition cases are identified by first_slick_lap when present, otherwise
    by the compatibility field actual_transition_lap. Decision-time errors are
    already scored against actual_pit_lap inside validate_case.
    """
    report = build_validation_report() if report is None else report
    if "first_slick_lap" in report.columns:
        transition_mask = report["first_slick_lap"].notna() | report["actual_transition_lap"].notna()
    else:
        transition_mask = report["actual_transition_lap"].notna()
    transition_cases = report[transition_mask].copy()
    controls = report[~transition_mask].copy()
    public_errors = pd.to_numeric(transition_cases["decision_time_error_laps"], errors="coerce").abs()
    public_within_two = int((public_errors <= 2).sum())
    rainfall_windows = pd.to_numeric(transition_cases["rainfall_only_window_start"], errors="coerce")
    controls_silent = int(controls["decision_time_window_start"].isna().sum())
    return {
        "wet_to_dry_cases": int(len(transition_cases)),
        "public_within_two": public_within_two,
        "public_mean_absolute_error": round(float(public_errors.mean()), 2) if not public_errors.empty else 0.0,
        "rainfall_only_windows": int(rainfall_windows.notna().sum()),
        "control_cases": int(len(controls)),
        "controls_silent": controls_silent,
    }


def write_validation_artifacts(output_dir: Path | None = None) -> dict[str, Path]:
    """Write validation CSV under reports/results (and inventory alongside)."""
    output_dir = output_dir or RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_validation_report()
    report_path = output_dir / "v2_cross_race_validation.csv"
    report.to_csv(report_path, index=False)

    inventory = pd.DataFrame(
        {
            "case_id": [case.case_id for case in V2_CASES],
            "race": [case.title for case in V2_CASES],
            "focal_driver": [case.focal_driver for case in V2_CASES],
            "purpose": [case.purpose for case in V2_CASES],
            "review_note": [case.review_note for case in V2_CASES],
        }
    )
    inventory_path = output_dir / "v2_case_inventory.csv"
    inventory.to_csv(inventory_path, index=False)
    return {"validation": report_path, "inventory": inventory_path}


if __name__ == "__main__":
    for name, path in write_validation_artifacts().items():
        print(f"{name}: {path}")
