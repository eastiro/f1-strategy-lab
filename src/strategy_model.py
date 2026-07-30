from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .weather_timing import prepare_driver_lap_timing


COMPOUND_BASE_DELTA = {
    "SOFT": -0.30,
    "MEDIUM": 0.0,
    "HARD": 0.25,
    "INTERMEDIATE": 4.5,
    "WET": 8.0,
}

COMPOUND_DEGRADATION = {
    "SOFT": 0.055,
    "MEDIUM": 0.040,
    "HARD": 0.030,
    "INTERMEDIATE": 0.075,
    "WET": 0.09,
}

COMPOUND_WET_PENALTY = {
    # Public race data cannot identify a reliable wet-performance ordering for
    # the three slick compounds. Treat their main wet-track penalty equally.
    "SOFT": 10.5,
    "MEDIUM": 10.5,
    "HARD": 10.5,
    "INTERMEDIATE": 1.4,
    "WET": 0.8,
}

COMPOUND_WET_RISK = {
    "SOFT": 1.0,
    "MEDIUM": 1.0,
    "HARD": 1.0,
    "INTERMEDIATE": 0.2,
    "WET": 0.1,
}

INTERMEDIATE_DRYING_THRESHOLD = 0.50
INTERMEDIATE_WET_WEAR_FLOOR = 0.15


@dataclass(frozen=True)
class StintPlan:
    compound: str
    start_lap: int
    end_lap: int
    tyre_age_at_start: int = 0


@dataclass(frozen=True)
class StrategyResult:
    total_time: float
    laps: pd.DataFrame
    risk_score: float = 0.0
    risk_level: str = "低"


def intermediate_wear_exposure(
    stint: StintPlan,
    lap_number: int,
    wetness_by_lap: dict[int, float],
) -> float:
    """Convert an intermediate stint into drying-weighted wear laps."""
    exposure = 0.0
    for previous_lap in range(stint.start_lap, lap_number):
        wetness = max(0.0, min(1.0, wetness_by_lap.get(previous_lap, 0.0)))
        drying_factor = max(0.0, (INTERMEDIATE_DRYING_THRESHOLD - wetness) / INTERMEDIATE_DRYING_THRESHOLD)
        exposure += INTERMEDIATE_WET_WEAR_FLOOR + (1.0 - INTERMEDIATE_WET_WEAR_FLOOR) * drying_factor
    return exposure


def estimate_driver_baseline(laps: pd.DataFrame, driver: str) -> float:
    driver_laps = laps[
        (laps["name_acronym"] == driver)
        & (laps["clean_lap"])
        & (laps["lap_duration"].notna())
    ]
    if driver_laps.empty:
        raise ValueError(f"No clean laps found for driver {driver}.")
    return float(driver_laps["lap_duration"].median())


def simulate_strategy(
    total_laps: int,
    baseline_lap_time: float,
    stints: list[StintPlan],
    pit_loss: float = 22.0,
    fuel_effect_per_lap: float = -0.035,
    lap_adjustments: dict[int, float] | None = None,
    wetness_by_lap: dict[int, float] | None = None,
) -> StrategyResult:
    lap_adjustments = lap_adjustments or {}
    wetness_by_lap = wetness_by_lap or {}
    rows = []
    for lap_number in range(1, total_laps + 1):
        stint = find_stint(stints, lap_number)
        tyre_age = stint.tyre_age_at_start + lap_number - stint.start_lap
        compound = stint.compound.upper()
        compound_delta = COMPOUND_BASE_DELTA.get(compound, 0.0)
        fuel_delta = fuel_effect_per_lap * (lap_number - 1)
        pit_delta = pit_loss if lap_number in pit_laps(stints) else 0.0
        wetness = max(0.0, min(1.0, wetness_by_lap.get(lap_number, 0.0)))
        wear_exposure = max(tyre_age, 0)
        if compound == "INTERMEDIATE":
            wear_exposure = intermediate_wear_exposure(stint, lap_number, wetness_by_lap)
        degradation = COMPOUND_DEGRADATION.get(compound, 0.05) * wear_exposure
        wet_penalty = wetness * COMPOUND_WET_PENALTY.get(compound, 8.0)
        weather_delta = lap_adjustments.get(lap_number, 0.0) + wet_penalty
        lap_risk = wetness * COMPOUND_WET_RISK.get(compound, 0.7)
        lap_time = baseline_lap_time + compound_delta + degradation + fuel_delta + pit_delta + weather_delta
        rows.append(
            {
                "lap_number": lap_number,
                "compound": compound,
                "tyre_age": tyre_age,
                "wear_exposure": wear_exposure,
                "degradation": degradation,
                "wetness": wetness,
                "wet_penalty": wet_penalty,
                "weather_delta": weather_delta,
                "risk_score": lap_risk,
                "simulated_lap_time": lap_time,
                "pit_lap": pit_delta > 0,
            }
        )
    result = pd.DataFrame(rows)
    risk_score = float(result["risk_score"].mean()) if not result.empty else 0.0
    return StrategyResult(
        total_time=float(result["simulated_lap_time"].sum()),
        laps=result,
        risk_score=risk_score,
        risk_level=risk_level(risk_score),
    )


def find_stint(stints: list[StintPlan], lap_number: int) -> StintPlan:
    for stint in stints:
        if stint.start_lap <= lap_number <= stint.end_lap:
            return stint
    raise ValueError(f"No stint covers lap {lap_number}.")


def pit_laps(stints: list[StintPlan]) -> set[int]:
    return {stint.start_lap for stint in stints if stint.start_lap > 1}


def build_wetness_curve(
    total_laps: int,
    start_lap: int,
    end_lap: int,
    peak_wetness: float = 1.0,
    ramp_fraction: float = 0.35,
) -> dict[int, float]:
    start_lap = max(1, min(total_laps, start_lap))
    end_lap = max(start_lap, min(total_laps, end_lap))
    peak_wetness = max(0.0, min(1.0, peak_wetness))
    ramp_fraction = max(0.05, min(0.49, ramp_fraction))
    span = max(1, end_lap - start_lap)

    wetness: dict[int, float] = {}
    for lap_number in range(start_lap, end_lap + 1):
        progress = (lap_number - start_lap) / span
        if progress <= ramp_fraction:
            intensity = progress / ramp_fraction
        elif progress >= 1 - ramp_fraction:
            intensity = (1 - progress) / ramp_fraction
        else:
            intensity = 1.0
        wetness[lap_number] = peak_wetness * max(0.0, min(1.0, intensity))
    return wetness


def build_wetness_from_weather(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    driver: str,
    total_laps: int,
    rain_gain_per_minute: float = 0.05,
    dry_loss_per_minute: float = 0.10,
    max_wetness: float = 0.65,
) -> dict[int, float]:
    if weather.empty or "rainfall" not in weather.columns or "date" not in weather.columns:
        return {}

    driver_laps = prepare_driver_lap_timing(laps, driver)
    if driver_laps.empty:
        return {}

    weather_state = weather.copy()
    weather_state["weather_time"] = pd.to_datetime(
        weather_state["date"],
        utc=True,
        format="mixed",
    )
    weather_state["rainfall"] = pd.to_numeric(weather_state["rainfall"], errors="coerce").fillna(0)
    weather_state["track_temperature"] = pd.to_numeric(weather_state.get("track_temperature", 25.0), errors="coerce").fillna(25.0)
    weather_state = weather_state.sort_values("weather_time")

    first_lap_time = driver_laps["lap_start_time"].min()
    last_lap_time = driver_laps["lap_end_time"].max()
    weather_state = weather_state[
        (weather_state["weather_time"] >= first_lap_time - pd.Timedelta(minutes=20))
        & (weather_state["weather_time"] <= last_lap_time + pd.Timedelta(minutes=2))
    ].copy()
    if weather_state.empty:
        return {}

    wetness_values = []
    wetness = 0.0
    previous_time = weather_state["weather_time"].iloc[0]
    for _, row in weather_state.iterrows():
        current_time = row["weather_time"]
        minutes = max(1.0, (current_time - previous_time).total_seconds() / 60)
        track_temp = float(row["track_temperature"])
        dry_multiplier = 1.0 + max(0.0, track_temp - 25.0) / 25.0
        if int(row["rainfall"]) == 1:
            wetness += rain_gain_per_minute * minutes
        else:
            wetness -= dry_loss_per_minute * dry_multiplier * minutes
        wetness = max(0.0, min(1.0, wetness))
        wetness_values.append(wetness)
        previous_time = current_time

    weather_state["wetness"] = wetness_values

    max_wetness = max(0.0, min(1.0, max_wetness))
    wetness_by_lap: dict[int, float] = {}
    for _, lap in driver_laps.iterrows():
        lap_number = int(lap["lap_number"])
        # Final-stint tyre calls are judged by the state a car exits into, not
        # by the average conditions across a pit lap that began before the stop.
        nearest_index = (weather_state["weather_time"] - lap["lap_end_time"]).abs().idxmin()
        lap_wetness = float(weather_state.loc[nearest_index, "wetness"])
        if 1 <= lap_number <= total_laps:
            wetness_by_lap[lap_number] = lap_wetness * max_wetness

    return wetness_by_lap


def estimate_pace_residuals(
    laps: pd.DataFrame,
    drivers: list[str],
    fuel_effect_per_lap: float = -0.035,
) -> pd.DataFrame:
    residual_rows = []
    selected_laps = laps[
        (laps["name_acronym"].isin(drivers))
        & (laps["clean_lap"])
        & (laps["lap_number"] > 1)
        & (laps["lap_duration"].notna())
        & (laps["compound"].isin(COMPOUND_BASE_DELTA))
    ].copy()
    if selected_laps.empty:
        return pd.DataFrame()

    selected_laps["tyre_age_at_lap_start"] = pd.to_numeric(
        selected_laps["tyre_age_at_lap_start"],
        errors="coerce",
    ).fillna(0)
    selected_laps["lap_number"] = selected_laps["lap_number"].astype(int)

    for driver in drivers:
        driver_laps = selected_laps[selected_laps["name_acronym"] == driver].copy()
        if driver_laps.empty:
            continue
        baseline = estimate_driver_baseline(laps, driver)
        for _, lap in driver_laps.iterrows():
            compound = str(lap["compound"]).upper()
            tyre_age = max(0.0, float(lap["tyre_age_at_lap_start"]))
            expected = (
                baseline
                + COMPOUND_BASE_DELTA.get(compound, 0.0)
                + COMPOUND_DEGRADATION.get(compound, 0.05) * tyre_age
                + fuel_effect_per_lap * (int(lap["lap_number"]) - 1)
            )
            residual_rows.append(
                {
                    "lap_number": int(lap["lap_number"]),
                    "name_acronym": driver,
                    "compound": compound,
                    "lap_duration": float(lap["lap_duration"]),
                    "expected_lap_time": expected,
                    "pace_residual": float(lap["lap_duration"]) - expected,
                }
            )

    residuals = pd.DataFrame(residual_rows)
    if residuals.empty:
        return residuals

    median_by_lap = residuals.groupby("lap_number", as_index=False)["pace_residual"].median()
    median_by_lap = median_by_lap.rename(columns={"pace_residual": "field_median_residual"})
    residuals = residuals.merge(median_by_lap, on="lap_number", how="left")
    smooth_map = (
        residuals[["lap_number", "field_median_residual"]]
        .drop_duplicates()
        .sort_values("lap_number")
        .assign(
            field_median_residual_smooth=lambda frame: frame["field_median_residual"].rolling(
                3,
                center=True,
                min_periods=1,
            ).mean()
        )[["lap_number", "field_median_residual_smooth"]]
    )
    residuals = residuals.merge(smooth_map, on="lap_number", how="left")
    return residuals


def build_track_state_diagnostics(
    rainfall_wetness_by_lap: dict[int, float],
    residuals: pd.DataFrame,
    total_laps: int,
    residual_threshold_seconds: float = 0.5,
) -> pd.DataFrame:
    """Place rainfall-derived wetness and shared pace residuals side by side."""
    track_state = pd.DataFrame({"lap_number": range(1, total_laps + 1)})
    track_state["rainfall_wetness"] = track_state["lap_number"].map(rainfall_wetness_by_lap).fillna(0.0)

    if residuals.empty:
        track_state["shared_residual"] = 0.0
        track_state["driver_count"] = 0
        track_state["residual_signal"] = 0.0
        return track_state

    shared = (
        residuals.groupby("lap_number", as_index=False)
        .agg(
            shared_residual=("field_median_residual_smooth", "first"),
            driver_count=("name_acronym", "nunique"),
        )
        .sort_values("lap_number")
    )
    track_state = track_state.merge(shared, on="lap_number", how="left")
    track_state["shared_residual"] = track_state["shared_residual"].interpolate(limit_direction="both").fillna(0.0)
    track_state["driver_count"] = track_state["driver_count"].fillna(0)

    dry_reference = track_state.loc[
        track_state["rainfall_wetness"] <= 0.02,
        "shared_residual",
    ]
    dry_baseline = float(dry_reference.median()) if not dry_reference.empty else float(track_state["shared_residual"].median())
    residual_excess = (track_state["shared_residual"] - dry_baseline - residual_threshold_seconds).clip(lower=0.0)
    positive_excess = residual_excess[residual_excess > 0]
    residual_scale = float(positive_excess.quantile(0.9)) if not positive_excess.empty else 1.0
    residual_scale = max(2.0, residual_scale)

    coverage = (track_state["driver_count"] / 3.0).clip(lower=0.0, upper=1.0)
    track_state["residual_signal"] = (residual_excess / residual_scale).clip(lower=0.0, upper=1.0) * coverage
    track_state["residual_signal"] = track_state["residual_signal"].rolling(
        3,
        center=True,
        min_periods=1,
    ).mean()

    return track_state


def build_reconstructed_track_state(
    rainfall_wetness_by_lap: dict[int, float],
    diagnostics: pd.DataFrame,
    total_laps: int,
    rainfall_weight: float = 0.35,
    minimum_driver_count: int = 3,
    residual_ramp_laps: int = 3,
) -> dict[int, float]:
    """Blend rainfall evidence with shared pace only for retrospective analysis.

    Shared pace slows immediately when rain starts, often before the rainfall
    proxy represents accumulated surface water. Its contribution therefore
    ramps in only after consecutive wet-proxy laps instead of jumping on the
    first observed shower.
    """
    rainfall_weight = max(0.0, min(1.0, rainfall_weight))
    residual_ramp_laps = max(1, int(residual_ramp_laps))
    maximum_residual_weight = 1.0 - rainfall_weight
    diagnostic_by_lap = diagnostics.set_index("lap_number") if not diagnostics.empty else pd.DataFrame()
    reconstructed: dict[int, float] = {}
    consecutive_wet_laps = 0

    for lap_number in range(1, total_laps + 1):
        rainfall_wetness = max(0.0, min(1.0, rainfall_wetness_by_lap.get(lap_number, 0.0)))
        consecutive_wet_laps = consecutive_wet_laps + 1 if rainfall_wetness > 0.05 else 0
        if lap_number not in diagnostic_by_lap.index:
            reconstructed[lap_number] = rainfall_wetness
            continue

        diagnostic = diagnostic_by_lap.loc[lap_number]
        driver_count = float(diagnostic.get("driver_count", 0.0))
        residual_signal = float(diagnostic.get("residual_signal", 0.0))
        if rainfall_wetness > 0.05 and driver_count >= minimum_driver_count:
            ramp_progress = (
                1.0
                if residual_ramp_laps == 1
                else max(0.0, min(1.0, (consecutive_wet_laps - 1) / (residual_ramp_laps - 1)))
            )
            residual_weight = maximum_residual_weight * ramp_progress
            reconstructed[lap_number] = (
                (1.0 - residual_weight) * rainfall_wetness + residual_weight * residual_signal
            )
        else:
            reconstructed[lap_number] = rainfall_wetness

    return reconstructed


def build_drying_scenarios(
    wetness_by_lap: dict[int, float],
    total_laps: int,
    shift_laps: int = 2,
) -> dict[str, dict[int, float]]:
    """Create drying variants without moving later re-wetting pulses.

    Each declining segment after the primary wet peak is shifted internally.
    Any later rise is preserved at its observed lap, so an uncertainty about
    drainage does not silently become an uncertainty about when rain returned.
    """
    base = {
        lap_number: max(0.0, min(1.0, wetness_by_lap.get(lap_number, 0.0)))
        for lap_number in range(1, total_laps + 1)
    }
    if not base:
        return {"干燥较快（-2 圈）": {}, "当前重建": {}, "干燥较慢（+2 圈）": {}}

    peak_wetness = max(base.values())
    peak_laps = [lap for lap, wetness in base.items() if wetness >= peak_wetness - 0.01]
    peak_end = max(peak_laps) if peak_laps else 1
    shift_laps = max(1, int(shift_laps))

    early = base.copy()
    late = base.copy()
    segment_start = peak_end
    for lap_number in range(peak_end + 1, total_laps + 2):
        is_rewetting = (
            lap_number <= total_laps
            and base[lap_number] > base[lap_number - 1] + 0.01
        )
        if lap_number <= total_laps and not is_rewetting:
            continue

        segment_end = lap_number - 1
        for segment_lap in range(segment_start + 1, segment_end + 1):
            early_source = min(segment_end, segment_lap + shift_laps)
            late_source = max(segment_start, segment_lap - shift_laps)
            early[segment_lap] = base[early_source]
            late[segment_lap] = base[late_source]
        segment_start = lap_number

    return {
        "干燥较快（-2 圈）": early,
        "当前重建": base,
        "干燥较慢（+2 圈）": late,
    }


def risk_level(score: float) -> str:
    if score >= 0.35:
        return "高"
    if score >= 0.18:
        return "中"
    return "低"
