from __future__ import annotations

import unittest

import pandas as pd

from src.plots import plot_strategy_weather_timeline, plot_v2_track_state_evidence
from src.race_control import extract_strategy_events
from src.strategy_model import (
    StintPlan,
    build_drying_scenarios,
    build_reconstructed_track_state,
    build_track_state_diagnostics,
    build_wetness_curve,
    build_wetness_from_weather,
    estimate_driver_baseline,
    estimate_pace_residuals,
    find_stint,
    risk_level,
    simulate_strategy,
)
from src.weather_timing import map_timestamps_to_laps, prepare_driver_lap_timing


class StrategyModelTests(unittest.TestCase):
    def test_strategy_event_extraction_groups_same_lap_messages(self) -> None:
        events = extract_strategy_events(
            [
                {"lap_number": 19, "message": "DRS DISABLED"},
                {"lap_number": 19, "message": "LOW GRIP CONDITIONS"},
                {"lap_number": 41, "message": "DRS ENABLED"},
                {"lap_number": 41, "message": "NORMAL GRIP CONDITIONS"},
                {"lap_number": 52, "message": "CHEQUERED FLAG"},
            ]
        )

        self.assertEqual(
            events,
            [
                {"lap": 19, "label": "DRS disabled · Low grip", "kind": "wet_warning"},
                {"lap": 41, "label": "DRS enabled · Normal grip", "kind": "dry_confirmation"},
            ],
        )

    def test_combined_timeline_renders_track_state_stints_and_events(self) -> None:
        stints = pd.DataFrame(
            [
                {"name_acronym": "NOR", "compound": "MEDIUM", "stint_lap_start": 1, "stint_lap_end": 2, "tyre_age_at_start": 0},
                {"name_acronym": "NOR", "compound": "INTERMEDIATE", "stint_lap_start": 3, "stint_lap_end": 4, "tyre_age_at_start": 0},
            ]
        )
        figure = plot_strategy_weather_timeline(
            stints,
            {1: 0.0, 2: 0.1, 3: 0.5, 4: 0.2},
            [{"lap": 3, "label": "Low grip", "kind": "wet_warning"}],
        )

        self.assertEqual(len(figure.data), 4)
        self.assertEqual(len(figure.layout.shapes), 2)

    def test_v2_evidence_labels_the_real_pit_anchor_without_window_label_clutter(self) -> None:
        state = pd.DataFrame(
            {
                "lap_number": [38, 39, 40, 41],
                "decision_time_wetness": [0.6, 0.5, 0.3, 0.1],
                "retrospective_wetness": [0.7, 0.5, 0.2, 0.1],
            }
        )

        figure = plot_v2_track_state_evidence(
            state,
            actual_transition_laps=[40],
            decision_window_start=41,
            retrospective_window_start=39,
            decision_window_end=39,
        )

        annotations = list(figure.layout.annotations)
        self.assertEqual([annotation.text for annotation in annotations], ["Pit L40"])
        self.assertEqual(len(figure.layout.shapes), 4)

    def test_simulation_applies_compound_wear_and_pit_loss(self) -> None:
        result = simulate_strategy(
            total_laps=3,
            baseline_lap_time=100.0,
            stints=[StintPlan("SOFT", 1, 2), StintPlan("HARD", 3, 3)],
            pit_loss=20.0,
            fuel_effect_per_lap=0.0,
        )

        self.assertAlmostEqual(result.total_time, 319.705, places=3)
        self.assertEqual(result.laps["compound"].tolist(), ["SOFT", "SOFT", "HARD"])
        self.assertEqual(result.laps["pit_lap"].tolist(), [False, False, True])

    def test_intermediate_wear_is_lower_when_the_track_is_wet(self) -> None:
        stints = [StintPlan("INTERMEDIATE", 1, 3)]
        wet_result = simulate_strategy(
            3,
            100.0,
            stints,
            fuel_effect_per_lap=0.0,
            wetness_by_lap={1: 1.0, 2: 1.0, 3: 1.0},
        )
        dry_result = simulate_strategy(
            3,
            100.0,
            stints,
            fuel_effect_per_lap=0.0,
            wetness_by_lap={1: 0.0, 2: 0.0, 3: 0.0},
        )

        self.assertLess(wet_result.laps.loc[2, "wear_exposure"], dry_result.laps.loc[2, "wear_exposure"])
        self.assertLess(wet_result.laps.loc[2, "degradation"], dry_result.laps.loc[2, "degradation"])

    def test_find_stint_rejects_an_uncovered_lap(self) -> None:
        with self.assertRaisesRegex(ValueError, "No stint covers lap 3"):
            find_stint([StintPlan("SOFT", 1, 2)], 3)

    def test_wetness_curve_clamps_inputs_and_uses_zero_at_window_edges(self) -> None:
        curve = build_wetness_curve(8, 0, 99, peak_wetness=2.0, ramp_fraction=0.9)

        self.assertEqual(set(curve), set(range(1, 9)))
        self.assertEqual(curve[1], 0.0)
        self.assertEqual(curve[8], 0.0)
        self.assertLessEqual(max(curve.values()), 1.0)

    def test_driver_baseline_uses_the_median_clean_lap(self) -> None:
        laps = pd.DataFrame(
            [
                {"name_acronym": "NOR", "clean_lap": True, "lap_duration": 90.0},
                {"name_acronym": "NOR", "clean_lap": True, "lap_duration": 92.0},
                {"name_acronym": "NOR", "clean_lap": False, "lap_duration": 120.0},
            ]
        )

        self.assertEqual(estimate_driver_baseline(laps, "NOR"), 91.0)
        with self.assertRaisesRegex(ValueError, "No clean laps found"):
            estimate_driver_baseline(laps, "HAM")

    def test_risk_level_has_stable_threshold_boundaries(self) -> None:
        self.assertEqual(risk_level(0.35), risk_level(0.36))
        self.assertNotEqual(risk_level(0.35), risk_level(0.34))
        self.assertEqual(risk_level(0.18), risk_level(0.19))
        self.assertNotEqual(risk_level(0.18), risk_level(0.17))

    def test_weather_timing_maps_events_to_containing_or_nearest_lap(self) -> None:
        laps = pd.DataFrame(
            [
                {"name_acronym": "NOR", "lap_number": 1, "date_start": "2024-01-01T12:00:00Z", "lap_duration": 90},
                {"name_acronym": "NOR", "lap_number": 2, "date_start": "2024-01-01T12:01:30Z", "lap_duration": None},
            ]
        )
        timing = prepare_driver_lap_timing(laps, "NOR")
        timestamps = pd.Series(["2024-01-01T12:00:30Z", "2024-01-01T12:04:00Z"])

        self.assertEqual(timing.loc[1, "lap_duration"], 90.0)
        self.assertEqual(map_timestamps_to_laps(timing, timestamps, total_laps=2), [1, 2])

    def test_weather_wetness_stays_bounded_and_rises_after_rainfall(self) -> None:
        laps = pd.DataFrame(
            [
                {"name_acronym": "NOR", "lap_number": 1, "date_start": "2024-01-01T12:00:00Z", "lap_duration": 100},
                {"name_acronym": "NOR", "lap_number": 2, "date_start": "2024-01-01T12:01:40Z", "lap_duration": 100},
            ]
        )
        weather = pd.DataFrame(
            [
                {"date": "2024-01-01T12:01:30Z", "rainfall": 1, "track_temperature": 25},
                {"date": "2024-01-01T12:03:00Z", "rainfall": 1, "track_temperature": 25},
            ]
        )

        wetness = build_wetness_from_weather(laps, weather, "NOR", total_laps=2)

        self.assertEqual(set(wetness), {1, 2})
        self.assertTrue(all(0.0 <= value <= 0.65 for value in wetness.values()))
        self.assertGreater(wetness[2], wetness[1])

    def test_pace_residuals_remove_the_explicit_baseline_terms(self) -> None:
        laps = pd.DataFrame(
            [
                {"name_acronym": "NOR", "clean_lap": True, "lap_number": 1, "lap_duration": 100.0, "compound": "MEDIUM", "tyre_age_at_lap_start": 0},
                {"name_acronym": "NOR", "clean_lap": True, "lap_number": 2, "lap_duration": 100.0, "compound": "MEDIUM", "tyre_age_at_lap_start": 0},
                {"name_acronym": "NOR", "clean_lap": True, "lap_number": 3, "lap_duration": 100.1, "compound": "MEDIUM", "tyre_age_at_lap_start": 1},
            ]
        )

        residuals = estimate_pace_residuals(laps, ["NOR"], fuel_effect_per_lap=-0.1)

        self.assertEqual(residuals["lap_number"].tolist(), [2, 3])
        self.assertAlmostEqual(residuals.loc[0, "pace_residual"], 0.1, places=5)
        self.assertAlmostEqual(residuals.loc[1, "pace_residual"], 0.26, places=5)

    def test_track_state_uses_rainfall_when_no_shared_pace_exists(self) -> None:
        diagnostics = build_track_state_diagnostics({1: 0.2, 2: 0.5}, pd.DataFrame(), total_laps=3)
        reconstructed = build_reconstructed_track_state({1: 0.2, 2: 0.5}, diagnostics, total_laps=3)

        self.assertEqual(diagnostics["shared_residual"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(diagnostics["residual_signal"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(reconstructed, {1: 0.2, 2: 0.5, 3: 0.0})

    def test_track_state_delays_residual_weight_at_the_start_of_rain(self) -> None:
        diagnostics = pd.DataFrame(
            {
                "lap_number": [1, 2, 3],
                "driver_count": [3, 3, 3],
                "residual_signal": [0.9, 0.9, 0.9],
            }
        )
        reconstructed = build_reconstructed_track_state(
            {1: 0.1, 2: 0.2, 3: 0.3},
            diagnostics,
            total_laps=3,
        )

        self.assertEqual(reconstructed[1], 0.1)
        self.assertLess(reconstructed[2], 0.655)
        self.assertAlmostEqual(reconstructed[3], 0.69, places=5)

    def test_drying_scenarios_shift_only_the_post_peak_recovery(self) -> None:
        scenarios = build_drying_scenarios({1: 0.0, 2: 0.8, 3: 0.6, 4: 0.2}, total_laps=4, shift_laps=1)

        self.assertEqual(scenarios["当前重建"][3], 0.6)
        self.assertLess(scenarios["干燥较快（-2 圈）"][3], scenarios["当前重建"][3])
        self.assertGreater(scenarios["干燥较慢（+2 圈）"][3], scenarios["当前重建"][3])

    def test_drying_scenarios_keep_later_rewetting_at_its_observed_lap(self) -> None:
        scenarios = build_drying_scenarios(
            {1: 0.8, 2: 0.6, 3: 0.3, 4: 0.55, 5: 0.4, 6: 0.1},
            total_laps=6,
            shift_laps=1,
        )

        self.assertEqual(scenarios["干燥较快（-2 圈）"][4], 0.55)
        self.assertEqual(scenarios["干燥较慢（+2 圈）"][4], 0.55)


if __name__ == "__main__":
    unittest.main()
