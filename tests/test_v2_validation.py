from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.config import PROCESSED_DIR
from src.errors import CaseDataUnavailableError
from src.evaluate_holdouts import summarise_signal_coverage
from src.fastf1_adapter import build_stints_from_laps
from src.fastf1_scanner import (
    count_wet_to_slick_transitions as count_fastf1_wet_to_slick_transitions,
)
from src.scan_openf1_holdouts import count_wet_to_slick_transitions
from src.strategy_model import (
    build_drying_scenarios,
    build_reconstructed_track_state,
    build_track_state_diagnostics,
    build_wetness_from_weather,
    estimate_pace_residuals,
)
from src.v2_validation import (
    V2_CASES,
    build_validation_metrics,
    build_validation_report,
    load_case_data,
)


class V2ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_validation_report().set_index("case_id")

    def test_case_set_contains_three_transitions_and_two_controls(self) -> None:
        self.assertEqual(len(self.report), 5)
        transition_cases = self.report[self.report["actual_transition_lap"].notna()]
        control_cases = self.report[self.report["actual_transition_lap"].isna()]
        self.assertEqual(len(transition_cases), 3)
        self.assertEqual(len(control_cases), 2)

    def test_public_signal_is_close_for_observed_wet_to_dry_cases(self) -> None:
        transition_cases = self.report[self.report["first_slick_lap"].notna()]
        errors = pd.to_numeric(transition_cases["decision_time_error_laps"], errors="coerce")
        self.assertTrue((errors.abs() <= 2).all())

    def test_controls_do_not_invent_a_public_switch_window(self) -> None:
        controls = self.report[self.report["first_slick_lap"].isna()]
        self.assertTrue(controls["decision_time_window_start"].isna().all())
        self.assertTrue(controls["retrospective_window_start"].isna().all())
        self.assertTrue(controls["failure_type"].notna().all())

    def test_silverstone_exposes_an_interval_not_a_single_rainfall_claim(self) -> None:
        silverstone = self.report.loc["silverstone_2024"]
        self.assertEqual(silverstone["rainfall_only_window_start"], 39)
        self.assertEqual(silverstone["decision_time_window_label"], "第 39-41 圈")

    def test_exploratory_cases_separate_pit_service_and_first_slick_laps(self) -> None:
        expected = {
            "silverstone_2024": (39, 40),
            "montreal_2024": (45, 46),
            "zandvoort_2023": (11, 12),
        }
        for case_id, (pit_lap, first_slick) in expected.items():
            row = self.report.loc[case_id]
            with self.subTest(case_id=case_id):
                self.assertEqual(int(row["actual_pit_lap"]), pit_lap)
                self.assertEqual(int(row["first_slick_lap"]), first_slick)
                # Compatibility field keeps first-slick semantics.
                self.assertEqual(int(row["actual_transition_lap"]), first_slick)

    def test_decision_time_error_is_scored_against_pit_service_lap(self) -> None:
        for case_id in ["silverstone_2024", "montreal_2024", "zandvoort_2023"]:
            row = self.report.loc[case_id]
            with self.subTest(case_id=case_id):
                signal = float(row["decision_time_window_start"])
                pit_lap = float(row["actual_pit_lap"])
                self.assertEqual(float(row["decision_time_error_laps"]), signal - pit_lap)
                # Must not silently score against first slick when pit exists.
                first_slick = float(row["first_slick_lap"])
                if first_slick != pit_lap:
                    self.assertNotEqual(
                        float(row["decision_time_error_laps"]),
                        signal - first_slick,
                    )

    def test_summary_metrics_include_controls_and_error(self) -> None:
        metrics = build_validation_metrics(self.report.reset_index())
        self.assertEqual(metrics["wet_to_dry_cases"], 3)
        self.assertEqual(metrics["public_within_two"], 3)
        self.assertEqual(metrics["controls_silent"], 2)
        # Errors are relative to pit service: Silverstone +2, Canada +2, Netherlands -1.
        self.assertAlmostEqual(metrics["public_mean_absolute_error"], 1.67, places=2)

    def test_required_portfolio_data_files_are_present(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        expected_slugs = [
            "2024_united_kingdom",
            "2024_canada",
            "2023_netherlands",
            "2024_netherlands",
            "2024_brazil",
        ]
        for slug in expected_slugs:
            self.assertTrue((project_root / "data" / "raw" / f"{slug}_race_control.json").exists())
            self.assertTrue((project_root / "data" / "processed" / f"{slug}_laps_clean.csv").exists())

    def test_missing_case_bundle_has_an_actionable_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            with patch("src.v2_validation.PROCESSED_DIR", Path(temporary_directory)):
                with self.assertRaisesRegex(CaseDataUnavailableError, "2024 United Kingdom") as raised:
                    load_case_data(V2_CASES[0])
        self.assertIn("2024_united_kingdom_laps_clean.csv", str(raised.exception))

    def test_invalid_case_bundle_names_the_missing_columns(self) -> None:
        case = V2_CASES[0]
        with TemporaryDirectory() as temporary_directory:
            processed_directory = Path(temporary_directory)
            pd.DataFrame([{"name_acronym": "NOR"}]).to_csv(
                processed_directory / f"{case.slug}_laps_clean.csv",
                index=False,
            )
            pd.DataFrame(
                [{"driver_number": 4, "stint_number": 1, "compound": "SOFT", "stint_lap_start": 1, "stint_lap_end": 3}]
            ).to_csv(processed_directory / f"{case.slug}_stints.csv", index=False)
            pd.DataFrame([{"date": "2024-01-01T00:00:00Z", "rainfall": 0}]).to_csv(
                processed_directory / f"{case.slug}_weather.csv",
                index=False,
            )
            with patch("src.v2_validation.PROCESSED_DIR", processed_directory):
                with self.assertRaisesRegex(CaseDataUnavailableError, "lap_duration"):
                    load_case_data(case)

    def test_silverstone_reconstruction_does_not_jump_on_the_first_rain_lap(self) -> None:
        slug = "2024_united_kingdom"
        laps = pd.read_csv(PROCESSED_DIR / f"{slug}_laps_clean.csv")
        weather = pd.read_csv(PROCESSED_DIR / f"{slug}_weather.csv")
        total_laps = int(laps["lap_number"].max())
        rainfall = build_wetness_from_weather(laps, weather, "NOR", total_laps)
        residuals = estimate_pace_residuals(laps, ["NOR", "HAM", "VER", "PIA"])
        diagnostics = build_track_state_diagnostics(rainfall, residuals, total_laps)
        reconstructed = build_reconstructed_track_state(rainfall, diagnostics, total_laps)

        self.assertAlmostEqual(reconstructed[18], rainfall[18], places=5)
        self.assertLess(reconstructed[19], 0.30)

    def test_silverstone_drying_scenarios_keep_the_late_rewetting_pulse_fixed(self) -> None:
        slug = "2024_united_kingdom"
        laps = pd.read_csv(PROCESSED_DIR / f"{slug}_laps_clean.csv")
        weather = pd.read_csv(PROCESSED_DIR / f"{slug}_weather.csv")
        total_laps = int(laps["lap_number"].max())
        rainfall = build_wetness_from_weather(laps, weather, "NOR", total_laps)
        residuals = estimate_pace_residuals(laps, ["NOR", "HAM", "VER", "PIA"])
        diagnostics = build_track_state_diagnostics(rainfall, residuals, total_laps)
        reconstructed = build_reconstructed_track_state(rainfall, diagnostics, total_laps)
        scenarios = build_drying_scenarios(reconstructed, total_laps)

        for scenario in scenarios.values():
            self.assertAlmostEqual(scenario[44], reconstructed[44], places=5)

    def test_fastf1_evaluation_bundles_and_report_are_present(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for slug in ["2021_emilia_romagna", "2021_hungary"]:
            self.assertTrue((project_root / "data" / "raw" / f"{slug}_fastf1_metadata.json").exists())
            self.assertTrue((project_root / "data" / "raw" / f"{slug}_race_control.json").exists())
            self.assertTrue((project_root / "data" / "processed" / f"{slug}_stints.csv").exists())
            self.assertTrue((project_root / "data" / "processed" / f"{slug}_weather.csv").exists())
        self.assertTrue((project_root / "reports" / "results" / "v2_fastf1_evaluation_2021.csv").exists())

    def test_holdout_scanner_counts_every_eligible_driver_transition(self) -> None:
        stints = [
            {"driver_number": 1, "stint_number": 1, "compound": "INTERMEDIATE"},
            {"driver_number": 1, "stint_number": 2, "compound": "SOFT"},
            {"driver_number": 2, "stint_number": 1, "compound": "WET"},
            {"driver_number": 2, "stint_number": 2, "compound": "MEDIUM"},
            {"driver_number": 3, "stint_number": 1, "compound": "SOFT"},
            {"driver_number": 3, "stint_number": 2, "compound": "INTERMEDIATE"},
        ]
        self.assertEqual(count_wet_to_slick_transitions(stints), 2)

    def test_state_machine_coverage_keeps_abstentions_out_of_accuracy(self) -> None:
        results = pd.DataFrame(
            [
                {"signal_eligible": True, "public_signal_error_laps": 2},
                {"signal_eligible": False, "public_signal_error_laps": None},
                {"signal_eligible": False, "public_signal_error_laps": -8},
            ]
        )
        summary = summarise_signal_coverage(results)
        self.assertEqual(summary["events"], 3)
        self.assertEqual(summary["signals_issued"], 1)
        self.assertEqual(summary["abstentions"], 2)
        self.assertEqual(summary["issued_within_two_laps"], 1)

    def test_fastf1_scanner_counts_compound_changes_by_driver_stint(self) -> None:
        laps = pd.DataFrame(
            [
                {"Driver": "AAA", "LapNumber": 1, "Stint": 1, "Compound": "INTERMEDIATE"},
                {"Driver": "AAA", "LapNumber": 2, "Stint": 1, "Compound": "INTERMEDIATE"},
                {"Driver": "AAA", "LapNumber": 3, "Stint": 2, "Compound": "MEDIUM"},
                {"Driver": "BBB", "LapNumber": 1, "Stint": 1, "Compound": "WET"},
                {"Driver": "BBB", "LapNumber": 2, "Stint": 2, "Compound": "SOFT"},
                {"Driver": "CCC", "LapNumber": 1, "Stint": 1, "Compound": "SOFT"},
                {"Driver": "CCC", "LapNumber": 2, "Stint": 2, "Compound": "INTERMEDIATE"},
            ]
        )
        self.assertEqual(count_fastf1_wet_to_slick_transitions(laps), 2)

    def test_fastf1_adapter_builds_v2_stint_schema(self) -> None:
        laps = pd.DataFrame(
            [
                {"Driver": "AAA", "DriverNumber": 1, "LapNumber": 1, "Stint": 1, "Compound": "INTERMEDIATE"},
                {"Driver": "AAA", "DriverNumber": 1, "LapNumber": 2, "Stint": 1, "Compound": "INTERMEDIATE"},
                {"Driver": "AAA", "DriverNumber": 1, "LapNumber": 3, "Stint": 2, "Compound": "MEDIUM"},
            ]
        )
        stints = build_stints_from_laps(laps)
        self.assertEqual(stints.to_dict("records"), [
            {"driver_number": 1, "stint_number": 1, "compound": "INTERMEDIATE", "stint_lap_start": 1, "stint_lap_end": 2, "driver": "AAA"},
            {"driver_number": 1, "stint_number": 2, "compound": "MEDIUM", "stint_lap_start": 3, "stint_lap_end": 3, "driver": "AAA"},
        ])


if __name__ == "__main__":
    unittest.main()
