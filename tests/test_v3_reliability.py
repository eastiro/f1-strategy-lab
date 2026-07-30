from __future__ import annotations

import unittest

import pandas as pd

from src.v3_reliability import build_reliability_events, summarize_reliability_events


class V3ReliabilityTests(unittest.TestCase):
    def test_combines_signal_and_abstention_rows(self) -> None:
        challenge = pd.DataFrame(
            [
                {
                    "year": 2025,
                    "country": "Australia",
                    "driver": "ALB",
                    "actual_transition_lap": 33,
                    "public_signal_window_start": 32,
                    "public_signal_error_laps": -1,
                    "event_quality": "single_rain_phase",
                },
                {
                    "year": 2025,
                    "country": "Spain",
                    "driver": "NOR",
                    "actual_transition_lap": 20,
                    "public_signal_window_start": None,
                    "public_signal_error_laps": None,
                    "event_quality": "multi_rain_phase",
                },
            ]
        )
        fastf1 = pd.DataFrame(
            [
                {
                    "year": 2021,
                    "country": "Imola",
                    "driver": "ALO",
                    "actual_transition_lap": 29,
                    "state_machine_outcome": "signal",
                    "public_signal_error_laps": 8,
                    "event_quality": "race_control_adjacent",
                    "nearby_race_control_phases": "safety_car",
                }
            ]
        )

        events = build_reliability_events(challenge, fastf1)

        self.assertEqual(events["outcome"].tolist(), ["signal_accurate", "abstain", "signal_inaccurate"])
        self.assertEqual(events.loc[1, "reason_group"], "multi_rain_phase")
        self.assertEqual(events.loc[2, "reason_group"], "race_control_adjacent")

    def test_summary_is_empty_for_missing_inputs(self) -> None:
        summary = summarize_reliability_events(build_reliability_events())
        self.assertTrue(summary.empty)
        self.assertIn("事件数", summary.columns)

    def test_summary_preserves_source_and_outcome_counts(self) -> None:
        challenge = pd.DataFrame(
            [
                {
                    "year": 2025,
                    "country": "Australia",
                    "driver": "ALB",
                    "actual_transition_lap": 33,
                    "public_signal_window_start": 32,
                    "public_signal_error_laps": -1,
                    "event_quality": "single_rain_phase",
                },
                {
                    "year": 2025,
                    "country": "Spain",
                    "driver": "NOR",
                    "actual_transition_lap": 20,
                    "public_signal_window_start": None,
                    "public_signal_error_laps": None,
                    "event_quality": "multi_rain_phase",
                },
            ]
        )
        fastf1 = pd.DataFrame(
            [
                {
                    "year": 2021,
                    "country": "Imola",
                    "driver": "ALO",
                    "actual_transition_lap": 29,
                    "state_machine_outcome": "signal",
                    "public_signal_error_laps": 8,
                    "event_quality": "race_control_adjacent",
                }
            ]
        )

        summary = summarize_reliability_events(build_reliability_events(challenge, fastf1))
        counts = {
            (row.source, row.outcome): int(row.事件数)
            for row in summary.itertuples()
        }
        self.assertEqual(
            counts,
            {
                ("2021 FastF1", "signal_inaccurate"): 1,
                ("2025 挑战集", "abstain"): 1,
                ("2025 挑战集", "signal_accurate"): 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
