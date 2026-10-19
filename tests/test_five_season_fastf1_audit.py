"""Focused tests for the five-season (2021-2025) FastF1 audit.

These tests are written with the standard-library ``unittest`` framework so they
are discovered by ``python -m unittest discover -s tests -v``. (pytest also runs
``unittest.TestCase`` subclasses, so ``python -m pytest tests/test_five_season_fastf1_audit.py``
exercises the same coverage.)

They guard the *boundaries* the tasks require:
- the five required season / exposure labels are present and correct;
- every scanned session is recorded; unavailable sessions are never silently dropped;
- abstentions are retained and never scored as hits;
- zero-signal seasons report ``N/A``, never ``0.0``;
- the timing anchor is correctly represented and no fallback is called pit-service;
- the 2021 frozen-rule transition/signal/abstention baseline remains reproducible;
- the report and protocol preserve the non-pooled / non-live / non-V1-validation
  boundary.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import scripts.build_five_season_report as report_builder
from src.fastf1_five_season_runner import EXPOSURE_LABELS, SEASONS, short_slug_name

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "reports", "results")

AUDIT_CSV = os.path.join(RESULTS_DIR, "v3_fastf1_audit_2021_2025.csv")
EVENTS_CSV = os.path.join(RESULTS_DIR, "v3_fastf1_events_2021_2025.csv")

EXPECTED_EXPOSURE = {
    2021: "previously audited",
    2022: "new pre-registered extension",
    2023: "partially development-exposed",
    2024: "partially development-exposed",
    2025: "development-exposed",
}


class TestFiveSeasonExposureAndIntegrity(unittest.TestCase):
    audit = None
    events = None

    @classmethod
    def setUpClass(cls):
        if os.path.exists(AUDIT_CSV):
            cls.audit = pd.read_csv(AUDIT_CSV)
        if os.path.exists(EVENTS_CSV):
            cls.events = pd.read_csv(EVENTS_CSV)

    def _require_audit(self):
        if self.audit is None:
            self.skipTest("combined audit CSV not generated yet")

    def _require_events(self):
        if self.events is None:
            self.skipTest("combined events CSV not generated yet")

    # ---- exposure labels ----
    def test_season_labels_match_expected_exposure_table(self):
        self.assertEqual(EXPOSURE_LABELS, EXPECTED_EXPOSURE)
        self.assertEqual(SEASONS, [2021, 2022, 2023, 2024, 2025])

    def test_short_slug_name_maps_known_events_to_existing_bundle_names(self):
        self.assertEqual(short_slug_name("Hungarian Grand Prix"), "Hungary")
        self.assertEqual(short_slug_name("British Grand Prix"), "United Kingdom")
        self.assertEqual(short_slug_name("Dutch Grand Prix"), "Netherlands")
        self.assertEqual(short_slug_name("Emilia Romagna Grand Prix"), "Emilia Romagna")
        self.assertEqual(short_slug_name("Canadian Grand Prix"), "Canada")
        self.assertEqual(short_slug_name("Brazilian Grand Prix"), "Brazil")

    # ---- output integrity (require the scan to have been run) ----
    def test_all_five_seasons_present_in_audit(self):
        self._require_audit()
        present = set(self.audit["season"].unique().tolist())
        self.assertEqual(present, set(SEASONS))

    def test_every_season_carries_its_exposure_label(self):
        self._require_audit()
        for season in SEASONS:
            sub = self.audit[self.audit["season"] == season]
            self.assertFalse(sub.empty, f"season {season} missing from audit")
            self.assertTrue((sub["exposure_label"] == EXPECTED_EXPOSURE[season]).all())

    def test_audit_records_unavailable_sessions_instead_of_dropping(self):
        self._require_audit()
        self.assertTrue(self.audit["selection_status"].notna().all())
        self.assertTrue((self.audit["selection_status"] != "").all())

    def test_2021_reproduces_existing_candidate_race_set(self):
        self._require_events()
        e21 = self.events[self.events["season"] == 2021]
        races = set(e21["race"].tolist())
        self.assertEqual(races, {"Emilia Romagna", "Hungary"})

    def test_events_retain_abstentions_and_do_not_score_them(self):
        self._require_events()
        abstain = self.events[self.events["state_machine_outcome"] == "abstain"]
        self.assertFalse(abstain.empty, "expected some abstentions to be retained")
        self.assertTrue(abstain["signal_eligible"].eq(False).all())
        self.assertTrue(abstain["signed_error_laps"].isna().all())
        self.assertTrue(abstain["absolute_error_laps"].isna().all())

    def test_only_single_rain_phase_issues_signals(self):
        self._require_events()
        signals = self.events[self.events["state_machine_outcome"] == "signal"]
        self.assertTrue((signals["event_quality"] == "single_rain_phase").all())
        self.assertTrue(signals["signed_error_laps"].notna().all())
        self.assertTrue(signals["absolute_error_laps"].notna().all())

    # ---- Task 12 timing-anchor correctness ----
    def test_timing_anchor_columns_present_in_events(self):
        self._require_events()
        for col in (
            "pit_service_lap",
            "timing_anchor_lap",
            "timing_anchor_source",
            "reporting_signed_error_laps",
            "reporting_absolute_error_laps",
        ):
            self.assertIn(col, self.events.columns)

    def test_no_fallback_is_labelled_pit_service(self):
        self._require_events()
        fallback = self.events[self.events["timing_anchor_source"] == "first_slick_fallback"]
        if not fallback.empty:
            self.assertTrue(
                fallback["pit_service_lap"].isna().all(),
                "first_slick_fallback rows must have a null pit_service_lap",
            )
        pit = self.events[self.events["timing_anchor_source"] == "pit_service"]
        if not pit.empty:
            self.assertTrue(
                pit["pit_service_lap"].notna().all(),
                "pit_service rows must carry a non-null pit_service_lap",
            )

    def test_2021_frozen_baseline_remains_reproducible(self):
        self._require_events()
        e21 = self.events[self.events["season"] == 2021]
        self.assertEqual(set(e21["race"].tolist()), {"Emilia Romagna", "Hungary"})
        issued = e21[e21["state_machine_outcome"] == "signal"]
        self.assertGreater(len(issued), 0, "2021 must retain its issued signals")


class TestZeroSignalSeasonsReportNA(unittest.TestCase):
    def test_zero_signal_seasons_report_na_not_zero(self):
        if not (os.path.exists(AUDIT_CSV) and os.path.exists(EVENTS_CSV)):
            self.skipTest("CSVs not generated yet")
        audit = pd.read_csv(AUDIT_CSV)
        events = pd.read_csv(EVENTS_CSV)
        for year in (2022, 2024, 2025):
            summary = report_builder._season_summary(audit, events, year)
            self.assertEqual(summary["signals"], 0, f"{year} should have no issued signals")
            self.assertEqual(
                summary["mae"],
                "N/A (no issued signals)",
                f"{year} MAE must be N/A, never 0.0",
            )
        for year in (2021, 2023):
            summary = report_builder._season_summary(audit, events, year)
            self.assertGreater(summary["signals"], 0, f"{year} should have signals")
            self.assertIsInstance(summary["mae"], float, f"{year} MAE should be numeric")


class TestReportNonPooledBoundary(unittest.TestCase):
    def test_report_and_protocol_preserve_non_pooled_boundary(self):
        report = os.path.join(PROJECT_ROOT, "reports", "research", "v3_five_season_fastf1_report.md")
        protocol = os.path.join(PROJECT_ROOT, "reports", "research", "v3_five_season_fastf1_protocol.md")
        for path in (report, protocol):
            if not os.path.exists(path):
                self.skipTest(f"{os.path.basename(path)} not written yet")
            with open(path, encoding="utf-8") as fh:
                text = fh.read().lower()
            self.assertIn("retrospective robustness audit", text)
            self.assertNotIn("five-year accuracy", text)
            self.assertNotIn("pooled accuracy", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
