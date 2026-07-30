from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

from streamlit.testing.v1 import AppTest

from src.v2_validation import V2_CASES, build_case_track_state, build_validation_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"
PORTFOLIO_APP = PROJECT_ROOT / "app" / "portfolio_app.py"

EXPECTED_FACT_PAIRS = {
    "silverstone_2024": (39, 40),
    "montreal_2024": (45, 46),
    "zandvoort_2023": (11, 12),
}


class PortfolioDualAnchorReportTests(unittest.TestCase):
    """The shared report must expose both anchor columns and keep the facts."""

    def setUp(self) -> None:
        self.report = build_validation_report()

    def test_report_exposes_both_anchor_columns(self) -> None:
        for column in ("actual_pit_lap", "first_slick_lap"):
            with self.subTest(column=column):
                self.assertIn(column, self.report.columns)

    def test_three_factual_pairs_remain_correct(self) -> None:
        for case_id, (pit, slick) in EXPECTED_FACT_PAIRS.items():
            row = self.report[self.report["case_id"] == case_id].iloc[0]
            with self.subTest(case_id=case_id):
                self.assertEqual(int(row["actual_pit_lap"]), pit)
                self.assertEqual(int(row["first_slick_lap"]), slick)


class PortfolioDualAnchorSourceTests(unittest.TestCase):
    """The portfolio source must render both anchors, not a single switch label."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PORTFOLIO_APP.read_text(encoding="utf-8")

    def test_table_uses_dual_anchor_fields(self) -> None:
        self.assertIn('"Pit service"', self.source)
        self.assertIn('"First slick"', self.source)
        self.assertIn('report["actual_pit_lap"]', self.source)
        self.assertIn('report["first_slick_lap"]', self.source)

    def test_single_switch_label_removed_from_table(self) -> None:
        self.assertNotIn('"Observed switch"', self.source)

    def test_chart_annotations_explain_both_anchors(self) -> None:
        self.assertIn("Pit service lap", self.source)
        self.assertIn("First slick lap", self.source)

    def test_explanatory_copy_present(self) -> None:
        self.assertIn("is scored against the pit service lap", self.source)


def _load_portfolio_module():
    """Import portfolio_app with streamlit stubbed so its pure chart helper is testable."""
    fake_st = mock.MagicMock()
    fake_st.cache_data.return_value = lambda fn: fn
    fake_st.columns.side_effect = lambda n: tuple(mock.MagicMock() for _ in range(int(n)))
    fake_st.stop.side_effect = lambda: None
    fake_st.selectbox.return_value = "silverstone_2024"
    with mock.patch.dict(sys.modules, {"streamlit": fake_st}):
        spec = importlib.util.spec_from_file_location("portfolio_app_dualanchor", str(PORTFOLIO_APP))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class PortfolioDualAnchorChartTests(unittest.TestCase):
    """The evidence chart must show solid (pit) and dashed (first slick) marks."""

    def test_canadian_case_shows_dual_markers(self) -> None:
        module = _load_portfolio_module()
        case = next(c for c in V2_CASES if c.case_id == "montreal_2024")
        state, transitions = build_case_track_state(case)
        fig = module.make_evidence_chart(
            state, transitions, public_latest=None, public_earliest=None, retrospective=None
        )
        shapes = list(fig.layout.shapes)
        green = [s for s in shapes if getattr(getattr(s, "line", None), "color", None) == "#16a34a"]
        solid_x = {
            int(s.x0)
            for s in green
            if getattr(getattr(s, "line", None), "dash", "solid") in (None, "solid")
        }
        dashed_x = {
            int(s.x0) for s in green if getattr(getattr(s, "line", None), "dash", None) == "dash"
        }
        self.assertIn(45, solid_x, "pit service lap should be a solid green mark at lap 45")
        self.assertIn(46, dashed_x, "first slick lap should be a dashed green mark at lap 46")

    def test_same_lap_does_not_draw_duplicate_marks(self) -> None:
        module = _load_portfolio_module()
        case = next(c for c in V2_CASES if c.case_id == "montreal_2024")
        state, _ = build_case_track_state(case)
        # Build a transition where pit == first slick to verify no duplicate.
        transitions = [{"actual_pit_lap": 10, "first_slick_lap": 10}]
        fig = module.make_evidence_chart(
            state, transitions, public_latest=None, public_earliest=None, retrospective=None
        )
        shapes = list(fig.layout.shapes)
        green = [s for s in shapes if getattr(getattr(s, "line", None), "color", None) == "#16a34a"]
        self.assertEqual(len(green), 1, "same-lap case must draw exactly one green mark")


class BothEntryPointsAppTest(unittest.TestCase):
    """Both Streamlit entry points must run without raising."""

    def test_chinese_and_portfolio_apps_run(self) -> None:
        for path in (STREAMLIT_APP, PORTFOLIO_APP):
            with self.subTest(path=path.name):
                at = AppTest.from_file(str(path), default_timeout=30)
                at.run()
                self.assertFalse(at.exception, f"{path.name} raised an exception")


if __name__ == "__main__":
    unittest.main()
