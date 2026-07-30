"""Focused tests for the V3.1 decision-chain visibility (Task 06).

These tests intentionally avoid re-implementing the model. They check that the
pure helper ``build_model_trace`` and the Chinese cross-race page source expose
the four required decision-chain concepts, and that the per-case trace text
stays truthful (no team-causality claims, clean control cases).
"""

from __future__ import annotations

import unittest

from src.v2_trace import build_model_trace
from src.v2_validation import V2_CASES, build_validation_report

try:
    from streamlit.testing.v1 import AppTest

    _HAS_APP_TEST = True
except Exception:  # pragma: no cover - streamlit missing in some envs
    _HAS_APP_TEST = False

APP_PATH = "app/streamlit_app.py"
CONTROL_CASE_ID = "zandvoort_2024_control"


def _report_row(case_id: str):
    report = build_validation_report()
    return report[report["case_id"] == case_id].iloc[0]


class DecisionChainSourceTest(unittest.TestCase):
    def test_static_band_concepts_present_in_source(self):
        with open(APP_PATH, encoding="utf-8") as fh:
            source = fh.read()
        for phrase in ["审计看什么", "审计怎么判断", "审计输出什么", "如何核对"]:
            self.assertIn(phrase, source, f"static band missing concept: {phrase}")

    def test_dynamic_trace_labels_present_in_source(self):
        with open(APP_PATH, encoding="utf-8") as fh:
            source = fh.read()
        for label in ["公开输入", "规则判断", "审计输出", "赛后核对"]:
            self.assertIn(label, source, f"dynamic trace missing label: {label}")

    def test_helper_returns_four_stages(self):
        trace = build_model_trace(_report_row("montreal_2024"))
        for key in ["inputs", "rule", "output", "check"]:
            self.assertIn(key, trace)
        self.assertTrue(trace["has_switch"])


class CanadianTraceTest(unittest.TestCase):
    def setUp(self):
        self.trace = build_model_trace(_report_row("montreal_2024"))

    def test_output_shows_l47(self):
        self.assertIn("第 47 圈", self.trace["output"])

    def test_check_shows_l45_pit_service(self):
        self.assertIn("第 45 圈", self.trace["check"])

    def test_no_team_causality_wording(self):
        full = " ".join(self.trace[k] for k in ("inputs", "rule", "output", "check"))
        self.assertNotIn("车队", full)
        self.assertIn("不是模型输入", self.trace["check"])
        # The trace must not claim to know why the team pitted on lap 45.
        self.assertIn("不解释为什么真实进站发生在此圈", self.trace["check"])


class NetherlandsTraceTest(unittest.TestCase):
    def setUp(self):
        self.trace = build_model_trace(_report_row("zandvoort_2023"))

    def test_explains_later_rain_phase_boundary(self):
        full = " ".join(self.trace[k] for k in ("inputs", "rule", "output", "check"))
        self.assertIn("后续", full)
        self.assertIn("独立降雨", full)


class ControlCaseTraceTest(unittest.TestCase):
    def setUp(self):
        self.trace = build_model_trace(_report_row(CONTROL_CASE_ID))

    def test_no_switch_and_no_error_number(self):
        self.assertFalse(self.trace["has_switch"])
        self.assertEqual(self.trace["output"], "不判断（对湿转干保持沉默）。")
        # No fabricated error number for a control case.
        self.assertNotIn("误差", self.trace["check"])
        self.assertNotIn("第", self.trace["check"])
        self.assertIn("对照", self.trace["inputs"])


@unittest.skipUnless(_HAS_APP_TEST, "streamlit AppTest unavailable")
class ChineseAppDecisionChainRenderTest(unittest.TestCase):
    def _fresh_select(self, case_id: str):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        self.assertFalse(at.exception, f"page raised before selection: {at.exception}")
        selector = next(s for s in at.selectbox if s.key == "v2_case_selector")
        selector.set_value(case_id).run()
        self.assertFalse(
            at.exception,
            f"page raised on selection {case_id}: {at.exception}",
        )

    def test_silverstone_renders(self):
        self._fresh_select("silverstone_2024")

    def test_canada_renders(self):
        self._fresh_select("montreal_2024")

    def test_netherlands_renders(self):
        self._fresh_select("zandvoort_2023")

    def test_control_renders(self):
        self._fresh_select(CONTROL_CASE_ID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
