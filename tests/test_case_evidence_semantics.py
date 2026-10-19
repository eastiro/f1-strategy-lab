from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.plots import plot_v2_track_state_evidence
from src.v2_validation import V2_CASES, build_validation_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"


class CaseEvidenceSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_validation_report().set_index("case_id")
        cls.app_source = STREAMLIT_APP.read_text(encoding="utf-8")

    def test_uk_canada_netherlands_pit_and_first_slick(self) -> None:
        self.assertEqual(int(self.report.loc["silverstone_2024", "actual_pit_lap"]), 39)
        self.assertEqual(int(self.report.loc["silverstone_2024", "first_slick_lap"]), 40)
        self.assertEqual(int(self.report.loc["montreal_2024", "actual_pit_lap"]), 45)
        self.assertEqual(int(self.report.loc["montreal_2024", "first_slick_lap"]), 46)
        self.assertEqual(int(self.report.loc["zandvoort_2023", "actual_pit_lap"]), 11)
        self.assertEqual(int(self.report.loc["zandvoort_2023", "first_slick_lap"]), 12)

    def test_chart_legend_names_cover_required_semantics(self) -> None:
        state = pd.DataFrame(
            {
                "lap_number": list(range(1, 21)),
                "decision_time_wetness": [0.1] * 20,
                "retrospective_wetness": [0.2] * 20,
            }
        )
        fig = plot_v2_track_state_evidence(
            state,
            decision_window_start=12,
            decision_window_end=10,
            retrospective_window_start=11,
            actual_pit_lap=11,
            first_slick_lap=12,
            to_compound="SOFT",
            field_pit_laps=[9, 10, 11, 12],
            later_rain_start=14,
        )
        names = {trace.name for trace in fig.data if getattr(trace, "name", None)}
        required = {
            "蓝色实线：仅降雨代理",
            "浅蓝带：公开信号区间",
            "蓝色点线：保守公开信号点",
            "橙色实线：赛后重建",
            "橙色点线：赛后重建窗口",
            "绿色实线：真实进站服务圈",
            "绿色虚线：首个光头胎圈",
        }
        self.assertTrue(required.issubset(names), f"missing legend entries: {required - names}")
        # Old ambiguous single green label must not be the only explanation.
        self.assertNotIn("绿虚线：真实换回光头胎", names)

    def test_page_explains_2021_versus_2025_and_avoids_old_green_only_copy(self) -> None:
        self.assertIn("为什么一个结果来自 2025，另一个来自 2021？", self.app_source)
        self.assertIn("真实进站服务圈", self.app_source)
        self.assertIn("首个光头胎圈", self.app_source)
        self.assertIn("保守公开信号点", self.app_source)
        self.assertIn("不能解释车队为什么选择在第 45 圈进站", self.app_source)
        self.assertIn("不是“下雨前就换干胎”的反例", self.app_source)
        self.assertNotIn("绿虚线：真实换回光头胎", self.app_source)

    def test_case_set_still_has_three_transitions(self) -> None:
        self.assertEqual(len(V2_CASES), 5)
        transition_cases = self.report[self.report["first_slick_lap"].notna()]
        self.assertEqual(len(transition_cases), 3)


if __name__ == "__main__":
    unittest.main()
