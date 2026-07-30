"""Task 14: V4.1 最后一停策略判断摘要（先给答案，再给图表）的聚焦测试。

覆盖：
- 摘要位于比赛背景之后、热力图视角控件之前；
- 它使用已有稳健性对象（robust_best / dominant_alternatives /
  scenario_results_for），而不是重新计算或硬编码答案；
- 无稳健替代方案时，页面明确说“没有方案在三种设定中都优于基准”；
- 三情景名称和结果读取自已有 scenario_results_for(robust_best)；
- 页面保留“不是实际策略预测”的边界；
- 下方热力图的当前重建结论被明确标为“仅当前重建下”。

不修改任何模拟公式或数据：strategy_model.py / plots.py / V2-V3 均不动。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.conclusion_summary import build_strategy_judgment, describe_delta

APP_PATH = "app/streamlit_app.py"


def _make_stint(start_lap: int) -> object:
    class _Stint:
        def __init__(self, lap: int) -> None:
            self.start_lap = lap

    return _Stint(start_lap)


def _make_robust(slick_pit: int, compound: str, worst: float, median: float) -> pd.Series:
    return pd.Series(
        {
            "slick_pit": slick_pit,
            "compound": compound,
            "worst_case": worst,
            "median_delta": median,
        }
    )


def _make_scenario_fn(slick_pit: int, compound: str, deltas: dict[str, float]):
    def _fn(candidate: pd.Series) -> pd.DataFrame:
        # 复刻 app 里的 scenario_results_for：按候选的圈数与配方返回三情景结果。
        assert int(candidate["slick_pit"]) == slick_pit
        assert candidate["compound"] == compound
        return pd.DataFrame(
            [{"scenario": s, "delta": d} for s, d in deltas.items()]
        )

    return _fn


EMPTY_DOMINANT = pd.DataFrame(columns=["slick_pit", "compound", "worst_case", "median_delta"])


class ConclusionHelperTest(unittest.TestCase):
    def test_question_baseline_reads_from_objects_not_hardcoded(self) -> None:
        # 用非默认的圈数/窗口，证明文字来自传入对象而非硬编码 40。
        robust = _make_robust(37, "MEDIUM", 0.30, -0.80)
        deltas = {"干燥较快（-2 圈）": -1.5, "当前赛后重建": -0.8, "干燥较慢（+2 圈）": 0.3}
        j = build_strategy_judgment(
            robust_best=robust,
            dominant_alternatives=EMPTY_DOMINANT,
            reference_final_stint=_make_stint(37),
            robust_compound_label="中胎",
            reference_compound_label="软胎",
            pit_window_start=34,
            pit_window_end=42,
            scenario_results_for=_make_scenario_fn(37, "MEDIUM", deltas),
        )
        self.assertIn("第 37 圈换软胎", j["question_baseline"])
        self.assertIn("第 34-42 圈", j["question_baseline"])
        # 不应出现默认 40 圈的硬编码文本
        self.assertNotIn("第 40 圈", j["question_baseline"])

    def test_empty_dominant_says_no_alternative_dominates(self) -> None:
        robust = _make_robust(40, "MEDIUM", 0.30, -0.80)
        deltas = {"干燥较快（-2 圈）": -1.5, "当前赛后重建": -0.8, "干燥较慢（+2 圈）": 0.3}
        j = build_strategy_judgment(
            robust_best=robust,
            dominant_alternatives=EMPTY_DOMINANT,
            reference_final_stint=_make_stint(40),
            robust_compound_label="中胎",
            reference_compound_label="软胎",
            pit_window_start=37,
            pit_window_end=45,
            scenario_results_for=_make_scenario_fn(40, "MEDIUM", deltas),
        )
        self.assertEqual(j["robustness_kind"], "warning")
        self.assertIn("没有替代方案都优于", j["robustness_conclusion"])
        self.assertIn("最接近稳健的是第 40 圈换中胎", j["robustness_conclusion"])
        self.assertIn("干燥较慢（+2 圈）", j["robustness_conclusion"])
        self.assertIn("失去优势", j["robustness_conclusion"])
        self.assertIn("不是实际策略预测", j["robustness_conclusion"])

    def test_nonempty_dominant_reports_count_and_closest(self) -> None:
        robust = _make_robust(40, "MEDIUM", -0.50, -1.00)
        deltas = {"干燥较快（-2 圈）": -2.0, "当前赛后重建": -1.0, "干燥较慢（+2 圈）": -0.5}
        dominant = pd.DataFrame(
            [{"slick_pit": 39, "compound": "SOFT", "worst_case": -0.4, "median_delta": -0.9}]
        )
        j = build_strategy_judgment(
            robust_best=robust,
            dominant_alternatives=dominant,
            reference_final_stint=_make_stint(40),
            robust_compound_label="中胎",
            reference_compound_label="软胎",
            pit_window_start=37,
            pit_window_end=45,
            scenario_results_for=_make_scenario_fn(40, "MEDIUM", deltas),
        )
        self.assertEqual(j["robustness_kind"], "info")
        self.assertIn("有 1 个替代方案都优于", j["robustness_conclusion"])
        self.assertIn("最接近稳健的是第 40 圈换中胎", j["robustness_conclusion"])
        self.assertIn("不等于真实最优", j["robustness_conclusion"])

    def test_three_scenario_table_from_scenario_results_for(self) -> None:
        robust = _make_robust(40, "MEDIUM", -0.50, -1.00)
        deltas = {"干燥较快（-2 圈）": -2.0, "当前赛后重建": -1.0, "干燥较慢（+2 圈）": -0.5}
        j = build_strategy_judgment(
            robust_best=robust,
            dominant_alternatives=EMPTY_DOMINANT,
            reference_final_stint=_make_stint(40),
            robust_compound_label="中胎",
            reference_compound_label="软胎",
            pit_window_start=37,
            pit_window_end=45,
            scenario_results_for=_make_scenario_fn(40, "MEDIUM", deltas),
        )
        table = j["three_scenario_table"]
        self.assertEqual(list(table["干燥设定"]), [
            "干燥较快（-2 圈）",
            "当前赛后重建",
            "干燥较慢（+2 圈）",
        ])
        # 数值直接来自 scenario_results_for 的 delta，而非硬编码
        self.assertEqual(list(table["数值（秒）"]), ["-2.00", "-1.00", "-0.50"])
        self.assertEqual(list(table["相对基准"]), [
            describe_delta(-2.0),
            describe_delta(-1.0),
            describe_delta(-0.5),
        ])

    def test_worst_case_text_matches_robust_best(self) -> None:
        robust = _make_robust(40, "MEDIUM", -0.50, -1.00)
        deltas = {"干燥较快（-2 圈）": -2.0, "当前赛后重建": -1.0, "干燥较慢（+2 圈）": -0.5}
        j = build_strategy_judgment(
            robust_best=robust,
            dominant_alternatives=EMPTY_DOMINANT,
            reference_final_stint=_make_stint(40),
            robust_compound_label="中胎",
            reference_compound_label="软胎",
            pit_window_start=37,
            pit_window_end=45,
            scenario_results_for=_make_scenario_fn(40, "MEDIUM", deltas),
        )
        self.assertEqual(j["worst_case_text"], describe_delta(-0.50))


class ConclusionChinesePageTest(unittest.TestCase):
    def _source(self) -> str:
        with open(APP_PATH, encoding="utf-8") as fh:
            return fh.read()

    def test_summary_after_race_background_before_heatmap_radio(self) -> None:
        source = self._source()
        self.assertIn("#### 比赛背景", source)
        self.assertIn("最后一停：策略判断", source)
        self.assertIn('"热力图视角"', source)
        pos_bg = source.index("#### 比赛背景")
        pos_summary = source.index("最后一停：策略判断")
        pos_radio = source.index('"热力图视角"')
        self.assertLess(pos_bg, pos_summary, "summary must come after race background")
        self.assertLess(pos_summary, pos_radio, "summary must come before heatmap radio")

    def test_current_reconstruction_heading_marked_only_current(self) -> None:
        source = self._source()
        self.assertIn(
            "当前重建下表现最好的候选方案（仅当前重建下）",
            source,
        )

    def test_heatmap_guide_not_second_contradictory_conclusion(self) -> None:
        source = self._source()
        self.assertIn("不是第二个相互矛盾的结论", source)

    def test_boundary_not_actual_prediction_retained(self) -> None:
        source = self._source()
        # 摘要固定结论与热力图引导都保留“不是实际策略预测”的边界
        self.assertIn("不是实际策略预测", source)


try:
    from streamlit.testing.v1 import AppTest

    _HAS_APP_TEST = True
except Exception:  # pragma: no cover
    _HAS_APP_TEST = False


@unittest.skipUnless(_HAS_APP_TEST, "streamlit AppTest unavailable")
class ConclusionChineseAppTest(unittest.TestCase):
    def _collect_text(self, at) -> str:
        parts = []
        for el in getattr(at, "subheader", []):
            parts.append(str(getattr(el, "value", el)))
        for el in getattr(at, "info", []):
            parts.append(str(getattr(el, "value", el)))
        for el in getattr(at, "warning", []):
            parts.append(str(getattr(el, "value", el)))
        for el in getattr(at, "caption", []):
            parts.append(str(getattr(el, "value", el)))
        for el in getattr(at, "write", []):
            parts.append(str(getattr(el, "value", el)))
        for el in getattr(at, "markdown", []):
            parts.append(str(getattr(el, "value", el)))
        return "\n".join(parts)

    def test_page_renders_with_judgment_block(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        self.assertFalse(at.exception, f"page raised: {at.exception}")
        text = self._collect_text(at)
        self.assertIn("最后一停：策略判断", text)
        # 固定稳健性判断文案（warning 或 info 之一）出现
        self.assertTrue(
            ("没有替代方案都优于" in text) or ("个替代方案都优于" in text),
            "fixed robustness conclusion text missing from rendered page",
        )
        # 热力图引导
        self.assertIn("不是第二个相互矛盾的结论", text)
        # 当前重建结论被标为仅当前重建下
        self.assertIn("仅当前重建下", text)

    def test_three_scenario_table_rendered_with_three_rows(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=120)
        at.run()
        self.assertFalse(at.exception, f"page raised: {at.exception}")
        found = False
        for el in getattr(at, "dataframe", []):
            data = getattr(el, "value", None)
            if data is None:
                continue
            if "干燥设定" in getattr(data, "columns", []) and len(data) == 3:
                found = True
                break
        self.assertTrue(found, "three-scenario robust_best table (3 rows) not rendered")


if __name__ == "__main__":
    unittest.main(verbosity=2)
