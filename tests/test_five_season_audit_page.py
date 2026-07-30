"""Task 13: presentation of the 2021-2025 five-season audit on the Chinese page.

Covers the read-only display helper and the Chinese ``公开信号审计`` page behavior:
the final five-season conclusion, the per-season table read from the Task 12
product, the demoted 2025/2021 diagnostics, and graceful degradation when the
product is missing.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.five_season_display import (
    FiveSeasonDataMissing,
    load_five_season_audit,
)

APP_PATH = "app/streamlit_app.py"

try:
    from streamlit.testing.v1 import AppTest

    _HAS_APP_TEST = True
except Exception:  # pragma: no cover - streamlit missing in some envs
    _HAS_APP_TEST = False


class FiveSeasonHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = load_five_season_audit()

    def test_five_seasons_present_once_each(self) -> None:
        seasons = [r["season"] for r in self.data["seasons"]]
        self.assertEqual(seasons, [2021, 2022, 2023, 2024, 2025])

    def test_totals_are_coverage_not_accuracy(self) -> None:
        s = self.data["summary"]
        self.assertEqual(s["events"], 173)
        self.assertEqual(s["signals"], 27)
        self.assertEqual(s["abstentions"], 146)
        # coverage, not a 27/173 accuracy figure
        self.assertNotEqual(s["signals"], s["events"])

    def test_zero_signal_seasons_show_na_not_zero(self) -> None:
        by_year = {r["season"]: r for r in self.data["seasons"]}
        for yr in (2022, 2024, 2025):
            self.assertIsNone(by_year[yr]["mae"])
            self.assertEqual(by_year[yr]["mae_label"], "N/A（该季未发出信号）")
        # issued-signal seasons still report a numeric MAE (never 0.0)
        self.assertEqual(by_year[2021]["mae_label"], "4.69 圈")
        self.assertEqual(by_year[2023]["mae_label"], "0.43 圈")

    def test_research_position_labels(self) -> None:
        by_year = {r["season"]: r for r in self.data["seasons"]}
        self.assertEqual(by_year[2021]["research_position"], "既有审计赛季")
        self.assertEqual(by_year[2022]["research_position"], "新增预注册扩展")
        self.assertEqual(by_year[2023]["research_position"], "规则形成期已接触")
        self.assertEqual(by_year[2025]["research_position"], "规则形成期已接触")
        # 2022 means something different from 2023-2025
        self.assertNotEqual(
            by_year[2022]["research_position"], by_year[2023]["research_position"]
        )

    def test_anchor_counts_match_protocol(self) -> None:
        s = self.data["summary"]
        self.assertEqual(s["pit_service"], 153)
        self.assertEqual(s["first_slick_fallback"], 20)

    def test_missing_product_raises_gracefully(self) -> None:
        with self.assertRaises(FiveSeasonDataMissing):
            load_five_season_audit(product_path="nonexistent_path.csv")


class FiveSeasonChinesePageTest(unittest.TestCase):
    def _source(self) -> str:
        with open(APP_PATH, encoding="utf-8") as fh:
            return fh.read()

    def test_five_season_block_before_diagnostics(self) -> None:
        source = self._source()
        # V4.3: five-season audit table is in exp_audit (expander 1),
        # diagnostics in exp_diag (expander 2). Both are still present.
        self.assertIn("查看五赛季审计表与口径", source)
        self.assertIn("规则形成过程中的两个诊断（不是最终五年结论）", source)
        pos_block = source.index("查看五赛季审计表与口径")
        pos_diag = source.index("规则形成过程中的两个诊断（不是最终五年结论）")
        self.assertLess(
            pos_block, pos_diag, "audit expander must precede diagnostics expander in source"
        )
        # graceful degradation only: no st.stop inside the audit block
        self.assertNotIn("st.stop()", source[pos_block : pos_block + 1000])

    def test_source_keeps_boundaries(self) -> None:
        source = self._source()
        self.assertIn("不能与本五赛季 FastF1 表合并", source)
        self.assertIn("不是实时策略建议", source)
        self.assertIn("也不验证 Silverstone 模拟器的物理模型", source)

    def test_diagnostics_demoted_not_merged(self) -> None:
        source = self._source()
        self.assertIn("更不构成任何单一准确率", source)
        self.assertIn("1/53", source)
        self.assertIn("允许弃权", source)


@unittest.skipUnless(_HAS_APP_TEST, "streamlit AppTest unavailable")
class FiveSeasonChineseAppTest(unittest.TestCase):
    def test_page_renders_with_summary_metrics(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=90)
        at.run()
        self.assertFalse(at.exception, f"page raised: {at.exception}")
        metric_by_label = {m.label: str(m.value) for m in at.metric}
        self.assertEqual(metric_by_label.get("湿转干事件"), "173")
        self.assertEqual(metric_by_label.get("发出公开信号"), "27")
        self.assertEqual(metric_by_label.get("明确弃权"), "146")

    def test_page_renders_without_stopping_on_audit_tab(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=90)
        at.run()
        self.assertFalse(at.exception, f"page raised: {at.exception}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
