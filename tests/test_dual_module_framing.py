"""Focused tests for the dual-module framing (Task 08).

These tests check that the agreed two-module framing (Module 1: Silverstone
2024 simulator; Module 2: public-signal reliability audit) is stated to a
first-time visitor, that the audit is honestly relabeled (not a model / not a
cross-race validator), and that the 1/53 negative result is interpreted plainly
without being spun as success or used to invalidate the separate simulator.

They avoid re-implementing any model logic.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHINESE_APP = REPO_ROOT / "app" / "streamlit_app.py"
ENGLISH_APP = REPO_ROOT / "app" / "portfolio_app.py"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _every_occurrence_has_prefix(text: str, target: str, prefixes: tuple[str, ...]) -> bool:
    """Return True if every ``target`` in ``text`` is immediately preceded by
    one of ``prefixes`` (no standalone / opposite-polarity occurrence)."""
    start = 0
    while True:
        idx = text.find(target, start)
        if idx == -1:
            return True
        before = text[max(0, idx - max(len(p) for p in prefixes)) : idx]
        if not any(before.endswith(p) for p in prefixes):
            return False
        start = idx + len(target)


try:
    from streamlit.testing.v1 import AppTest

    _HAS_APP_TEST = True
except Exception:  # pragma: no cover - streamlit missing in some envs
    _HAS_APP_TEST = False


class ChineseFramingSourceTest(unittest.TestCase):
    def setUp(self):
        self.src = _read(CHINESE_APP)

    def test_two_module_framing_present(self):
        for phrase in [
            "两块",
            "独立",
            "公开信号审计",
            "不验证",
            "通用实时决策系统",
        ]:
            self.assertIn(phrase, self.src, f"missing framing phrase: {phrase}")

    def test_no_false_validation_claim(self):
        # The audit must never be presented as validating the simulator's physics.
        self.assertIn(
            "审计不验证模拟器的物理模型",
            self.src,
            "honest negation of 'audit validates simulator physics' missing",
        )
        # '通用实时决策系统' must only ever appear in a negative context
        # ('不是' = is not; '包装成' = prevent packaging it *as* one).
        self.assertTrue(
            _every_occurrence_has_prefix(self.src, "通用实时决策系统", ("不是", "包装成")),
            "'通用实时决策系统' appears in a positive / unguarded context",
        )

    def test_audit_tab_relabeled(self):
        self.assertIn("公开信号审计", self.src)
        self.assertNotIn("跨比赛验证", self.src, "old tab label '跨比赛验证' still present")
        self.assertNotIn("模型决策链", self.src, "old '模型决策链' label still present")

    def test_audit_rule_and_record_labels(self):
        self.assertIn("公开信号审计规则", self.src)
        self.assertIn("本案例的信号审计记录", self.src)

    def test_1_53_interpretation_explicit(self):
        for phrase in ["证伪", "1/53", "没有证伪", "V1 单场假设模拟器", "没有证明"]:
            self.assertIn(phrase, self.src, f"1/53 interpretation missing phrase: {phrase}")


class EnglishFramingSourceTest(unittest.TestCase):
    def setUp(self):
        self.src = _read(ENGLISH_APP)

    def test_two_module_framing_present(self):
        for phrase in [
            "two independent modules",
            "public-signal reliability audit",
            "does not validate",
            "general live decision system",
        ]:
            self.assertIn(phrase, self.src, f"missing English framing phrase: {phrase}")

    def test_no_false_validation_claim(self):
        # 'general live decision system' must only appear after a negation.
        self.assertTrue(
            _every_occurrence_has_prefix(
                self.src, "general live decision system", ("not a ", "neither module is a ")
            ),
            "'general live decision system' appears without negation",
        )
        self.assertIn(
            "does not validate the simulator's physics",
            self.src,
            "honest negation of 'audit validates simulator' missing",
        )

    def test_opening_not_trained_live_model(self):
        # The opening deck question no longer implies a trained/live model.
        self.assertNotIn(
            "where a model should remain silent",
            self.src,
            "opening still implies a trained/live model",
        )


@unittest.skipUnless(_HAS_APP_TEST, "streamlit AppTest unavailable")
class FramingRenderTest(unittest.TestCase):
    def test_english_page_renders(self):
        at = AppTest.from_file(ENGLISH_APP, default_timeout=60)
        at.run()
        self.assertFalse(at.exception, f"English page raised: {at.exception}")

    def test_chinese_page_renders(self):
        at = AppTest.from_file(CHINESE_APP, default_timeout=60)
        at.run()
        self.assertFalse(at.exception, f"Chinese page raised: {at.exception}")

    def test_audit_selector_works_under_renamed_tab(self):
        at = AppTest.from_file(CHINESE_APP, default_timeout=60)
        at.run()
        self.assertFalse(at.exception, f"Chinese page raised before select: {at.exception}")
        selector = next(s for s in at.selectbox if s.key == "v2_case_selector")
        selector.set_value("montreal_2024").run()
        self.assertFalse(
            at.exception,
            f"audit selector raised on montreal_2024: {at.exception}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
