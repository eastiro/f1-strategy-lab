from __future__ import annotations

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP = PROJECT_ROOT / "app" / "streamlit_app.py"


class CrossRaceValidationPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = STREAMLIT_APP.read_text(encoding="utf-8")

    def test_tab_label_is_audit_not_validation(self) -> None:
        self.assertIn('"公开信号审计"', self.source)
        self.assertNotIn('"跨比赛验证"', self.source)
        self.assertNotIn('"V2 跨比赛验证"', self.source)

    def test_reader_narrative_sections_are_present(self) -> None:
        # V4.3: sections reorganized into three expanders. All content preserved.
        required_sections = [
            "查看五赛季审计表与口径",
            "查看规则形成与失败诊断",
            "查看案例图、原始字段与研究文档",
            "模型在找什么",
            "最终发现",
            "规则形成过程中的两个诊断（不是最终五年结论）",
            "先通过一场比赛理解",
            "两个独立实验：2025 压力测试与 2021 弃权评估",
            "研究明细",
        ]
        for section in required_sections:
            with self.subTest(section=section):
                self.assertIn(section, self.source)

    def test_core_honest_finding_is_stated(self) -> None:
        self.assertIn("没有稳定泛化", self.source)
        self.assertIn("拒绝强行判断", self.source)
        self.assertIn("不是预测一个精确进站圈", self.source)

    def test_three_stages_are_not_merged_into_one_accuracy(self) -> None:
        self.assertIn("更不构成任何单一准确率", self.source)
        self.assertIn("不能与本五赛季 FastF1 表合并", self.source)
        self.assertNotIn("7/86", self.source)
        self.assertNotIn("模型可信度全景", self.source)

    def test_case_evidence_comes_before_aggregate_experiments(self) -> None:
        # V4.3: experiments are in exp_diag (expander 2), case evidence in
        # exp_case (expander 3). The three expanders appear in order 1-2-3.
        exp_case_index = self.source.index("查看案例图、原始字段与研究文档")
        exp_diag_index = self.source.index("查看规则形成与失败诊断")
        self.assertLess(
            exp_diag_index, exp_case_index,
            "experiment expander must appear before case-evidence expander"
        )
        # Inside exp_diag: 2025 pressure test must come before 2021 evaluation
        challenge_index = self.source.index("2025 简单规则压力测试")
        fastf1_index = self.source.index("2021 弃权状态机评估")
        self.assertLess(challenge_index, fastf1_index)

    def test_key_concepts_are_explained_in_reader_language(self) -> None:
        required_phrases = [
            "真实进站服务圈",
            "首个光头胎圈",
            "仅降雨代理",
            "保守公开信号点",
            "赛后使用完整多人圈速残差得到的重建",
            "前后 2 圈",
            "压力测试",
            "状态机",
            "弃权",
            "覆盖率",
            "未识别",
            "车手级",
            "为什么一个结果来自 2025，另一个来自 2021？",
        ]
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.source)

    def test_research_details_remain_available(self) -> None:
        required_expanders = [
            "五场探索案例表",
            "完整验证字段",
            "2025 事件类型诊断",
            "联合事件分布图（不是统一准确率）",
            "研究文档下载",
            "这些比赛为什么被选中",
        ]
        for label in required_expanders:
            with self.subTest(label=label):
                self.assertIn(label, self.source)

    def test_case_selector_and_downloads_keep_stable_keys(self) -> None:
        self.assertIn('key="v2_case_selector"', self.source)
        self.assertIn('key="download_v2_research_summary_zh"', self.source)
        self.assertIn('key="download_v2_research_summary_en"', self.source)
        self.assertIn('key="download_v2_holdout_diagnostic_2025"', self.source)


if __name__ == "__main__":
    unittest.main()
