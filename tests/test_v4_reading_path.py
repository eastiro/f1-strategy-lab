"""Task 16: V4 first-read path fixes on the Chinese page.

Verifies the four reading-path fixes (A/B/C/D) without touching any model,
formula, data, statistic, or the English site. These are pure text /
information-architecture clarifications.
"""

from __future__ import annotations

import unittest
from pathlib import Path

APP_PATH = "app/streamlit_app.py"

try:
    import pandas as pd
    from streamlit.testing.v1 import AppTest

    _HAS_APP_TEST = True
except Exception:  # pragma: no cover - streamlit/pandas missing in some envs
    _HAS_APP_TEST = False


class V4ReadingPathSourceTest(unittest.TestCase):
    def _source(self) -> str:
        with open(APP_PATH, encoding="utf-8") as fh:
            return fh.read()

    # --- A. five-season immediate definitions -----------------------------------
    def test_A_five_season_definitions(self) -> None:
        source = self._source()
        self.assertIn("是车手级湿转干转场记录，不是比赛场次，也不是准确率", source)
        self.assertIn("弃权 = 规则在条件复杂时拒绝输出车手级信号", source)
        self.assertIn("不是车手退赛", source)

    # --- B. V4.1 bridging text ------------------------------------------------
    def test_B_v41_bridge_text(self) -> None:
        source = self._source()
        self.assertIn("最接近稳健", source)
        self.assertIn("当前重建下最好", source)
        self.assertIn("不等于一定快过", source)
        self.assertIn("真实最优", source)

    # --- C. removed duplicate three-scenario dataframe, kept chart ------------
    def test_C_duplicate_dataframe_removed(self) -> None:
        source = self._source()
        self.assertNotIn("scenario_display = selected_scenario_results", source)
        self.assertNotIn('"干燥速度设定"', source)
        # the heatmap exploration chart is kept
        self.assertIn("plot_scenario_breakdown(selected_scenario_results)", source)
        # the V4.1 summary helper is still the single source of the three-scenario table
        self.assertIn("build_strategy_judgment(", source)

    # --- D. 2021 detail cards collapsed into an expander ----------------------
    def test_D_2021_details_in_collapsed_expander(self) -> None:
        source = self._source()
        detail_idx = source.index('fastf1_left.metric("车手级事件"')
        exp_idx = source.index("detail_expander = exp_diag.expander(", 0, detail_idx)
        self.assertLess(
            exp_idx, detail_idx, "2021 detail cards must live inside the collapsed expander"
        )
        self.assertIn(
            "查看 2021 初始状态机扫描细节（与上方五赛季表中的 2021 是同一批事件）", source
        )
        self.assertIn("不是新的五赛季结果，也不构成第二套准确率", source)
        # collapsed by default
        self.assertIn("expanded=False", source)
        detail_block = source[exp_idx : source.index("# 复核路径 3", detail_idx)]
        # The expander is an explicit container. Its children must be written
        # through that container, otherwise Streamlit renders an empty bar.
        self.assertIn("detail_expander.columns(3)", detail_block)
        self.assertNotIn("research_tab.columns(3)", detail_block)
        self.assertNotIn("exp_diag.columns(3)", detail_block)
        self.assertNotIn("research_tab.write(", detail_block)
        self.assertNotIn("research_tab.caption(", detail_block)
        self.assertIn("detail_expander.write(", detail_block)
        self.assertIn("detail_expander.caption(", detail_block)

    # --- V4.2. 差距来源解释块（当前赛后重建） ---------------------------------
    def test_V42_position_after_bridge_before_heatmap(self) -> None:
        source = self._source()
        bridge_idx = source.index("两者都是候选方案之间的比较，不等于一定快过")
        v42_idx = source.index('conclusion_tab.subheader("这点差距从哪里来？（当前赛后重建）")')
        radio_idx = source.index("heatmap_view_label = conclusion_tab.radio(")
        self.assertLess(bridge_idx, v42_idx, "V4.2 block must come after the V4.1 caliber bridge")
        self.assertLess(v42_idx, radio_idx, "V4.2 block must come before the heatmap radio")

    def test_V42_boundary_text_present(self) -> None:
        source = self._source()
        self.assertIn("本区块解释的是「当前重建下表现最好的候选方案」", source)
        self.assertIn("不等同于上方按三种干燥设定排序的「最接近稳健」方案", source)
        self.assertIn("不代表真实比赛因果", source)
        self.assertIn("这些是简化模拟公式的分项相减，帮助复核本模型为什么给出这个差距", source)
        self.assertIn("不解释 McLaren 当时为什么做出策略选择", source)

    def test_V42_expander_contains_table_and_boundary(self) -> None:
        source = self._source()
        exp_idx = source.index(
            'detail_expander = conclusion_tab.expander("查看模型内分段明细", expanded=False)'
        )
        radio_idx = source.index("heatmap_view_label = conclusion_tab.radio(")
        block = source[exp_idx:radio_idx]
        # 表格与边界说明必须写在 expander 容器自身上（detail_expander.xxx），
        # 不能写回 conclusion_tab.xxx 跑到外层形成空框（Task 16 同款 bug）。
        self.assertIn("detail_expander.dataframe(pd.DataFrame(_rows)", block)
        self.assertIn("detail_expander.caption(", block)
        self.assertNotIn("conclusion_tab.dataframe(pd.DataFrame(_rows)", block)
        self.assertIn("这些是简化模拟公式的分项相减，帮助复核本模型为什么给出这个差距", block)
        # 默认收起。
        self.assertIn("expanded=False", block)
        self.assertNotIn("expanded=True", block)


@unittest.skipUnless(_HAS_APP_TEST, "streamlit AppTest unavailable")
class V4ReadingPathAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.at = AppTest.from_file(APP_PATH, default_timeout=120)
        self.at.run()
        self.assertFalse(self.at.exception, f"page raised: {self.at.exception}")

    def _all_text(self) -> str:
        parts = []
        for name in ("markdown", "text", "caption", "info", "warning", "success", "subheader"):
            for el in getattr(self.at, name, []):
                v = getattr(el, "value", None)
                if isinstance(v, str):
                    parts.append(v)
        return "\n".join(parts)

    def test_A_visible_definitions(self) -> None:
        text = self._all_text()
        self.assertIn("是车手级湿转干转场记录，不是比赛场次，也不是准确率", text)
        self.assertIn("弃权 = 规则在条件复杂时拒绝输出车手级信号", text)
        self.assertIn("不是车手退赛", text)

    def test_B_visible_bridge(self) -> None:
        text = self._all_text()
        self.assertIn("最接近稳健", text)
        self.assertIn("当前重建下最好", text)
        self.assertIn("不等于一定快过", text)
        self.assertIn("真实最优", text)

    def test_C_only_one_three_scenario_dataframe(self) -> None:
        count = 0
        has_duplicate = False
        for el in getattr(self.at, "dataframe", []):
            val = getattr(el, "value", None)
            if not isinstance(val, pd.DataFrame):
                continue
            if "干燥设定" in val.columns and len(val) == 3:
                count += 1
            if "干燥速度设定" in val.columns:
                has_duplicate = True
        self.assertEqual(count, 1, "exactly one 3-row three-scenario dataframe (V4.1 summary)")
        self.assertFalse(has_duplicate, "heatmap duplicate scenario dataframe must be removed")

    def test_D_collapsed_expander_present(self) -> None:
        found = None
        for el in getattr(self.at, "expander", []):
            label = getattr(el, "label", "")
            if "查看 2021 初始状态机扫描细节" in label:
                found = el
                break
        self.assertIsNotNone(found, "2021 detail expander must exist and render")
        self.assertEqual(
            len(found.metric),
            3,
            "2021 metric cards must be children of the expander, not siblings below it",
        )

    def test_V42_renders_with_boundaries(self) -> None:
        text = self._all_text()
        self.assertIn("这点差距从哪里来？（当前赛后重建）", text)
        self.assertIn("当前赛后重建", text)
        self.assertIn("模型内", text)
        self.assertIn("不代表真实比赛因果", text)
        self.assertIn("不解释 McLaren 当时为什么做出策略选择", text)
        # V4.2 expander renders
        found = None
        for el in getattr(self.at, "expander", []):
            if "分段明细" in str(getattr(el, "label", "")):
                found = el
                break
        self.assertIsNotNone(found, "V4.2 '查看模型内分段明细' expander must render")
        # V4.2 phase breakdown table renders on the page (阶段 column, 2 rows).
        has_phase_table = any(
            isinstance(getattr(el, "value", None), pd.DataFrame)
            and "阶段" in getattr(getattr(el, "value", None), "columns", [])
            for el in getattr(self.at, "dataframe", [])
        )
        self.assertTrue(has_phase_table, "V4.2 model-internal phase breakdown table must render")

    def test_V42_expander_owns_table_and_caption_children(self) -> None:
        # Task 18 验收缺口 2：expander 必须真正拥有 phase 表与边界 caption 子元素，
        # 而不是把它们写成外层 tab 的 sibling（否则重演空白折叠框）。
        found = None
        for el in getattr(self.at, "expander", []):
            if "分段明细" in str(getattr(el, "label", "")):
                found = el
                break
        self.assertIsNotNone(found, "V4.2 '查看模型内分段明细' expander must render")
        self.assertFalse(
            getattr(found, "proto", None) and getattr(found.proto, "expanded", True),
            "V4.2 expander must be collapsed by default (expanded=False)",
        )
        # 子元素：phase 明细表 + 边界 caption 都在 expander 内部。
        self.assertTrue(
            len(found.dataframe) >= 1,
            "V4.2 phase breakdown dataframe must be a child of the expander",
        )
        self.assertTrue(
            len(found.caption) >= 1,
            "V4.2 boundary caption must be a child of the expander",
        )
        phase_df = None
        for df_el in found.dataframe:
            val = getattr(df_el, "value", None)
            if isinstance(val, pd.DataFrame) and "阶段" in val.columns:
                phase_df = val
                break
        self.assertIsNotNone(phase_df, "expander child dataframe must carry the 阶段 column")
        caption_text = "\n".join(
            getattr(c, "value", "") for c in found.caption
        )
        self.assertIn("简化模拟公式的分项相减", caption_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
