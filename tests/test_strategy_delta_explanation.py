"""Task 17 纯辅助模块测试：strategy_delta_explanation.build_delta_explanation。

覆盖验收要求 1：
- 提前换 / 晚换 / 同圈换 三种边界；
- 分段边界正确；
- 五项分解与 simulated_lap_time 总差距精确相等（1e-9）；
- 不会双算 weather_delta；
- 输入缺列或圈数不对齐时给出明确错误，不静默填零。
"""

import unittest

import pandas as pd

from src.strategy_delta_explanation import (
    DeltaExplanationError,
    build_delta_explanation,
)

COMPOUND_BASE_DELTA = {"SOFT": -0.3, "MEDIUM": 0.0, "HARD": 0.25, "INTERMEDIATE": 4.5}
PIT_LOSS = 22.0


def _make_laps(
    pit: int,
    final_compound: str,
    total_laps: int,
    inter_pit: int = 10,
    *,
    degradation_fn=None,
    wet_penalty_fn=None,
    bogus_weather_delta=False,
):
    """构造与 simulate_strategy 同构的逐圈 DataFrame。

    仅保留模型实际写出的列（lap_number / compound / pit_lap / degradation /
    wet_penalty / simulated_lap_time），由 helper 自行从 compound / pit_lap 推导
    compound_delta / pit_delta。simulated_lap_time 设为
    compound_delta + degradation + pit_delta + wet_penalty（baseline/fuel/adjustments 归零），
    因此候选减基准的差必然等于四项分解之和。
    """
    def _default_degradation(lap: int, compound: str) -> float:
        return 0.01 * lap * (1 if compound in ("SOFT", "MEDIUM") else 2)

    def _default_wet_penalty(lap: int, compound: str) -> float:
        return 0.0

    if degradation_fn is None:
        degradation_fn = _default_degradation
    if wet_penalty_fn is None:
        wet_penalty_fn = _default_wet_penalty

    rows = []
    for lap in range(1, total_laps + 1):
        # 简化结构：进站前统一 MEDIUM，进站后切换为最终配方。
        # 足以让早期圈（min(cand_pit, ref_pit) 之前）在 candidate / reference 间完全一致，
        # 从而干净地隔离出「换胎错位」与「两者都已换回干胎后」两个阶段。
        compound = "MEDIUM" if lap < pit else final_compound
        pit_lap = lap == pit
        compound_delta = COMPOUND_BASE_DELTA[compound]
        pit_delta = PIT_LOSS if pit_lap else 0.0
        degradation = degradation_fn(lap, compound)
        wet_penalty = wet_penalty_fn(lap, compound)
        simulated = compound_delta + degradation + pit_delta + wet_penalty
        row = {
            "lap_number": lap,
            "compound": compound,
            "pit_lap": pit_lap,
            "degradation": degradation,
            "wet_penalty": wet_penalty,
            "simulated_lap_time": simulated,
        }
        if bogus_weather_delta:
            # 故意给出一个与 wet_penalty 不一致的 weather_delta，验证它不会被重复计入。
            row["weather_delta"] = wet_penalty + 99.0
        rows.append(row)
    return pd.DataFrame(rows)


class TestSegmentBoundaries(unittest.TestCase):
    def test_early_pit_builds_mismatch_and_slicks_phases(self):
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52)
        out = build_delta_explanation(
            cand, ref, candidate_pit=20, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        names = [p["name"] for p in out["phases"]]
        self.assertEqual(names, ["换胎错位阶段", "两者都已换回干胎后"])
        self.assertEqual(out["phases"][0]["lap_start"], 20)
        self.assertEqual(out["phases"][0]["lap_end"], 30)
        self.assertEqual(out["phases"][1]["lap_start"], 31)
        self.assertEqual(out["phases"][1]["lap_end"], 52)
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)

    def test_late_pit_builds_mismatch_and_slicks_phases(self):
        cand = _make_laps(pit=40, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="SOFT", total_laps=52)
        out = build_delta_explanation(
            cand, ref, candidate_pit=40, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        names = [p["name"] for p in out["phases"]]
        self.assertEqual(names, ["换胎错位阶段", "两者都已换回干胎后"])
        self.assertEqual(out["phases"][0]["lap_start"], 30)
        self.assertEqual(out["phases"][0]["lap_end"], 40)
        self.assertEqual(out["phases"][1]["lap_start"], 41)
        self.assertEqual(out["phases"][1]["lap_end"], 52)
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)

    def test_same_pit_hides_mismatch_phase(self):
        # 同圈换且最终配方相同 -> 两套策略完全一致，差距接近零，只显示“两者都已换回干胎后”。
        cand = _make_laps(pit=30, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="SOFT", total_laps=52)
        out = build_delta_explanation(
            cand, ref, candidate_pit=30, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        names = [p["name"] for p in out["phases"]]
        self.assertEqual(names, ["两者都已换回干胎后"])
        self.assertTrue(out["gap_near_zero"])
        self.assertIsNone(out["dominant_phase_name"])
        self.assertAlmostEqual(out["total_gap"], 0.0, delta=1e-9)
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)

    def test_same_pit_different_compound_shows_pit_lap_phase_and_conserves(self):
        # 边界：同圈换但配方不同。规范不显示“换胎错位阶段”，但必须新增“同圈换胎当圈”
        # 以覆盖换胎当圈的差异，否则显示阶段之和 != 总差距（Task 18 验收缺口 1）。
        cand = _make_laps(pit=30, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52)
        out = build_delta_explanation(
            cand, ref, candidate_pit=30, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        names = [p["name"] for p in out["phases"]]
        self.assertEqual(names, ["同圈换胎当圈", "两者都已换回干胎后"])
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)
        self.assertTrue(out["decomposition_ok"])

    def test_same_pit_same_compound_omits_pit_lap_phase_and_total_gap_zero(self):
        # 同圈同胎：当圈四项差值全为零，省略“同圈换胎当圈”，守恒仍成立（总差距为 0）。
        cand = _make_laps(pit=30, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="SOFT", total_laps=52)
        out = build_delta_explanation(
            cand, ref, candidate_pit=30, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        names = [p["name"] for p in out["phases"]]
        self.assertEqual(names, ["两者都已换回干胎后"])
        self.assertAlmostEqual(out["total_gap"], 0.0, delta=1e-9)
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)


class TestDecompositionEqualsTotalGap(unittest.TestCase):
    def test_five_item_breakdown_matches_simulated_lap_time_gap(self):
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52)
        out = build_delta_explanation(
            cand, ref, candidate_pit=20, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        for phase in out["phases"]:
            component_sum = (
                phase["wet_penalty"]
                + phase["degradation"]
                + phase["compound_delta"]
                + phase["pit_delta"]
            )
            self.assertAlmostEqual(component_sum, phase["total_gap"], delta=1e-9)
        self.assertTrue(out["decomposition_ok"])

    def test_phase_sums_equal_overall_total_gap_when_pits_differ(self):
        # 提前换：阶段覆盖 [min, total]，早期圈差距为零，故阶段和 == 总差距。
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52)
        out = build_delta_explanation(
            cand, ref, candidate_pit=20, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)


class TestNoWeatherDeltaDoubleCount(unittest.TestCase):
    def test_bogus_weather_delta_is_ignored(self):
        # 故意在输入里放一个与 wet_penalty 不一致的 weather_delta 列；
        # helper 从不读取 weather_delta，分解仍精确等于 simulated_lap_time 差。
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=52, bogus_weather_delta=True)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52, bogus_weather_delta=True)
        out = build_delta_explanation(
            cand, ref, candidate_pit=20, reference_pit=30,
            compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
        )
        for phase in out["phases"]:
            component_sum = (
                phase["wet_penalty"]
                + phase["degradation"]
                + phase["compound_delta"]
                + phase["pit_delta"]
            )
            self.assertAlmostEqual(component_sum, phase["total_gap"], delta=1e-9)
        self.assertTrue(out["decomposition_ok"])


class TestInputValidation(unittest.TestCase):
    def test_missing_column_raises(self):
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52)
        bad = cand.drop(columns=["simulated_lap_time"])
        with self.assertRaises(DeltaExplanationError):
            build_delta_explanation(
                bad, ref, candidate_pit=20, reference_pit=30,
                compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
            )

    def test_missing_reference_column_raises(self):
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52).drop(columns=["degradation"])
        with self.assertRaises(DeltaExplanationError):
            build_delta_explanation(
                cand, ref, candidate_pit=20, reference_pit=30,
                compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
            )

    def test_misaligned_lap_counts_raise(self):
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=51)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52)
        with self.assertRaises(DeltaExplanationError):
            build_delta_explanation(
                cand, ref, candidate_pit=20, reference_pit=30,
                compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
            )

    def test_misaligned_lap_numbers_raise(self):
        cand = _make_laps(pit=20, final_compound="SOFT", total_laps=52)
        ref = _make_laps(pit=30, final_compound="MEDIUM", total_laps=52)
        ref_missing_first = ref[ref["lap_number"] != 1].reset_index(drop=True)
        with self.assertRaises(DeltaExplanationError):
            build_delta_explanation(
                cand, ref_missing_first, candidate_pit=20, reference_pit=30,
                compound_base_delta=COMPOUND_BASE_DELTA, pit_loss=PIT_LOSS,
            )


class TestRealModelRegression(unittest.TestCase):
    """C.2：真实 simulate_strategy 输出回归，证明映射与模型总时间差完全一致。

    不修改 strategy_model.py、不新增列、不复制/调参模型常量；
    compound_base_delta 直接取自模型同名常量，pit_loss 用模型默认值 22.0。
    """

    @staticmethod
    def _run(cand_pit, cand_compound, ref_pit, ref_compound, total_laps=52):
        from src.strategy_model import (
            COMPOUND_BASE_DELTA,
            StintPlan,
            simulate_strategy,
        )

        ref = simulate_strategy(
            total_laps=total_laps,
            baseline_lap_time=90.0,
            stints=[
                StintPlan("MEDIUM", 1, ref_pit - 1),
                StintPlan(ref_compound, ref_pit, total_laps),
            ],
        )
        cand = simulate_strategy(
            total_laps=total_laps,
            baseline_lap_time=90.0,
            stints=[
                StintPlan("MEDIUM", 1, cand_pit - 1),
                StintPlan(cand_compound, cand_pit, total_laps),
            ],
        )
        # 校验：模型 laps 确实不输出 compound_delta / pit_delta 列（仅 compound / pit_lap）。
        for laps in (ref.laps, cand.laps):
            assert "compound_delta" not in laps.columns
            assert "pit_delta" not in laps.columns
        out = build_delta_explanation(
            candidate_laps=cand.laps,
            reference_laps=ref.laps,
            candidate_pit=cand_pit,
            reference_pit=ref_pit,
            compound_base_delta=COMPOUND_BASE_DELTA,
            pit_loss=22.0,
            total_laps=total_laps,
        )
        return ref, cand, out

    def test_total_gap_equals_model_total_time_difference(self):
        ref, cand, out = self._run(
            cand_pit=20, cand_compound="SOFT", ref_pit=30, ref_compound="MEDIUM"
        )
        self.assertAlmostEqual(
            out["total_gap"], cand.total_time - ref.total_time, delta=1e-9
        )

    def test_phase_sums_equal_total_gap_and_each_phase_decomposes(self):
        ref, cand, out = self._run(
            cand_pit=20, cand_compound="SOFT", ref_pit=30, ref_compound="MEDIUM"
        )
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)
        for phase in out["phases"]:
            component_sum = (
                phase["wet_penalty"]
                + phase["degradation"]
                + phase["compound_delta"]
                + phase["pit_delta"]
            )
            self.assertAlmostEqual(component_sum, phase["total_gap"], delta=1e-9)
        self.assertTrue(out["decomposition_ok"])

    def test_same_pit_different_compound_real_model_conserves(self):
        # 真实模型下“同圈换胎但配方不同”也必须守恒（覆盖 Task 18 缺口 1）。
        ref, cand, out = self._run(
            cand_pit=30, cand_compound="SOFT", ref_pit=30, ref_compound="MEDIUM"
        )
        phase_sum = sum(p["total_gap"] for p in out["phases"])
        self.assertAlmostEqual(phase_sum, out["total_gap"], delta=1e-9)
        # 真实模型下 compound_delta 来自同源常量，实打实地非零。
        self.assertGreater(abs(out["total_gap"]), 1e-6)
        names = [p["name"] for p in out["phases"]]
        self.assertEqual(names, ["同圈换胎当圈", "两者都已换回干胎后"])
        for phase in out["phases"]:
            component_sum = (
                phase["wet_penalty"]
                + phase["degradation"]
                + phase["compound_delta"]
                + phase["pit_delta"]
            )
            self.assertAlmostEqual(component_sum, phase["total_gap"], delta=1e-9)


if __name__ == "__main__":
    unittest.main()
