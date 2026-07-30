"""V4.2 差距来源解释层：纯只读辅助（无 Streamlit 依赖）。

本模块只接收「已经生成」的 candidate / reference 两套逐圈 DataFrame，
以及两者的最后一停圈，输出分段表与中文摘要所需数据。

它**不**调用 simulate_strategy、不修改 strategy_model.py、不引入任何新参数，
也不另起一套重算路径。所有数字都是对既有 ``simulate_strategy(...)`` 结果的
模型内逐圈项相减。

关于 ``compound_delta`` / ``pit_delta`` 列
-----------------------------------------
``simulate_strategy(...)`` 的 ``laps`` DataFrame 当前**不直接存储**
``compound_delta`` 与 ``pit_delta`` 两列，但它存储了推导这两项所需的全部信息：

* ``compound`` 列 —— 与模型内部 ``compound_delta = COMPOUND_BASE_DELTA[compound]`` 完全一致；
* ``pit_lap`` 列（True 表示该圈进站）—— 与模型内部 ``pit_delta = pit_loss if pit else 0`` 完全一致。

因此本模块在输入缺少这两列时，用调用方传入的 ``compound_base_delta`` /
``pit_loss``（即模型自身使用的同一组常量）从 ``compound`` / ``pit_lap`` 精确重算，
得到与模型内部逐项相减完全等价的结果。**这不修改模型、不新增参数、不另起重算路径。**
若输入已自带 ``compound_delta`` / ``pit_delta`` 列，则优先直接使用它们。

分解项的边界（关键，避免重复计算）
-----------------------------------
本解释层只有**四项**可相加的模型内分解项：

* ``wet_penalty``（光头胎湿地惩罚差）
* ``degradation``（轮胎衰退差）
* ``compound_delta``（轮胎配方基础差）
* ``pit_delta``（进站损失差）

``simulated_lap_time`` 是**守恒核对的总数**（候选总用时 − 基准总用时），
不是第五个可相加分解项；``weather_delta`` **绝不**进入分段表或计算，
因为它已包含 ``wet_penalty``，重算会重复计入湿地惩罚。

``compound_delta`` / ``pit_delta`` **不是** 当前 ``simulate_strategy(...)``
laps 的输出字段（这是 Task 17 验收时确认的前提偏差）。缺失时一律以调用方
传入的、与模型**同源**的 ``COMPOUND_BASE_DELTA`` / ``DEFAULT_PIT_LOSS``
从已有 ``compound`` / ``pit_lap`` 精确推导；本模块不复制、不调参这些常量，
也不为“字段漂亮”向模型输出新增列。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# 模型内部 simulate_strategy 实际写出的列（用于对齐 / 校验）。
_LAP_ID_COLUMN = "lap_number"
# 必须存在的“基础”列；缺列时显式报错，绝不静默填零。
_REQUIRED_BASE_COLUMNS = [
    "lap_number",
    "compound",
    "pit_lap",
    "degradation",
    "wet_penalty",
    "simulated_lap_time",
]
# 可由 compound / pit_lap 推导的列。
_DERIVABLE_COLUMNS = ["compound_delta", "pit_delta"]

# 分解项（除总差距外）。顺序即报告列顺序。
_DECOMPOSITION_ITEMS = [
    ("wet_penalty", "光头胎湿地惩罚差"),
    ("degradation", "轮胎衰退差"),
    ("compound_delta", "轮胎配方基础差"),
    ("pit_delta", "进站损失差"),
]

GAP_TOLERANCE = 1e-9


class DeltaExplanationError(ValueError):
    """输入无法在对齐前提下完成分解时抛出（不静默填零）。"""


def _require_columns(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in _REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise DeltaExplanationError(
            f"{name} laps 缺少必需列：{missing}；无法在不静默填零的前提下完成分解。"
        )
    if df.empty:
        raise DeltaExplanationError(f"{name} laps 为空；无法完成分解。")


def _ensure_derived_columns(
    df: pd.DataFrame,
    compound_base_delta: dict[str, float],
    pit_loss: float,
) -> pd.DataFrame:
    """补齐 compound_delta / pit_delta。优先用输入已有列，否则从 compound / pit_lap 推导。"""
    out = df.copy()
    if "compound_delta" not in out.columns:
        if "compound" not in out.columns:
            raise DeltaExplanationError("缺少 compound_delta 且没有 compound 列可用于推导。")
        # .get(..., 0.0) 与模型内部 COMPOUND_BASE_DELTA.get(compound, 0.0) 一致。
        out["compound_delta"] = out["compound"].astype(str).str.upper().map(
            lambda c: float(compound_base_delta.get(c, 0.0))
        )
    if "pit_delta" not in out.columns:
        if "pit_lap" not in out.columns:
            raise DeltaExplanationError("缺少 pit_delta 且没有 pit_lap 列可用于推导。")
        out["pit_delta"] = out["pit_lap"].astype(float) * float(pit_loss)
    return out


def _align(candidate_laps: pd.DataFrame, reference_laps: pd.DataFrame) -> pd.DataFrame:
    """按 lap_number 对齐两套逐圈数据，返回带逐圈候选减基准差值的合并表。

    若圈集合不完全一致（缺圈 / 圈号不对齐），显式报错，绝不外连接填零。
    """
    req = candidate_laps[
        ["lap_number", "compound_delta", "degradation", "wet_penalty", "pit_delta", "simulated_lap_time"]
    ].rename(columns=lambda c: f"cand_{c}")
    ref = reference_laps[
        ["lap_number", "compound_delta", "degradation", "wet_penalty", "pit_delta", "simulated_lap_time"]
    ].rename(columns=lambda c: f"ref_{c}")
    merged = req.merge(
        ref,
        left_on="cand_lap_number",
        right_on="ref_lap_number",
        how="outer",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        mismatched = sorted(
            merged.loc[
                merged["_merge"] != "both",
                ["cand_lap_number", "ref_lap_number"],
            ]
            .bfill(axis=1)
            .iloc[:, 0]
            .dropna()
            .astype(int)
            .tolist()
        )
        raise DeltaExplanationError(
            "candidate 与 reference 的圈集合未对齐（圈数或圈号不一致），"
            f"无法在不静默填零的前提下相减；问题圈：{mismatched}"
        )
    merged = merged.sort_values("cand_lap_number").reset_index(drop=True)
    for col in ["compound_delta", "degradation", "wet_penalty", "pit_delta", "simulated_lap_time"]:
        merged[f"d_{col}"] = merged[f"cand_{col}"] - merged[f"ref_{col}"]
    return merged


def _phase_row(name: str, lap_start: int, lap_end: int, seg: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "lap_start": int(lap_start),
        "lap_end": int(lap_end),
        "total_gap": float(seg["d_simulated_lap_time"].sum()),
    }
    for col, _label in _DECOMPOSITION_ITEMS:
        row[col] = float(seg[f"d_{col}"].sum())
    return row


def build_delta_explanation(
    candidate_laps: pd.DataFrame,
    reference_laps: pd.DataFrame,
    candidate_pit: int,
    reference_pit: int,
    compound_base_delta: dict[str, float],
    pit_loss: float = 22.0,
    total_laps: int | None = None,
) -> dict[str, Any]:
    """计算「当前重建下最优候选」相对「Norris 真实基准」的模型内差距来源。

    参数
    ----
    candidate_laps, reference_laps
        已生成的逐圈 DataFrame（来自 simulate_strategy(...).laps）。
    candidate_pit, reference_pit
        候选 / 基准的最后一停圈（整数）。
    compound_base_delta, pit_loss
        与模型内部一致的常量；用于从 compound / pit_lap 推导 compound_delta / pit_delta。
    total_laps
        比赛总圈数；缺省时由数据最大圈号推断。

    返回
    ----
    dict，含：
        total_gap           候选相对基准的总差距（秒，候选减基准）
        candidate_pit / reference_pit
        phases              分段列表（每个非空阶段一行）
        dominant_phase_name 主导阶段名（差距接近零时为 None）
        gap_near_zero       bool
        decomposition_ok    每阶段「四项分解 == 该阶段 simulated_lap_time 总差」均成立
    """
    _require_columns(candidate_laps, "candidate")
    _require_columns(reference_laps, "reference")

    cand = _ensure_derived_columns(candidate_laps, compound_base_delta, pit_loss)
    ref = _ensure_derived_columns(reference_laps, compound_base_delta, pit_loss)

    merged = _align(cand, ref)

    if total_laps is None:
        total_laps = int(merged["cand_lap_number"].max())
    else:
        total_laps = int(total_laps)

    low = min(candidate_pit, reference_pit)
    high = max(candidate_pit, reference_pit)

    phases: list[dict[str, Any]] = []
    if candidate_pit != reference_pit:
        # 阶段一：换胎错位阶段（两套策略最后一停圈不同才显示）
        seg = merged[(merged["cand_lap_number"] >= low) & (merged["cand_lap_number"] <= high)]
        phases.append(_phase_row("换胎错位阶段", low, high, seg))
        # 阶段二：两者都已换回干胎后
        if high + 1 <= total_laps:
            seg = merged[(merged["cand_lap_number"] >= high + 1) & (merged["cand_lap_number"] <= total_laps)]
            phases.append(_phase_row("两者都已换回干胎后", high + 1, total_laps, seg))
    else:
        # 同圈换胎：不显示“换胎错位阶段”。
        pit_lap = int(candidate_pit)
        # 该圈四项模型内差值是否非零（湿地惩罚 / 衰退 / 配方基础 / 进站损失）。
        pit_seg = merged[merged["cand_lap_number"] == pit_lap]
        if not pit_seg.empty:
            four_delta_abs = sum(
                abs(float(pit_seg[f"d_{col}"].iloc[0])) for col, _ in _DECOMPOSITION_ITEMS
            )
            if four_delta_abs > GAP_TOLERANCE:
                # 同圈不同胎（或当圈存在任一模型内差异）→ 必须显式显示当圈，
                # 否则该圈的差距会被排除在所有阶段之外，破坏“阶段和 = 总差距”。
                phases.append(_phase_row("同圈换胎当圈", pit_lap, pit_lap, pit_seg))
        # 两者都已换回干胎后（同圈换胎也从 pit_lap + 1 起算）
        if pit_lap + 1 <= total_laps:
            seg = merged[(merged["cand_lap_number"] >= pit_lap + 1) & (merged["cand_lap_number"] <= total_laps)]
            phases.append(_phase_row("两者都已换回干胎后", pit_lap + 1, total_laps, seg))

    # 严格不变量：每个阶段「四项分解之和」必须等于该阶段 simulated_lap_time 总差。
    decomposition_ok = True
    for ph in phases:
        component_sum = sum(ph[col] for col, _ in _DECOMPOSITION_ITEMS)
        if abs(component_sum - ph["total_gap"]) > GAP_TOLERANCE:
            decomposition_ok = False
            raise DeltaExplanationError(
                f"阶段「{ph['name']}」四项分解之和与 simulated_lap_time 总差距不一致："
                f"{component_sum:.6f} vs {ph['total_gap']:.6f}；检查输入列或推导常量。"
            )

    # 总差距：所有圈逐圈 simulated_lap_time 之差（= 候选总用时 - 基准总用时）。
    total_gap = float(merged["d_simulated_lap_time"].sum())

    # 守恒不变量：所有显示阶段之和必须等于总差距。
    # 绝不把“未显示阶段的差距”悄悄留在标题总数里。
    phase_sum = float(sum(ph["total_gap"] for ph in phases))
    if abs(phase_sum - total_gap) > GAP_TOLERANCE:
        raise DeltaExplanationError(
            "显示阶段之和（"
            f"{phase_sum:.6f}）与总差距（{total_gap:.6f}）不相等；"
            "存在未被任何阶段覆盖的非零差距圈，拒绝静默发布。"
        )

    gap_near_zero = abs(total_gap) <= GAP_TOLERANCE
    dominant_phase_name: str | None = None
    if not gap_near_zero and phases:
        dominant_phase_name = max(phases, key=lambda p: abs(p["total_gap"]))["name"]

    return {
        "total_gap": total_gap,
        "candidate_pit": int(candidate_pit),
        "reference_pit": int(reference_pit),
        "phases": phases,
        "dominant_phase_name": dominant_phase_name,
        "gap_near_zero": gap_near_zero,
        "decomposition_ok": decomposition_ok,
    }
