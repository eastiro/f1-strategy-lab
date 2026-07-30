"""Pure presentation helper: derive the model decision-chain trace from a
validation report row.

This module intentionally contains **no model logic and no new formulas**.
It only reads fields already produced by ``src.v2_validation`` so the rendered
"model trace" stays fully testable without Streamlit and can never drift from
the underlying data.
"""

from __future__ import annotations

import pandas as pd


def _opt_int(value: object) -> int | None:
    """Return an int or None for a possibly-missing numeric cell."""
    return None if pd.isna(value) else int(value)


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value)
    return text if text and text.lower() != "nan" else None


def build_model_trace(row: pd.Series) -> dict[str, str]:
    """Return the four-stage model trace for one validation case row.

    The four stages are, in order:
    1. ``inputs``  — 公开输入 (what public data was available)
    2. ``rule``    — 规则判断 (whether a usable window formed)
    3. ``output``  — 模型输出 (the conservative public-signal point or 不判断)
    4. ``check``   — 赛后核对 (post-race comparison against pit service)

    Control cases (no wet-to-slick event) yield a clean "no switch / no error
    number" trace rather than a fabricated error.
    """
    pit = _opt_int(row.get("actual_pit_lap"))
    first_slick = _opt_int(row.get("first_slick_lap"))
    if first_slick is None:
        first_slick = _opt_int(row.get("actual_transition_lap"))
    has_switch = pit is not None or first_slick is not None

    signal_start = _opt_int(row.get("decision_time_window_start"))
    signal_end = _opt_int(row.get("decision_time_window_end"))
    dry_lap = _opt_int(row.get("dry_track_confirmation_lap"))
    dry_source = _opt_str(row.get("dry_track_confirmation_source"))
    rainfall_only = _opt_int(row.get("rainfall_only_window_start"))
    error = row.get("decision_time_error_laps")
    failure_type = _opt_str(row.get("failure_type")) or ""
    rainfall_records = _opt_int(row.get("rainfall_records")) or 0

    if not has_switch:
        rainfall_text = "有降雨记录" if rainfall_records > 0 else "没有降雨记录"
        return {
            "inputs": f"本场为对照：{rainfall_text}，但没有湿胎转光头胎事件。",
            "rule": "公开证据未形成可用湿转干窗口 → 模型不判断。",
            "output": "不判断（对湿转干保持沉默）。",
            "check": "没有真实进站/光头胎锚点，无需赛后核对。",
            "has_switch": False,
        }

    # --- Real case with an observed wet-to-slick switch ---
    # 1. 公开输入
    bits: list[str] = []
    if rainfall_records > 0:
        bits.append("公开降雨记录存在")
    if dry_lap is not None:
        src_text = dry_source if dry_source else "赛会/赛道条件信号"
        bits.append(f"第 {dry_lap} 圈有{src_text}")
    if rainfall_only is None and dry_lap is not None:
        bits.append("仅凭降雨不足以判定可用窗口，需赛会条件信号补足")
    if not bits:
        bits.append("公开降雨记录与赛会/赛道条件信号")
    inputs = "公开输入：" + "；".join(bits) + "。"

    # 2. 规则判断
    # The conservative public-signal point is the *later* boundary of the
    # public-signal interval (see app legend). Report fields may store the
    # later lap under either window_start or window_end, so use min/max.
    if signal_start is not None and signal_end is not None:
        signal_lo = min(signal_start, signal_end)
        signal_hi = max(signal_start, signal_end)
        window_text = f"第 {signal_hi} 圈" if signal_lo == signal_hi else f"第 {signal_lo}–{signal_hi} 圈"
        rule = (
            f"公开证据在 {window_text} 形成可用湿转干恢复窗口"
            f"（保守公开信号点为较晚的第 {signal_hi} 圈）。"
        )
        if "多阶段天气" in failure_type:
            rule += "本场后续还有独立降雨阶段，不混入当前验证事件。"
    else:
        rule = "公开证据不足以形成可用湿转干窗口 → 模型不判断。"

    # 3. 模型输出
    if signal_start is not None and signal_end is not None:
        signal_hi = max(signal_start, signal_end)
        output = f"保守公开信号点：约第 {signal_hi} 圈开始考虑换回光头胎。"
    else:
        output = "不判断。"

    # 4. 赛后核对
    if pit is not None:
        anchor_text = f"真实进站服务圈第 {pit} 圈"
        if first_slick is not None and first_slick != pit:
            anchor_text += f"（首个光头胎圈第 {first_slick} 圈）"
    elif first_slick is not None:
        anchor_text = f"首个光头胎圈第 {first_slick} 圈（本场无独立 pit 锚点）"
    else:
        anchor_text = "无可用锚点"

    if signal_end is not None and error is not None and pd.notna(error):
        sign = "+" if int(error) >= 0 else ""
        check = (
            f"{anchor_text}；与模型输出相差 {sign}{int(error)} 圈。"
            "此事实为赛后核对锚点，不是模型输入，也不解释为什么真实进站发生在此圈。"
        )
    elif signal_end is not None:
        check = (
            f"{anchor_text}；与模型输出无法量化比较。"
            "此事实为赛后核对锚点，不是模型输入。"
        )
    else:
        check = f"{anchor_text}；模型未发出信号，无误差可计算。"

    return {
        "inputs": inputs,
        "rule": rule,
        "output": output,
        "check": check,
        "has_switch": True,
    }
