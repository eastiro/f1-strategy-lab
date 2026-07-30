"""V4.1 最后一停策略判断摘要的纯展示辅助（只读已有稳健性对象，不重算公式）。

本模块不依赖 streamlit，只把 app/streamlit_app.py 里已经算好的
``robust_best`` / ``dominant_alternatives`` / ``scenario_results_for(robust_best)``
组织成面向访问者的文字与表格。所有数值都来自调用方传入的既有结果，绝不在本
模块重新模拟或硬编码答案。

“稳健”在本任务里只指：同一条赛后重建曲线的干燥段在 -2 / 0 / +2 圈三种既定偏移
下，一个候选方案是否仍优于真实策略基准。本模块不引入新的物理模型或统计置信区间。
"""
from __future__ import annotations

import pandas as pd


def describe_delta(value: float) -> str:
    """把相对基准的秒数差转成“约快 / 约慢 X 秒”的读法。"""
    return f"约快 {-value:.1f} 秒" if value < 0 else f"约慢 {value:.1f} 秒"


def build_strategy_judgment(
    robust_best: "pd.Series",
    dominant_alternatives: "pd.DataFrame",
    reference_final_stint: object,
    robust_compound_label: str,
    reference_compound_label: str,
    pit_window_start: int,
    pit_window_end: int,
    scenario_results_for,
) -> dict:
    """返回摘要区块所需的全部文字与表格。

    参数都是 app 里已经算好的对象，本函数只读取、不重算、不硬编码。

    Returns
    -------
    dict with keys:
        question_baseline        —— 一行问题+基准事实
        robustness_conclusion    —— 固定的稳健性判断正文（warning 或 info 文案）
        robustness_kind          —— "warning" / "info"
        three_scenario_table     —— robust_best 在三种设定下的三行对照表
        worst_case_text          —— 最差情景差距的格式化读法
        reference_lap / robust_lap / reference_compound_label / robust_compound_label
    """
    reference_lap = int(reference_final_stint.start_lap)
    robust_lap = int(robust_best["slick_pit"])
    robust_worst = float(robust_best["worst_case"])

    # 直接复用调用方已算好的函数，而不是重新计算。
    robust_scenario_results = scenario_results_for(robust_best)
    robust_scenario_lookup = robust_scenario_results.set_index("scenario")["delta"].to_dict()

    question_baseline = (
        f"基准是 Norris 第 {reference_lap} 圈换{reference_compound_label}。"
        f"这里仅比较最后一停改在第 {pit_window_start}-{pit_window_end} 圈、"
        f"换软/中/硬胎的候选方案。"
    )

    if len(dominant_alternatives) == 0:
        # 找出在哪种设定下 robust_best 失去优势（delta >= 0）。
        if float(robust_scenario_lookup.get("干燥较快（-2 圈）", 0)) >= 0:
            losing_setting = "干燥较快（-2 圈）"
        elif float(robust_scenario_lookup.get("干燥较慢（+2 圈）", 0)) >= 0:
            losing_setting = "干燥较慢（+2 圈）"
        else:
            losing_setting = "当前重建"
        losing_delta = float(robust_scenario_lookup.get(losing_setting, robust_worst))
        robustness_conclusion = (
            f"在干燥速度 -2 / 0 / +2 圈三种设定下，没有替代方案都优于第 {reference_lap} 圈换"
            f"{reference_compound_label}的基准。"
            f"最接近稳健的是第 {robust_lap} 圈换{robust_compound_label}，"
            f"它在「{losing_setting}」设定下失去优势（{describe_delta(losing_delta)}）。"
            f"这仍是模型在赛后重建下的稳健性估计，不是实际策略预测，也不代表真实比赛中一定无解。"
        )
        robustness_kind = "warning"
    else:
        robustness_conclusion = (
            f"在干燥速度 -2 / 0 / +2 圈三种设定下，有 {len(dominant_alternatives)} 个替代方案都优于第 "
            f"{reference_lap} 圈换{reference_compound_label}的基准。"
            f"其中按最差情景优先排序，最接近稳健的是第 {robust_lap} 圈换{robust_compound_label}。"
            f"「最接近稳健」指它在三种设定里最不容易变慢，不等于真实最优或实际策略预测。"
        )
        robustness_kind = "info"

    three_scenario_rows = []
    for label in ["干燥较快（-2 圈）", "当前赛后重建", "干燥较慢（+2 圈）"]:
        val = float(robust_scenario_lookup.get(label, float(robust_best["median_delta"])))
        three_scenario_rows.append(
            {
                "干燥设定": label,
                "相对基准": describe_delta(val),
                "数值（秒）": f"{val:+.2f}",
            }
        )
    three_scenario_table = pd.DataFrame(three_scenario_rows)

    worst_case_text = describe_delta(robust_worst)

    return {
        "question_baseline": question_baseline,
        "robustness_conclusion": robustness_conclusion,
        "robustness_kind": robustness_kind,
        "three_scenario_table": three_scenario_table,
        "worst_case_text": worst_case_text,
        "reference_lap": reference_lap,
        "robust_lap": robust_lap,
        "reference_compound_label": reference_compound_label,
        "robust_compound_label": robust_compound_label,
    }
