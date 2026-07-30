"""Read-only presentation helper for the 2021-2025 five-season FastF1 audit.

This module ONLY reads the Task 12 generated combined events product
(``reports/results/v3_fastf1_events_2021_2025.csv``). It never scans, downloads,
or generates any FastF1 data, and it never mutates the product.

Every number shown on the Chinese audit page is derived here from the product,
so the page cannot drift from the audited figures. The mapping of season to the
Chinese research-position wording is presentation text keyed by season, not
research data, and is fixed by the Task 13 product decision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PRODUCT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "results"
    / "v3_fastf1_events_2021_2025.csv"
)

# Presentation wording keyed by season (Task 13 product decision).
RESEARCH_POSITION = {
    2021: "既有审计赛季",
    2022: "新增预注册扩展",
    2023: "规则形成期已接触",
    2024: "规则形成期已接触",
    2025: "规则形成期已接触",
}

SEASONS_ORDER = [2021, 2022, 2023, 2024, 2025]


class FiveSeasonDataMissing(Exception):
    """Raised when the five-season audit product is absent or unusable."""


def load_five_season_audit(product_path=None):
    """Load the five-season audit summary and per-season rows from the product.

    Returns a dict::

        {
            "summary": {
                "events": int, "signals": int, "abstentions": int,
                "pit_service": int, "first_slick_fallback": int,
            },
            "seasons": [
                {
                    "season": int,
                    "research_position": str,
                    "events": int, "signals": int, "abstentions": int,
                    "pit_service": int, "first_slick_fallback": int,
                    "mae": Optional[float],
                    "mae_label": str,   # "4.69 圈" or "N/A（该季未发出信号）"
                },
                ...
            ],
        }

    Raises FiveSeasonDataMissing if the product is missing or malformed.
    """
    path = Path(product_path) if product_path is not None else _PRODUCT
    if not path.exists():
        raise FiveSeasonDataMissing(f"未找到五赛季审计产物：{path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # surface read failures as our own error
        raise FiveSeasonDataMissing(f"读取五赛季审计产物失败：{exc}") from exc

    required = {
        "season",
        "state_machine_outcome",
        "timing_anchor_source",
        "reporting_absolute_error_laps",
    }
    if df.empty or not required.issubset(df.columns):
        raise FiveSeasonDataMissing("五赛季审计产物缺少必要列或为空")

    seasons = []
    total = {
        "events": 0,
        "signals": 0,
        "abstentions": 0,
        "pit_service": 0,
        "first_slick_fallback": 0,
    }
    for yr in SEASONS_ORDER:
        if yr not in df["season"].values:
            # A complete audit covers every season; skip only if absent.
            continue
        g = df[df["season"] == yr]
        g_issued = (
            g["state_machine_outcome"].astype(str).str.strip().str.lower().eq("signal")
        )
        events = len(g)
        signals = int(g_issued.sum())
        abstentions = events - signals
        pit = int((g["timing_anchor_source"] == "pit_service").sum())
        fb = int((g["timing_anchor_source"] == "first_slick_fallback").sum())
        abs_err = g.loc[g_issued, "reporting_absolute_error_laps"]
        if signals > 0 and abs_err.notna().any():
            mae = round(float(abs_err.dropna().mean()), 2)
            mae_label = f"{mae:.2f} 圈"
        else:
            mae = None
            mae_label = "N/A（该季未发出信号）"

        seasons.append(
            {
                "season": yr,
                "research_position": RESEARCH_POSITION.get(yr, ""),
                "events": events,
                "signals": signals,
                "abstentions": abstentions,
                "pit_service": pit,
                "first_slick_fallback": fb,
                "mae": mae,
                "mae_label": mae_label,
            }
        )
        total["events"] += events
        total["signals"] += signals
        total["abstentions"] += abstentions
        total["pit_service"] += pit
        total["first_slick_fallback"] += fb

    return {"summary": total, "seasons": seasons}
