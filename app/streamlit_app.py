from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conclusion_summary import build_strategy_judgment
from src.config import (
    DEFAULT_COUNTRY,
    DEFAULT_YEAR,
    PROCESSED_DIR,
    RAW_DIR,
    RESEARCH_DIR,
    RESULTS_DIR,
)
from src.errors import CaseDataUnavailableError
from src.five_season_display import FiveSeasonDataMissing, load_five_season_audit
from src.openf1_client import make_race_slug
from src.plots import (
    plot_compound_comparison,
    plot_cumulative_strategy_delta,
    plot_lap_pace,
    plot_pace_residuals,
    plot_reliability_event_map,
    plot_reliability_reason_counts,
    plot_scenario_breakdown,
    plot_sensitivity_heatmap,
    plot_stint_timeline,
    plot_strategy_weather_timeline,
    plot_v2_track_state_evidence,
    plot_weather_residual_comparison,
    plot_wetness_penalty,
    plot_wetness_sources,
)
from src.race_control import load_strategy_events
from src.strategy_delta_explanation import build_delta_explanation
from src.strategy_model import (
    COMPOUND_BASE_DELTA,
    COMPOUND_DEGRADATION,
    COMPOUND_WET_PENALTY,
    StintPlan,
    build_drying_scenarios,
    build_reconstructed_track_state,
    build_track_state_diagnostics,
    build_wetness_curve,
    build_wetness_from_weather,
    estimate_driver_baseline,
    estimate_pace_residuals,
    intermediate_wear_exposure,
    simulate_strategy,
)
from src.v2_trace import build_model_trace
from src.v2_validation import (
    V2_CASES,
    build_case_track_state,
    build_validation_metrics,
    build_validation_report,
    field_wet_to_slick_wave,
    later_rainfall_phase_start,
)
from src.v3_reliability import build_reliability_events, summarize_reliability_events
from src.weather_timing import prepare_driver_lap_timing

DEFAULT_INTER_PIT = 28
DEFAULT_SLICK_PIT = 40
DEFAULT_PIT_LOSS = 22.0
DEFAULT_RAIN_START = 18
DEFAULT_RAIN_END = 40
DEFAULT_PEAK_WETNESS = 0.65
DEFAULT_LAUNCH_PENALTY = 4.5


def real_stints_for_driver(laps: pd.DataFrame, stints_real: pd.DataFrame, driver: str) -> list[StintPlan]:
    if stints_real.empty:
        return []

    driver_numbers = laps.loc[laps["name_acronym"] == driver, "driver_number"].dropna().unique()
    if len(driver_numbers) == 0:
        return []

    driver_stints = stints_real[stints_real["driver_number"].isin(driver_numbers)].copy()
    driver_stints = driver_stints.dropna(subset=["compound", "stint_lap_start", "stint_lap_end"])
    driver_stints = driver_stints.sort_values("stint_number")
    return [
        StintPlan(
            str(row["compound"]).upper(),
            int(row["stint_lap_start"]),
            int(row["stint_lap_end"]),
        )
        for _, row in driver_stints.iterrows()
    ]


def summarize_weather_by_lap(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    driver: str,
    total_laps: int,
) -> pd.DataFrame:
    if weather.empty or "date" not in weather.columns:
        return pd.DataFrame(
            {
                "lap_number": range(1, total_laps + 1),
                "openf1_rainfall": 0,
                "track_temperature": None,
                "humidity": None,
            }
        )

    driver_laps = prepare_driver_lap_timing(laps, driver)

    weather_state = weather.copy()
    weather_state["weather_time"] = pd.to_datetime(weather_state["date"], utc=True, errors="coerce")
    weather_state["rainfall"] = pd.to_numeric(weather_state.get("rainfall", 0), errors="coerce").fillna(0)
    weather_state["track_temperature"] = pd.to_numeric(
        weather_state.get("track_temperature", pd.NA),
        errors="coerce",
    )
    weather_state["humidity"] = pd.to_numeric(weather_state.get("humidity", pd.NA), errors="coerce")
    weather_state = weather_state.dropna(subset=["weather_time"]).sort_values("weather_time")

    rows = []
    for lap_number in range(1, total_laps + 1):
        lap = driver_laps[driver_laps["lap_number"] == lap_number]
        if lap.empty:
            rows.append({"lap_number": lap_number})
            continue

        lap_start_time = lap["lap_start_time"].iloc[0]
        lap_end_time = lap["lap_end_time"].iloc[0]
        lap_weather = weather_state[
            (weather_state["weather_time"] >= lap_start_time)
            & (weather_state["weather_time"] <= lap_end_time)
        ]
        if lap_weather.empty and not weather_state.empty:
            nearest_index = (weather_state["weather_time"] - lap_start_time).abs().idxmin()
            lap_weather = weather_state.loc[[nearest_index]]

        rows.append(
            {
                "lap_number": lap_number,
                "openf1_rainfall": int(lap_weather["rainfall"].max()) if not lap_weather.empty else 0,
                "track_temperature": round(float(lap_weather["track_temperature"].mean()), 1)
                if not lap_weather.empty and lap_weather["track_temperature"].notna().any()
                else None,
                "humidity": round(float(lap_weather["humidity"].mean()), 1)
                if not lap_weather.empty and lap_weather["humidity"].notna().any()
                else None,
            }
        )

    return pd.DataFrame(rows)


def read_optional_csv(path: Path, label: str) -> pd.DataFrame:
    """Load supporting data without exposing a Python traceback to visitors."""
    if not path.exists():
        st.warning(f"未找到{label}，相关图表将显示为空。")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        st.warning(f"无法读取{label}（{path.name}）：{error}。相关图表将显示为空。")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_v2_validation() -> pd.DataFrame:
    return build_validation_report()


@st.cache_data(show_spinner=False)
def load_v2_challenge_set() -> pd.DataFrame:
    path = RESULTS_DIR / "v2_holdout_v23_2025.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_v2_fastf1_evaluation() -> pd.DataFrame:
    path = RESULTS_DIR / "v2_fastf1_evaluation_2021.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_v2_case_track_state(case_id: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    case = next(case for case in V2_CASES if case.case_id == case_id)
    return build_case_track_state(case)


@st.cache_data(show_spinner=False)
def load_five_season_audit_cached() -> object | None:
    """Read-only entry point for the five-season audit display.

    Returns the helper dict, or ``None`` when the Task 12 product is missing so
    the page can degrade gracefully (no ``st.stop()``, no traceback).
    """
    try:
        return load_five_season_audit()
    except FiveSeasonDataMissing:
        return None


st.set_page_config(page_title="F1 Strategy Lab", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      :root {
        --paper: #fbfaf8;
        --panel: #ffffff;
        --panel-soft: #f5f2ed;
        --rule: #e7e3db;
        --rule-strong: #d0c9be;
        --ink: #15120f;
        --body: #3e3933;
        --muted: #6f6a62;
        --faint: #9d968c;
        --red: #c8102e;
        --red-wash: #fcf3f4;
        --teal: #16766f;
        --sans: "Segoe UI Variable Display", "Segoe UI", Inter, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
        --mono: "Cascadia Mono", Consolas, ui-monospace, "SF Mono", monospace;
      }
      html { scroll-behavior: smooth; }
      .stApp, .stApp button, .stApp input, .stApp textarea { font-family: var(--sans); }
      .stApp { background: var(--paper); color: var(--ink); }
      .stAppHeader { display: none; }
      [data-testid="stHeaderActionElements"], [data-testid="stDeployButton"], [data-testid="stBaseButton-header"], #MainMenu, footer { display: none; }
      .block-container { max-width: 1040px; margin-inline: auto; padding: 3.2rem 1.8rem 6rem; }
      h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -.012em !important; text-wrap: balance; }
      h1 { font-size: 3.15rem !important; font-weight: 800 !important; line-height: 1 !important; }
      h2, h3 { padding-left: 0; margin-top: 2.6rem !important; }
      h2 { font-size: 1.82rem !important; font-weight: 750 !important; }
      h3 { font-size: 1.42rem !important; font-weight: 720 !important; letter-spacing: -.008em !important; }
      h4 { font-size: 1.06rem !important; font-weight: 700 !important; }
      [data-testid="stMarkdownContainer"] p { font-size: 1.01rem; line-height: 1.76; letter-spacing: .002em; color: var(--body); }

      .lab-hero { margin: 0 0 2.1rem; }
      .lab-kicker { display: flex; align-items: center; gap: .6rem; color: var(--red); font-family: var(--mono); font-size: .72rem; font-weight: 600; letter-spacing: .18em; text-transform: uppercase; }
      .lab-kicker-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--red); }
      .lab-version { margin-left: .15rem; padding: .1rem .42rem; border: 1px solid var(--rule-strong); border-radius: 2px; color: var(--muted); font-size: .66rem; letter-spacing: .08em; }
      .lab-title-row { display: flex; align-items: baseline; justify-content: space-between; gap: 2rem; margin: 1.15rem 0 0; padding-bottom: 1.1rem; border-bottom: 1px solid var(--rule-strong); }
      .lab-title { margin: 0; color: var(--ink); font-size: 3.15rem; font-weight: 800; line-height: .98; letter-spacing: -.028em; }
      .lab-subtitle { color: var(--muted); font-family: var(--mono); font-size: .84rem; font-weight: 500; letter-spacing: .01em; white-space: nowrap; }
      .lab-module-strip { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr) minmax(0,.82fr); gap: 0; border-bottom: 1px solid var(--rule); }
      .lab-module { display: grid; grid-template-columns: 1.85rem 1fr; gap: .7rem; padding: 1.15rem 1.5rem 1.15rem 0; min-width: 0; }
      .lab-module + .lab-module { padding-left: 1.5rem; border-left: 1px solid var(--rule); }
      .lab-module-index { color: var(--faint); font-family: var(--mono); font-size: .76rem; font-weight: 500; padding-top: .1rem; }
      .lab-module strong { display: block; color: var(--ink); font-size: 1rem; font-weight: 700; margin-bottom: .34rem; }
      .lab-module span { display: block; color: var(--muted); font-size: .89rem; line-height: 1.62; }
      .lab-module-divider { display: none; }
      .lab-boundary { display: flex; flex-direction: column; justify-content: center; padding: 1.15rem 0 1.15rem 1.5rem; border-left: 1px solid var(--rule); }
      .lab-boundary strong { color: var(--teal); font-family: var(--mono); font-size: .7rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .4rem; }
      .lab-boundary span { color: var(--muted); font-size: .87rem; line-height: 1.6; }
      .lab-route { margin: 1.15rem 0 0; color: var(--muted); font-size: .87rem; line-height: 1.7; }
      .lab-route b { color: var(--ink); font-weight: 650; white-space: nowrap; }

      [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--rule); border-radius: 2px; padding: 1.05rem 1.15rem; min-height: 104px; box-shadow: none; }
      [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .82rem !important; font-weight: 550 !important; letter-spacing: .02em !important; }
      [data-testid="stMetricLabel"] p { font-size: .82rem !important; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--faint) !important; font-size: .86rem !important; line-height: 1.62 !important; opacity: 1 !important; }
      [data-testid="stMetricValue"] { color: var(--ink); font-family: var(--mono); font-size: 1.95rem; font-weight: 600; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
      [data-testid="stMetricDelta"] { font-family: var(--mono); font-size: .84rem; font-variant-numeric: tabular-nums; }
      [data-testid="stMarkdownContainer"] { color: var(--body); }
      [data-testid="stDataFrame"] { background: var(--panel); border: 1px solid var(--rule); border-radius: 2px; overflow: hidden; box-shadow: none; }
      [data-testid="stDataFrame"] [role="columnheader"] { font-family: var(--mono); font-size: .74rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
      [data-testid="stPlotlyChart"] { background: var(--panel); border: 1px solid var(--rule); border-radius: 2px; overflow: hidden; box-shadow: none; }
      hr { border-color: var(--rule) !important; }

      [data-testid="stTabs"] > div > div:first-child { position: sticky; top: 0; z-index: 80; background: rgba(251,250,248,.95); backdrop-filter: saturate(180%) blur(12px); }
      [data-testid="stTabs"] [data-baseweb="tab-list"] { position: relative; gap: 1.9rem; min-height: 3rem; border-bottom: 1px solid var(--rule-strong); background: transparent; }
      [data-testid="stTabs"] button { min-height: 3rem; padding: .55rem .02rem; color: var(--muted); font-size: .95rem; font-weight: 600; letter-spacing: -.005em !important; white-space: nowrap; }
      [data-testid="stTabs"] button:hover { color: var(--ink); }
      [data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink); font-weight: 700; }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--red); height: 2px; }

      [data-testid="stAlert"] { border-radius: 2px; border: 1px solid var(--rule); border-left: 2px solid var(--teal); background: #f2f6f5; box-shadow: none; }
      [data-testid="stAlert"] p { line-height: 1.68; }
      [data-testid="stExpander"] { border: 1px solid var(--rule) !important; border-radius: 2px !important; background: var(--panel); overflow: hidden; box-shadow: none !important; }
      [data-testid="stExpander"] summary { min-height: 2.9rem; }
      [data-testid="stExpander"] summary p { color: var(--body) !important; font-size: .94rem !important; font-weight: 600 !important; }
      [data-testid="stExpander"] summary:hover { background: var(--panel-soft); }

      [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-size: .88rem !important; font-weight: 600; letter-spacing: .01em; }
      [data-baseweb="select"] > div, [data-baseweb="input"] > div { border-color: var(--rule-strong) !important; border-radius: 2px !important; background: var(--panel) !important; }
      [data-baseweb="tag"] { background: var(--ink) !important; border-radius: 2px !important; font-family: var(--mono); font-size: .8rem; }
      section[data-testid="stSidebar"] { background: var(--panel-soft); border-right: 1px solid var(--rule); }
      .stButton > button, .stDownloadButton > button { border-radius: 2px; border-color: var(--rule-strong); background: var(--panel); color: var(--ink); font-weight: 600; box-shadow: none; transition: border-color .15s ease, color .15s ease; }
      .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--red); color: var(--red); background: var(--panel); }
      .stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid var(--red); outline-offset: 1px; }
      [data-testid="stSlider"] [role="slider"] { box-shadow: none; }
      code, kbd, pre { font-family: var(--mono); }

      .lab-rule { display: flex; align-items: center; gap: 1.1rem; margin: 1.35rem 0 0; }
      .lab-rule-meta { color: var(--faint); font-family: var(--mono); font-size: .68rem; font-weight: 500; letter-spacing: .14em; white-space: nowrap; }
      .lab-rule-track { position: relative; flex: 1; height: 1px; background: var(--rule-strong); }
      .lab-rule-track::before { content: ""; position: absolute; left: 0; right: 0; top: -3px; height: 7px; background: repeating-linear-gradient(90deg, var(--rule-strong) 0 1px, transparent 1px 5.8%); opacity: .5; }
      .lab-rule-pin { position: absolute; top: 50%; width: 7px; height: 7px; margin: -3.5px 0 0 -3.5px; border-radius: 50%; background: var(--red); box-shadow: 0 0 0 3px rgba(200,16,46,.09); }
      .lab-rule-pin.ghost { background: var(--paper); border: 1.5px solid var(--rule-strong); box-shadow: none; }

      /* ── reference pass: measure, hierarchy, tab rail ── */
      h1 { font-size: 3.05rem !important; letter-spacing: -.04em !important; line-height: 1.02 !important; }
      h2 { font-size: 1.5rem !important; line-height: 1.3 !important; margin-top: 2.1rem !important; letter-spacing: -.02em !important; }
      h3 { font-size: 1.08rem !important; line-height: 1.4 !important; }
      [data-testid="stMarkdownContainer"] p { max-width: 46em; font-size: .95rem; line-height: 1.78; }
      .lab-note, .lab-boundary p, [data-testid="stCaptionContainer"] p { max-width: none; }
      [data-testid="stTabs"] [data-baseweb="tab-list"] { border-bottom: 1px solid var(--rule); gap: 1.7rem; }
      [data-testid="stTabs"] button { font-size: .84rem; letter-spacing: .01em; padding: .35rem 0 .6rem; }
      [data-testid="stTabs"] button[aria-selected="true"] { font-weight: 700; }
      [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; letter-spacing: -.03em; }
      [data-testid="stExpander"] { border: 1px solid var(--rule) !important; border-radius: 2px !important; background: transparent !important; }
      hr, [data-testid="stDivider"] { border-color: var(--rule) !important; }

      @media (max-width: 1100px) {
        .lab-module-strip { grid-template-columns: minmax(0,1fr) minmax(0,1fr); }
        .lab-boundary { grid-column: 1 / -1; border-left: 0; border-top: 1px solid var(--rule); padding-left: 0; }
      }
      @media (max-width: 900px) {
        .block-container { padding: 1.5rem 1.1rem 4rem; }
        .lab-title-row { align-items: start; flex-direction: column; gap: .5rem; margin-top: .9rem; }
        .lab-title { font-size: 2.3rem; }
        .lab-subtitle { font-size: .78rem; white-space: normal; }
        .lab-module-strip { grid-template-columns: 1fr; }
        .lab-module { padding: .9rem 0; }
        .lab-module + .lab-module { padding-left: 0; border-left: 0; border-top: 1px solid var(--rule); }
        .lab-boundary { grid-column: auto; padding: .9rem 0; }
        .lab-route { font-size: .86rem; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0; justify-content: space-between; overflow-x: auto; }
        [data-testid="stTabs"] button { flex: 1 0 auto; min-width: 0; padding-left: .3rem; padding-right: .3rem; font-size: .86rem; }
        h1 { font-size: 2.35rem !important; }
        h3 { font-size: 1.28rem !important; }
        [data-testid="stMetric"] { min-height: 96px; }
      }

    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <section class="lab-hero">
      <div class="lab-kicker"><span class="lab-kicker-dot"></span>F1 Strategy Research Lab <span class="lab-version">V4</span></div>
      <div class="lab-title-row">
        <h1 class="lab-title">F1 Strategy Lab</h1>
        <div class="lab-subtitle">2024 英国大奖赛 · Norris 最后一停 · 公开数据赛后研究</div>
      </div>
      <div class="lab-rule">
        <span class="lab-rule-meta">SILVERSTONE · 52 LAPS</span>
        <div class="lab-rule-track">
          <span class="lab-rule-pin ghost" style="left:74.9%"></span>
          <span class="lab-rule-pin" style="left:76.9%"></span>
        </div>
        <span class="lab-rule-meta">WET &rarr; DRY</span>
      </div>
      <div class="lab-module-strip">
        <div class="lab-module">
          <div class="lab-module-index">01</div>
          <div><strong>银石策略沙盒</strong><span>比较 Norris 最后一停的换胎圈数与干胎选择，并检查结论对干燥速度假设是否敏感。</span></div>
        </div>
        <div class="lab-module-divider"></div>
        <div class="lab-module">
          <div class="lab-module-index">02</div>
          <div><strong>公开信号审计</strong><span>跨比赛检查公开信号何时可以发声、何时应该弃权，主动呈现方法的适用边界。</span></div>
        </div>
        <div class="lab-boundary"><strong>研究边界</strong><span>两块内容独立；审计不验证模拟器的物理模型，两块内容都不是通用实时决策系统。</span></div>
      </div>
      <div class="lab-route">想探究 Norris 最后一停的不同选择，从 <b>比赛结论</b> 或 <b>自由实验</b> 开始；想判断公开信号能否支持湿转干换胎，打开 <b>公开信号审计</b>。</div>
    </section>
    """,
    unsafe_allow_html=True,
)

slug = make_race_slug(DEFAULT_YEAR, DEFAULT_COUNTRY)
lap_path = PROCESSED_DIR / f"{slug}_laps_clean.csv"
stint_path = PROCESSED_DIR / f"{slug}_stints.csv"
weather_path = PROCESSED_DIR / f"{slug}_weather.csv"

if not lap_path.exists():
    st.warning("还没有找到处理后的比赛数据。请先运行数据下载和清洗命令。")
    st.code(
        "python -m src.download_openf1 --year 2024 --country \"United Kingdom\"\n"
        "python -m src.prepare_dataset --year 2024 --country \"United Kingdom\"",
        language="powershell",
    )
    st.stop()

try:
    laps = pd.read_csv(lap_path)
except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
    st.error(f"无法读取主比赛圈速数据（{lap_path.name}）：{error}。请重新生成该比赛的数据文件。")
    st.stop()

stints_real = read_optional_csv(stint_path, "轮胎分段数据")
weather = read_optional_csv(weather_path, "天气数据")
drivers = sorted(laps["name_acronym"].dropna().unique().tolist())
default_drivers = [driver for driver in ["HAM", "NOR", "VER", "PIA"] if driver in drivers]
driver_story_notes = {
    "HAM": "本场冠军，用来当最终结果参照",
    "NOR": "主分析对象，问题集中在他最后一停",
    "VER": "后段使用硬胎，是关键替代路线参照",
    "PIA": "Norris 的队友，用来观察 McLaren 的策略环境",
}
total_laps = int(laps["lap_number"].max())

conclusion_tab, experiment_tab, model_tab, research_tab = st.tabs(
    ["比赛结论", "自由实验", "模型与数据", "公开信号审计"]
)
conclusion_tab.caption("先看真实策略、可能更快的替代方案，以及它们在天气变化下是否仍然成立。")
experiment_tab.caption("这里可以自由改参数做假设实验；它不会改变比赛结论页固定的 Norris 主线分析。")
model_tab.caption("查看天气推断、轮胎参数、圈速残差和模型边界。")
research_tab.caption("公开信号什么时候足以提示“可以开始考虑换回光头胎”，什么时候不该被相信。")

conclusion_tab.markdown("#### 比赛背景")
conclusion_tab.write(
    "2024 英国大奖赛后段经历由湿转干的窗口。Hamilton 第 39 圈换软胎并最终夺冠，"
    "Norris 第 40 圈换软胎，Verstappen 第 39 圈换硬胎追近。"
    "这个项目只聚焦一个问题：Norris 最后一停的换胎时机和干胎选择，是否可能改变结果。"
)
conclusion_tab.caption(
    "这里的模拟不是车队级预测系统，而是一个用公开数据解释策略取舍的分析实验。"
)
with conclusion_tab.expander("为什么只看这四位车手？"):
    st.write(
        "本页默认只看四位关键车手："
        + "；".join(
            f"{driver}：{driver_story_notes.get(driver, '策略参照')}"
            for driver in default_drivers
        )
        + "。自由实验页可以选择更多车手。"
    )

experiment_tab.markdown("#### 实验参数")
control_left, control_middle, control_right = experiment_tab.columns(3)
with control_left:
    selected_driver = st.selectbox("车手", drivers, index=drivers.index("NOR") if "NOR" in drivers else 0)
    compare_drivers = st.multiselect("对比车手", drivers, default=default_drivers)
    st.caption("这里会影响自由实验里的圈速图和真实策略时间线；比赛结论页固定讲 Norris 主线。")
with control_middle:
    inter_pit = st.slider("实验：换半雨胎圈数", min_value=2, max_value=total_laps - 2, value=min(DEFAULT_INTER_PIT, total_laps - 2))
    slick_pit_min = min(inter_pit + 1, total_laps - 1)
    slick_pit = st.slider("实验：换回干胎圈数", min_value=slick_pit_min, max_value=total_laps - 1, value=max(slick_pit_min, min(DEFAULT_SLICK_PIT, total_laps - 1)))
    final_compound = st.selectbox("最后一段轮胎", ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE"], index=0)
with control_right:
    pit_loss = st.slider("单次进站损失（秒）", min_value=15.0, max_value=35.0, value=DEFAULT_PIT_LOSS, step=0.5)
    weather_mode = st.radio("天气来源", ["赛后重建（降雨 + 多人圈速）", "手动实验"], horizontal=True)
    if weather_mode == "手动实验":
        rain_start, rain_end = st.slider("雨势影响窗口", min_value=1, max_value=total_laps, value=(DEFAULT_RAIN_START, DEFAULT_RAIN_END))
        peak_wetness = st.slider("最大雨势强度", min_value=0.0, max_value=1.0, value=DEFAULT_PEAK_WETNESS, step=0.05)
    else:
        rain_start, rain_end = DEFAULT_RAIN_START, DEFAULT_RAIN_END
        peak_wetness = DEFAULT_PEAK_WETNESS
        st.caption("默认用 OpenF1 降雨记录和多人共同圈速重建赛后赛道状态；手动实验模式才显示湿滑状态滑条。")
launch_penalty = DEFAULT_LAUNCH_PENALTY
experiment_tab.caption(
    f"第一圈起步损失固定为 {DEFAULT_LAUNCH_PENALTY:.1f} 秒；本场没有车手使用全雨胎。"
)

left, right = experiment_tab.columns([1.2, 1])

with left:
    st.plotly_chart(plot_lap_pace(laps, compare_drivers), width="stretch")

baseline = estimate_driver_baseline(laps, selected_driver)
stints = [
    StintPlan("MEDIUM", 1, inter_pit - 1),
    StintPlan("INTERMEDIATE", inter_pit, slick_pit - 1),
    StintPlan(final_compound, slick_pit, total_laps),
]
lap_adjustments = {1: launch_penalty}
rainfall_wetness_by_lap = build_wetness_from_weather(laps, weather, selected_driver, total_laps)
track_state_drivers = default_drivers or [selected_driver]
track_state_residuals = estimate_pace_residuals(laps, track_state_drivers)
track_state_diagnostics = build_track_state_diagnostics(
    rainfall_wetness_by_lap,
    track_state_residuals,
    total_laps,
)
real_weather_wetness_by_lap = build_reconstructed_track_state(
    rainfall_wetness_by_lap,
    track_state_diagnostics,
    total_laps,
)
track_state_diagnostics["reconstructed_wetness"] = track_state_diagnostics["lap_number"].map(
    real_weather_wetness_by_lap
).fillna(0.0)
manual_wetness_by_lap = build_wetness_curve(total_laps, rain_start, rain_end, peak_wetness=peak_wetness)
if weather_mode == "赛后重建（降雨 + 多人圈速）":
    wetness_by_lap = real_weather_wetness_by_lap
else:
    wetness_by_lap = manual_wetness_by_lap
real_weather_label = "当前使用：赛后重建状态" if weather_mode == "赛后重建（降雨 + 多人圈速）" else "赛后重建状态（参考）"
manual_weather_label = "当前使用：手动实验" if weather_mode == "手动实验" else "手动实验曲线（未使用）"
weather_source_df = pd.DataFrame(
    [
        {
            "lap_number": lap_number,
            "source": source,
            "wetness": wetness_source.get(lap_number, 0.0),
        }
        for lap_number in range(1, total_laps + 1)
        for source, wetness_source in [
            (real_weather_label, real_weather_wetness_by_lap),
            (manual_weather_label, manual_wetness_by_lap),
        ]
    ]
)
weather_lap_summary = summarize_weather_by_lap(laps, weather, selected_driver, total_laps)
weather_lap_summary["降雨记录湿滑参考"] = weather_lap_summary["lap_number"].map(rainfall_wetness_by_lap).fillna(0.0).round(1)
weather_lap_summary["赛后重建湿滑状态"] = weather_lap_summary["lap_number"].map(real_weather_wetness_by_lap).fillna(0.0).round(1)
weather_lap_summary["手动实验湿滑指数"] = weather_lap_summary["lap_number"].map(manual_wetness_by_lap).fillna(0.0).round(1)
wetness_penalty_df = pd.DataFrame(
    [
        {
            "lap_number": lap_number,
            "tyre_category": tyre_category,
            "wetness": wetness_by_lap.get(lap_number, 0.0),
            "wet_penalty": wetness_by_lap.get(lap_number, 0.0) * COMPOUND_WET_PENALTY[compound],
        }
        for lap_number in range(1, total_laps + 1)
        for tyre_category, compound in [
            ("光头胎（软/中/硬）", "MEDIUM"),
            ("半雨胎", "INTERMEDIATE"),
        ]
    ]
)
driver_real = laps[(laps["name_acronym"] == selected_driver) & (laps["lap_duration"].notna())]
real_total = float(driver_real["lap_duration"].sum())
reference_stints = real_stints_for_driver(laps, stints_real, selected_driver) or stints
reference_wetness_by_lap = real_weather_wetness_by_lap or build_wetness_curve(
    total_laps,
    DEFAULT_RAIN_START,
    DEFAULT_RAIN_END,
    peak_wetness=DEFAULT_PEAK_WETNESS,
)
reference_lap_adjustments = {1: DEFAULT_LAUNCH_PENALTY}
reference_simulation = simulate_strategy(
    total_laps,
    baseline,
    reference_stints,
    pit_loss=DEFAULT_PIT_LOSS,
    lap_adjustments=reference_lap_adjustments,
    wetness_by_lap=reference_wetness_by_lap,
)
simulation = simulate_strategy(
    total_laps,
    baseline,
    stints,
    pit_loss=pit_loss,
    lap_adjustments=lap_adjustments,
    wetness_by_lap=wetness_by_lap,
)
delta = simulation.total_time - reference_simulation.total_time
calibrated_total = real_total + delta
display_laps = simulation.laps.copy()
calibration_offset_per_lap = (real_total - reference_simulation.total_time) / total_laps
display_laps["simulated_lap_time"] = display_laps["simulated_lap_time"] + calibration_offset_per_lap

with right:
    st.metric("相对真实策略", f"{delta:+.1f}s", "负数代表更快")
    st.metric("车手基准单圈", f"{baseline:.3f}s")
    st.metric("轮胎天气不匹配指数", simulation.risk_level, f"{simulation.risk_score:.2f}")
    st.caption(
        f"校准后的模拟完赛用时约为 {calibrated_total:,.1f}s。这里更重要的是相对差距，"
        "不是模型能否复刻真实总用时。"
    )
    st.caption(
        "真实策略基准使用 OpenF1 降雨记录推断的模型湿滑指数。你改换胎圈数、轮胎或天气模式时，"
        "这里显示当前假设相对真实策略的变化。"
    )
    st.plotly_chart(
        plot_cumulative_strategy_delta(reference_simulation.laps, simulation.laps),
        width="stretch",
    )

experiment_tab.subheader("当前实验使用的模型赛道湿滑状态")
experiment_tab.caption(
    "这里的指数不是官方水深或毫米降雨量，而是模型内部的 0-1 湿滑指数；它用来把天气换算成圈速惩罚。"
)
if weather_mode == "手动实验":
    experiment_tab.info(
        "当前天气模式：手动实验。圈速模拟使用你设定的雨势影响窗口和最大雨势强度；"
        "赛后重建曲线仍保留为参考，不会被手动设置改写。"
    )
else:
    experiment_tab.success(
        "当前天气模式：赛后重建。圈速模拟使用降雨记录与多人圈速残差重建的状态；"
        "手动雨势参数当前未生效。"
    )
weather_left, weather_right = experiment_tab.columns([1.35, 0.65])
with weather_left:
    st.plotly_chart(plot_wetness_sources(weather_source_df), width="stretch")
with weather_right:
    active_max_wetness = max(wetness_by_lap.values()) if wetness_by_lap else 0.0
    real_max_wetness = max(real_weather_wetness_by_lap.values()) if real_weather_wetness_by_lap else 0.0
    rain_laps = weather_lap_summary.loc[
        weather_lap_summary["openf1_rainfall"].fillna(0) > 0,
        "lap_number",
    ].tolist()
    rain_lap_text = (
        f"第 {min(rain_laps)}-{max(rain_laps)} 圈附近"
        if rain_laps
        else "OpenF1 未记录到降雨圈"
    )
    st.metric("当前使用峰值湿滑指数", f"{active_max_wetness:.1f}")
    st.metric("赛后重建峰值", f"{real_max_wetness:.1f}")
    st.caption(f"OpenF1 rainfall 标记：{rain_lap_text}。")
    st.caption(
        "手动实验不会改变真实天气基准，只会改变当前模拟策略和下方轮胎惩罚。"
    )
with experiment_tab.expander("查看按圈整理的真实天气与手动雨势"):
    weather_table = weather_lap_summary.rename(
        columns={
            "lap_number": "圈数",
            "openf1_rainfall": "OpenF1 是否降雨",
            "track_temperature": "赛道温度",
            "humidity": "空气湿度",
        }
    )
    st.dataframe(weather_table, width="stretch", hide_index=True)

experiment_tab.subheader("最后一段轮胎对比")
comparison_rows = []
for compound_option in ["SOFT", "MEDIUM", "HARD"]:
    option_stints = [
        StintPlan("MEDIUM", 1, inter_pit - 1),
        StintPlan("INTERMEDIATE", inter_pit, slick_pit - 1),
        StintPlan(compound_option, slick_pit, total_laps),
    ]
    option_result = simulate_strategy(
        total_laps,
        baseline,
        option_stints,
        pit_loss=pit_loss,
        lap_adjustments=lap_adjustments,
        wetness_by_lap=wetness_by_lap,
    )
    option_delta = option_result.total_time - reference_simulation.total_time
    comparison_rows.append(
        {
            "最后一段轮胎": compound_option,
            "估计完赛用时（秒）": round(real_total + option_delta, 2),
            "相对真实策略差距（秒）": round(option_delta, 2),
            "轮胎天气不匹配程度": option_result.risk_level,
            "不匹配指数": round(option_result.risk_score, 3),
        }
    )

comparison_df = pd.DataFrame(comparison_rows).sort_values("估计完赛用时（秒）")
best_option = comparison_df.iloc[0]
comparison_left, comparison_right = experiment_tab.columns([0.9, 1.1])
with comparison_left:
    st.plotly_chart(plot_compound_comparison(comparison_df), width="stretch")
with comparison_right:
    st.success(
        "当前参数下最快："
        f"{best_option['最后一段轮胎']}，"
        f"相对真实策略 {best_option['相对真实策略差距（秒）']:+.2f} 秒，"
        f"轮胎天气不匹配：{best_option['轮胎天气不匹配程度']}。"
    )
    st.caption("这里固定当前进站时机和雨势设置，只改变最后一段轮胎，用来观察轮胎选择本身的影响。")
    st.caption("结果是轮胎-天气适宜性的启发式比较，不是事故概率或真实车队策略预测；安全车、DRS、交通和轮胎温度不在模型中。")
    with st.expander("查看三种轮胎的详细数值"):
        st.dataframe(comparison_df, width="stretch", hide_index=True)

conclusion_tab.subheader("如果早一圈或晚一圈换回干胎，会发生什么？")
conclusion_tab.write(
    "把它当成一张策略地图：横轴是第几圈换回干胎，纵轴是最后用哪种干胎；每一个格子都是一套从那一圈跑到终点的完整方案。"
    "我们再把同一条赛后重建曲线的干燥段向前或向后移动两圈，观察候选方案对这项模型假设是否敏感。"
)
story_driver = "NOR" if "NOR" in drivers else drivers[0]
story_baseline = estimate_driver_baseline(laps, story_driver)
story_reference_stints = real_stints_for_driver(laps, stints_real, story_driver)
story_rainfall_wetness_by_lap = build_wetness_from_weather(
    laps,
    weather,
    story_driver,
    total_laps,
)
story_track_state_residuals = estimate_pace_residuals(laps, default_drivers or [story_driver])
story_track_state_diagnostics = build_track_state_diagnostics(
    story_rainfall_wetness_by_lap,
    story_track_state_residuals,
    total_laps,
)
story_weather_wetness_by_lap = build_reconstructed_track_state(
    story_rainfall_wetness_by_lap,
    story_track_state_diagnostics,
    total_laps,
)
drying_scenarios = build_drying_scenarios(story_weather_wetness_by_lap, total_laps)
reference_intermediate_stint = next(
    (stint for stint in story_reference_stints if stint.compound == "INTERMEDIATE"),
    story_reference_stints[1],
)
reference_final_stint = story_reference_stints[-1]
analysis_inter_pit = reference_intermediate_stint.start_lap
analysis_slick_pit = reference_final_stint.start_lap
pit_window_start = max(analysis_inter_pit + 1, analysis_slick_pit - 3)
pit_window_end = min(total_laps - 1, analysis_slick_pit + 5)
sensitivity_rows = []
# V4.2 解释层所需的逐圈 DataFrame：仅保留「当前重建」场景，供下方差距来源区块读取。
# 不另起重算路径，只持有既有循环已经生成的 StrategyResult.laps 引用。
current_scenario_reference_laps = None
current_scenario_candidate_laps = {}
for scenario_name, scenario_wetness in drying_scenarios.items():
    scenario_reference = simulate_strategy(
        total_laps,
        story_baseline,
        story_reference_stints,
        pit_loss=DEFAULT_PIT_LOSS,
        lap_adjustments=reference_lap_adjustments,
        wetness_by_lap=scenario_wetness,
    )
    if scenario_name == "当前重建":
        current_scenario_reference_laps = scenario_reference.laps
    for candidate_pit in range(pit_window_start, pit_window_end + 1):
        for compound_option in ["SOFT", "MEDIUM", "HARD"]:
            candidate_stints = [
                StintPlan("MEDIUM", 1, analysis_inter_pit - 1),
                StintPlan("INTERMEDIATE", analysis_inter_pit, candidate_pit - 1),
                StintPlan(compound_option, candidate_pit, total_laps),
            ]
            candidate_result = simulate_strategy(
                total_laps,
                story_baseline,
                candidate_stints,
                pit_loss=DEFAULT_PIT_LOSS,
                lap_adjustments=reference_lap_adjustments,
                wetness_by_lap=scenario_wetness,
            )
            if scenario_name == "当前重建":
                current_scenario_candidate_laps[(candidate_pit, compound_option)] = candidate_result.laps
            sensitivity_rows.append(
                {
                    "scenario": scenario_name,
                    "slick_pit": candidate_pit,
                    "compound": compound_option,
                    "delta": candidate_result.total_time - scenario_reference.total_time,
                }
            )

sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_summary = (
    sensitivity_df.groupby(["slick_pit", "compound"], as_index=False)
    .agg(
        median_delta=("delta", "median"),
        best_case=("delta", "min"),
        worst_case=("delta", "max"),
    )
)
current_weather_results = sensitivity_df[
    sensitivity_df["scenario"] == "当前重建"
][["slick_pit", "compound", "delta"]].rename(columns={"delta": "current_delta"})
sensitivity_summary = sensitivity_summary.merge(
    current_weather_results,
    on=["slick_pit", "compound"],
    how="left",
)
alternative_summary = sensitivity_summary[
    ~(
        (sensitivity_summary["slick_pit"] == reference_final_stint.start_lap)
        & (sensitivity_summary["compound"] == reference_final_stint.compound)
    )
]
robust_best = alternative_summary.sort_values(["worst_case", "median_delta"]).iloc[0]
dominant_alternatives = alternative_summary[alternative_summary["worst_case"] < 0]
scenario_order = {"干燥较快（-2 圈）": 0, "当前重建": 1, "干燥较慢（+2 圈）": 2}
compound_display = {"SOFT": "软胎", "MEDIUM": "中胎", "HARD": "硬胎"}
reference_compound_label = compound_display.get(
    reference_final_stint.compound,
    reference_final_stint.compound,
)
robust_compound_label = compound_display.get(
    robust_best["compound"],
    robust_best["compound"],
)
intermediate_crossover = (
    COMPOUND_BASE_DELTA["INTERMEDIATE"] - COMPOUND_BASE_DELTA["MEDIUM"]
) / (
    COMPOUND_WET_PENALTY["MEDIUM"] - COMPOUND_WET_PENALTY["INTERMEDIATE"]
)
soft_intermediate_crossover = (
    COMPOUND_BASE_DELTA["INTERMEDIATE"] - COMPOUND_BASE_DELTA["SOFT"]
) / (
    COMPOUND_WET_PENALTY["SOFT"] - COMPOUND_WET_PENALTY["INTERMEDIATE"]
)


def describe_delta(value: float) -> str:
    return f"约快 {-value:.1f} 秒" if value < 0 else f"约慢 {value:.1f} 秒"


def scenario_results_for(candidate: pd.Series) -> pd.DataFrame:
    results = sensitivity_df[
        (sensitivity_df["slick_pit"] == candidate["slick_pit"])
        & (sensitivity_df["compound"] == candidate["compound"])
    ].copy()
    results["scenario_order"] = results["scenario"].map(scenario_order)
    return results.sort_values("scenario_order")


# ---------------------------------------------------------------------------
# V4.1 最后一停：策略判断（先给答案，再给图表）
# 固定稳健性结论：始终使用 dominant_alternatives / robust_best /
# scenario_results_for(robust_best)，不随下方热力图视角切换而改变。
# 所有数值来自上面已算好的对象，本区块不重新模拟、不硬编码答案。
# ---------------------------------------------------------------------------
conclusion_tab.subheader("最后一停：策略判断")
judgment = build_strategy_judgment(
    robust_best=robust_best,
    dominant_alternatives=dominant_alternatives,
    reference_final_stint=reference_final_stint,
    robust_compound_label=robust_compound_label,
    reference_compound_label=reference_compound_label,
    pit_window_start=pit_window_start,
    pit_window_end=pit_window_end,
    scenario_results_for=scenario_results_for,
)
conclusion_tab.write(judgment["question_baseline"])
if judgment["robustness_kind"] == "warning":
    conclusion_tab.warning(judgment["robustness_conclusion"])
else:
    conclusion_tab.info(judgment["robustness_conclusion"])
conclusion_tab.write("**最接近稳健的候选方案在三种干燥设定下的结果：**")
conclusion_tab.dataframe(judgment["three_scenario_table"], width="stretch", hide_index=True)
judge_left, judge_right = conclusion_tab.columns([1, 2])
with judge_left:
    st.metric("最差情景差距", judgment["worst_case_text"])
    st.caption("「最差情景」取三种设定里最慢的结果；负数代表比基准更快。")
with judge_right:
    conclusion_tab.caption(
        "「稳健」仅指：同一条赛后重建曲线的干燥段在 -2 / 0 / +2 圈三种既定偏移下，"
        "该候选是否仍优于真实策略基准。它不是对真实比赛或车队策略的预测。"
    )

conclusion_tab.caption(
    "「最接近稳健」按三种干燥设定中最差结果排序；「当前重建下最好」只看当前这条重建曲线。"
    "两者都是候选方案之间的比较，不等于一定快过第 40 圈软胎基准，更不等于真实最优。"
)

# ---------------------------------------------------------------------------
# V4.2 这点差距从哪里来？（当前赛后重建下，最优候选相对基准的模型内差距来源）
# 解释层：复用上面已经算好的「当前重建」场景 reference / candidate 逐圈 DataFrame，
# 不再重新模拟、不改公式。所有数字都是模型内逐圈公式项相减，不是真实车队因果解释。
# ---------------------------------------------------------------------------
conclusion_tab.subheader("这点差距从哪里来？（当前赛后重建）")
conclusion_tab.info(
    "本区块解释的是「当前重建下表现最好的候选方案」相对 Norris 第 "
    f"{reference_final_stint.start_lap} 圈换{reference_compound_label}基准的差距；"
    "它不等同于上方按三种干燥设定排序的「最接近稳健」方案，且不代表真实比赛因果。"
)
if current_scenario_reference_laps is not None and current_scenario_candidate_laps:
    current_reconstruction_best = alternative_summary.sort_values(
        ["current_delta", "worst_case"]
    ).iloc[0]
    _v4p2_pit = int(current_reconstruction_best["slick_pit"])
    _v4p2_compound = current_reconstruction_best["compound"]
    _v4p2_candidate_laps = current_scenario_candidate_laps.get((_v4p2_pit, _v4p2_compound))
    if _v4p2_candidate_laps is not None:
        explanation = build_delta_explanation(
            candidate_laps=_v4p2_candidate_laps,
            reference_laps=current_scenario_reference_laps,
            candidate_pit=_v4p2_pit,
            reference_pit=int(reference_final_stint.start_lap),
            compound_base_delta=COMPOUND_BASE_DELTA,
            pit_loss=DEFAULT_PIT_LOSS,
            total_laps=total_laps,
        )
        _gap = explanation["total_gap"]
        _sign = "快" if _gap < 0 else "慢"
        _cand_compound_label = compound_display.get(_v4p2_compound, _v4p2_compound)
        conclusion_tab.write(
            f"当前赛后重建下，表现最好的候选方案是第 {_v4p2_pit} 圈换{_cand_compound_label}，"
            f"相对 Norris 第 {reference_final_stint.start_lap} 圈换{reference_compound_label}基准，"
            f"模型内总差距约 {_sign}{abs(_gap):.1f} 秒。"
        )
        if explanation["gap_near_zero"]:
            conclusion_tab.write(
                "差距接近零，没有单一主导阶段；这本身也是模型在赛后重建下对这一对策略的判断。"
            )
        else:
            _dom = next(
                p for p in explanation["phases"] if p["name"] == explanation["dominant_phase_name"]
            )
            if _dom["name"] == "两者都已换回干胎后":
                _range = f"第 {_dom['lap_start']} 圈到终点"
            else:
                _range = f"第 {_dom['lap_start']}-{_dom['lap_end']} 圈"
            conclusion_tab.write(
                f"差距主要集中在「{_dom['name']}」（{_range}），"
                "来自模型内逐圈公式项相减，不代表真实车手或车队的动机。"
            )
        detail_expander = conclusion_tab.expander("查看模型内分段明细", expanded=False)
        _rows = []
        for p in explanation["phases"]:
            _rows.append(
                {
                    "阶段": p["name"],
                    "圈数": f"{p['lap_start']}-{p['lap_end']}",
                    "相对基准总差距（秒）": round(p["total_gap"], 2),
                    "光头胎湿地惩罚差": round(p["wet_penalty"], 2),
                    "轮胎衰退差": round(p["degradation"], 2),
                    "轮胎配方基础差": round(p["compound_delta"], 2),
                    "进站损失差": round(p["pit_delta"], 2),
                }
            )
        detail_expander.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
        detail_expander.caption(
            "这些是简化模拟公式的分项相减，帮助复核本模型为什么给出这个差距；"
            "它们不是实测轮胎状态、真实水深，也不解释 McLaren 当时为什么做出策略选择。"
        )

heatmap_view_label = conclusion_tab.radio(
    "热力图视角",
    ["当前赛后重建", "干燥速度 ±2 圈敏感性（取最差结果）"],
    horizontal=True,
)
heatmap_view = "current" if heatmap_view_label == "当前赛后重建" else "worst"
current_fast_alternatives = alternative_summary[alternative_summary["current_delta"] < 0]
if heatmap_view == "current":
    selected_best = alternative_summary.sort_values(["current_delta", "worst_case"]).iloc[0]
    selected_heading = "当前重建下表现最好的候选方案（仅当前重建下）"
else:
    selected_best = robust_best
    selected_heading = "最接近稳健的替代方案"

selected_compound_label = compound_display.get(
    selected_best["compound"],
    selected_best["compound"],
)
selected_scenario_results = scenario_results_for(selected_best)
scenario_delta_lookup = selected_scenario_results.set_index("scenario")["delta"].to_dict()
current_scenario_delta = float(scenario_delta_lookup.get("当前重建", selected_best["median_delta"]))
faster_drying_delta = float(scenario_delta_lookup.get("干燥较快（-2 圈）", selected_best["worst_case"]))
slower_drying_delta = float(scenario_delta_lookup.get("干燥较慢（+2 圈）", selected_best["best_case"]))

if heatmap_view == "current":
    if current_scenario_delta < 0:
        conclusion_tab.success(
            f"先看结论：在当前赛后重建下，第 {int(selected_best['slick_pit'])} 圈换{selected_compound_label}"
            f"是候选方案中表现最好的一个，相对基准{describe_delta(current_scenario_delta)}。"
            f"但当干燥段快两圈时，它会{describe_delta(faster_drying_delta)}，因此这是一项敏感度结果，不是实际策略预测。"
        )
    else:
        conclusion_tab.info(
            "先看结论：在当前赛后重建下，测试窗口内没有候选方案表现得比 Norris 的基准策略更好。"
            "这不代表真实比赛中一定无解，只代表当前简化模型没有找到明确收益。"
        )
elif dominant_alternatives.empty:
    conclusion_tab.info(
        f"先看结论：在干燥速度 ±2 圈的敏感性设定下，没有替代方案在三种设定中都优于真实策略。"
        f"最接近稳健的是第 {int(selected_best['slick_pit'])} 圈换{selected_compound_label}，"
        f"当前重建下{describe_delta(current_scenario_delta)}，干燥较快时{describe_delta(faster_drying_delta)}。"
    )
else:
    conclusion_tab.success(
        f"先看结论：在干燥速度 ±2 圈的敏感性设定下，模型有 {len(dominant_alternatives)} 个候选方案优于基准。"
        f"其中最稳健的是第 {int(selected_best['slick_pit'])} 圈换{selected_compound_label}。"
    )

conclusion_tab.info(
    f"基准：Norris 真实在第 {reference_final_stint.start_lap} 圈换{reference_compound_label}，记为 0.00 秒。"
    f"这里固定他第 {analysis_inter_pit} 圈换半雨胎，只扫描最后一停的候选圈数（第 {pit_window_start}-{pit_window_end} 圈）。"
    + (
        "当前视角：负数代表模型在当前赛后重建下估计更快，正数代表更慢。"
        if heatmap_view == "current"
        else "敏感性视角：每一格取三种干燥速度设定里最慢的那个结果；负数仍代表更快。"
    )
)
conclusion_tab.caption(
    "模型简化说明：这里不建模安全车、DRS、交通影响、车手失误或轮胎温度；所有结果都是赛后重建下的敏感度估算，不是实际策略预测。"
)
transition_lap = reference_final_stint.start_lap
transition_wetness = story_weather_wetness_by_lap.get(transition_lap, 0.0)
transition_inter_age = transition_lap - analysis_inter_pit
transition_inter_wear = intermediate_wear_exposure(
    reference_intermediate_stint,
    transition_lap,
    story_weather_wetness_by_lap,
)
fresh_soft_cost = (
    COMPOUND_BASE_DELTA["SOFT"]
    + transition_wetness * COMPOUND_WET_PENALTY["SOFT"]
)
fresh_intermediate_cost = (
    COMPOUND_BASE_DELTA["INTERMEDIATE"]
    + transition_wetness * COMPOUND_WET_PENALTY["INTERMEDIATE"]
)
used_intermediate_cost = fresh_intermediate_cost + (
    COMPOUND_DEGRADATION["INTERMEDIATE"] * transition_inter_wear
)

model_tab.subheader(f"第 {transition_lap} 圈换软胎，模型到底在比较什么？")
transition_left, transition_middle, transition_right = model_tab.columns(3)
with transition_left:
    st.metric("当圈模型湿滑指数", f"{transition_wetness:.1f}")
    st.caption("这是降雨记录推断出的内部指数，不是官方水深。")
with transition_middle:
    fresh_delta = fresh_soft_cost - fresh_intermediate_cost
    st.metric("新软胎 vs 新半雨胎", f"{fresh_delta:+.2f}s/圈")
    st.caption("正数代表新软胎更慢；此处不含胎龄。")
with transition_right:
    used_delta = fresh_soft_cost - used_intermediate_cost
    st.metric("新软胎 vs 已用半雨胎", f"{used_delta:+.2f}s/圈")
    st.caption(
        f"半雨胎实际跑了 {transition_inter_age} 圈；湿地折算磨损为 {transition_inter_wear:.2f} 圈。"
        "正数代表新软胎更慢。"
    )
model_tab.caption(
    "因此，0.00 秒只是把 Norris 的真实整套策略设为校准基准，不是模型独立证明“第 40 圈新软胎在湿地最快”。"
    f"当前参数中，新软胎与新半雨胎的交叉点约为 {soft_intermediate_crossover:.2f}；"
    "这个小拆解显示的是当前湿地低磨损、赛道转干后加速磨损的半雨胎假设。"
)
with model_tab.expander("模型说明：为什么更早换半雨胎不一定更快？"):
    early_example_lap = max(2, analysis_inter_pit - 3)
    early_example_wetness = story_weather_wetness_by_lap.get(early_example_lap, 0.0)
    st.write(
        f"当前简化模型里，半雨胎在偏干赛道上有基础速度损失。根据轮胎参数，模型湿滑指数大约超过 "
        f"**{intermediate_crossover:.2f}** 后，半雨胎才开始比中胎更划算。"
        f"以第 {early_example_lap} 圈为例，推断湿滑指数约为 **{early_example_wetness:.1f}**，"
        "此时提前换半雨胎仍可能得不偿失。"
    )
    st.caption(
        "因此，提前进站只有在赛道已经足够湿时才可能获益；否则会先承担半雨胎在偏干赛道上的速度损失。"
        "这个交叉点来自当前简化参数，不是真实F1轮胎的官方阈值。"
    )
conclusion_tab.info(
    "上方已经给出固定的稳健性判断；下面的热力图用于继续查看每个假设下的候选方案，"
    "不是第二个相互矛盾的结论。"
)
sensitivity_left, sensitivity_right = conclusion_tab.columns([1.1, 0.9])
with sensitivity_left:
    st.plotly_chart(
        plot_sensitivity_heatmap(
            sensitivity_summary,
            reference_pit=reference_final_stint.start_lap,
            reference_compound=reference_final_stint.compound,
            view=heatmap_view,
        ),
        width="stretch",
    )
with sensitivity_right:
    if heatmap_view == "current":
        st.success(f"当前重建下有 {len(current_fast_alternatives)} 个候选方案优于真实策略。")
    elif dominant_alternatives.empty:
        st.warning("保守结论：没有“稳赢”的替代方案")
    else:
        st.success(f"敏感性结论：有 {len(dominant_alternatives)} 个候选方案在三种干燥速度设定下都优于基准。")
    st.write(
        f"{selected_heading}是"
        f"**第 {int(selected_best['slick_pit'])} 圈换{selected_compound_label}**。"
    )
    st.plotly_chart(plot_scenario_breakdown(selected_scenario_results), width="stretch")
    st.caption(
        "上图展示当前热力图选中候选方案在三种干燥设定下的结果；需要精确数值时，"
        "默认参考上方「最后一停：策略判断」里的固定三情景表。"
        "左边切换视角后，右边推荐的方案也会跟着改变；三种设定都来自同一条重建曲线的干燥段 ±2 圈偏移，不是独立天气预测。"
    )
    with st.expander("查看全部敏感性结果"):
        sensitivity_detail = sensitivity_df.rename(
            columns={
                "scenario": "天气场景",
                "slick_pit": "换回干胎圈数",
                "compound": "最终轮胎",
                "delta": "相对真实策略差距（秒）",
            }
        )
        sensitivity_detail["相对真实策略差距（秒）"] = sensitivity_detail["相对真实策略差距（秒）"].round(2)
        st.dataframe(sensitivity_detail, width="stretch", hide_index=True)

timing_values = sensitivity_summary[
    sensitivity_summary["compound"] == reference_final_stint.compound
]["median_delta"]
compound_values = sensitivity_summary[
    sensitivity_summary["slick_pit"] == reference_final_stint.start_lap
]["median_delta"]
timing_span = float(timing_values.max() - timing_values.min())
compound_span = float(compound_values.max() - compound_values.min())
conclusion_tab.markdown("#### 模型结论")
conclusion_left, conclusion_middle = conclusion_tab.columns(2)
with conclusion_left:
    st.metric("换胎时机影响跨度", f"{timing_span:.2f} 秒")
    st.caption(f"固定使用{reference_compound_label}，比较第 {pit_window_start}-{pit_window_end} 圈进站。")
with conclusion_middle:
    st.metric("轮胎选择影响跨度", f"{compound_span:.2f} 秒")
    st.caption(f"固定第 {reference_final_stint.start_lap} 圈进站，比较软胎、中胎和硬胎。")

if timing_span > compound_span:
    conclusion_tab.info(
        f"在这组候选方案里，换回光头胎的时机比最后选哪一种光头胎更敏感，"
        f"模型跨度约多 {timing_span - compound_span:.2f} 秒。"
    )
else:
    conclusion_tab.info(
        f"在这组候选方案里，最后选哪一种光头胎比早晚一圈进站更敏感，"
        f"模型跨度约多 {compound_span - timing_span:.2f} 秒。"
    )

conclusion_tab.subheader("真实策略参考")
if stints_real.empty:
    conclusion_tab.info("还没有找到真实 stint 数据。")
else:
    driver_lookup = laps[["driver_number", "name_acronym", "full_name"]].dropna().drop_duplicates()
    all_real_strategy = stints_real.merge(driver_lookup, on="driver_number", how="left")
    all_real_strategy = all_real_strategy.dropna(subset=["stint_lap_start", "stint_lap_end"])
    all_real_strategy = all_real_strategy.sort_values(["name_acronym", "stint_number"])
    story_real_strategy = all_real_strategy[
        all_real_strategy["name_acronym"].isin(default_drivers or [story_driver])
    ].copy()
    story_strategy_events = load_strategy_events(RAW_DIR / f"{slug}_race_control.json")
    experiment_reference_drivers = list(dict.fromkeys([selected_driver] + (compare_drivers or [])))
    experiment_real_strategy = all_real_strategy[
        all_real_strategy["name_acronym"].isin(experiment_reference_drivers)
    ].copy()
    experiment_real_strategy["driver_order"] = pd.Categorical(
        experiment_real_strategy["name_acronym"],
        categories=experiment_reference_drivers,
        ordered=True,
    )
    experiment_real_strategy = experiment_real_strategy.sort_values(["driver_order", "stint_number"])
    simulated_driver_label = f"模拟 {selected_driver}"
    simulated_strategy = pd.DataFrame(
        [
            {
                "name_acronym": simulated_driver_label,
                "stint_number": 1,
                "compound": "MEDIUM",
                "stint_lap_start": 1,
                "stint_lap_end": inter_pit - 1,
                "tyre_age_at_start": 0,
            },
            {
                "name_acronym": simulated_driver_label,
                "stint_number": 2,
                "compound": "INTERMEDIATE",
                "stint_lap_start": inter_pit,
                "stint_lap_end": slick_pit - 1,
                "tyre_age_at_start": 0,
            },
            {
                "name_acronym": simulated_driver_label,
                "stint_number": 3,
                "compound": final_compound,
                "stint_lap_start": slick_pit,
                "stint_lap_end": total_laps,
                "tyre_age_at_start": 0,
            },
        ]
    )
    timeline_strategy = pd.concat([experiment_real_strategy, simulated_strategy], ignore_index=True)
    conclusion_tab.plotly_chart(
        plot_strategy_weather_timeline(
            story_real_strategy,
            story_weather_wetness_by_lap,
            story_strategy_events,
            title="比赛天气、轮胎策略与关键赛会信号",
        ),
        width="stretch",
    )
    conclusion_tab.caption("顶部三角可悬停查看赛会事件：红色表示湿地风险提示，蓝色表示干地确认；下方颜色为实际轮胎配方。")
    experiment_tab.subheader("当前实验与所选车手真实策略")
    experiment_tab.caption(
        f"这里固定显示 {selected_driver} 的真实策略和模拟策略；"
        f"对比车手会作为额外参照加入。当前参照：{', '.join(experiment_reference_drivers)}。"
    )
    experiment_tab.plotly_chart(
        plot_stint_timeline(
            timeline_strategy,
            wetness_by_lap=wetness_by_lap,
            title="所选车手真实策略 vs 当前模拟策略（蓝色背景越深表示模型湿滑指数越高）",
        ),
        width="stretch",
    )

    with conclusion_tab.expander("查看原始策略表"):
        table = story_real_strategy.copy()
        table["起始圈"] = table["stint_lap_start"].astype(int)
        table["结束圈"] = table["stint_lap_end"].astype(int)
        table = table.rename(
            columns={
                "name_acronym": "车手",
                "stint_number": "段",
                "compound": "轮胎",
                "tyre_age_at_start": "起始胎龄",
            }
        )
        st.dataframe(
            table[["车手", "段", "轮胎", "起始圈", "结束圈", "起始胎龄"]],
            width="stretch",
            hide_index=True,
        )

experiment_tab.subheader("当前实验结果")
if abs(delta) < 0.05:
    experiment_tab.write(
        f"当前模拟策略基本复刻了 {selected_driver} 的真实策略基准。"
        "你可以调整换胎圈数或最后一段轮胎，观察策略本身带来的相对变化。"
    )
elif delta < 0:
    experiment_tab.write(
        f"这个简化模型估计：当前模拟策略比 {selected_driver} 的真实策略基准快 {-delta:.1f} 秒。"
        "它只能作为粗略信号，不代表真实车队级别的策略预测。"
    )
else:
    experiment_tab.write(
        f"这个简化模型估计：当前模拟策略比 {selected_driver} 的真实策略基准慢 {delta:.1f} 秒。"
        "它只能作为粗略信号，不代表真实车队级别的策略预测。"
    )

model_tab.caption(
    "当前模型包含：真实策略基准校准 + 车手基准节奏 + 轮胎类型差异 + 干胎温和线性衰退 + 半雨胎在赛道转干时加速的磨损 + 进站损失 + 湿滑指数曲线 + 第一圈起步损失。"
)
model_tab.caption(
    "轮胎天气不匹配指数不是事故概率，而是衡量当前轮胎和模型湿滑指数是否匹配；三种光头胎在湿滑里使用同一套主要惩罚，不假设硬胎天生比软胎更适合雨天。"
)
model_tab.caption(
    "赛后重建按每圈结束时的赛道状态取样，避免把进站前的降雨平均进车手出站后实际面对的赛道条件。"
)
model_tab.caption(
    "默认转干速度用第 39-40 圈四位前排车手回到光头胎的真实窗口做校准；圈速残差只用于交叉验证，不会直接改动天气曲线。"
)

model_tab.subheader("天气推断与圈速异常对照")
state_left, state_right = model_tab.columns([1.2, 0.8])
with state_left:
    st.plotly_chart(plot_weather_residual_comparison(track_state_diagnostics), width="stretch")
with state_right:
    st.write(
        "实线蓝线是模型真正使用的赛后重建状态；虚线蓝线是原始降雨记录参考；黄线是多位前排车手共同的异常慢圈信号。"
    )
    st.caption(
        "只有至少三位参考车手同时支持、且降雨曲线已有湿滑信号时，黄线才会校正实线蓝线。它用于赛后重建，不能直接叫作真实水深或实时预测。"
    )
    peak_wetness_value = float(track_state_diagnostics["reconstructed_wetness"].max())
    peak_laps = track_state_diagnostics.loc[
        track_state_diagnostics["reconstructed_wetness"] >= peak_wetness_value - 0.01,
        "lap_number",
    ]
    peak_window = f"第 {int(peak_laps.min())}-{int(peak_laps.max())} 圈"
    st.metric("天气推断峰值区间", peak_window, f"{peak_wetness_value:.2f} 湿滑指数")

model_tab.subheader("模型赛道湿滑状态如何影响圈速")
rain_left, rain_right = model_tab.columns([1.2, 0.8])
with rain_left:
    st.plotly_chart(plot_wetness_penalty(wetness_penalty_df), width="stretch")
with rain_right:
    current_wet_penalty = float(simulation.laps["wet_penalty"].sum())
    reference_wet_penalty = float(reference_simulation.laps["wet_penalty"].sum())
    st.metric("当前策略累计雨势惩罚", f"{current_wet_penalty:.1f}s", f"{current_wet_penalty - reference_wet_penalty:+.1f}s 相比真实策略")
    st.caption(
        "模型赛道湿滑状态会按轮胎类别换算成圈速惩罚：三种光头胎在湿地里都承受相同的主要惩罚，半雨胎惩罚更小。"
        "当前默认曲线来自 OpenF1 降雨记录与多人圈速的赛后重建，不等同于真实水膜厚度或官方降雨量。"
    )

model_tab.subheader("真实圈速残差诊断")
residual_drivers = sorted(set((compare_drivers or track_state_drivers) + [selected_driver]))
pace_residuals = estimate_pace_residuals(laps, residual_drivers)
residual_left, residual_right = model_tab.columns([1.2, 0.8])
with residual_left:
    st.plotly_chart(plot_pace_residuals(pace_residuals, selected_driver), width="stretch")
with residual_right:
    st.caption(
        "残差 = 真实单圈 - 干地模型预期。它可以帮助我们发现模型没解释掉的部分，"
        "比如雨势、交通、轮胎状态、车手失误或安全车影响。"
    )
    driver_residuals = pace_residuals[pace_residuals["name_acronym"] == selected_driver] if not pace_residuals.empty else pd.DataFrame()
    final_stint_residuals = driver_residuals[driver_residuals["lap_number"] >= slick_pit] if not driver_residuals.empty else pd.DataFrame()
    if not final_stint_residuals.empty:
        early_final = float(final_stint_residuals.head(3)["pace_residual"].mean())
        late_final = float(final_stint_residuals.tail(3)["pace_residual"].mean())
        st.metric("最后一段残差变化", f"{late_final - early_final:+.2f}s", "末三圈 - 前三圈")
    st.caption(
        "单个车手的残差不会进入天气模型；只有至少三位参考车手共同的残差信号才会参与赛后重建。"
    )

with model_tab.expander("模型边界说明：当前原型暂时没建模的因素"):
    st.write(
        "这一栏不是报错，而是告诉读者这个原型的边界。当前暂时没有建模安全车、虚拟安全车、"
        "交通/脏空气、DRS、车手失误、锁死/冲出赛道、每套胎的新旧程度差异，"
        "以及真实天气的非线性快速变化。"
    )

with model_tab.expander("查看当前模拟逐圈明细"):
    detail_table = display_laps.rename(
        columns={
            "lap_number": "圈数",
            "compound": "轮胎",
            "tyre_age": "胎龄",
            "wear_exposure": "模型有效磨损圈数",
            "degradation": "轮胎衰退损失（秒）",
            "wetness": "模型湿滑指数",
            "wet_penalty": "雨势惩罚（秒）",
            "weather_delta": "天气相关损失（秒）",
            "risk_score": "轮胎天气不匹配指数",
            "simulated_lap_time": "校准后模拟圈速（秒）",
            "pit_lap": "进站圈",
        }
    )
    st.dataframe(detail_table, width="stretch", hide_index=True)


def format_transition_lap(value: object) -> str:
    return "无转换事件" if pd.isna(value) else f"第 {int(value)} 圈"


def format_window_start(value: object, actual_transition: object, missing_label: str = "未识别") -> str:
    if pd.isna(actual_transition):
        return "不适用（对照）"
    return missing_label if pd.isna(value) else f"第 {int(value)} 圈"


def compact_validation_outcome(row: pd.Series) -> str:
    if pd.isna(row.get("actual_pit_lap")) and pd.isna(row.get("actual_transition_lap")):
        return "对照：未生成窗口"
    error = row.get("decision_time_error_laps")
    if pd.notna(error) and abs(float(error)) <= 2:
        return "公开信号落在真实进站前后 2 圈内"
    return "需要人工复核"


def format_optional_lap(value: object, missing: str = "—") -> str:
    return missing if pd.isna(value) else f"第 {int(value)} 圈"


def case_boundary_note(case_id: str, row: pd.Series) -> str:
    """Plain-language model boundary for the selected exploratory case."""
    if case_id == "montreal_2024":
        return (
            "本项目只能核对公开信号离真实进站服务有多远，"
            "不能解释车队为什么选择在第 45 圈进站；"
            "交通、位置、轮胎温度、对手策略和完整战术都不在模型里。"
        )
    if case_id == "zandvoort_2023":
        return (
            "当前验证只看开局第一次湿转干：第 11 圈进站 / 第 12 圈软胎发生在第一段雨后恢复期；"
            "后续再次降雨是另一个阶段，不是“下雨前就换干胎”的反例。"
        )
    if case_id == "silverstone_2024":
        return (
            "图中绿色标记区分进站服务圈与首个光头胎圈；"
            "模型信号只表示公开证据何时允许开始观察换胎，不是进站命令。"
        )
    if pd.isna(row.get("actual_pit_lap")) and pd.isna(row.get("first_slick_lap")):
        return "对照案例：没有湿胎转光头胎事件，模型应保持沉默。"
    return "模型信号用于解释公开赛道状态证据，不是车队应立刻进站的指令。"


def research_section(title: str) -> None:
    """Render a clear section heading inside the cross-race validation tab."""
    research_tab.markdown(f"### {title}")


try:
    v2_report = load_v2_validation()
except CaseDataUnavailableError as error:
    research_tab.error(f"无法加载公开信号审计数据：{error}")
    st.stop()

challenge_set = load_v2_challenge_set()
fastf1_evaluation = load_v2_fastf1_evaluation()
v2_metrics = build_validation_metrics(v2_report)

challenge_errors = (
    pd.to_numeric(challenge_set["public_signal_error_laps"], errors="coerce").abs()
    if not challenge_set.empty
    else pd.Series(dtype=float)
)
challenge_within_two = int((challenge_errors <= 2).sum()) if not challenge_errors.empty else 0
challenge_mae = float(challenge_errors.mean()) if not challenge_errors.empty else 0.0
sustained_events = (
    challenge_set[challenge_set["event_class"] == "sustained_wet_to_dry"]
    if not challenge_set.empty and "event_class" in challenge_set.columns
    else pd.DataFrame()
)
sustained_errors = (
    pd.to_numeric(sustained_events["public_signal_error_laps"], errors="coerce").abs()
    if not sustained_events.empty
    else pd.Series(dtype=float)
)
sustained_mae = float(sustained_errors.mean()) if not sustained_errors.empty else 0.0

issued_fastf1 = (
    fastf1_evaluation[fastf1_evaluation["state_machine_outcome"] == "signal"]
    if not fastf1_evaluation.empty
    else pd.DataFrame()
)
issued_fastf1_errors = (
    pd.to_numeric(issued_fastf1["public_signal_error_laps"], errors="coerce").abs()
    if not issued_fastf1.empty
    else pd.Series(dtype=float)
)
fastf1_within_two = int((issued_fastf1_errors <= 2).sum()) if not issued_fastf1_errors.empty else 0
fastf1_issued_mae = float(issued_fastf1_errors.mean()) if not issued_fastf1_errors.empty else 0.0
fastf1_abstentions = (
    len(fastf1_evaluation) - len(issued_fastf1) if not fastf1_evaluation.empty else 0
)


# ===========================================================================
# V4.3: Research Boundary — "Public Signals Are Not Enough for Per-Driver
# Pit-Stop Prediction"
# ===========================================================================

# ===========================================================================
# 研究边界：公开信号不足以做逐车手换胎预测
# ===========================================================================
research_section("研究边界：公开信号不足以做逐车手换胎预测")

research_tab.write(
    "这是一项跨比赛公开信号尝试，**没有支持**稳定预测每位车手何时该从湿胎换回光头胎；"
    "它不是 Silverstone 模拟器的验证，也不是实时进站建议。"
)

# --- 173 / 27 / 146 覆盖范围（从五赛季审计产物只读）---
_fsa = load_five_season_audit_cached()
if _fsa is None:
    research_tab.warning(
        "暂未找到五赛季回顾审计产物（reports/results/v3_fastf1_events_2021_2025.csv），"
        "研究边界区块无法显示覆盖数字与证据链；下方复核路径和其余页面内容不受影响。"
    )
    _v4p3_has_audit = False
else:
    _v4p3_has_audit = True
    _fs_summary = _fsa["summary"]
    _fs_m1, _fs_m2, _fs_m3 = research_tab.columns(3)
    _fs_m1.metric("湿转干事件", _fs_summary["events"])
    _fs_m2.metric("发出公开信号", _fs_summary["signals"])
    _fs_m3.metric("明确弃权", _fs_summary["abstentions"])
    research_tab.caption(
        "这 173 / 27 / 146 是车手级湿转干转场记录，不是比赛场次，也不是准确率。"
        "弃权 = 规则在条件复杂时拒绝输出车手级信号；不是车手退赛，也不表示当时没有下雨。"
    )

    # --- 三条简短证据链（从已有五赛季产物只读推导，不手写新口径）---
    # 信号在比赛中的分布：从原始 CSV 读取（与 load_five_season_audit 同一数据源）。
    import pandas as _v4p3_pd
    _v4p3_raw = _v4p3_pd.read_csv(
        Path(__file__).resolve().parents[1]
        / "reports" / "results" / "v3_fastf1_events_2021_2025.csv"
    )
    _v4p3_sig = _v4p3_raw[
        _v4p3_raw["state_machine_outcome"]
        .astype(str).str.strip().str.lower()
        .eq("signal")
    ]
    _v4p3_by_race = _v4p3_sig.groupby(["season", "race"]).size()
    _v4p3_race_lines = []
    for (_yr, _race), _cnt in _v4p3_by_race.items():
        _v4p3_race_lines.append(f"{_yr} {_race}（{_cnt}）")

    research_tab.write(
        "**证据一：** 五年里大多数记录属于复杂情况（起步/重启、赛会控制相邻、多阶段降雨等），"
        "因此规则选择弃权而非强行判断。"
    )
    research_tab.write(
        "**证据二：** 27 次信号全部集中在两个单阶段降雨比赛："
        + "\u3001".join(_v4p3_race_lines)
        + "\u3002不能据此声称跨比赛泛化。"
    )
    research_tab.write(
        "**证据三：** 复杂雨战的起步/重启、赛会控制相邻、多阶段降雨等条件，"
        "会让公开信号不足以给出逐车手换胎圈。"
    )

    research_tab.info(
        "因此，项目核心保留为 **2024 银石的透明赛后策略沙盒**；"
        "公开信号审计只提供\u201c哪里不该相信简单规则\u201d的证据边界，"
        "而不是第二个相互矛盾的跨比赛预测声明。"
    )

# ===========================================================================
# 复核路径 1：查看五赛季审计表与口径
# ===========================================================================
exp_audit = research_tab.expander("查看五赛季审计表与口径", expanded=False)
if _v4p3_has_audit:
    _fs_rows = [
        {
            "赛季": r["season"],
            "研究位置": r["research_position"],
            "湿转干事件": r["events"],
            "发出信号": r["signals"],
            "弃权": r["abstentions"],
            "公开进站锚点": r["pit_service"],
            "信号误差": r["mae_label"],
        }
        for r in _fsa["seasons"]
    ]
    exp_audit.dataframe(pd.DataFrame(_fs_rows), width="stretch", hide_index=True)
    exp_audit.caption(
        "五年中共有 153 行使用独立 FastF1 进站服务锚点；2025 的 20 行没有得到独立进站记录，"
        "诚实标记为\u300cfirst-slick 回退\u300d，不能假装是进站服务圈。"
    )
    exp_audit.info(
        "2022 是最干净的新增扩展，因为是在查看结果前加入；该年没有发出逐车手信号，说明规则选择不假装确定。\n"
        "2023 的低误差不能当成一般化证明，因为该赛季在规则形成期已经接触过。\n"
        "现有单独的 2025 OpenF1 1/53 结果是另一套协议，不能与本五赛季 FastF1 表合并、平均或计算总准确率。\n"
        "五赛季审计是赛后回顾，不是实时策略建议，也不验证 Silverstone 模拟器的物理模型。"
    )
else:
    exp_audit.warning("暂未找到五赛季回顾审计产物，该路径无法展开。")

# ===========================================================================
# 复核路径 2：查看规则形成与失败诊断
# ===========================================================================
exp_diag = research_tab.expander("查看规则形成与失败诊断", expanded=False)

# --- 2a. 公开信号审计规则（四步）---
exp_diag.markdown("### 公开信号审计规则")
exp_diag.write(
    "下面的四步是本项目公开信号审计规则的完整运转过程，不是某一场比赛的结论，"
    "也不是实时轮胎决策模型。它先读公开信号，再判断是否该发信号，最后只在赛后对照真实进站动作。"
)
_chain_steps = [
    ("1. 审计看什么", "公开降雨记录，以及赛会/赛道条件信号（如 DRS 恢复、正常抓地确认）。"),
    ("2. 审计怎么判断", "只在公开证据构成一段连贯的湿转干恢复窗口时才发信号；证据不足或混乱时拒绝强行判断。"),
    (
        "3. 审计输出什么",
        "一个\u201c开始考虑换回光头胎\u201d的保守公开信号点；证据不足时输出\u201c不判断\u201d。",
    ),
    ("4. 如何核对", "只在赛后，把信号点与真实进站服务圈对照；真实进站动作是核对锚点，不是这条公开信号规则的输入。"),
]
_chain_cols = exp_diag.columns(4)
for _col, (_title, _body) in zip(_chain_cols, _chain_steps):
    _col.markdown(f"**{_title}**")
    _col.caption(_body)

# --- 2b. 模型在找什么 ---
exp_diag.markdown("### 模型在找什么")
exp_diag.write(
    "赛道从湿地逐渐变干时，车手不会只看\u201c有没有下雨\u201d就决定换回光头胎。"
    "本页检查的是：用公开数据里能看到的线索，能不能标出一个**值得开始观察换胎**的区间。"
)
exp_diag.write(
    "它输出的不是唯一正确的进站圈，也不模拟车队完整决策系统。"
    "核对时使用的\u201c真实进站服务圈\u201d，是公开 pit 记录里的进站服务；"
    "\u201c首个光头胎圈\u201d是逐圈数据里新胎第一次出现，通常在 pit-out lap。"
    "两者都只用于事后对照，不是模型输入的答案。"
)
exp_diag.caption(
    "本页分开两类信息：一类是比赛当时已经公开可见的信号；另一类只能在赛后用完整数据复盘。"
)

# --- 2c. 最终发现 ---
exp_diag.markdown("### 最终发现")
exp_diag.info(
    "简单公开信号规则在少量探索案例中表现不错，但放到更大的压力测试中没有稳定泛化。"
    "加入弃权机制后，模型会在复杂事件中拒绝强行判断，但已经发出的信号仍然存在明显误差。"
    "项目的价值是展示规则什么时候可能有用、什么时候不应该被相信，而不是预测一个精确进站圈。"
)
exp_diag.write(
    "**已经证明的：** 可以诚实记录成功、失败和\u201c证据不足时不判断\u201d。"
    " **没有证明的：** 不能稳定预测每位车手的精确换胎圈，也不能当作实时进站指令。"
)

# --- 2d. 规则形成过程中的两个诊断（不是最终五年结论）---
exp_diag.markdown("### 规则形成过程中的两个诊断（不是最终五年结论）")
exp_diag.write(
    "下面三个阶段是研究的递进：探索案例用于形成假设；而 2025 OpenF1 压力测试与 2021 FastF1 弃权扫描，"
    "是规则形成过程中的两个诊断\u2014\u2014前者暴露简单规则的失败（1/53），后者促使项目采用"
    "\u201c允许弃权\u201d的状态机。"
    "两者数据来源和评价协议不同，只用于展示研究过程与失败边界，"
    "既不能彼此相加，也不能与上方完整五赛季审计合并或替代它，更不构成任何单一准确率。"
)
stage_summary = pd.DataFrame(
    [
        {
            "阶段": "探索案例",
            "结果": (
                f"{v2_metrics['public_within_two']}/{v2_metrics['wet_to_dry_cases']} 个湿转干案例"
                "落在真实进站服务前后 2 圈内"
            ),
            "读者应理解的含义": "用来形成假设，不是正式准确率",
        },
        {
            "阶段": "2025 压力测试",
            "结果": (
                f"简单规则只有 {challenge_within_two}/{len(challenge_set)} "
                "落在前后 2 圈内"
                if not challenge_set.empty
                else "暂无 2025 压力测试数据"
            ),
            "读者应理解的含义": "原来的简单规则未能泛化",
        },
        {
            "阶段": "2021 弃权评估",
            "结果": (
                f"{len(issued_fastf1)}/{len(fastf1_evaluation)} 发出信号，"
                f"{fastf1_within_two}/{len(issued_fastf1)} 在前后 2 圈内，"
                f"{fastf1_abstentions}/{len(fastf1_evaluation)} 弃权"
                if not fastf1_evaluation.empty
                else "暂无 2021 评估数据"
            ),
            "读者应理解的含义": "弃权减少强行判断，但没有解决已发信号的全部误差",
        },
    ]
)
exp_diag.dataframe(stage_summary, width="stretch", hide_index=True)
exp_diag.caption(
    "\u201c前后 2 圈\u201d是宽松的核对标准：模型给出的观察窗口（保守公开信号点）"
    "与真实进站服务圈相差不超过两圈。"
    "它衡量的是观察信号是否靠近真实进站动作，不是要求模型点中唯一进站圈。"
)
with exp_diag.expander("为什么一个结果来自 2025，另一个来自 2021？", expanded=True):
    st.markdown(
        """
1. **2025** 用来压力测试并否定已经冻结的简单公开信号规则（OpenF1，53 个车手级事件）。
2. 允许弃权的状态机是在看过这次失败之后才形成的，因此 **不能再拿 2025 当\u201c未见成绩\u201d**。
3. 当时 OpenF1 可用目录里没有新的合格未见样本，项目才引入 **FastF1 作为独立历史来源**。
4. **2021** 是预先冻结的 FastF1 全赛季扫描，不是看完结果后挑出\u201c好看\u201d的年份；它只提供有限的独立来源覆盖率证据，不是终极跨年份泛化证明。
5. 所以页面同时出现 2025 与 2021，是**研究阶段递进**，不是 2025 对 2021 的性能对比，也不是统一准确率。
"""
    )

# --- 2e. 两个独立实验：2025 压力测试与 2021 弃权评估 ---
exp_diag.markdown("### 两个独立实验：2025 压力测试与 2021 弃权评估")
exp_diag.write(
    "下面是两个独立实验。它们使用不同规则、不同数据来源，必须分开阅读，不能合并成一个总准确率。"
)

exp_diag.markdown("#### 2025 简单规则压力测试")
if challenge_set.empty:
    exp_diag.info("暂无 2025 压力测试数据。")
else:
    exp_diag.write(
        "压力测试的意思是：先冻结探索阶段形成的简单规则，再用更多新事件检查它能否推广。"
        "这里的事件是**车手级**湿胎转光头胎记录，不等于同等数量的独立比赛。"
    )
    challenge_left, challenge_middle, challenge_right = exp_diag.columns(3)
    challenge_left.metric("车手级事件", len(challenge_set))
    challenge_middle.metric("落在前后 2 圈内", f"{challenge_within_two}/{len(challenge_set)}")
    challenge_right.metric("平均绝对误差", f"{challenge_mae:.2f} 圈")
    exp_diag.write(
        f"结果很明确：{len(challenge_set)} 个事件里只有 {challenge_within_two} 个落在真实转换前后两圈内，"
        f"平均绝对误差约 {challenge_mae:.2f} 圈。这是失败诊断，不是新的预测功能。"
    )
    exp_diag.info(
        "这条 1/53 结果到底在说什么：\u2460 它**证伪**的是\u2014\u2014把这套简单公开信号规则当成"
        "通用的\u201c逐车手湿转干换胎预测器\u201d并不可靠；\u2461 审计仍然有用：它告诉我们公开确认常常"
        "过晚或模糊，从而**防止**把银石模拟器或这条规则包装成通用实时决策系统；"
        "\u2462 它**没有证伪**独立的 V1 单场假设模拟器，也**没有证明**那个模拟器的物理假设正确。"
        "模拟器与审计是分开的两件事，失败的是\u201c通用预测\u201d这一声称，不是整个项目。"
    )
    exp_diag.write(
        "为什么简单规则会失效：同一套半雨胎可能跨越多次降雨、起步/重启，或安全车等赛会控制阶段。"
        "早期的 DRS 或正常抓地提示，到真正换胎时往往已经过期。"
        f"即使只看持续湿胎分段，平均误差仍有 {sustained_mae:.2f} 圈，"
        "所以不是靠排除起跑短分段就能解决。"
    )
    exp_diag.caption(
        "\u201c覆盖率\u201d此时几乎不是问题\u2014\u2014简单规则会对大多数事件给出一个窗口；"
        "真正的问题是这些窗口经常偏得太远。"
    )

exp_diag.markdown("#### 2021 弃权状态机评估")
if fastf1_evaluation.empty:
    exp_diag.info("暂无 2021 弃权状态机评估数据。")
else:
    exp_diag.write(
        "面对复杂事件，模型需要一组明确规则来决定\u201c发信号还是不判断\u201d。"
        "这组按事件所处阶段做决定的规则，就是**状态机**。"
    )
    exp_diag.write(
        "当证据不足或条件过于混乱时，模型会选择不判断，而不是硬给一个看似精确的换胎圈。"
        "这种\u201c模型选择不判断\u201d的结果叫作**弃权**。弃权不等于预测正确，只表示拒绝强行输出。"
    )
    exp_diag.info(
        "2021 在这里用于展示规则为何加入\u300c弃权\u300d；它不是新的五赛季结果，也不构成第二套准确率。"
    )
    detail_expander = exp_diag.expander(
        "查看 2021 初始状态机扫描细节（与上方五赛季表中的 2021 是同一批事件）",
        expanded=False,
    )
    fastf1_left, fastf1_middle, fastf1_right = detail_expander.columns(3)
    fastf1_left.metric("车手级事件", len(fastf1_evaluation))
    fastf1_middle.metric("发出信号 / 弃权", f"{len(issued_fastf1)} / {fastf1_abstentions}")
    fastf1_right.metric(
        "已发信号落在前后 2 圈内",
        f"{fastf1_within_two}/{len(issued_fastf1)}" if len(issued_fastf1) else "不适用",
    )
    detail_expander.write(
        f"2021 年历史扫描保留了全部 {len(fastf1_evaluation)} 个事件："
        f"状态机发出 {len(issued_fastf1)} 个信号，对 {fastf1_abstentions} 个复杂事件弃权；"
        f"已发信号中有 {fastf1_within_two} 个落在前后两圈内，"
        f"已发信号平均绝对误差约 {fastf1_issued_mae:.2f} 圈。"
    )
    detail_expander.write(
        "这里必须同时看两件事：**覆盖率**（在多少事件上愿意发信号）和**已发信号误差**。"
        "减少预测次数可以避免乱猜，但不能包装成准确率提升。"
    )
    detail_expander.caption(
        "数据来源是 FastF1 历史赛季，与上面 2025 OpenF1 压力测试不是同一套评价协议。"
    )

# ===========================================================================
# 复核路径 3：查看案例图、原始字段与研究文档
# ===========================================================================
exp_case = research_tab.expander("查看案例图、原始字段与研究文档", expanded=False)

# --- 3a. 先通过一场比赛理解 ---
exp_case.markdown("### 先通过一场比赛理解")
exp_case.write(
    "在看整体数字之前，先选一场比赛看证据长什么样。"
    "图里会同时出现比赛当时可见的信息、赛后才能完成的复盘，以及真实进站动作。"
)

exp_case.markdown("**读图前先看这些标记**")
exp_case.markdown(
    """
- **蓝色实线**：只由公开降雨记录生成的基础赛道状态代理；不是实测水深。
- **浅蓝带**：公开证据允许开始考虑换回光头胎的区间；表示不同公开线索之间的时间范围。
- **蓝色点线**：浅蓝带中的**保守公开信号点**（较晚边界），也是当前评价使用的核对点。
- **橙色实线**：赛后使用完整多人圈速残差得到的重建；比赛当时不可用。
- **橙色点线**：赛后重建给出的恢复窗口；只用于复盘。
- **绿色实线**：真实进站服务圈（pit service lap），策略时间核对的锚点。
- **绿色虚线**：首个光头胎圈（first slick lap），逐圈记录里新胎第一次出现；通常是 pit-out lap。
"""
)
exp_case.caption(
    "模型信号解释的是赛道状态证据，不是命令车队立刻进站。"
    "若公开数据没有形成稳定窗口，会显示未识别：表示证据不足，不等于证明当时没有雨。"
)

case_roles = {
    "silverstone_2024": "湿转干主案例",
    "montreal_2024": "多次换胎压力测试",
    "zandvoort_2023": "多阶段天气",
    "zandvoort_2024_control": "干地对照",
    "sao_paulo_2024_control": "湿地对照",
}
case_labels = {
    case.case_id: f"{case.title} \u00b7 {case.focal_driver}（{case_roles.get(case.case_id, '\u9a8c\u8bc1\u6848\u4f8b')}）"
    for case in V2_CASES
}
selected_case_id = exp_case.selectbox(
    "选择验证案例",
    list(case_labels),
    format_func=lambda case_id: case_labels[case_id],
    key="v2_case_selector",
)
selected_case = next(case for case in V2_CASES if case.case_id == selected_case_id)
selected_case_report = v2_report[v2_report["case_id"] == selected_case_id].iloc[0]
selected_case_state, selected_case_transitions = load_v2_case_track_state(selected_case_id)

pit_lap = selected_case_report.get("actual_pit_lap")
first_slick = selected_case_report.get("first_slick_lap")
if pd.isna(first_slick):
    first_slick = selected_case_report.get("actual_transition_lap")
to_compound = selected_case_report.get("to_compound")
has_switch = pd.notna(pit_lap) or pd.notna(first_slick)
anchor_lap = (
    int(pit_lap)
    if pd.notna(pit_lap)
    else int(first_slick)
    if pd.notna(first_slick)
    else None
)

exp_case.markdown("**当前验证的事件**")
if not has_switch:
    exp_case.info("对照案例：没有湿胎转光头胎事件，因此没有 pit / first slick 锚点。")
else:
    compound_text = "光头胎" if pd.isna(to_compound) else str(to_compound)
    from_compound = "\u6e7f\u80ce/\u534a\u96e8\u80ce"
    if selected_case_transitions:
        from_compound = str(selected_case_transitions[0].get("from_compound", from_compound))
    summary_cols = exp_case.columns(4)
    summary_cols[0].metric("\u8f66\u624b\u4e0e\u8f6c\u6362", f"{selected_case.focal_driver} \u2192 {compound_text}")
    summary_cols[1].metric("\u771f\u5b9e\u8fdb\u7ad9\u670d\u52a1\u5708", format_optional_lap(pit_lap, "\u65e0\u72ec\u7acb pit \u951a\u70b9"))
    summary_cols[2].metric("\u9996\u4e2a\u5149\u5934\u80ce\u5708", format_optional_lap(first_slick))
    summary_cols[3].metric(
        "\u672c\u6b21\u6bd4\u8f83\u951a\u70b9",
        "\u8fdb\u7ad9\u670d\u52a1\u5708" if pd.notna(pit_lap) else "\u9996\u4e2a\u5149\u5934\u80ce\u5708\uff08\u65e0 pit\uff09",
    )
    if selected_case_id == "zandvoort_2023":
        phase_text = "\u5f00\u5c40\u7b2c\u4e00\u6bb5\u6e7f\u8f6c\u5e72\u6062\u590d\uff08\u4e0d\u662f\u5168\u573a\u540e\u7eed\u5929\u6c14\uff09"
    elif selected_case_id == "montreal_2024":
        phase_text = "\u540e\u6bb5\u6e7f\u80ce/\u534a\u96e8\u80ce\u8f6c\u56de\u5149\u5934\u80ce\u7684\u90a3\u4e00\u6b21\u8f6c\u6362"
    elif selected_case_id == "silverstone_2024":
        phase_text = "\u540e\u6bb5\u6700\u7ec8\u56de\u5230\u5149\u5934\u80ce"
    else:
        phase_text = "\u672c\u573a\u89c2\u5bdf\u7a97\u53e3\u5185\u7684\u6e7f\u8f6c\u5e72\uff08\u82e5\u5b58\u5728\uff09"
    exp_case.caption(f"\u5f53\u524d\u9a8c\u8bc1\u5173\u6ce8\u7684\u5929\u6c14\u9636\u6bb5\uff1a{phase_text}\u3002\u4ece {from_compound} \u6362\u5230 {compound_text}\u3002")
    exp_case.warning(case_boundary_note(selected_case_id, selected_case_report))

# Field wave + later rain for multi-phase clarity
field_wave = None
later_rain = None
field_pit_laps: list[int] = []
if has_switch and anchor_lap is not None:
    wet_start = selected_case_report.get("wet_stint_start")
    phase_start = int(wet_start) if pd.notna(wet_start) else max(1, anchor_lap - 10)
    phase_end = anchor_lap + 2
    try:
        field_wave = field_wet_to_slick_wave(selected_case, phase_start, phase_end + 5)
        field_pit_laps = list(field_wave.get("pit_laps") or [])
    except Exception:
        field_wave = None
        field_pit_laps = []
    total_case_laps = int(selected_case_state["lap_number"].max())
    later_rain = later_rainfall_phase_start(selected_case, anchor_lap, total_case_laps)

if field_wave and field_wave.get("event_count"):
    pit_min = field_wave.get("pit_lap_min")
    pit_max = field_wave.get("pit_lap_max")
    wave_bits = [
        f"同一阶段全场湿转干进站波次约第 {pit_min}-{pit_max} 圈"
        if pit_min is not None and pit_max is not None
        else f"同一阶段记录到 {field_wave['event_count']} 次湿转干"
    ]
    if selected_case_id == "montreal_2024":
        wave_bits.append("VER 第 45 圈、NOR 第 47 圈处于最后一批转换中；DRS 约第 40 圈恢复，正常抓地确认在第 47 圈。")
    elif selected_case_id == "zandvoort_2023":
        wave_bits.append("多位车手约在第 9-12 圈回到光头胎，VER 第 11 圈处于中间；DRS 第 10 圈恢复。")
    exp_case.caption(" ".join(wave_bits) + " 这只说明焦点车手位于真实场上波次之中，不推断车队动机。")

view_mode = "全场"
if has_switch and anchor_lap is not None:
    view_mode = exp_case.radio(
        "图表范围",
        ["聚焦当前事件", "查看全场"],
        horizontal=True,
        key=f"v2_case_view_{selected_case_id}",
        help="多阶段比赛建议先看聚焦视图，避免把后续降雨误读为当前事件背景。",
    )

focus_start = focus_end = None
if has_switch and anchor_lap is not None and view_mode == "聚焦当前事件":
    focus_start = max(1, anchor_lap - 10)
    focus_end = min(int(selected_case_state["lap_number"].max()), anchor_lap + 10)
    if later_rain is not None:
        focus_end = min(focus_end, max(anchor_lap + 3, int(later_rain) - 1))

decision_start = selected_case_report.get("decision_time_window_start")
decision_end = selected_case_report.get("decision_time_window_end")
retrospective_start = selected_case_report.get("retrospective_window_start")

# --- 信号审计记录 ---
_model_trace = build_model_trace(selected_case_report)
exp_case.markdown("**本案例的信号审计记录**")
for _label, _text in [
    ("公开输入", _model_trace["inputs"]),
    ("规则判断", _model_trace["rule"]),
    ("审计输出", _model_trace["output"]),
    ("赛后核对", _model_trace["check"]),
]:
    exp_case.markdown(f"- **{_label}：** {_text}")

exp_case.plotly_chart(
    plot_v2_track_state_evidence(
        selected_case_state,
        decision_window_start=None if pd.isna(decision_start) else int(decision_start),
        retrospective_window_start=None if pd.isna(retrospective_start) else int(retrospective_start),
        decision_window_end=None if pd.isna(decision_end) else int(decision_end),
        actual_pit_lap=None if pd.isna(pit_lap) else int(pit_lap),
        first_slick_lap=None if pd.isna(first_slick) else int(first_slick),
        to_compound=None if pd.isna(to_compound) else str(to_compound),
        focus_start=focus_start,
        focus_end=focus_end,
        later_rain_start=None if later_rain is None or view_mode == "聚焦当前事件" else int(later_rain),
        field_pit_laps=field_pit_laps or None,
    ),
    width="stretch",
)
if later_rain is not None and view_mode == "查看全场":
    exp_case.caption(
        f"灰色区域从约第 {int(later_rain)} 圈起表示后续独立降雨阶段，不属于当前验证事件。"
    )
exp_case.caption(
    "读图顺序：蓝实线（仅降雨代理）\u2192 浅蓝带与蓝色点线（公开信号区间与保守信号点）\u2192 "
    "橙实线/橙点线（赛后重建）\u2192 绿色实线（进站服务）与绿色虚线（首个光头胎圈）。"
)
exp_case.write(f"**本场证据说明：** {selected_case_report['evidence_note']}")
pit_source = selected_case_report.get("pit_anchor_source", "")
if isinstance(pit_source, str) and pit_source and pit_source != "pit":
    exp_case.caption(f"进站锚点状态：{pit_source}。")
exp_case.caption(f"证据类型：{selected_case_report.get('failure_type', '\u672a\u5206\u7c7b')}。")

with exp_case.expander("这些比赛为什么被选中"):
    st.dataframe(
        pd.DataFrame(
            {
                "比赛": [case.title for case in V2_CASES],
                "用途": [case.purpose for case in V2_CASES],
                "需要注意": [case.review_note for case in V2_CASES],
            }
        ),
        width="stretch",
        hide_index=True,
    )

# --- 3b. 研究明细（原 Section 6，内容全部保留）---
exp_case.markdown("### 研究明细")
exp_case.caption("以下内容保留完整研究证据，默认收起，不影响主叙事阅读。")

with exp_case.expander("五场探索案例表"):
    st.write(
        "这五场比赛用于形成假设，并检查干地/湿地对照是否被误触发。"
        "误差相对真实进站服务圈计算；首个光头胎圈只用于解释数据时间轴。"
        "探索集结果不能当成正式准确率。"
    )
    v2_overview = pd.DataFrame(
        {
            "比赛": v2_report["race"],
            "车手": v2_report["driver"],
            "真实进站服务": v2_report.get("actual_pit_lap", pd.Series([pd.NA] * len(v2_report))).map(
                lambda value: format_optional_lap(value, "无 / 对照")
            ),
            "首个光头胎圈": v2_report.get("first_slick_lap", v2_report["actual_transition_lap"]).map(
                lambda value: format_optional_lap(value, "无 / 对照")
            ),
            "公开信号窗口": v2_report.apply(
                lambda row: (
                    "不适用（对照）"
                    if pd.isna(row.get("actual_pit_lap")) and pd.isna(row.get("actual_transition_lap"))
                    else row["decision_time_window_label"]
                ),
                axis=1,
            ),
            "结果": v2_report.apply(compact_validation_outcome, axis=1),
        }
    )
    st.dataframe(v2_overview, width="stretch", hide_index=True)
    overview_left, overview_right = st.columns(2)
    overview_left.metric(
        "湿转干探索案例",
        f"{v2_metrics['public_within_two']}/{v2_metrics['wet_to_dry_cases']} 落在进站前后 2 圈内",
    )
    overview_right.metric(
        "对照未被误触发",
        f"{v2_metrics['controls_silent']}/{v2_metrics['control_cases']}",
    )

with exp_case.expander("完整验证字段"):
    v2_display_columns = [
        "race",
        "driver",
        "actual_pit_lap",
        "first_slick_lap",
        "actual_transition_lap",
        "to_compound",
        "rainfall_only_window_start",
        "decision_time_window_label",
        "retrospective_window_start",
        "classification",
        "failure_type",
    ]
    available = [column for column in v2_display_columns if column in v2_report.columns]
    v2_display = v2_report[available].rename(
        columns={
            "race": "\u6bd4\u8d5b",
            "driver": "\u89c2\u5bdf\u8f66\u624b",
            "actual_pit_lap": "\u771f\u5b9e\u8fdb\u7ad9\u670d\u52a1\u5708",
            "first_slick_lap": "\u9996\u4e2a\u5149\u5934\u80ce\u5708",
            "actual_transition_lap": "\u517c\u5bb9\u5b57\u6bb5_\u9996\u4e2a\u5149\u5934\u80ce",
            "to_compound": "\u771f\u5b9e\u6362\u5165\u8f6e\u80ce",
            "rainfall_only_window_start": "\u4ec5\u964d\u96e8\u4ee3\u7406\u7a97\u53e3",
            "decision_time_window_label": "\u51b3\u7b56\u65f6\u516c\u5f00\u4fe1\u53f7\u533a\u95f4",
            "retrospective_window_start": "\u8d5b\u540e\u91cd\u5efa\u7a97\u53e3",
            "classification": "\u5f53\u524d\u7ed3\u8bba",
            "failure_type": "\u8bc1\u636e\u7c7b\u578b",
        }
    )
    for column_name, source_name in [
        ("真实进站服务圈", "actual_pit_lap"),
        ("首个光头胎圈", "first_slick_lap"),
        ("兼容字段_首个光头胎", "actual_transition_lap"),
    ]:
        if column_name in v2_display.columns and source_name in v2_report.columns:
            v2_display[column_name] = v2_report[source_name].map(
                lambda value: format_optional_lap(value, "不适用 / 未识别")
            )
    if "仅降雨代理窗口" in v2_display.columns:
        v2_display["仅降雨代理窗口"] = v2_report.apply(
            lambda row: format_window_start(
                row["rainfall_only_window_start"],
                row.get("first_slick_lap", row.get("actual_transition_lap")),
            ),
            axis=1,
        ).to_list()
    if "赛后重建窗口" in v2_display.columns:
        v2_display["赛后重建窗口"] = v2_report.apply(
            lambda row: format_window_start(
                row["retrospective_window_start"],
                row.get("first_slick_lap", row.get("actual_transition_lap")),
            ),
            axis=1,
        ).to_list()
    st.dataframe(v2_display, width="stretch", hide_index=True)
    st.caption(
        "字段解释：真实进站服务圈 = 公开 pit 服务锚点；首个光头胎圈 = 逐圈记录首次出现新胎；"
        "兼容字段保留旧 actual_transition_lap（= 首个光头胎圈）。"
        "仅降雨代理窗口 = 只看降雨记录的基础判断；"
        "决策时公开信号区间 = 比赛当时可见的降雨 + DRS/抓地等公开线索；"
        "赛后重建窗口 = 赛后使用完整圈速进行复盘。"
        "误差按进站服务圈计算。未识别表示输入没有形成稳定窗口。"
    )

if not challenge_set.empty:
    with exp_case.expander("2025 事件类型诊断"):
        st.write(
            "把 53 个车手级事件按复杂条件拆开，可以看到失败集中在起步/重启、赛会控制相邻和多阶段降雨。"
        )
        quality_summary = (
            challenge_set.assign(abs_error=challenge_errors)
            .groupby("event_quality", as_index=False)
            .agg(
                事件数=("driver", "size"),
                平均绝对误差_圈=("abs_error", "mean"),
                前后两圈内=("abs_error", lambda values: int((values <= 2).sum())),
            )
        )
        quality_summary["平均绝对误差（圈）"] = quality_summary.pop("平均绝对误差_圈").round(2)
        quality_summary["事件类型"] = quality_summary.pop("event_quality").map(
            {
                "start_or_restart": "\u8d77\u8dd1\u6216\u91cd\u542f",
                "race_control_adjacent": "\u8d5b\u4f1a\u63a7\u5236\u76f8\u90bb",
                "multi_rain_phase": "\u591a\u6b21\u964d\u96e8\u9636\u6bb5",
                "single_rain_phase": "\u5355\u9636\u6bb5\u964d\u96e8",
            }
        )
        st.dataframe(
            quality_summary[["事件类型", "事件数", "平均绝对误差（圈）", "前后两圈内"]],
            width="stretch",
            hide_index=True,
        )
        clean_events = int((challenge_set["event_quality"] == "single_rain_phase").sum())
        st.caption(
            f"这 53 个事件里只有 {clean_events} 个属于单阶段降雨、且不受起步/重启或赛会控制干扰的"
            "\u201c干净\u201d场景。后续状态机因此把多数复杂事件标成弃权，而不是继续硬猜。"
        )

reliability_events = build_reliability_events(challenge_set, fastf1_evaluation)
with exp_case.expander("联合事件分布图（不是统一准确率）"):
    st.write(
        "这组图只用于观察成功、失败、弃权及其原因分布，不是统一准确率，"
        "也不能把两套评价协议直接合并比较。"
    )
    if reliability_events.empty:
        st.info("暂无可用于联合观察的事件数据。")
    else:
        source_summary = summarize_reliability_events(reliability_events)
        source_breakdown = (
            source_summary.pivot_table(
                index="source",
                columns="outcome",
                values="\u4e8b\u4ef6\u6570",
                aggfunc="sum",
                fill_value=0,
            )
            .rename(
                columns={
                    "signal_accurate": "\u843d\u5728 \u00b12 \u5708\u5185",
                    "signal_inaccurate": "\u8d85\u51fa \u00b12 \u5708",
                    "abstain": "\u5f03\u6743",
                }
            )
            .reset_index()
        )
        for column in ["\u843d\u5728 \u00b12 \u5708\u5185", "\u8d85\u51fa \u00b12 \u5708", "\u5f03\u6743"]:
            if column not in source_breakdown.columns:
                source_breakdown[column] = 0
        source_breakdown["合计"] = source_breakdown[["落在 \u00b12 圈内", "超出 \u00b12 圈", "弃权"]].sum(axis=1)
        source_breakdown = source_breakdown.rename(columns={"source": "数据来源"})[
            ["数据来源", "合计", "落在 \u00b12 圈内", "超出 \u00b12 圈", "弃权"]
        ]
        st.dataframe(source_breakdown, width="stretch", hide_index=True)
        st.caption(
            "上表按数据来源分开计数。2025 压力测试检查的是简单规则；2021 评估检查的是允许弃权的状态机。"
            "\u201c弃权\u201d和\u201c信号超出 \u00b12 圈\u201d是两种不同结果，不能混成同一种失败。"
        )
        st.plotly_chart(plot_reliability_event_map(reliability_events), width="stretch")
        st.caption(
            "散点图：每个点是一个车手级湿转干事件；纵轴是公开信号与真实换胎圈的差值。"
            "绿色 = 落在前后 2 圈内，红色 = 偏差更大，灰色 = 模型弃权。"
            "灰色点靠近零线只是为了可读，不代表误差为零。横轴只是事件排列顺序。"
        )
        st.plotly_chart(plot_reliability_reason_counts(reliability_events), width="stretch")
        st.caption(
            "原因统计图按数据来源和事件条件计数；悬停可看到信号与弃权的拆分。"
            "重点是看失败和弃权集中在哪些条件，而不是合成一个总准确率。"
        )

with exp_case.expander("研究文档下载"):
    summary_path = RESEARCH_DIR / "v2_research_summary.md"
    if summary_path.exists():
        st.download_button(
            "下载中文研究总结",
            data=summary_path.read_text(encoding="utf-8"),
            file_name="f1_strategy_lab_v2_research_summary.md",
            mime="text/markdown",
            key="download_v2_research_summary_zh",
        )
    english_summary_path = RESEARCH_DIR / "v2_research_summary_en.md"
    if english_summary_path.exists():
        st.download_button(
            "下载英文研究摘要",
            data=english_summary_path.read_text(encoding="utf-8"),
            file_name="f1_strategy_lab_v2_research_summary_en.md",
            mime="text/markdown",
            key="download_v2_research_summary_en",
        )
    challenge_diagnostic = RESEARCH_DIR / "v2_holdout_diagnostic_2025.md"
    if challenge_diagnostic.exists():
        st.download_button(
            "下载 2025 失败诊断报告",
            data=challenge_diagnostic.read_text(encoding="utf-8"),
            file_name="f1_strategy_lab_v2_holdout_diagnostic_2025.md",
            mime="text/markdown",
            key="download_v2_holdout_diagnostic_2025",
        )
    st.caption(
        "文档面向想复核研究过程的人；普通访客不下载也不影响阅读本页。"
    )
