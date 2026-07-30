from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2_validation import V2_CASES, build_case_track_state, build_validation_metrics, build_validation_report
from src.config import RESEARCH_DIR, RESULTS_DIR
from src.v3_reliability import build_reliability_events


CASE_LABELS = {
    "silverstone_2024": "2024 British GP - Norris",
    "montreal_2024": "2024 Canadian GP - Verstappen",
    "zandvoort_2023": "2023 Dutch GP - Verstappen",
    "zandvoort_2024_control": "2024 Dutch GP - dry-race control",
    "sao_paulo_2024_control": "2024 Sao Paulo GP - persistent-wet control",
}

CASE_ROLES = {
    "silverstone_2024": "Primary wet-to-dry case",
    "montreal_2024": "Wet-to-dry stress case",
    "zandvoort_2023": "Multi-phase weather case",
    "zandvoort_2024_control": "Dry-race control",
    "sao_paulo_2024_control": "Persistent-wet control",
}

CASE_FINDINGS = {
    "silverstone_2024": "Public evidence covers the observed switch.",
    "montreal_2024": "Rainfall alone is insufficient; public track confirmation arrives two laps after the switch.",
    "zandvoort_2023": "The early switch is covered, but later rain makes this a multi-phase case.",
    "zandvoort_2024_control": "No wet-to-slick window is generated in a dry race.",
    "sao_paulo_2024_control": "No wet-to-slick window is generated when the wet race never returns to slicks.",
}

CASE_TABLE_READOUTS = {
    "silverstone_2024": "Public evidence covers the switch",
    "montreal_2024": "Rainfall insufficient; confirmation late",
    "zandvoort_2023": "Early coverage; later rain returns",
    "zandvoort_2024_control": "No false switch in dry conditions",
    "sao_paulo_2024_control": "No false switch in persistent wet",
}


@st.cache_data(show_spinner=False)
def load_validation() -> pd.DataFrame:
    try:
        return build_validation_report()
    except (FileNotFoundError, OSError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_case_evidence(case_id: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    case = next(item for item in V2_CASES if item.case_id == case_id)
    return build_case_track_state(case)


@st.cache_data(show_spinner=False)
def load_challenge_set() -> pd.DataFrame:
    path = RESULTS_DIR / "v2_holdout_v23_2025.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_fastf1_evaluation() -> pd.DataFrame:
    path = RESULTS_DIR / "v2_fastf1_evaluation_2021.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def format_lap(value: object) -> str:
    if pd.isna(value):
        return "No observed switch"
    return f"Lap {int(value)}"


def format_interval(start: object, end: object) -> str:
    if pd.isna(start) and pd.isna(end):
        return "No public-signal window"
    if pd.isna(start):
        return format_lap(end)
    if pd.isna(end) or int(start) == int(end):
        return format_lap(start)
    earliest, latest = sorted([int(start), int(end)])
    return f"Laps {earliest}-{latest}"


def make_evidence_chart(
    state: pd.DataFrame,
    transitions: list[dict[str, object]],
    public_latest: object,
    public_earliest: object,
    retrospective: object,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=state["lap_number"],
            y=state["decision_time_wetness"],
            mode="lines",
            name="Rainfall-only baseline",
            line={"color": "#3b82f6", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=state["lap_number"],
            y=state["retrospective_wetness"],
            mode="lines",
            name="Retrospective reconstruction",
            line={"color": "#f59e0b", "width": 3},
        )
    )
    for index, transition in enumerate(transitions):
        pit_lap = transition.get("actual_pit_lap")
        first_slick = transition.get("first_slick_lap")
        same_lap = (
            pit_lap is not None
            and first_slick is not None
            and int(pit_lap) == int(first_slick)
        )
        if pit_lap is not None and not same_lap:
            fig.add_vline(
                x=int(pit_lap),
                line_color="#16a34a",
                line_width=2,
                line_dash="solid",
                annotation_text="Pit service lap" if index == 0 else None,
                annotation_position="top right",
            )
        if first_slick is not None and not same_lap:
            fig.add_vline(
                x=int(first_slick),
                line_color="#16a34a",
                line_width=2,
                line_dash="dash",
                annotation_text="First slick lap" if index == 0 else None,
                annotation_position="bottom left",
            )
        if same_lap:
            fig.add_vline(
                x=int(pit_lap),
                line_color="#16a34a",
                line_width=2,
                line_dash="solid",
                annotation_text="Pit service = first slick lap" if index == 0 else None,
                annotation_position="top right",
            )

    if not pd.isna(public_latest):
        latest = int(public_latest)
        earliest = int(public_earliest) if not pd.isna(public_earliest) else latest
        fig.add_vrect(
            x0=min(earliest, latest),
            x1=max(earliest, latest),
            fillcolor="#3b82f6",
            opacity=0.14,
            line_width=0,
            annotation_text="Public-signal interval",
            annotation_position="top left",
        )
        fig.add_vline(
            x=latest,
            line_color="#3b82f6",
            line_width=1,
            line_dash="dot",
        )
    if not pd.isna(retrospective):
        fig.add_vline(
            x=int(retrospective),
            line_color="#f59e0b",
            line_width=1,
            line_dash="dot",
            annotation_text="Retrospective window",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title="Track-state evidence around the tyre transition",
        xaxis_title="Lap",
        yaxis_title="Model track-state index",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#1d252e"},
        xaxis={"gridcolor": "#e6eaee", "linecolor": "#cbd2d8"},
        yaxis={"range": [0, 1], "gridcolor": "#e6eaee", "linecolor": "#cbd2d8"},
        hoverlabel={"bgcolor": "#1d252e", "font": {"color": "#f7f9fa"}},
    )
    return fig


st.set_page_config(page_title="F1 Strategy Lab", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      :root { --paper: #ffffff; --ink: #171d24; --body: #43505d; --muted: #677482; --line: #d7dde3; --panel: #ffffff; --canvas: #f3f5f7; --red: #c8102e; --cyan: #1b8783; --lab-font: Inter, "SF Pro Display", "Segoe UI", Arial, sans-serif; }
      .stApp, .stApp button, .stApp input, .stApp textarea { font-family: var(--lab-font); }
      .stApp { background: var(--canvas); color: var(--ink); }
      .stAppHeader { display: none; }
      [data-testid="stHeaderActionElements"], [data-testid="stDeployButton"], [data-testid="stBaseButton-header"], #MainMenu, footer { display: none; }
      .block-container { max-width: 1480px; padding: 3.5rem 2rem 5rem; }
      @media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
      h1, h2, h3 { color: var(--ink) !important; letter-spacing: 0 !important; }
      h1 { font-size: 3rem !important; font-weight: 750 !important; margin-bottom: .25rem !important; }
      h2 { font-size: 1.35rem !important; font-weight: 700 !important; margin-top: 2.8rem !important; padding-left: 0; }
      h3 { font-size: 1.05rem !important; }
      p, li, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] { color: var(--body); }
      .research-kicker { color: var(--red); font-size: .72rem; font-weight: 800; letter-spacing: .12rem; margin-bottom: .8rem; }
      .research-deck { max-width: 850px; color: #34404b; font-size: 1.05rem; line-height: 1.55; margin: 1rem 0 2.25rem; }
      [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); border-top: 3px solid var(--red); border-radius: 3px; padding: 1rem 1.1rem; min-height: 118px; box-shadow: 0 1px 2px rgba(18,31,43,.04); }
      [data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; }
      [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.72rem; font-weight: 700; }
      [data-testid="stDataFrame"] { background: var(--paper); border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
      [data-testid="stAlert"] { border-radius: 3px; border: 1px solid #c9e0df; background: #edf7f6; }
      .stButton > button, .stDownloadButton > button { border-radius: 3px; border: 1px solid #aeb8c2; background: #ffffff; color: var(--ink); }
      .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--red); color: var(--ink); }
      .stSelectbox > div > div { border-radius: 3px; }
      hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="research-kicker">FORMULA 1 / PUBLIC-DATA RESEARCH</div>', unsafe_allow_html=True)
st.title("F1 Strategy Lab")
st.caption("Explainable wet-to-dry tyre-transition research using public Formula 1 data")

st.markdown(
    '<div class="research-deck">Can public race data identify when a wet race returned to slick tyres, while also showing uncertainty and the cases where the method should remain silent?</div>',
    unsafe_allow_html=True,
)

st.info(
    "This portfolio is two independent modules: (1) a single-race strategy simulator (2024 Silverstone) "
    "that explores Norris's final wet-to-slick stop under explicit simplified assumptions; "
    "(2) a public-signal reliability audit that checks when public evidence agrees with real wet-to-slick "
    "pit actions, and when it is too late or too ambiguous to trust. The audit does not validate the simulator's physics, and neither module is a general live decision system."
)

report = load_validation()
if report.empty:
    st.error("The validation data could not be loaded. Regenerate the bundled case data before opening this portfolio page.")
    st.stop()
metrics = build_validation_metrics(report)
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Exploratory wet-to-dry cases", metrics["wet_to_dry_cases"])
metric_2.metric("Exploratory cases within +/-2 laps", f"{metrics['public_within_two']}/{metrics['wet_to_dry_cases']}")
metric_3.metric("Mean absolute error", f"{metrics['public_mean_absolute_error']:.2f} laps")
metric_4.metric("Controls left silent", f"{metrics['controls_silent']}/{metrics['control_cases']}")

st.caption(
    "Validation status: exploratory development set. The current cases shaped the method; "
    "future holdout races will be evaluated without changing the rules."
)

st.markdown("### Method")
st.write(
    "The project keeps three evidence modes separate. A rainfall-only baseline is deliberately "
    "kept visible. A decision-time public signal adds race-control confirmation such as normal-grip "
    "conditions or DRS being re-enabled. A retrospective reconstruction may use shared multi-driver "
    "pace residuals after the race, so it is never presented as live prediction."
)

st.markdown("### Cross-Race Validation")
display = pd.DataFrame(
    {
        "Race": [CASE_LABELS[case_id] for case_id in report["case_id"]],
        "Role": [CASE_ROLES[case_id] for case_id in report["case_id"]],
        "Pit service": [format_lap(value) for value in report["actual_pit_lap"]],
        "First slick": [format_lap(value) for value in report["first_slick_lap"]],
        "Public-signal interval": [
            format_interval(start, end)
            for start, end in zip(report["decision_time_window_start"], report["decision_time_window_end"])
        ],
        "Research readout": [CASE_TABLE_READOUTS[case_id] for case_id in report["case_id"]],
    }
)
st.dataframe(display, width="stretch", hide_index=True)
st.caption(
    "Pit service lap is the lap the car actually pitted; first slick lap is the first lap the new "
    "slick compound is logged (often the pit-out lap). They can differ by one lap. Strategy timing "
    "is scored against the pit service lap, not the first slick lap."
)

challenge = load_challenge_set()
if not challenge.empty:
    challenge_errors = pd.to_numeric(challenge["public_signal_error_laps"], errors="coerce").abs()
    sustained = challenge[challenge["event_class"] == "sustained_wet_to_dry"]
    sustained_errors = pd.to_numeric(sustained["public_signal_error_laps"], errors="coerce").abs()
    st.markdown("### 2025 Challenge-Set Result")
    challenge_1, challenge_2, challenge_3, challenge_4 = st.columns(4)
    challenge_1.metric("Raw transitions", len(challenge))
    challenge_2.metric("Within +/-2 laps", f"{int((challenge_errors <= 2).sum())}/{len(challenge)}")
    challenge_3.metric("Mean absolute error", f"{challenge_errors.mean():.2f} laps")
    challenge_4.metric("Sustained-event error", f"{sustained_errors.mean():.2f} laps")
    st.write(
        "This challenge set falsified the simple public-signal rule as a general per-driver switch predictor. "
        "Early DRS or normal-grip messages become stale when a single wet-tyre stint spans later rain, "
        "safety-car, or restart phases. The result is retained here as a failure analysis, not hidden behind the exploratory pilot."
    )
    quality_summary = (
        challenge.assign(abs_error=challenge_errors)
        .groupby("event_quality", as_index=False)
        .agg(
            Events=("driver", "size"),
            **{
                "Mean absolute error (laps)": ("abs_error", "mean"),
                "Within +/-2 laps": ("abs_error", lambda values: int((values <= 2).sum())),
            },
        )
    )
    quality_summary["Mean absolute error (laps)"] = quality_summary["Mean absolute error (laps)"].round(2)
    quality_summary["Event quality"] = quality_summary.pop("event_quality").map(
        {
            "start_or_restart": "Start or restart",
            "race_control_adjacent": "Race-control adjacent",
            "multi_rain_phase": "Multi-rain phase",
        }
    )
    st.dataframe(
        quality_summary[["Event quality", "Events", "Mean absolute error (laps)", "Within +/-2 laps"]],
        width="stretch",
        hide_index=True,
    )
    clean_events = int((challenge["event_quality"] == "single_rain_phase").sum())
    st.info(
        f"V2.4 state-machine gate: {clean_events}/{len(challenge)} events are clean enough to issue a driver-level signal. "
        "The remaining events are deliberately classified as ambiguous rather than forced into a prediction. "
        "This set has zero clean events, so it is useful for failure diagnosis but cannot measure V2.4 accuracy yet."
    )

fastf1_evaluation = load_fastf1_evaluation()
if not fastf1_evaluation.empty:
    issued = fastf1_evaluation[fastf1_evaluation["state_machine_outcome"] == "signal"]
    issued_errors = pd.to_numeric(issued["public_signal_error_laps"], errors="coerce").abs()
    st.markdown("### V2.4 Historical Coverage Result")
    coverage_1, coverage_2, coverage_3, coverage_4 = st.columns(4)
    coverage_1.metric("Observed transitions", len(fastf1_evaluation))
    coverage_2.metric("Signals issued", f"{len(issued)}/{len(fastf1_evaluation)}")
    coverage_3.metric("Issued within +/-2 laps", f"{int((issued_errors <= 2).sum())}/{len(issued)}")
    coverage_4.metric("Issued-signal MAE", f"{issued_errors.mean():.2f} laps")
    st.write(
        "A separately documented FastF1 historical scan selected both 2021 candidates: Emilia Romagna and Hungary. "
        "The state machine abstained on all 14 Hungary start/restart transitions and issued signals only for eligible Emilia Romagna events. "
        "Coverage and accuracy are shown together so abstention cannot be mistaken for success."
    )

reliability_events = build_reliability_events(challenge, fastf1_evaluation)
if not reliability_events.empty:
    reliability_accurate = int((reliability_events["outcome"] == "signal_accurate").sum())
    reliability_inaccurate = int((reliability_events["outcome"] == "signal_inaccurate").sum())
    reliability_abstain = int((reliability_events["outcome"] == "abstain").sum())
    st.markdown("### Reliability & Abstention")
    st.write(
        "V3 turns the state-machine result into an applicability view: it shows when the method issued a public-signal window, "
        "when that signal missed the observed switch, and when the method deliberately refused to guess."
    )
    reliability_1, reliability_2, reliability_3, reliability_4 = st.columns(4)
    reliability_1.metric("Evaluated events", len(reliability_events))
    reliability_2.metric("Signals within +/-2 laps", reliability_accurate)
    reliability_3.metric("Signals outside +/-2 laps", reliability_inaccurate)
    reliability_4.metric("Abstentions", reliability_abstain)
    st.info(
        "The important result is not a new accuracy score. It is the separation between a signal, a miss, and an abstention: "
        "the method is designed to expose when public evidence is too ambiguous for a responsible driver-level claim."
    )
    reason_labels = {
        "race_control_adjacent": "Race-control adjacent (SC/VSC/red flag context)",
        "multi_rain_phase": "Multi-rain phase (wet-dry cycles)",
        "start_or_restart": "Start or restart adjacent",
        "clean_or_other": "Single-phase or other (clean/uncategorized)",
    }
    reason_summary = (
        reliability_events.assign(reason=reliability_events["reason_group"].map(reason_labels))
        .groupby("reason", as_index=False)
        .size()
        .rename(columns={"reason": "Event condition", "size": "Events"})
        .sort_values("Events", ascending=False)
    )
    st.dataframe(reason_summary, width="stretch", hide_index=True)
    st.caption(
        "This panorama combines the frozen 2025 challenge set and the 2021 FastF1 scan. "
        "The condition groups are mutually exclusive: each event is assigned to the first applicable category. "
        "Race-control context includes nearby safety-car, VSC, or red-flag phases; multi-rain phase means the wet tyre stint spans more than one wet-dry cycle. "
        "It describes the method's evidence boundary; it is not a live strategy recommendation or a universal F1 accuracy benchmark."
    )

st.markdown("### Evidence Explorer")
selected_case_id = st.selectbox(
    "Select a validation case",
    [case.case_id for case in V2_CASES],
    format_func=lambda case_id: CASE_LABELS[case_id],
)
selected_row = report[report["case_id"] == selected_case_id].iloc[0]
state, transitions = load_case_evidence(selected_case_id)
st.plotly_chart(
    make_evidence_chart(
        state,
        transitions,
        selected_row.get("decision_time_window_start"),
        selected_row.get("decision_time_window_end"),
        selected_row.get("retrospective_window_start"),
    ),
    width="stretch",
)
st.caption(CASE_FINDINGS[selected_case_id])

st.markdown("### What This Project Does Not Claim")
st.write(
    "It does not measure water depth, forecast tyre strategy live, or reproduce every race variable. "
    "OpenF1 rainfall is binary, race-control messages are confirmation signals rather than optimal-strategy labels, "
    "and pace residuals can contain traffic, safety-car, tyre, and driver effects. The five-race case set is a "
    "prototype validation set, not a universal F1 strategy rule."
)

english_summary = RESEARCH_DIR / "v2_research_summary_en.md"
if english_summary.exists():
    st.download_button(
        "Download Research Summary",
        data=english_summary.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_research_summary_en.md",
        mime="text/markdown",
    )

protocol = RESEARCH_DIR / "v2_validation_protocol.md"
if protocol.exists():
    st.download_button(
        "Download Validation Protocol",
        data=protocol.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_validation_protocol.md",
        mime="text/markdown",
    )

v24_protocol = RESEARCH_DIR / "v2_4_evaluation_protocol.md"
if v24_protocol.exists():
    st.download_button(
        "Download V2.4 Evaluation Protocol",
        data=v24_protocol.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_4_evaluation_protocol.md",
        mime="text/markdown",
    )

candidate_scan = RESEARCH_DIR / "v2_candidate_scan_2026.md"
if candidate_scan.exists():
    st.download_button(
        "Download 2026 Availability Audit",
        data=candidate_scan.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_candidate_scan_2026.md",
        mime="text/markdown",
    )

historical_scan = RESEARCH_DIR / "v2_candidate_scan_2023.md"
if historical_scan.exists():
    st.download_button(
        "Download 2023 Historical Availability Audit",
        data=historical_scan.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_candidate_scan_2023.md",
        mime="text/markdown",
    )

historical_2024_scan = RESEARCH_DIR / "v2_candidate_scan_2024.md"
if historical_2024_scan.exists():
    st.download_button(
        "Download 2024 Historical Availability Audit",
        data=historical_2024_scan.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_candidate_scan_2024.md",
        mime="text/markdown",
    )

fastf1_report = RESEARCH_DIR / "v2_fastf1_evaluation_2021.md"
if fastf1_report.exists():
    st.download_button(
        "Download V2.4 FastF1 Evaluation",
        data=fastf1_report.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_fastf1_evaluation_2021.md",
        mime="text/markdown",
    )

challenge_diagnostic = RESEARCH_DIR / "v2_holdout_diagnostic_2025.md"
if challenge_diagnostic.exists():
    st.download_button(
        "Download 2025 Challenge-Set Diagnostic",
        data=challenge_diagnostic.read_text(encoding="utf-8"),
        file_name="f1_strategy_lab_v2_holdout_diagnostic_2025.md",
        mime="text/markdown",
    )
