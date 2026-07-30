from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


F1_FONT_STACK = "Inter, SF Pro Display, Segoe UI, Arial, sans-serif"
LAB_COLORS = {
    "ink": "#1d252e",
    "title": "#151b22",
    "body": "#485460",
    "muted": "#5d6975",
    "line": "#cbd2d8",
    "grid": "#e6eaee",
    "paper": "#ffffff",
    "f1_red": "#c8102e",
    "decision_blue": "#2563a8",
    "reconstruction_amber": "#c77800",
    "actual_green": "#16803a",
    "faster_blue": "#245b93",
    "slower_red": "#b9453d",
}
DEFAULT_MARGIN = {"l": 44, "r": 28, "t": 72, "b": 40}
STRATEGY_DELTA_COLORSCALE = [
    [0.0, "#245b93"],
    [0.38, "#a9c7e3"],
    [0.5, "#ffffff"],
    [0.62, "#ecc2bd"],
    [1.0, "#b9453d"],
]

pio.templates["f1_lab"] = go.layout.Template(
    layout={
        "paper_bgcolor": LAB_COLORS["paper"],
        "plot_bgcolor": LAB_COLORS["paper"],
        "font": {"color": LAB_COLORS["ink"], "family": F1_FONT_STACK, "size": 14},
        "title": {"font": {"color": LAB_COLORS["title"], "size": 17}},
        "hoverlabel": {"bgcolor": LAB_COLORS["ink"], "bordercolor": LAB_COLORS["ink"], "font": {"color": "#f7f9fa"}},
        "xaxis": {
            "gridcolor": LAB_COLORS["grid"],
            "linecolor": LAB_COLORS["line"],
            "zerolinecolor": LAB_COLORS["line"],
            "tickfont": {"color": LAB_COLORS["muted"]},
            "title": {"font": {"color": LAB_COLORS["body"]}},
        },
        "yaxis": {
            "gridcolor": LAB_COLORS["grid"],
            "linecolor": LAB_COLORS["line"],
            "zerolinecolor": LAB_COLORS["line"],
            "tickfont": {"color": LAB_COLORS["muted"]},
            "title": {"font": {"color": LAB_COLORS["body"]}},
        },
        "legend": {"font": {"color": "#34404b"}, "bgcolor": "rgba(0,0,0,0)"},
    }
)
pio.templates.default = "f1_lab"


def apply_lab_theme(fig: go.Figure) -> go.Figure:
    """Force a stable light chart surface even when Streamlit's theme changes."""
    fig.update_layout(
        template=None,
        paper_bgcolor=LAB_COLORS["paper"],
        plot_bgcolor=LAB_COLORS["paper"],
        font={"color": LAB_COLORS["ink"], "family": F1_FONT_STACK, "size": 14},
        title={"font": {"color": LAB_COLORS["title"], "size": 17}},
        hoverlabel={"bgcolor": LAB_COLORS["ink"], "bordercolor": LAB_COLORS["ink"], "font": {"color": "#f7f9fa"}},
        legend={"font": {"color": "#34404b"}, "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(
        automargin=True,
        gridcolor=LAB_COLORS["grid"],
        linecolor=LAB_COLORS["line"],
        zerolinecolor=LAB_COLORS["line"],
        tickfont={"color": LAB_COLORS["muted"]},
        title_font={"color": LAB_COLORS["body"]},
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor=LAB_COLORS["grid"],
        linecolor=LAB_COLORS["line"],
        zerolinecolor=LAB_COLORS["line"],
        tickfont={"color": LAB_COLORS["muted"]},
        title_font={"color": LAB_COLORS["body"]},
    )
    if not any(getattr(fig.layout.margin, edge, None) for edge in ("l", "r", "t", "b")):
        fig.update_layout(margin=DEFAULT_MARGIN)
    return fig


COMPOUND_COLORS = {
    "SOFT": "#e10600",
    "MEDIUM": "#ffd12e",
    "HARD": "#d9d9d9",
    "INTERMEDIATE": "#39b54a",
    "WET": "#1f78ff",
}


def plot_v2_track_state_evidence(
    state: pd.DataFrame,
    actual_transition_laps: list[int] | None = None,
    decision_window_start: int | None = None,
    retrospective_window_start: int | None = None,
    decision_window_end: int | None = None,
    *,
    actual_pit_lap: int | None = None,
    first_slick_lap: int | None = None,
    to_compound: str | None = None,
    focus_start: int | None = None,
    focus_end: int | None = None,
    later_rain_start: int | None = None,
    field_pit_laps: list[int] | None = None,
) -> go.Figure:
    """Show rainfall proxy, public-signal window, retrospective rebuild, and real anchors.

    Real anchors distinguish:
    - actual_pit_lap: public pit service lap (strategy timing)
    - first_slick_lap: first lap compound record showing the new slick
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=state["lap_number"],
            y=state["decision_time_wetness"],
            mode="lines",
            name="蓝色实线：仅降雨代理",
            line={"color": "#3b82f6", "width": 3},
            hovertemplate="第 %{x} 圈<br>仅降雨代理 %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=state["lap_number"],
            y=state["retrospective_wetness"],
            mode="lines",
            name="橙色实线：赛后重建",
            line={"color": "#f59e0b", "width": 3},
            hovertemplate="第 %{x} 圈<br>赛后重建 %{y:.2f}<extra></extra>",
        )
    )

    # Backward compatibility: old callers passed a list of stint/transition laps.
    if actual_pit_lap is None and actual_transition_laps:
        actual_pit_lap = int(actual_transition_laps[0])

    if decision_window_start is not None:
        band_end = decision_window_end if decision_window_end is not None else decision_window_start
        left = min(int(band_end), int(decision_window_start))
        right = max(int(band_end), int(decision_window_start))
        fig.add_vrect(
            x0=left,
            x1=right,
            fillcolor="#3b82f6",
            opacity=0.16,
            line_width=0,
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="浅蓝带：公开信号区间",
                line={"color": "#3b82f6", "width": 8},
                hoverinfo="skip",
            )
        )
        # Conservative public-signal point = later boundary used for scoring.
        fig.add_vline(
            x=int(decision_window_start),
            line_color="#1d4ed8",
            line_width=2,
            line_dash="dot",
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="蓝色点线：保守公开信号点",
                line={"color": "#1d4ed8", "width": 2, "dash": "dot"},
                hoverinfo="skip",
            )
        )

    if retrospective_window_start is not None:
        fig.add_vline(
            x=int(retrospective_window_start),
            line_color="#f59e0b",
            line_width=1.5,
            line_dash="dot",
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="橙色点线：赛后重建窗口",
                line={"color": "#f59e0b", "width": 1.5, "dash": "dot"},
                hoverinfo="skip",
            )
        )

    compound_label = (to_compound or "光头胎").upper()
    y_top = 0.97
    if actual_pit_lap is not None:
        fig.add_vline(
            x=int(actual_pit_lap),
            line_color="#16803a",
            line_width=2.5,
            line_dash="solid",
        )
        fig.add_annotation(
            x=int(actual_pit_lap),
            y=y_top,
            yref="paper",
            text=f"Pit L{int(actual_pit_lap)}",
            showarrow=False,
            font={"size": 12, "color": "#16803a"},
            bgcolor="rgba(255,255,255,0.85)",
            xanchor="left",
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="绿色实线：真实进站服务圈",
                line={"color": "#16803a", "width": 2.5},
                hoverinfo="skip",
            )
        )

    if first_slick_lap is not None and (
        actual_pit_lap is None or int(first_slick_lap) != int(actual_pit_lap)
    ):
        fig.add_vline(
            x=int(first_slick_lap),
            line_color="#16803a",
            line_width=2,
            line_dash="dash",
        )
        fig.add_annotation(
            x=int(first_slick_lap),
            y=0.88,
            yref="paper",
            text=f"首个 {compound_label} 圈 L{int(first_slick_lap)}",
            showarrow=False,
            font={"size": 12, "color": "#16803a"},
            bgcolor="rgba(255,255,255,0.85)",
            xanchor="left",
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="绿色虚线：首个光头胎圈",
                line={"color": "#16803a", "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    elif first_slick_lap is not None and actual_pit_lap is not None:
        # Same lap — still document first slick in legend via annotation.
        fig.add_annotation(
            x=int(first_slick_lap),
            y=0.88,
            yref="paper",
            text=f"Pit=首个 {compound_label} L{int(first_slick_lap)}",
            showarrow=False,
            font={"size": 12, "color": "#16803a"},
            bgcolor="rgba(255,255,255,0.85)",
            xanchor="left",
        )

    if field_pit_laps:
        unique_field = sorted({int(lap) for lap in field_pit_laps})
        fig.add_trace(
            go.Scatter(
                x=unique_field,
                y=[0.04] * len(unique_field),
                mode="markers",
                name="场上湿转干进站波次",
                marker={
                    "symbol": "line-ns",
                    "size": 12,
                    "color": "#64748b",
                    "line": {"width": 2, "color": "#64748b"},
                },
                hovertemplate="场上进站服务 第 %{x} 圈<extra></extra>",
            )
        )

    if later_rain_start is not None:
        fig.add_vrect(
            x0=int(later_rain_start),
            x1=float(state["lap_number"].max()),
            fillcolor="#94a3b8",
            opacity=0.12,
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=int(later_rain_start),
            y=0.5,
            text="后续独立降雨阶段（不属于本次验证）",
            showarrow=False,
            font={"size": 12, "color": "#475569"},
            bgcolor="rgba(255,255,255,0.8)",
            xanchor="left",
        )

    layout_kwargs: dict = {
        "title": "真实进站动作附近的赛道状态证据",
        "xaxis_title": "圈数",
        "yaxis_title": "模型赛道状态指数",
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": 1.18, "x": 0, "font": {"size": 12}},
        "margin": {"l": 20, "r": 20, "t": 110, "b": 20},
    }
    fig.update_layout(**layout_kwargs)
    fig.update_yaxes(range=[0, 1])
    if focus_start is not None and focus_end is not None:
        fig.update_xaxes(range=[focus_start, focus_end])
    return apply_lab_theme(fig)


def plot_lap_pace(laps: pd.DataFrame, drivers: list[str]) -> go.Figure:
    subset = laps[laps["name_acronym"].isin(drivers)].copy()
    hover_columns = ["compound", "pit_lap", "team_name"]
    if "tyre_age_at_lap_start" in subset.columns:
        hover_columns.append("tyre_age_at_lap_start")
    fig = px.line(
        subset,
        x="lap_number",
        y="lap_duration",
        color="name_acronym",
        line_dash="compound",
        hover_data=hover_columns,
        labels={
            "lap_number": "圈数",
            "lap_duration": "单圈用时（秒）",
            "name_acronym": "车手",
            "compound": "轮胎",
            "pit_lap": "进站圈",
            "team_name": "车队",
        },
        title="真实比赛单圈节奏",
    )
    fig.update_yaxes(title="单圈用时（秒，越低越快）", autorange="reversed")
    fig.update_xaxes(title="圈数")
    return apply_lab_theme(fig)


def plot_strategy_comparison(real_laps: pd.DataFrame, simulated_laps: pd.DataFrame, driver: str) -> go.Figure:
    driver_laps = real_laps[real_laps["name_acronym"] == driver].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=driver_laps["lap_number"],
            y=driver_laps["lap_duration"],
            mode="lines+markers",
            name=f"{driver} 真实",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=simulated_laps["lap_number"],
            y=simulated_laps["simulated_lap_time"],
            mode="lines+markers",
            name="模拟策略",
        )
    )
    if "wetness" in simulated_laps.columns:
        fig.add_trace(
            go.Scatter(
                x=simulated_laps["lap_number"],
                y=simulated_laps["wetness"],
                mode="lines",
                name="模型湿滑指数",
                yaxis="y2",
                line={"dash": "dot", "color": "#2f80ed"},
            )
        )
    fig.update_layout(
        title="真实策略 vs 模拟策略",
        xaxis_title="圈数",
        yaxis_title="单圈用时（秒，越低越快）",
        yaxis2={
            "title": "模型湿滑指数",
            "overlaying": "y",
            "side": "right",
            "range": [0, 1],
            "showgrid": False,
        },
    )
    fig.layout.yaxis.autorange = "reversed"
    return apply_lab_theme(fig)


def plot_cumulative_strategy_delta(reference_laps: pd.DataFrame, simulated_laps: pd.DataFrame) -> go.Figure:
    comparison = reference_laps[["lap_number", "simulated_lap_time"]].merge(
        simulated_laps[["lap_number", "simulated_lap_time"]],
        on="lap_number",
        suffixes=("_reference", "_current"),
    )
    comparison["cumulative_delta"] = (
        comparison["simulated_lap_time_current"] - comparison["simulated_lap_time_reference"]
    ).cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison["lap_number"],
            y=comparison["cumulative_delta"],
            mode="lines",
            fill="tozeroy",
            name="累计策略时间差",
            line={"color": "#2f80ed", "width": 3},
            hovertemplate="第 %{x} 圈<br>累计差距：%{y:+.2f} 秒<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#888888")
    fig.update_layout(
        title="当前策略相对真实策略基准<br>的累计时间差",
        xaxis_title="圈数",
        yaxis_title="累计差距（秒，负数代表更快）",
        height=360,
        margin={"l": 55, "r": 30, "t": 55, "b": 40},
    )
    return apply_lab_theme(fig)


def plot_stint_timeline(
    stints: pd.DataFrame,
    wetness_by_lap: dict[int, float] | None = None,
    title: str = "真实策略 vs 当前模拟策略<br>（蓝色背景越深表示模型湿滑指数越高）",
) -> go.Figure:
    fig = go.Figure()
    if stints.empty:
        return apply_lab_theme(fig)

    drivers = list(stints["name_acronym"].dropna().drop_duplicates())
    y_positions = {driver: index for index, driver in enumerate(reversed(drivers))}
    wetness_by_lap = wetness_by_lap or {}

    if wetness_by_lap:
        y_min = -0.5
        y_max = max(y_positions.values()) + 0.5
        for lap_number, wetness in sorted(wetness_by_lap.items()):
            alpha = 0.04 + 0.26 * max(0.0, min(1.0, wetness))
            fig.add_shape(
                type="rect",
                x0=lap_number - 0.5,
                x1=lap_number + 0.5,
                y0=y_min,
                y1=y_max,
                xref="x",
                yref="y",
                fillcolor=f"rgba(47, 128, 237, {alpha})",
                line_width=0,
                layer="below",
            )

    for _, stint in stints.iterrows():
        driver = stint["name_acronym"]
        compound = stint["compound"]
        start = int(stint["stint_lap_start"])
        end = int(stint["stint_lap_end"])
        tyre_age_value = pd.to_numeric(stint.get("tyre_age_at_start", 0), errors="coerce")
        tyre_age = int(tyre_age_value) if pd.notna(tyre_age_value) else 0
        y = y_positions[driver]
        color = COMPOUND_COLORS.get(compound, "#999999")
        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[y, y],
                mode="lines+markers",
                line={"width": 18, "color": color},
                marker={"size": 8, "color": color},
                name=f"{driver} {compound}",
                hovertemplate=(
                    f"车手：{driver}<br>"
                    f"轮胎：{compound}<br>"
                    f"圈数：{start}-{end}<br>"
                    f"起步胎龄：{tyre_age} 圈<extra></extra>"
                ),
                showlegend=False,
            )
        )
        font_color = "#111111" if compound in {"MEDIUM", "HARD"} else "#ffffff"
        fig.add_annotation(
            x=(start + end) / 2,
            y=y,
            text=f"{compound} {start}-{end}",
            showarrow=False,
            font={"size": 12, "color": font_color},
        )

    fig.update_layout(
        title=title,
        xaxis_title="圈数",
        yaxis={
            "tickmode": "array",
            "tickvals": list(y_positions.values()),
            "ticktext": list(y_positions.keys()),
            "title": "",
        },
        height=max(260, 75 * len(drivers)),
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
    )
    fig.update_xaxes(range=[0, max(stints["stint_lap_end"].max() + 2, 54)], dtick=5)
    return apply_lab_theme(fig)


def plot_strategy_weather_timeline(
    stints: pd.DataFrame,
    wetness_by_lap: dict[int, float],
    strategy_events: list[dict[str, object]] | None = None,
    title: str = "Race conditions, strategy, and key race-control events",
) -> go.Figure:
    """Place modeled track state, tyre stints, and public strategy signals on one shared lap axis."""
    if stints.empty:
        return apply_lab_theme(go.Figure())

    strategy_events = strategy_events or []
    total_laps = int(stints["stint_lap_end"].max())
    drivers = list(stints["name_acronym"].dropna().drop_duplicates())
    y_positions = {driver: index for index, driver in enumerate(reversed(drivers))}
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.26, 0.74],
        vertical_spacing=0.08,
    )

    laps = list(range(1, total_laps + 1))
    wetness = [max(0.0, min(1.0, wetness_by_lap.get(lap, 0.0))) for lap in laps]
    fig.add_trace(
        go.Bar(
            x=laps,
            y=wetness,
            marker={
                "color": wetness,
                "colorscale": [[0.0, "#eff5fa"], [1.0, LAB_COLORS["decision_blue"]]],
                "cmin": 0,
                "cmax": 1,
            },
            hovertemplate="Lap %{x}<br>Model track-state index: %{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for _, stint in stints.iterrows():
        driver = stint["name_acronym"]
        compound = str(stint["compound"]).upper()
        start = int(stint["stint_lap_start"])
        end = int(stint["stint_lap_end"])
        tyre_age_value = pd.to_numeric(stint.get("tyre_age_at_start", 0), errors="coerce")
        tyre_age = int(tyre_age_value) if pd.notna(tyre_age_value) else 0
        fig.add_trace(
            go.Scatter(
                x=[start, end],
                y=[y_positions[driver], y_positions[driver]],
                mode="lines+markers",
                line={"width": 16, "color": COMPOUND_COLORS.get(compound, "#999999")},
                marker={"size": 7, "color": COMPOUND_COLORS.get(compound, "#999999")},
                hovertemplate=(
                    f"Driver: {driver}<br>Tyre: {compound}<br>"
                    f"Laps: {start}-{end}<br>Starting tyre age: {tyre_age}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    event_colors = {
        "wet_warning": LAB_COLORS["slower_red"],
        "dry_confirmation": LAB_COLORS["decision_blue"],
        "race_control": "#a36a00",
    }
    if strategy_events:
        fig.add_trace(
            go.Scatter(
                x=[int(event["lap"]) for event in strategy_events],
                y=[1.03] * len(strategy_events),
                mode="markers",
                marker={
                    "symbol": "triangle-down",
                    "size": 9,
                    "color": [event_colors.get(str(event["kind"]), LAB_COLORS["muted"]) for event in strategy_events],
                },
                customdata=[[str(event["label"]), str(event["kind"])] for event in strategy_events],
                hovertemplate="Lap %{x}<br>%{customdata[0]}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    for event in strategy_events:
        lap = int(event["lap"])
        color = event_colors.get(str(event["kind"]), LAB_COLORS["muted"])
        fig.add_vline(
            x=lap,
            line_color=color,
            line_width=1,
            line_dash="dot",
        )

    fig.update_xaxes(title="Lap", range=[0.5, total_laps + 0.5], dtick=5, row=2, col=1)
    fig.update_yaxes(title="Track state", range=[0, 1.12], showgrid=False, row=1, col=1)
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(y_positions.values()),
        ticktext=list(y_positions.keys()),
        title="",
        showgrid=False,
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        height=max(390, 92 * len(drivers) + 160),
        margin={"l": 56, "r": 28, "t": 88, "b": 42},
    )
    return apply_lab_theme(fig)


def plot_compound_comparison(comparison: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if comparison.empty:
        return apply_lab_theme(fig)

    delta_column = "相对真实策略差距（秒）" if "相对真实策略差距（秒）" in comparison.columns else "相对真实差距（秒）"
    colors = [COMPOUND_COLORS.get(compound, "#999999") for compound in comparison["最后一段轮胎"]]
    fig.add_trace(
        go.Bar(
            x=comparison["最后一段轮胎"],
            y=comparison[delta_column],
            marker_color=colors,
            text=comparison[delta_column].map(lambda value: f"{value:+.2f}s"),
            textposition="outside",
            hovertemplate=(
                "轮胎：%{x}<br>"
                "相对真实策略差距：%{y:+.2f} 秒<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#666666")
    fig.update_layout(
        title="最后一段轮胎方案对比",
        xaxis_title="最后一段轮胎",
        yaxis_title="相对真实策略差距（秒，越低越快）",
        height=230,
        margin={"l": 50, "r": 30, "t": 45, "b": 35},
    )
    return apply_lab_theme(fig)


def plot_wetness_penalty(penalties: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if penalties.empty:
        return apply_lab_theme(fig)

    fig = px.line(
        penalties,
        x="lap_number",
        y="wet_penalty",
        color="tyre_category",
        color_discrete_map={
            "光头胎（软/中/硬）": "#d35400",
            "半雨胎": "#2e8b57",
        },
        labels={
            "lap_number": "圈数",
            "wet_penalty": "模型湿滑状态带来的单圈惩罚（秒）",
            "tyre_category": "轮胎类别",
        },
        title="当前模型赛道湿滑状态<br>光头胎与半雨胎的单圈惩罚",
    )
    fig.update_layout(
        height=280,
        margin={"l": 50, "r": 25, "t": 50, "b": 40},
        legend_title_text="轮胎类别",
    )
    fig.update_yaxes(rangemode="tozero")
    return apply_lab_theme(fig)


def plot_wetness_sources(sources: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sources.empty:
        return apply_lab_theme(fig)

    source_colors = {
        "当前使用：赛后重建状态": "#2f80ed",
        "赛后重建状态（参考）": "#7aaeea",
        "当前使用：真实天气推断": "#2f80ed",
        "真实天气推断（参考）": "#7aaeea",
        "当前使用：手动实验": "#9b51e0",
        "手动实验曲线（未使用）": "#c8a2f0",
    }
    for source_name, source_df in sources.groupby("source", sort=False):
        is_active = source_name.startswith("当前使用")
        fig.add_trace(
            go.Scatter(
                x=source_df["lap_number"],
                y=source_df["wetness"],
                mode="lines",
                name=source_name,
                line={
                    "color": source_colors.get(source_name, "#666666"),
                    "width": 4 if is_active else 2,
                    "dash": "solid" if is_active else "dot",
                },
                hovertemplate=(
                    "圈数：%{x}<br>"
                    "湿滑指数：%{y:.1f}<br>"
                    f"{source_name}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="当前实验雨势：真实天气 vs 手动实验",
        xaxis_title="圈数",
        yaxis_title="模型湿滑指数（0-1）",
        yaxis={"range": [0, 1]},
        height=280,
        margin={"l": 55, "r": 30, "t": 55, "b": 40},
        legend={"orientation": "h", "y": 1.18, "x": 0},
    )
    return apply_lab_theme(fig)


def plot_pace_residuals(residuals: pd.DataFrame, driver: str) -> go.Figure:
    fig = go.Figure()
    if residuals.empty:
        return apply_lab_theme(fig)

    driver_residuals = residuals[residuals["name_acronym"] == driver].copy()
    field_residuals = (
        residuals[["lap_number", "field_median_residual_smooth"]]
        .drop_duplicates()
        .sort_values("lap_number")
    )

    if not field_residuals.empty:
        fig.add_trace(
            go.Scatter(
                x=field_residuals["lap_number"],
                y=field_residuals["field_median_residual_smooth"],
                mode="lines",
                name="对比车手中位残差",
                line={"color": "#f2c94c", "width": 3},
            )
        )

    if not driver_residuals.empty:
        fig.add_trace(
            go.Scatter(
                x=driver_residuals["lap_number"],
                y=driver_residuals["pace_residual"],
                mode="lines+markers",
                name=f"{driver} 残差",
                line={"color": "#2f80ed", "width": 2},
                marker={"size": 5},
                hovertemplate=(
                    "圈数：%{x}<br>"
                    "残差：%{y:+.2f} 秒<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line_dash="dot", line_color="#888888")
    fig.update_layout(
        title="真实圈速残差<br>真实单圈 - 干地模型预期",
        xaxis_title="圈数",
        yaxis_title="残差（秒，越高代表真实圈速越慢）",
        height=320,
        margin={"l": 55, "r": 30, "t": 55, "b": 40},
    )
    return apply_lab_theme(fig)


def plot_weather_residual_comparison(track_state: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if track_state.empty:
        return apply_lab_theme(fig)

    fig.add_trace(
        go.Scatter(
            x=track_state["lap_number"],
            y=track_state["rainfall_wetness"],
            mode="lines",
            name="降雨记录推断",
            line={"color": "#2f80ed", "width": 2, "dash": "dot"},
        )
    )
    if "reconstructed_wetness" in track_state.columns:
        fig.add_trace(
            go.Scatter(
                x=track_state["lap_number"],
                y=track_state["reconstructed_wetness"],
                mode="lines",
                name="赛后重建状态",
                line={"color": "#1f5fbf", "width": 4},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=track_state["lap_number"],
            y=track_state["residual_signal"],
            mode="lines",
            name="多人共同残差信号",
            line={"color": "#f2c94c", "width": 2},
        )
    )
    fig.update_layout(
        title="天气推断与多人圈速残差对照<br>（两者不直接混合）",
        xaxis_title="圈数",
        yaxis_title="湿滑指数推断 / 异常信号",
        yaxis={"range": [0, 1]},
        height=360,
        margin={"l": 55, "r": 30, "t": 95, "b": 40},
        legend={"orientation": "h", "y": 1.10, "x": 0},
    )
    return apply_lab_theme(fig)


def plot_sensitivity_heatmap(
    summary: pd.DataFrame,
    reference_pit: int | None = None,
    reference_compound: str | None = None,
    view: str = "current",
) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return apply_lab_theme(fig)

    if view == "worst":
        value_column = "worst_case"
        cell_label = "最差"
        colorbar_title = "最差结果"
        chart_title = "干燥速度敏感性：每个策略在三种设定中，最差会怎样？"
        hover_label = "三种干燥速度设定中的最差结果"
    else:
        value_column = "current_delta" if "current_delta" in summary.columns else "median_delta"
        cell_label = "当前"
        colorbar_title = "当前重建"
        chart_title = "当前赛后重建下，哪些候选策略表现更好？"
        hover_label = "当前赛后重建结果"

    pivot = summary.pivot(index="compound", columns="slick_pit", values=value_column)
    pivot = pivot.reindex([compound for compound in ["SOFT", "MEDIUM", "HARD"] if compound in pivot.index])
    text = pivot.map(lambda value: f"{cell_label}<br>{value:+.2f}s")
    if (
        reference_pit in pivot.columns
        and reference_compound in pivot.index
    ):
        text.loc[reference_compound, reference_pit] = "真实策略<br>0.00s"
    display_labels = {
        "SOFT": "软胎",
        "MEDIUM": "中胎",
        "HARD": "硬胎",
    }
    x_labels = [
        f"第 {int(lap)} 圈<br>真实"
        if lap == reference_pit
        else f"第 {int(lap)} 圈"
        for lap in pivot.columns
    ]
    color_limit = max(0.25, float(pivot.abs().max().max()))
    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=[display_labels.get(compound, compound) for compound in pivot.index],
            z=pivot.values,
            text=text.values,
            texttemplate="%{text}",
            colorscale=STRATEGY_DELTA_COLORSCALE,
            zmin=-color_limit,
            zmax=color_limit,
            zmid=0,
            colorbar={"title": colorbar_title},
            hovertemplate=(
                "换回干胎：%{x}<br>"
                "最终轮胎：%{y}<br>"
                f"{hover_label}：%{{z:+.2f}} 秒"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=chart_title,
        xaxis_title="假设换回干胎的圈数",
        yaxis_title="最终轮胎",
        height=300,
        margin={"l": 70, "r": 30, "t": 55, "b": 45},
    )
    return apply_lab_theme(fig)


def plot_scenario_breakdown(scenario_results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if scenario_results.empty:
        return apply_lab_theme(fig)

    results = scenario_results.copy()
    results["label"] = results["delta"].map(
        lambda value: f"快 {-value:.2f}s" if value < 0 else f"慢 {value:.2f}s"
    )
    colors = [LAB_COLORS["faster_blue"] if value < 0 else LAB_COLORS["slower_red"] for value in results["delta"]]

    fig.add_trace(
        go.Bar(
            x=results["scenario"],
            y=results["delta"],
            marker_color=colors,
            text=results["label"],
            textposition="outside",
            hovertemplate="干燥速度设定：%{x}<br>相对真实策略：%{y:+.2f} 秒<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#666666")
    fig.update_layout(
        title="推荐候选方案在三种干燥速度设定下<br>的结果",
        xaxis_title="干燥速度设定",
        yaxis_title="相对真实策略（秒，负数更快）",
        height=250,
        margin={"l": 55, "r": 25, "t": 50, "b": 40},
        showlegend=False,
    )
    return apply_lab_theme(fig)


def plot_reliability_event_map(events: pd.DataFrame) -> go.Figure:
    """Show when the evaluation issued a signal, missed, or abstained."""
    fig = go.Figure()
    if events.empty:
        return apply_lab_theme(fig)

    data = events.copy().reset_index(drop=True)
    data["event_index"] = range(1, len(data) + 1)
    data["event_label"] = data.apply(
        lambda row: f"{row['year']} {row['country']} · {row['driver']}",
        axis=1,
    )
    data["error_plot"] = data["public_signal_error_laps"].fillna(0.0)
    data.loc[data["outcome"] == "abstain", "error_plot"] = 0.0
    outcome_labels = {
        "signal_accurate": "已发信号：±2 圈内",
        "signal_inaccurate": "已发信号：超出 ±2 圈",
        "abstain": "弃权：没有发出信号",
    }
    outcome_colors = {
        "signal_accurate": LAB_COLORS["actual_green"],
        "signal_inaccurate": LAB_COLORS["slower_red"],
        "abstain": "#8b949e",
    }
    reason_symbols = {
        "race_control_adjacent": "diamond",
        "multi_rain_phase": "triangle-up",
        "start_or_restart": "square",
        "clean_or_other": "circle",
    }
    reason_labels = {
        "race_control_adjacent": "赛会控制相邻",
        "multi_rain_phase": "多阶段降雨",
        "start_or_restart": "起步/重启相邻",
        "clean_or_other": "单阶段或其他",
    }
    for outcome in ["signal_accurate", "signal_inaccurate", "abstain"]:
        subset = data[data["outcome"] == outcome]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["event_index"],
                y=subset["error_plot"],
                mode="markers",
                name=outcome_labels[outcome],
                marker={
                    "color": outcome_colors[outcome],
                    "size": 9,
                    "opacity": 0.88,
                    "symbol": [reason_symbols.get(value, "circle") for value in subset["reason_group"]],
                    "line": {"color": "#ffffff", "width": 0.8},
                },
                customdata=subset[["event_label", "source", "reason_group", "public_signal_error_laps"]],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "来源：%{customdata[1]}<br>"
                    "原因：%{customdata[2]}<br>"
                    "误差：%{customdata[3]:+.0f} 圈<extra>"
                    + outcome_labels[outcome]
                    + "</extra>"
                ),
            )
        )
    finite_errors = pd.to_numeric(data["public_signal_error_laps"], errors="coerce").dropna()
    limit = max(4.0, float(finite_errors.abs().max()) if not finite_errors.empty else 4.0)
    fig.add_hline(y=2, line_dash="dot", line_color="#9aa4ad", line_width=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="#9aa4ad", line_width=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#cbd2d8", line_width=1)
    fig.update_layout(
        title="信号 / 误判 / 弃权全景",
        xaxis_title="事件顺序（悬停查看比赛与车手）",
        yaxis_title="公开信号相对真实换胎的误差（圈）",
        xaxis={"tickmode": "linear", "dtick": max(1, len(data) // 10)},
        yaxis={"range": [-limit, limit]},
        height=380,
        margin={"l": 65, "r": 25, "t": 62, "b": 55},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    return apply_lab_theme(fig)


def plot_reliability_reason_counts(events: pd.DataFrame) -> go.Figure:
    """Show which event conditions account for the evaluation set."""
    fig = go.Figure()
    if events.empty:
        return apply_lab_theme(fig)

    reason_labels = {
        "race_control_adjacent": "赛会控制相邻",
        "multi_rain_phase": "多阶段降雨",
        "start_or_restart": "起步/重启相邻",
        "clean_or_other": "单阶段或其他",
    }
    reason_colors = {
        "race_control_adjacent": "#c77800",
        "multi_rain_phase": "#8b5cf6",
        "start_or_restart": "#64748b",
        "clean_or_other": "#16803a",
    }
    grouped = events.groupby(["source", "reason_group", "outcome"], dropna=False).size().reset_index(name="count")
    totals = grouped.groupby(["source", "reason_group"], as_index=False)["count"].sum()
    for reason, label in reason_labels.items():
        subset = totals[totals["reason_group"] == reason]
        if subset.empty:
            continue
        detail = grouped[grouped["reason_group"] == reason].groupby("source").apply(
            lambda frame: ", ".join(
                f"{row.outcome}: {int(row.count)}"
                for row in frame.itertuples()
            ),
            include_groups=False,
        )
        fig.add_trace(
            go.Bar(
                x=subset["source"],
                y=subset["count"],
                name=label,
                marker_color=reason_colors[reason],
                customdata=[detail.get(source, "") for source in subset["source"]],
                hovertemplate="来源：%{x}<br>事件数：%{y}<br>结果拆分：%{customdata}<extra>" + label + "</extra>",
            )
        )
    fig.update_layout(
        title="这些事件为什么会变难？",
        xaxis_title="评估来源",
        yaxis_title="事件数",
        barmode="stack",
        height=320,
        margin={"l": 55, "r": 25, "t": 58, "b": 45},
        legend={"orientation": "h", "y": 1.14, "x": 0},
    )
    return apply_lab_theme(fig)
