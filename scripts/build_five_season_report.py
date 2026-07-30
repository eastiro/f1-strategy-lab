"""Build the five-season FastF1 audit report from the combined CSVs.

Reads ``reports/results/v3_fastf1_audit_2021_2025.csv`` and
``reports/results/v3_fastf1_events_2021_2025.csv`` and writes
``reports/research/v3_five_season_fastf1_report.md``.

The report is intentionally descriptive and non-pooled: it shows per-season
tables, the 2022 extension separately, exposure labels, and explicit non-claims.
No "five-year accuracy" percentage is ever computed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
REPORT_PATH = PROJECT_ROOT / "reports" / "research" / "v3_five_season_fastf1_report.md"

SEASONS = [2021, 2022, 2023, 2024, 2025]
EXPOSURE = {
    2021: "previously audited",
    2022: "new pre-registered extension",
    2023: "partially development-exposed",
    2024: "partially development-exposed",
    2025: "development-exposed",
}


def _season_summary(audit: pd.DataFrame, events: pd.DataFrame, year: int) -> dict[str, object]:
    a = audit[audit["season"] == year]
    e = events[events["season"] == year]
    status_counts = a["selection_status"].value_counts().to_dict()
    signals = e[e["state_machine_outcome"] == "signal"]
    abstain = e[e["state_machine_outcome"] == "abstain"]
    # Reporting error is measured against the *honest* timing anchor: an
    # independent FastF1 pit-service lap when available, else the first-slick
    # fallback. It is computed only over issued signals.
    rep_err = pd.to_numeric(signals.get("reporting_absolute_error_laps"), errors="coerce").dropna()
    within2 = int((rep_err <= 2).sum()) if not rep_err.empty else 0
    mae = round(float(rep_err.mean()), 2) if not rep_err.empty else "N/A (no issued signals)"
    pit = int((e["timing_anchor_source"] == "pit_service").sum())
    fallback = int((e["timing_anchor_source"] == "first_slick_fallback").sum())
    return {
        "season": year,
        "exposure": EXPOSURE[year],
        "sessions_scanned": int(len(a)),
        "candidates": int((a["selection_status"] == "candidate").sum()),
        "no_rainfall": int((a["selection_status"] == "no_rainfall_record").sum()),
        "other_non_candidate": int(
            a["selection_status"].isin(
                ["no_wet_to_slick_transition", "timing_laps_unavailable", "weather_unavailable",
                 "session_unavailable", "bundle_error", "evaluation_error"]
            ).sum()
        ),
        "raw_transitions": int(len(e)),
        "signals": int(len(signals)),
        "abstentions": int(len(abstain)),
        "pit_anchors": pit,
        "fallback_anchors": fallback,
        "within_2_laps": within2,
        "mae": mae,
        "status_counts": status_counts,
    }


def build() -> None:
    audit = pd.read_csv(RESULTS_DIR / "v3_fastf1_audit_2021_2025.csv")
    events = pd.read_csv(RESULTS_DIR / "v3_fastf1_events_2021_2025.csv")

    summaries = [_season_summary(audit, events, y) for y in SEASONS]

    lines: list[str] = []
    lines.append("# V3 Five-Season (2021–2025) FastF1 Robustness Audit — Report")
    lines.append("")
    lines.append(
        "> This five-season output is a **retrospective robustness audit**. It does "
        "not create one combined accuracy claim, validate V1's physics, or establish a "
        "real-time pit-decision capability."
    )
    lines.append("")

    # Per-season table
    lines.append("## 1. Per-season summary (primary view)")
    lines.append("")
    lines.append(
        "MAE and the issued ±2-lap count are computed **only over issued signals**, "
        "measured against the honest timing anchor (independent pit-service lap when "
        "FastF1 yields one, otherwise the first-slick fallback). Seasons that issued "
        "no signals show `N/A (no issued signals)` — never `0.0`."
    )
    lines.append("")
    header = (
        "| Season | Exposure | Scanned | Candidates | No-rain | Other non-cand. | "
        "Transitions | Signals | Abstain | Pit anc. | Fallback | Issued ±2 laps | MAE (laps) |"
    )
    lines.append(header)
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for s in summaries:
        lines.append(
            f"| {s['season']} | {s['exposure']} | {s['sessions_scanned']} | {s['candidates']} | "
            f"{s['no_rainfall']} | {s['other_non_candidate']} | {s['raw_transitions']} | "
            f"{s['signals']} | {s['abstentions']} | {s['pit_anchors']} | {s['fallback_anchors']} | "
            f"{s['within_2_laps']} | {s['mae']} |"
        )
    lines.append("")

    # Timing-anchor method (honest pit vs first-slick) — Task 12 correction.
    lines.append("## 1b. Timing-anchor method (honest pit vs first-slick)")
    lines.append("")
    lines.append(
        "Task 11 reported the issued-signal distance as if every anchor were an observed "
        "**pit-service** lap. That was false: the frozen V2.4 rule's `actual_transition_lap` "
        "is first-slick / stint-boundary semantics, and the FastF1 adapter does not emit a "
        "`*_pit.csv`. Task 12 adds a reporting-only anchor helper that reads the cached "
        "FastF1 session directly and, for every observed wet-to-slick transition, applies a "
        "**deterministic** rule:"
    )
    lines.append("")
    lines.append(
        "1. `wet_stint_start = first_slick_lap - wet_stint_laps` (exact: the first slick "
        "lap is one after the wet stint ends)."
    )
    lines.append(
        "2. Candidate pit laps = FastF1 laps of the same driver with `PitInTime` not null, "
        "whose lap lies in `[wet_stint_start, first_slick_lap]` (extended by +1 lap for a "
        "timing mismatch)."
    )
    lines.append(
        "3. The latest candidate at or before `first_slick_lap` is the pit stop that fitted "
        "the new slick compound."
    )
    lines.append(
        "4. If such a pit stop exists -> `timing_anchor_source = pit_service`; otherwise -> "
        "`first_slick_fallback`. A fallback is **never** labelled as a pit-service anchor, and "
        "no event is dropped or invented to create a pit lap."
    )
    lines.append("")
    lines.append(
        "Reporting error = `signal_lap - timing_anchor_lap`. The frozen `signed_error_laps` / "
        "`absolute_error_laps` (measured from `first_slick_lap`) are retained unchanged in the "
        "event CSV; the new `reporting_*` columns sit alongside them."
    )
    lines.append("")

    # 2022 extension separately
    lines.append("## 2. 2022 new pre-registered extension")
    lines.append("")
    s22 = next(s for s in summaries if s["season"] == 2022)
    lines.append(
        f"2022 is the only newly pre-registered season in this extension. It was scanned "
        f"with the same FastF1 source and frozen V2.4 rule as every other year: "
        f"{s22['sessions_scanned']} sessions scanned, {s22['candidates']} candidate(s), "
        f"{s22['raw_transitions']} raw transition(s), {s22['signals']} signal(s), "
        f"{s22['abstentions']} abstention(s). It is highlighted separately and must not be "
        f"declared conclusive by itself."
    )
    lines.append("")

    # Exposure labels
    lines.append("## 3. Exposure labels")
    lines.append("")
    for y in SEASONS:
        lines.append(f"- **{y}**: `{EXPOSURE[y]}`")
    lines.append("")

    # Quarantined OpenF1 note
    lines.append("## 4. Quarantined note — 2025 OpenF1 `1/53` is NOT combined")
    lines.append("")
    lines.append(
        "The pre-existing 2025 OpenF1 simple-rule pressure diagnostic (`1/53`) remains "
        "visible and unchanged. It is a **different source and a different rule**; it "
        "falsifies the earlier simple rule. It is **never added to, averaged with, or "
        "pooled into** this FastF1 audit. The two answer different, narrower questions."
    )
    lines.append("")

    # Strongest valid conclusion
    lines.append("## 5. Strongest valid conclusion (coverage / abstention / error / limits)")
    lines.append("")
    tot_cand = sum(s["candidates"] for s in summaries)
    tot_trans = sum(s["raw_transitions"] for s in summaries)
    tot_sig = sum(s["signals"] for s in summaries)
    tot_abs = sum(s["abstentions"] for s in summaries)
    lines.append(
        f"Across the five seasons, {tot_cand} candidate wet-to-slick race sessions were "
        f"evaluated, yielding {tot_trans} observed driver transitions: the frozen V2.4 "
        f"rule issued {tot_sig} public signal(s) and abstained on {tot_abs}. The reported "
        f"error distributions describe **how far an issued signal landed from the independent "
        f"pit-service anchor when FastF1 yields one, or the first-slick fallback otherwise** "
        f"(see §1b) — not a generalization rate. Per-season results (Section 1) "
        f"remain the primary view; the mixed-exposure seasons are not collapsed into a "
        f"single accuracy figure."
    )
    lines.append("")

    # Non-claims
    lines.append("## 6. Explicit non-claims")
    lines.append("")
    lines.append("- This audit does **not** validate V1's tyre/weather physics.")
    lines.append(
        "- It does **not** establish a real-time, per-lap pit-decision capability for teams."
    )
    lines.append(
        "- It does **not** produce a single combined accuracy figure for the five seasons, "
        "nor a generalization rate."
    )
    lines.append(
        "- No threshold, abstention state, or transition was retuned, selected, or removed "
        "based on these results."
    )
    lines.append("")

    # Per-season status breakdown (full availability accounting)
    lines.append("## 7. Session-status accounting by season")
    lines.append("")
    for s in summaries:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(s["status_counts"].items()))
        lines.append(f"- **{s['season']}** ({s['exposure']}): {counts}")
    lines.append("")

    # Descriptive all-season aggregate (raw counts only)
    lines.append("## 8. Descriptive event accounting across mixed-exposure seasons")
    lines.append("")
    lines.append(
        "The following is **descriptive event accounting** (raw counts), not a success "
        "rate or generalization score:"
    )
    lines.append("")
    lines.append(
        f"- Total scanned sessions: {sum(s['sessions_scanned'] for s in summaries)}"
    )
    lines.append(f"- Total candidates: {tot_cand}")
    lines.append(f"- Total observed transitions: {tot_trans}")
    lines.append(f"- Total issued signals: {tot_sig}")
    lines.append(f"- Total abstentions: {tot_abs}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Reproducibility: generated by `scripts/build_five_season_report.py` from "
        "`v3_fastf1_audit_2021_2025.csv` and `v3_fastf1_events_2021_2025.csv`. The "
        "frozen rule source hashes are recorded in "
        "`v3_five_season_fastf1_protocol.md`."
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH} ({len(lines)} lines)")


if __name__ == "__main__":
    build()
