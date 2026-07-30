# V3 Five-Season (2021–2025) FastF1 Robustness Audit — Report

> This five-season output is a **retrospective robustness audit**. It does not create one combined accuracy claim, validate V1's physics, or establish a real-time pit-decision capability.

## 1. Per-season summary (primary view)

MAE and the issued ±2-lap count are computed **only over issued signals**, measured against the honest timing anchor (independent pit-service lap when FastF1 yields one, otherwise the first-slick fallback). Seasons that issued no signals show `N/A (no issued signals)` — never `0.0`.

| Season | Exposure | Scanned | Candidates | No-rain | Other non-cand. | Transitions | Signals | Abstain | Pit anc. | Fallback | Issued ±2 laps | MAE (laps) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | previously audited | 22 | 2 | 16 | 4 | 33 | 13 | 20 | 33 | 0 | 0 | 4.69 |
| 2022 | new pre-registered extension | 22 | 2 | 16 | 4 | 34 | 0 | 34 | 34 | 0 | 0 | N/A (no issued signals) |
| 2023 | partially development-exposed | 22 | 1 | 17 | 4 | 15 | 14 | 1 | 15 | 0 | 14 | 0.43 |
| 2024 | partially development-exposed | 24 | 2 | 19 | 3 | 38 | 0 | 38 | 38 | 0 | 0 | N/A (no issued signals) |
| 2025 | development-exposed | 24 | 3 | 18 | 3 | 53 | 0 | 53 | 33 | 20 | 0 | N/A (no issued signals) |

## 1b. Timing-anchor method (honest pit vs first-slick)

Task 11 reported the issued-signal distance as if every anchor were an observed **pit-service** lap. That was false: the frozen V2.4 rule's `actual_transition_lap` is first-slick / stint-boundary semantics, and the FastF1 adapter does not emit a `*_pit.csv`. Task 12 adds a reporting-only anchor helper that reads the cached FastF1 session directly and, for every observed wet-to-slick transition, applies a **deterministic** rule:

1. `wet_stint_start = first_slick_lap - wet_stint_laps` (exact: the first slick lap is one after the wet stint ends).
2. Candidate pit laps = FastF1 laps of the same driver with `PitInTime` not null, whose lap lies in `[wet_stint_start, first_slick_lap]` (extended by +1 lap for a timing mismatch).
3. The latest candidate at or before `first_slick_lap` is the pit stop that fitted the new slick compound.
4. If such a pit stop exists -> `timing_anchor_source = pit_service`; otherwise -> `first_slick_fallback`. A fallback is **never** labelled as a pit-service anchor, and no event is dropped or invented to create a pit lap.

Reporting error = `signal_lap - timing_anchor_lap`. The frozen `signed_error_laps` / `absolute_error_laps` (measured from `first_slick_lap`) are retained unchanged in the event CSV; the new `reporting_*` columns sit alongside them.

## 2. 2022 new pre-registered extension

2022 is the only newly pre-registered season in this extension. It was scanned with the same FastF1 source and frozen V2.4 rule as every other year: 22 sessions scanned, 2 candidate(s), 34 raw transition(s), 0 signal(s), 34 abstention(s). It is highlighted separately and must not be declared conclusive by itself.

## 3. Exposure labels

- **2021**: `previously audited`
- **2022**: `new pre-registered extension`
- **2023**: `partially development-exposed`
- **2024**: `partially development-exposed`
- **2025**: `development-exposed`

## 4. Quarantined note — 2025 OpenF1 `1/53` is NOT combined

The pre-existing 2025 OpenF1 simple-rule pressure diagnostic (`1/53`) remains visible and unchanged. It is a **different source and a different rule**; it falsifies the earlier simple rule. It is **never added to, averaged with, or pooled into** this FastF1 audit. The two answer different, narrower questions.

## 5. Strongest valid conclusion (coverage / abstention / error / limits)

Across the five seasons, 10 candidate wet-to-slick race sessions were evaluated, yielding 173 observed driver transitions: the frozen V2.4 rule issued 27 public signal(s) and abstained on 146. The reported error distributions describe **how far an issued signal landed from the independent pit-service anchor when FastF1 yields one, or the first-slick fallback otherwise** (see §1b) — not a generalization rate. Per-season results (Section 1) remain the primary view; the mixed-exposure seasons are not collapsed into a single accuracy figure.

## 6. Explicit non-claims

- This audit does **not** validate V1's tyre/weather physics.
- It does **not** establish a real-time, per-lap pit-decision capability for teams.
- It does **not** produce a single combined accuracy figure for the five seasons, nor a generalization rate.
- No threshold, abstention state, or transition was retuned, selected, or removed based on these results.

## 7. Session-status accounting by season

- **2021** (previously audited): candidate=2, no_rainfall_record=16, no_wet_to_slick_transition=4
- **2022** (new pre-registered extension): candidate=2, no_rainfall_record=16, no_wet_to_slick_transition=4
- **2023** (partially development-exposed): candidate=1, no_rainfall_record=17, no_wet_to_slick_transition=4
- **2024** (partially development-exposed): candidate=2, no_rainfall_record=19, no_wet_to_slick_transition=3
- **2025** (development-exposed): candidate=3, no_rainfall_record=18, no_wet_to_slick_transition=3

## 8. Descriptive event accounting across mixed-exposure seasons

The following is **descriptive event accounting** (raw counts), not a success rate or generalization score:

- Total scanned sessions: 114
- Total candidates: 10
- Total observed transitions: 173
- Total issued signals: 27
- Total abstentions: 146

---

Reproducibility: generated by `scripts/build_five_season_report.py` from `v3_fastf1_audit_2021_2025.csv` and `v3_fastf1_events_2021_2025.csv`. The frozen rule source hashes are recorded in `v3_five_season_fastf1_protocol.md`.