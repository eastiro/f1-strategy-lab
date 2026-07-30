# V3 Five-Season FastF1 Robustness Audit — Protocol (pre-registered before scanning)

**Status:** frozen before any 2022–2025 scan was performed.
**Owner framing:** Codex (research framing). **Executor:** WorkBuddy (data/protocol/evaluator pass).
**Question:** retrospective robustness audit only — not a combined accuracy score, not a live predictor.

---

## 1. Exact research question

> Under one frozen FastF1 normalization path and the existing frozen V2.4
> public-signal state-machine rule, what do complete 2021–2025 season audits
> show about event availability, issued signals, abstentions, and issued-signal
> distance from an independent observed pit-service lap when available, otherwise
> an explicitly labelled first-slick fallback?

This is a **five-season retrospective robustness audit**. It is **not** one combined accuracy figure across the five seasons, a random representative sample, a live prediction benchmark, a claim about team decision quality, or validation of V1 physics.

## 2. Non-claims (must stay visible in the report)

- The five-season output does **not** create one combined accuracy claim.
- It does **not** validate V1's physics.
- It does **not** establish a real-time pit-decision capability.
- It does **not** retune, select, or remove any threshold / state / transition.
- The pre-existing 2025 OpenF1 simple-rule `1/53` result is a *different source
  and different rule* diagnostic and is **never combined** with this FastF1 audit.

## 3. Why FastF1 for all five years (uniform-source reason)

The project previously held two non-comparable evidence streams:

- 2025 OpenF1 simple-rule pressure diagnostic (`1/53`, OpenF1, separate protocol).
- 2021 FastF1 V2.4 coverage evaluation (state-machine rule, FastF1, source-specific
  protocol).

This task creates a **new, uniform five-season table** by using **FastF1 for every
2021–2025 season** and the **same already-frozen V2.4 state machine** for every
eligible event. Existing results are kept visible and unchanged; they are not
added or averaged.

## 4. Frozen-rule source hashes (recorded before scanning)

These files implement the evaluator / rule and the scanner / adapter. Their
bytes are frozen for this task; none were modified while executing it.

| File | SHA-256 (lowercase) |
| --- | --- |
| `src/evaluate_holdouts.py` | `10406aaa81ced90b601515233828d1a20c34ea5dcd76ab55fe03d30d55947317` |
| `src/fastf1_scanner.py` | `ed3bd2ef4252c8cf3090f1c0a807f4e183b3e1eeeaf8e8644009bbd2857b36b6` |
| `src/fastf1_adapter.py` | `b2ccabb45c8a1d2b315cf8b37cb246f8592a7384587abfcfa3d57e6b645dbbe7` |
| `src/v2_validation.py` | `831e77eaca0659e901efe8ce5e161933bae60ae120955f606ff683935030684a` |
| `src/openf1_client.py` | `ba6fed5f56b7661baa4c81e854c70d0ef4d8876fc099e545e0d9ad5df6db3dd8` |
| `src/weather_timing.py` | `56df5c3bb65e8e394f4a1ada6512a6a5d428e89d94ee5982ece244bc67bd41d2` |
| `src/strategy_model.py` | `711eaf10d1d5a55ca71fedf2eda9d50f38ef60d15e78e91b67e58e07703fc177` |

### Frozen evaluator entry point

`evaluate_holdouts.evaluate_v24_state_machine_race(year, country)` — called exactly
as-is, never edited. Its behaviour (verbatim from the frozen source):

- Reads normalized bundle `{slug}_laps_clean.csv`, `{slug}_stints.csv`,
  `{slug}_weather.csv`, `{slug}_race_control.json` from `data/processed` / `data/raw`.
- For each driver, finds observed wet→slick transitions
  (`observed_wet_to_slick_transitions` in `v2_validation.py`).
- `event_quality` classification (frozen):
  - `start_or_restart` if `wet_stint_laps <= 2 or actual_lap <= 5`
  - else `race_control_adjacent` if nearby race-control phases (red flag / VSC / SC)
  - else `multi_rain_phase` if `rain_episodes_before_transition >= 2`
  - else `single_rain_phase`
- `signal_eligible = (event_quality == "single_rain_phase")`
- `state_machine_outcome = "signal" if signal_eligible else "abstain"`
- `public_signal_error_laps` is `None` unless a signal is issued, then
  `public_latest - actual_lap` (signed).

### Candidate definition (frozen)

A race session is a **candidate** iff it has (a) at least one FastF1 rainfall
record and (b) at least one observed `INTERMEDIATE`/`WET` → `SOFT`/`MEDIUM`/`HARD`
compound transition in its lap data.

- `SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}`
- `WET_COMPOUNDS = {"INTERMEDIATE", "WET"}`

Implemented by `fastf1_scanner.scan_year` → `count_wet_to_slick_transitions`, reused
unchanged. The frozen rule's `actual_transition_lap` is **first-slick /
stint-boundary** semantics (the first lap whose compound record shows the new
slick), equal to `first_slick_lap` for wet→slick events — it is **not** a
pit-service lap. Task 12 (post-hoc, reporting-only) adds a separate, honest
timing anchor that derives an **independent FastF1 pit-service lap** when the
cached session yields one, and otherwise retains `first_slick_lap` as an
explicit `first_slick_fallback`. Neither the candidate definition nor the V2.4
rule is changed by that addition.

## 5. Season list and exposure labels (frozen)

| Season | Exposure label | Reason |
| --- | --- | --- |
| 2021 | `previously audited` | Existing FastF1 V2.4 result already exists. |
| 2022 | `new pre-registered extension` | No prior project scan/evaluation of this season. |
| 2023 | `partially development-exposed` | 2023 Netherlands was used in an OpenF1 exploratory case. |
| 2024 | `partially development-exposed` | 2024 Silverstone/Canada/Netherlands/Brazil were used in development. |
| 2025 | `development-exposed` | The OpenF1 challenge diagnostic informed the later V2.4 abstention direction. |

Consequences (frozen reporting rules):

- A five-year summary may report **coverage and error distributions**, never an
  untouched holdout score or a single generalization rate.
- `2022` is the only newly pre-registered season; it may be highlighted separately
  but must not be declared conclusive by itself.
- `2021` Turkish GP technical fixture: **see Section 6**.

## 6. Exclusions (recorded before scanning)

**Existing exclusions:** only the 2021 Turkish GP fixture.

> **Discrepancy note (honest provenance).** This repository contains **no
> code-level 2021 Turkey fixture or exclusion**. `scan_year` treats the 2021
> Turkish Grand Prix like any other race; the existing `v2_fastf1_audit_2021.csv`
> records it as `selection_status = no_rainfall_record` (0 rainfall records). Its
> session is **retained in the audit as `no_rainfall_record`**, hence it never
> becomes a candidate or a score row — satisfying the *intent* of the
> pre-existing exclusion without introducing new code. No fabricated fixture was
> added. This is an audit-provenance clarification, not a rule or data change, and
> it is carried forward unchanged into the Task 12 outputs. Flagged here for Codex
> to confirm whether an explicit annotation is desired.

All other sessions are scanned; unavailable/failed sessions are recorded with an
explicit status, never silently dropped.

## 7. Complete scan plan

- **Tooling:** new resumable runner `src/fastf1_five_season_runner.py`
  (module run via `python -m src.fastf1_five_season_runner`). It reuses the
  existing `scan_year`, `save_fastf1_bundle`, and `evaluate_v24_state_machine_race`
  exactly; it adds no rule logic.
- **Scan scope:** every conventional `Race` session (`"R"`) in 2021, 2022, 2023,
  2024, 2025 (all rounds). No hand-picking of wet races.
- **Per-season batching (resumable):** `--year Y --round-start A --round-end B`.
  Completed rounds are skipped on re-run (idempotent append).
- **Per candidate:** normalize bundle with `save_fastf1_bundle(year, event_name,
  country_slug=short)` (reuses existing `data/processed` bundles when the slug
  matches), then `evaluate_v24_state_machine_race(year, short)`.
- **Outputs (per year, then combined):**
  - `reports/results/v3_fastf1_audit_{year}.csv` — every scanned session + status.
  - `reports/results/v3_fastf1_events_{year}.csv` — every evaluated driver transition.
  - `reports/results/v3_fastf1_audit_2021_2025.csv` — union of all audit rows.
  - `reports/results/v3_fastf1_events_2021_2025.csv` — union of all event rows.
  - `reports/research/v3_five_season_fastf1_report.md` — readable results.
- **Slug mapping:** FastF1 `EventName` (e.g. `"Hungarian Grand Prix"`) → short
  slug-name (e.g. `"Hungary"`) so bundle filenames match the existing
  `2021_hungary`, `2024_united_kingdom`, `2023_netherlands`, etc. bundles and the
  `make_race_slug` contract used by the frozen evaluator. The full `EventName` is
  still passed to `fastf1.get_session` for reliable resolution and is recorded in
  the `event_name` column for provenance.

## 8. Reporting rules (frozen)

1. A five-row season table with session-status totals, raw transitions, signals,
   abstentions, issued-signal within-±2-lap count, and MAE.
2. A separate line/table for `2022 new pre-registered extension`.
3. The exposure label of every season.
4. **No** combined accuracy figure for all five seasons is reported.
5. A separate, clearly quarantined note that 2025 OpenF1 `1/53` is a different
   source/rule diagnostic and is not combined.
6. The strongest valid conclusion, expressed only in terms of coverage,
   abstention, error, public-record evidence availability, and limits.
7. Explicit non-claims about V1 physics and real-time team recommendations.

If a descriptive all-season aggregate is shown, it is labelled
`descriptive event accounting across mixed-exposure seasons` with raw counts,
never a success rate or generalization score. Per-season results remain primary.

## 9. Reproducibility

Exact commands are recorded in the execution record appended to
`tasks/external-agent/EXTERNAL_AGENT_TASK_11.md` and summarized in the report.
The runner is deterministic given the FastF1 cache; re-running a year reproduces
its rows (idempotent append skips completed rounds).

## 10. Execution-environment note (added during execution — runtime only, no frozen-rule change)

The network access to FastF1 changed **twice** during execution; the runner was
adjusted at runtime (new file only, no frozen-rule byte touched) to match:

- **Phase A — 2026-07-22 (first attempts):** this sandbox's egress IP received
  **HTTP 403** from the FastF1 primary host `livetiming.formula1.com` (AWS S3
  backend) on every endpoint. FastF1's built-in mirror fallback only triggers on
  a **raised exception**, not on the silent hang the 403 produced, so a direct
  scan stalled indefinitely. Mitigation: `src/fastf1_five_season_runner.py` set
  `fastf1._api.base_url = fastf1._api.base_url_mirror` (official mirror
  `https://livetiming-mirror.fastf1.dev`). This served 2021–2022 only.
- **Phase B — 2026-07-23 (actual execution of 2023–2025):** the primary host was
  **restored** and returned **HTTP 200** for 2021–2025 (verified: race-control
  payloads of 13025 / 12532 / 19273 bytes for 2023 / 2024 / 2025). The mirror,
  however, **only serves ≤ 2022 and 404s for 2023–2025**, so forcing it was wrong.
  The runner was reverted to **FastF1's default primary host (no `base_url`
  override)**. FastF1's own built-in mirror fallback (on a raised exception)
  still applies if the primary 403s again.
- **Durable anti-hang safeguard (both phases):** FastF1 routes *every* HTTP
  request through `fastf1.req.Cache._cached_request`, which forwards `**kwargs`
  to the session and **never sets a `timeout`**; under burst load a host can
  *silently hang* the connection (observed: all three 2023–2025 scans stalled
  >50 min mid-fetch). A first attempt patched `requests.Session.get/post`, but
  that is bypassed by the cached-session class, so it was dead code. The working
  fix wraps `Cache._cached_request` itself and injects `_REQUEST_TIMEOUT = 90` on
  every request (`src/fastf1_five_season_runner.py` and the Task-12 anchor
  helper both apply it; idempotent). A hung fetch then becomes a fast
  `requests.exceptions.Timeout`, which FastF1 logs as "Failed to load …" and the
  runner's pre-warm retry loop then recovers from. This is runtime-only; no
  frozen file is touched.

- **Cache:** scans read/write the project cache `data/fastf1_cache` (not FastF1's
  default TEMP cache) so the audit is reproducible and reuses already-cached
  seasons (2021 + 2022 were fully cached beforehand; 2023/2024/2025 were fetched
  during the 2026-07-23 execution).
- **Rate-limit handling:** any host can rate-limit a burst of uncached stream
  fetches (last to arrive: weather / race-control). The runner pre-warms each
  session with a retry loop (4 attempts, `sleep(2)` between sessions) so all
  required streams (weather + race-control included) land in cache before the
  frozen `scan_year` / `save_fastf1_bundle` / `evaluate_v24_state_machine_race`
  read from it. A session whose pre-warm still fails is recorded with an explicit
  status, never silently dropped.
- **Interpreter:** the scan runs on the system Python that has FastF1 3.8.3
  installed (`C:/Users/EASTIRO/AppData/Local/Programs/Python/Python313/python.exe`);
  the managed 3.13.12 interpreter in this workspace does not have FastF1.
