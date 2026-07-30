# F1 Strategy Lab V2.3 Validation Protocol

## Status

The five cases currently shown in the app are an **exploratory development
set**. They were used to design the evidence taxonomy and should not be
presented as a held-out accuracy benchmark.

V2.3 freezes the current method before any new holdout cases are evaluated.

## Frozen Decision-Time Rules

For each observed wet-tyre-to-slick transition:

1. Build the rainfall-only track-state proxy from OpenF1 weather records.
2. Use the current fresh slick/intermediate crossover threshold from the
   explicit tyre model.
3. Require two consecutive slick-compatible laps after a modelled wet phase.
4. Read public race-control messages available during the race.
5. Prefer `NORMAL GRIP CONDITIONS`; if absent, use `DRS ENABLED` as a weaker
   dry-track confirmation.
6. Show the interval between rainfall recovery and public confirmation when
   both are available.
7. Do not use shared pace residuals in the decision-time result.

The retrospective reconstruction remains a separate explanatory result and is
not scored as live decision evidence.

## Challenge-Set Inclusion Rule

A future race enters the holdout set only if all of the following are true:

- OpenF1 provides complete Race session, lap, stint, weather, and race-control
  records.
- At least one driver has an observed `INTERMEDIATE` or `WET` to `SOFT`,
  `MEDIUM`, or `HARD` stint transition.
- The race is not used to change thresholds, confirmation priority, minimum
  duration, or case-selection logic.

For a challenge-set race, every eligible wet-to-slick driver transition must be
reported. No driver may be removed after results are seen.

## V2.3 Challenge-Set Record

The frozen rule was evaluated on every qualifying 2025 OpenF1 Race session:
Australia (16 transitions), United Kingdom (17), and Belgium (20). The full
53-transition result was `1/53` within plus or minus two laps with `12.62`
laps mean absolute error. These races are therefore no longer eligible as an
unseen set for V2.4; they are retained as diagnostic data.

## V2.4 Pre-Registered Direction

Before looking for a new evaluation set, V2.4 will convert the rule from a
single race-level confirmation into an event-level state machine:

1. Reset dry-track evidence after each new rainfall episode.
2. Mark safety-car, virtual-safety-car, red-flag, start, and restart windows
   as **ambiguous**, rather than treating them as ordinary drying evidence.
3. Report an explicit **abstain** outcome when a driver's wet stint crosses
   multiple weather phases or an ambiguous race-control window.
4. Score both accuracy *and coverage* on a future untouched set: how often the
   model issues a signal, and how accurate it is when it does.

V2.4 may be tuned only on the already-seen 2025 diagnostic data. The next
scanned qualifying races must remain untouched until this state-machine rule
and its abstention thresholds are frozen in writing.

## Reported Metrics

- Number of observed transitions.
- Mean absolute lap error of the decision-time public signal.
- Fraction of transitions within plus or minus two laps.
- Number of dry and persistent-wet controls left without a false switch window.
- Evidence taxonomy for every case.

## Current Exploratory Set

| Case | Role | Status |
| --- | --- | --- |
| 2024 British GP | Primary wet-to-dry case | Exploratory |
| 2024 Canadian GP | Wet-to-dry stress case | Exploratory |
| 2023 Dutch GP | Multi-phase weather case | Exploratory |
| 2024 Dutch GP | Dry-race control | Exploratory |
| 2024 Sao Paulo GP | Persistent-wet control | Exploratory |

## Claim Boundary

The current `3/3 within plus or minus two laps` result applies only to the
three selected exploratory wet-to-dry cases. It is useful evidence that the
public-signal approach is worth testing, not a general accuracy claim.
