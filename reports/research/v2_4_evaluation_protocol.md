# F1 Strategy Lab V2.4 Evaluation Protocol

## Purpose

V2.3 showed that a single race-level dry-track message is not a reliable
per-driver slick-switch predictor in disrupted or multi-phase wet races. V2.4
therefore tests a narrower question:

> Can a public-data rule identify the situations where it has enough clean,
> decision-time evidence to issue a wet-to-dry signal, and abstain elsewhere?

## Frozen Event-State Rule

For every observed `INTERMEDIATE` or `WET` to slick transition, the rule first
assigns exactly one event state:

| State | Condition | V2.4 action |
| --- | --- | --- |
| `start_or_restart` | Wet stint lasts at most two laps, or switch occurs by lap 5 | Abstain |
| `race_control_adjacent` | Safety car, virtual safety car, or red flag appears within three laps of the switch | Abstain |
| `multi_rain_phase` | Two or more distinct rainfall episodes occur before the switch | Abstain |
| `single_rain_phase` | None of the above | Issue a signal |

For `single_rain_phase` only:

1. Map rainfall observations to the driver's laps.
2. Reset the earliest usable dry-track confirmation after the latest rainfall
   observation in that wet stint.
3. Combine rainfall recovery and the first later `NORMAL GRIP CONDITIONS` or,
   if absent, `DRS ENABLED` confirmation.
4. Report the latest available boundary as the signal lap, plus the evidence
   interval when both inputs exist.

The rule does not use pace residuals, future laps, manual case exclusions, or
driver-specific tuning.

## Evaluation Contract

- The 2025 Australia, United Kingdom, and Belgium races are development-only
  diagnostics. They may not be used as V2.4 holdout results.
- Before another scan, this document and the implementation in
  `src/evaluate_holdouts.py` are frozen together.
- Every qualifying race in a newly scanned year is retained; every observed
  wet-to-slick transition is included.
- The report must show total events, signals issued, abstentions, coverage,
  issued-signal MAE, and issued signals within plus or minus two laps.
- An empty eligible subset is reported as **insufficient evaluation coverage**,
  not as success or failure on accuracy.

## 2025 Diagnostic Expectation

The existing 53-event challenge set contains no `single_rain_phase` events:
23 start/restart, 16 race-control-adjacent, and 14 multi-rain-phase events.
V2.4 should issue zero signals there. This confirms that the rule's abstention
path is active; it does not establish V2.4 predictive accuracy.

## Historical Unseen-Set Extension

The 2026 availability audit produced no eligible candidates. The next scan may
therefore use the 2023 historical Race directory, with one pre-registered
exclusion: the Netherlands, which is already part of the V2 exploratory set.
No result from the 2023 scan may change the state definitions, reset rule,
three-lap race-control radius, or coverage calculation above.

The same historical extension now applies to the 2024 Race directory. These
races are excluded before results are inspected because their OpenF1 bundles
were already downloaded during V2 development: Canada, Monaco, Netherlands,
Brazil, and United Kingdom. No result from the 2024 scan may change the frozen
rule or its reported metrics.
