# V2.4 FastF1 Historical-Source Protocol

## Why a Second Public Source

OpenF1's currently available 2023, 2024, and 2026 directories provide no
unseen eligible V2.4 event after the development-race exclusions. FastF1 is
therefore introduced as a separate historical-source route, not a replacement
for the existing OpenF1 evidence model.

FastF1 exposes session laps, weather data, and race-control messages for
historical F1 sessions. The adapter uses it only to scan for candidate events
and preserves the source label in every audit output.

## Frozen 2021 Scan

The first FastF1 scan covers every 2021 conventional Race session. The 2021
Turkish Grand Prix is excluded from any evaluation before results are inspected
because it was loaded once solely to verify the adapter's public field schema.
It may still appear in the availability audit.

Candidate selection requires both:

1. At least one FastF1 weather row with `Rainfall=True`.
2. At least one observed driver compound change from `INTERMEDIATE` or `WET`
   to `SOFT`, `MEDIUM`, or `HARD`, inferred from first valid lap of each stint.

The V2.4 event-state rule, abstention logic, and coverage metrics remain
unchanged. A selected FastF1 candidate must be converted into a documented
source-specific bundle before it can be scored; no metric is inferred from the
candidate scan alone.

## Frozen 2021 Result

The candidate scan selected Emilia Romagna and Hungary. Across all 33 observed
transitions, V2.4 issued 13 signals and abstained on 20. Six of the 13 issued
signals were within plus or minus two laps, with an issued-signal MAE of 3.69
laps. The row-level output and per-race breakdown are preserved in
`../results/v2_fastf1_evaluation_2021.csv` and `v2_fastf1_evaluation_2021.md`.

## Pre-Registered 2020 Extension

The next FastF1 evaluation scans every conventional 2020 Race session, with no
race exclusions. The V2.4 event states, rain-reset rule, three-lap
race-control radius, candidate criteria, and all reported coverage metrics are
frozen exactly as above. The 2021 result may not be used to tune any parameter
or select a 2020 race.

## Pre-Registered 2019 Extension

After the complete 2020 audit, the next scan covers every conventional 2019
Race session with no exclusions. The V2.4 event states, candidate criteria,
rain-reset rule, race-control radius, and coverage metrics remain unchanged.
The 2021 and 2020 outcomes may not be used to tune the rule or select a 2019
race.
