# V2.4 FastF1 2021 Historical Evaluation

## Design

FastF1 was introduced only after the OpenF1 availability audits produced no
unseen candidates. The adapter was verified using the 2021 Turkish Grand Prix,
which was pre-excluded from evaluation. The remaining full-season candidate
scan selected exactly two races: Emilia Romagna and Hungary. Both are included
below; no driver transition was removed.

## Coverage Result

| Metric | Result |
| --- | ---: |
| Observed wet-to-slick transitions | 33 |
| Signals issued | 13 |
| Abstentions | 20 |
| Coverage | 39.4% |
| Issued signals within plus or minus two laps | 6 / 13 |
| Issued-signal MAE | 3.69 laps |

## By Race

| Race | Events | Signals | Abstentions | Within plus or minus two laps | Issued-signal MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2021 Emilia Romagna | 19 | 13 | 6 | 6 | 3.69 laps |
| 2021 Hungary | 14 | 0 | 14 | 0 | Not applicable |

Hungary is a useful check of the abstention path: all 14 transitions occur at
the start/restart boundary and are withheld. Emilia Romagna supplies the 13
single-rain-phase issued signals; six additional transitions near safety-car
context are withheld.

## Interpretation

This is not a claim that the model predicts every wet-to-dry tyre call. It is
the first honest V2.4 coverage result: the state machine refuses to score
conditions it categorizes as ambiguous, and its issued signals remain a
moderate-error public-data estimate rather than a race-team recommendation.

The full row-level result is in `../results/v2_fastf1_evaluation_2021.csv`; the source
scanner audits are `../results/v2_fastf1_audit_2021_rounds_01_11.csv` and
`../results/v2_fastf1_audit_2021_rounds_12_22.csv`.
