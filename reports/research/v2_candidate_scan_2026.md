# V2.4 2026 Candidate-Scan Record

## Protocol Status

This scan happened after `v2_4_evaluation_protocol.md` froze the V2.4
event-state and abstention rules. It is an availability audit, not a model
result.

## OpenF1 Race-Session Audit

The OpenF1 2026 Race-session directory returned 24 listed sessions at the
time of the scan:

| Selection status | Sessions |
| --- | ---: |
| Missing weather data | 15 |
| No rainfall record | 8 |
| Rainfall but no wet-to-slick transition | 1 |
| Eligible candidate | 0 |

No 2026 race may be reported as a V2.4 evaluation case from this scan. The
absence of candidates is preserved rather than replaced with a hand-picked
race.

## Next Data Step

Use a historical OpenF1 season not used during V2 development as a separate
pre-registered unseen set, excluding every race already downloaded for the
exploratory cases. Keep the same event-state rule and report coverage before
any accuracy result.
