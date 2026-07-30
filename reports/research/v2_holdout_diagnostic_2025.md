# V2.3 2025 Challenge-Set Diagnostic

## Why This Exists

The original five V2 cases were used to develop the method. After freezing the
V2.3 decision-time rules, the project scanned every 2025 OpenF1 Race session
using the pre-defined inclusion rule: rainfall records plus at least one
observed wet-tyre-to-slick transition.

The scan selected Australia, Great Britain, and Belgium. All eligible driver
transitions were retained. No cases were removed after results were seen.

## Result

| Metric | V2.3 challenge-set result |
| --- | --- |
| Raw observed transitions | 53 |
| Public signal available | 53 |
| Within plus or minus two laps | 1 / 53 |
| Mean absolute error | 12.62 laps |
| Sustained wet-stint transitions | 30 |
| Sustained-event mean absolute error | 14.27 laps |

The original public-signal rule does not generalize to this broader challenge
set. The exploratory `3/3 within plus or minus two laps` result must therefore
not be read as general accuracy.

## Event-Quality Breakdown

| Event quality | Events | Mean absolute error | Within plus or minus two laps |
| --- | ---: | ---: | ---: |
| Start or restart | 23 | 10.48 laps | 0 |
| Race-control adjacent | 16 | 19.81 laps | 1 |
| Multi-rain phase | 14 | 7.93 laps | 0 |

Race-control-adjacent transitions are the most difficult group. A single
race-level DRS or grip message cannot distinguish a driver who changes tyres
because of a safety-car opportunity from a driver responding to a stable track
transition. This supports treating safety-car, virtual-safety-car, red-flag,
and restart phases as explicit ambiguous states rather than ordinary weather
evidence.

## Failure Mechanism

The rule used the first dry-track confirmation after an intermediate or wet
stint began. In multi-phase races, the same wet-tyre stint can span an earlier
dry interval, a later rain interval, safety-car periods, and a later slick
switch. An early `DRS ENABLED` or `NORMAL GRIP` message then becomes stale
evidence for the later transition.

For example, the 2025 Australian GP had DRS re-enabled on lap 12, but many
drivers switched from intermediates to slicks around laps 33-34 after later
weather and safety-car phases. Treating the lap-12 message as the relevant
confirmation produces an error of more than 20 laps.

## V2.4 Diagnostic Attempt

An exploratory reset-aware variant required confirmation after the most recent
raw rainfall observation. On this already-seen challenge set it reduced mean
absolute error to 9.30 laps, but remained inadequate. This is a diagnostic
result, not a new holdout score.

The first V2.4 state-machine gate then classifies an event as eligible only
when it is a single-rain-phase transition outside a start/restart or nearby
race-control window. It deliberately abstains on the other states instead of
pretending a race-level dry-track message can identify an individual driver's
switch point. In this particular 2025 diagnostic set, all 53 transitions are
ambiguous: 23 are start/restart events, 16 are race-control-adjacent, and 14
span multiple rain phases. The gate therefore issues `0/53` signals. This is
not a success metric; it is evidence that the next evaluation set needs some
clean, eligible events before accuracy and coverage can be meaningfully scored.

## What Changed in the Project Claim

The project no longer claims that public race-control confirmation can predict
the individual slick-switch lap across races. Its stronger and more defensible
claim is now:

> Public F1 data can support an explainable evidence reconstruction, but simple
> race-level dry-track confirmation is insufficient for a general per-driver
> tyre-switch model under multi-phase weather and race-control disruptions.

## Next Research Step

Before evaluating another unseen set, define a new event-level state machine
that explicitly handles rain resets, safety-car/red-flag phases, and whether a
driver's tyre stint spans multiple weather phases. Any V2.4 rule must then be
tested on a new, untouched candidate set.
