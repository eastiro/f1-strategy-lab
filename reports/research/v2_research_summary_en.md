# F1 Strategy Lab V2 Research Summary

## Question

Can an explainable model built from public F1 data reconstruct a wet-to-dry
switch back to slick tyres across multiple races, and state clearly when the
available evidence is too weak for a conclusion?

## Scope

The project is a retrospective strategy-analysis prototype, not a live race
predictor. It deliberately distinguishes between information that was public
during the race and information only available after the race.

## Evidence Modes

| Mode | Inputs | Intended use |
| --- | --- | --- |
| Rainfall-only baseline | OpenF1 binary rainfall and temperature records | Conservative weather baseline |
| Decision-time public signal | Rainfall baseline plus public race-control dry-track confirmation | Contemporary public evidence, not a recommendation engine |
| Retrospective reconstruction | Rainfall evidence plus shared multi-driver pace residuals | Post-race explanation only |

## Case Set and Results

Real events are now split into two clocks:

- **Pit service lap** (`actual_pit_lap`): public pit stop service lap; the
  strategy-timing anchor used for scoring.
- **First slick lap** (`first_slick_lap`): first lap whose compound record shows
  the new slick (often the pit-out lap). Used to explain the data timeline only.

Compatibility field `actual_transition_lap` equals `first_slick_lap`.

| Race | Pit service | First slick | Rainfall-only | Public signal | Retrospective | Error vs pit | Evidence type |
| --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2024 British GP, NOR | 39 | 40 | Lap 39 | Laps 39-41 | Lap 39 | +2 | Public signal coverage sufficient |
| 2024 Canadian GP, VER | 45 | 46 | No window | Lap 47 | Lap 30 | +2 | Rainfall evidence insufficient |
| 2023 Dutch GP, VER | 11 | 12 | No window | Lap 10 | Lap 19 | -1 | Multi-phase weather (first recovery only) |
| 2024 Dutch GP, NOR | — | — | No window | No window | No window | — | Dry-race control |
| 2024 Sao Paulo GP, VER | — | — | No window | No window | No window | — | Persistent-wet control |

The public-signal result ends within two laps of the **pit service** anchor in
all three wet-to-dry cases (3/3, MAE 1.67). This should not be read as a pure
weather prediction: Canada and 2023 Netherlands rainfall-only baselines both
fail to establish a sufficiently wet track state. The improved timing comes from
public race-control confirmation. The project measures distance from public
evidence to the real pit action; it does not explain why a team chose that lap.

For the 2023 Dutch GP, lap 11/12 belong to the first recovery pack; later rain
is a separate phase, not proof that the driver switched “before the rain.”

## 2025 Challenge-Set Result

After the V2.3 rule was frozen, a broader 2025 candidate screen produced 53
raw wet-to-slick driver transitions across Australia, Great Britain, and
Belgium. The simple public-signal rule was within two laps only once and had a
12.62-lap mean absolute error. This falsified its use as a general per-driver
switch predictor. The diagnostic showed why: a wet-tyre stint can span several
weather and race-control phases, so an earlier DRS or normal-grip confirmation
may be stale by the time a later slick switch occurs.

## Main Finding

The most useful output is not one claimed optimal lap. It is an evidence chain:

1. Separate public pit service from first slick appearance in lap data.
2. Keep the rainfall-only baseline visible.
3. Add public track-condition confirmation without using future pace data.
4. Show a range when the signals disagree.
5. Preserve controls and failures instead of forcing a strategy window.

The exploratory pilot therefore supports a narrower conclusion: public signals
can help reconstruct race context, but they need an event-level state machine
before they can be evaluated as a general tyre-switch detector.

## V2.4 Protocol and Coverage

V2.4 freezes an event-state gate before any further evaluation. It abstains on
start/restart events, transitions near a safety-car/virtual-safety-car/red-flag
phase, and wet stints that span more than one rainfall episode. Only a clean,
single-rain-phase event may receive a driver-level public signal. The 2025
diagnostic set contains no such clean events, so V2.4 issues `0/53` signals
there. This is correctly reported as insufficient evaluation coverage, not as
an accuracy result.

An availability audit of the 2026 OpenF1 Race directory also found no eligible
unseen candidate at the time of the scan. The project preserves these empty
results and pre-registers a historical unseen-set scan rather than selecting a
convenient race after the fact.

## V3 Reliability Panorama

V3 reorganizes the frozen V2.4 outputs into an applicability view. Across the
2025 challenge set and the 2021 FastF1 scan, 86 events are evaluated: 7 issued
signals land within plus or minus two laps, 59 issued signals fall outside that
window, and 20 events are abstentions. The interface keeps these outcomes and
the event conditions visible together.

This is not a new accuracy score. It answers a different question: when does
the method issue a public-signal window, when does it miss, and when does it
recognize that the evidence is too ambiguous for a responsible driver-level
claim? The reliability panorama is designed to make that boundary legible to a
first-time reader.

## Limitations

- Binary rainfall does not measure water depth, local intensity, or track grip.
- Race-control messages are condition confirmation, not a direct tyre-choice model.
- The tyre model uses simplified pace, wear, and fuel assumptions.
- Pace residuals are retrospective and can include non-weather effects.
- The five-race sample is too small for a general F1 strategy claim.
