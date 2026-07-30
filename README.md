# F1 Strategy Lab

An explainable, public-data Formula 1 strategy research project.

F1 Strategy Lab began with one narrow question: could a different final
wet-to-slick call have improved Lando Norris's result at the 2024 British Grand
Prix? The final V4 build keeps that case study separate from a broader audit of
whether public race signals can reliably support the same decision across races.

The interface is Chinese-first. This README provides the short English research
story for reviewers.

## Final Findings

### Module 1: 2024 British GP strategy sandbox

- Norris's real baseline is lap 40 on soft tyres.
- No tested alternative beats that baseline under all three drying assumptions.
- Lap 39 on soft tyres is the closest robust alternative, but it is about 0.5
  seconds slower in the current reconstruction and about 1.7 seconds slower
  under the slower-drying scenario.
- The simulator is an explicit sensitivity model, not a reconstruction of
  McLaren's private data or a claim about the uniquely optimal real decision.

### Module 2: public-signal reliability audit

- The five-season audit contains 173 driver-level wet-to-slick transitions.
- The conservative rule issues 27 signals and abstains 146 times.
- All 27 signals are concentrated in two unusual races: Emilia Romagna 2021
  and the Netherlands 2023.
- The result does not support a stable cross-race, per-driver pit prediction.
  That negative result is part of the project: public data can explain evidence
  boundaries without pretending to be a live strategy engine.

## What The App Does

- Reconstructs the 2024 Silverstone race context from OpenF1 data.
- Lets visitors vary the final stop lap, slick compound, and drying scenario.
- Compares candidates against Norris's real strategy across three assumptions.
- Explains the model-internal lap-time difference by phase.
- Audits a transparent public-signal rule across five seasons.
- Shows when the rule issues a signal, when it fails, and when it refuses to
  make a driver-level claim.

## Research Boundary

This is a post-race research tool, not a live race service.

- OpenF1 rainfall is a binary observation, not water depth.
- Race-control messages confirm public track conditions; they do not prove an
  optimal tyre decision.
- Tyre degradation, fuel effect, intermediate wear, and pit loss are simplified.
- Pace residuals can include traffic, safety-car, driver, and tyre effects.
- The public-signal audit does not validate the simulator's physical model.
- No private team telemetry, tyre-temperature data, or real-time prediction is
  claimed.

## Data And Reproducibility

Public sources:

- [OpenF1](https://openf1.org/) timing, stint, weather, driver, pit, and
  race-control records.
- [FastF1](https://docs.fastf1.dev/) historical timing and session records.

Processed case data and audit outputs are checked into the repository so the
published app does not require a live API request at startup. Detailed source
interpretation is documented in [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py --server.port 8506
```

On Windows, `启动中文网站.bat` starts the same app and opens the browser
automatically.

Open `http://localhost:8506/`.

## Verification

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

The final local V4 verification passes 148 tests, including strategy-model
boundaries, phase-delta conservation, five-season audit totals, reader-path
wording, and both Streamlit entry points.

## Project Structure

```text
app/streamlit_app.py        Final V4 interactive research app
src/strategy_model.py       Simplified tyre and track-state model
src/conclusion_summary.py   Three-scenario strategy conclusion
src/strategy_delta_explanation.py
                             Phase-level explanation of strategy gaps
src/five_season_display.py  Five-season audit presentation helpers
data/                       Bundled public race data
reports/research/           Research protocols and diagnostics
reports/results/            Reproducible audit outputs
tests/                      Model, data, and interface regression tests
```

## Deployment

The public entry point is `app/streamlit_app.py`. Deployment instructions are
in [`DEPLOYMENT.md`](DEPLOYMENT.md).

Current research build: **V4 final two-module case study**.
