# Data Sources and Reproducibility

## Primary Source

This project uses the public OpenF1 API for race-session data.

For each included race, the download client requests:

- `sessions`
- `drivers`
- `laps`
- `pit`
- `stints`
- `weather`
- `race_control`

The client is implemented in `src/openf1_client.py`. It retries transient API
failures and stores the raw response alongside a processed CSV version.

## Included Validation Data

The portfolio build includes five race bundles:

- 2024 United Kingdom
- 2024 Canada
- 2023 Netherlands
- 2024 Netherlands
- 2024 Brazil

The exact expected files are documented in `data/README.md` and checked by the
test suite.

## Secondary Historical Source: FastF1

V2.4 uses FastF1 only for a separately documented historical-source route
after the OpenF1 availability audit found no unseen candidate. The source
adapter in `src/fastf1_adapter.py` normalizes FastF1 race laps, weather, and
race-control messages into the V2 input schema and writes source metadata next
to each bundle.

The first frozen historical scan covers the 2021 season. It retains both
eligible candidates, Emilia Romagna and Hungary, and reports coverage and
abstention alongside error. See `reports/research/v2_4_fastf1_source_protocol.md` and
`reports/research/v2_fastf1_evaluation_2021.md`.

## Interpretation Rules

- OpenF1 `rainfall` is treated as a binary observation, not measured rainfall
  volume or track water depth.
- Race-control messages such as `NORMAL GRIP CONDITIONS` and `DRS ENABLED` are
  used as public track-condition confirmation. They are not treated as direct
  labels of an optimal tyre decision.
- Shared pace residuals are used only in the retrospective reconstruction. They
  are not used in the decision-time public-signal result.
- FastF1's `Rainfall` field is also treated as a binary observation. FastF1
  results are labeled as a separate source and are not silently mixed into the
  OpenF1 exploratory benchmark.

## Re-download

```powershell
python -m src.download_openf1 --year 2024 --country "United Kingdom"
python -m src.prepare_dataset --year 2024 --country "United Kingdom"
```

Before publishing a new version, verify OpenF1's current usage terms and data
availability for the intended deployment.
