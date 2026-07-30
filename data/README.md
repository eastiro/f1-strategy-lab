# Portfolio Data Manifest

The V2 portfolio app is reproducible from five OpenF1 race bundles:

- 2024 United Kingdom
- 2024 Canada
- 2023 Netherlands
- 2024 Netherlands
- 2024 Brazil

The V2.3 challenge-set diagnostic additionally keeps three 2025 bundles:

- 2025 Australia
- 2025 United Kingdom
- 2025 Belgium

For each bundle, the repository keeps the required raw OpenF1 session, driver,
lap, pit, stint, weather, and race-control JSON files plus processed CSV files.
The `.gitignore` file explicitly allows these eight bundles while continuing to
ignore unrelated downloaded races.

To refresh one bundle:

```powershell
python -m src.download_openf1 --year 2024 --country "United Kingdom"
python -m src.prepare_dataset --year 2024 --country "United Kingdom"
```

Use the country names above exactly as the OpenF1 session API expects. For the
2024 Sao Paulo Grand Prix, the OpenF1 country field is `Brazil`.
