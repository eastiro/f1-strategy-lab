#!/usr/bin/env bash
# Resumable five-season (2021-2025) FastF1 robustness audit runner.
# Uses the system python that has FastF1 3.8.3 installed. Forces the FastF1
# mirror (network fix) and the project cache via the runner module itself.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="C:/Users/EASTIRO/AppData/Local/Programs/Python/Python313/python.exe"
LOG="reports/results/_v3_scan.log"
: > "$LOG"
for YEAR in 2021 2022 2023 2024 2025; do
  echo "===== YEAR $YEAR START $(date) =====" >> "$LOG"
  "$PY" -m src.fastf1_five_season_runner --year "$YEAR" >> "$LOG" 2>&1
  echo "===== YEAR $YEAR END   $(date) =====" >> "$LOG"
done
echo "===== COMBINE START $(date) =====" >> "$LOG"
"$PY" -m src.fastf1_five_season_runner --combine >> "$LOG" 2>&1
echo "===== COMBINE END   $(date) =====" >> "$LOG"
