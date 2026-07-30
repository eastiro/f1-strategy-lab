#!/usr/bin/env bash
# Run the remaining Task-11 seasons (2022-2025) serially, then combine.
# Mirror + project cache are forced inside the runner module.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="C:/Users/EASTIRO/AppData/Local/Programs/Python/Python313/python.exe"
LOG="reports/results/_v3_scan_final.log"
: > "$LOG"
for YEAR in 2022 2023 2024 2025; do
  echo "===== YEAR $YEAR START $(date) =====" >> "$LOG"
  "$PY" -m src.fastf1_five_season_runner --year "$YEAR" >> "$LOG" 2>&1
  echo "===== YEAR $YEAR END   $(date) =====" >> "$LOG"
done
echo "===== COMBINE START $(date) =====" >> "$LOG"
"$PY" -m src.fastf1_five_season_runner --combine >> "$LOG" 2>&1
echo "===== COMBINE END   $(date) =====" >> "$LOG"
echo "ALL DONE" >> "$LOG"
