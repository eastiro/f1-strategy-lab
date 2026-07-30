from __future__ import annotations

import json
import re
from pathlib import Path


EVENT_RULES = (
    ("NORMAL GRIP", "Normal grip", "dry_confirmation"),
    ("LOW GRIP", "Low grip", "wet_warning"),
    ("DRS ENABLED", "DRS enabled", "dry_confirmation"),
    ("DRS DISABLED", "DRS disabled", "wet_warning"),
    ("VIRTUAL SAFETY", "VSC", "race_control"),
    ("SAFETY CAR", "Safety car", "race_control"),
    ("RED FLAG", "Red flag", "race_control"),
)


def extract_strategy_events(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep only race-control messages that add strategy context to a timeline."""
    events_by_lap: dict[int, list[tuple[str, str]]] = {}
    for record in records:
        lap = record.get("lap_number")
        message = str(record.get("message", "")).upper()
        if not isinstance(lap, (int, float)) or lap <= 0:
            continue
        for needle, label, kind in EVENT_RULES:
            if re.search(rf"\b{re.escape(needle)}\b", message):
                events_by_lap.setdefault(int(lap), []).append((label, kind))
                break

    events: list[dict[str, object]] = []
    for lap, labels_and_kinds in sorted(events_by_lap.items()):
        labels = list(dict.fromkeys(label for label, _ in labels_and_kinds))
        kinds = {kind for _, kind in labels_and_kinds}
        kind = "wet_warning" if "wet_warning" in kinds else "dry_confirmation" if "dry_confirmation" in kinds else "race_control"
        events.append({"lap": lap, "label": " · ".join(labels), "kind": kind})
    return events


def load_strategy_events(path: Path) -> list[dict[str, object]]:
    """Load timeline-ready strategy events from one OpenF1 race-control bundle."""
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return extract_strategy_events(records if isinstance(records, list) else [])
