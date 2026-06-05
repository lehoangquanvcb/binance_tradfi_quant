"""Retraining scheduler helper."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

STATE = Path("outputs/retraining_state.json")


def should_retrain(frequency_days: int = 7) -> bool:
    if not STATE.exists(): return True
    state = json.loads(STATE.read_text())
    last = datetime.fromisoformat(state.get("last_retrain"))
    return datetime.utcnow() - last >= timedelta(days=frequency_days)


def mark_retrained(model_version: str) -> dict:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    state = {"last_retrain": datetime.utcnow().isoformat(), "model_version": model_version}
    STATE.write_text(json.dumps(state, indent=2))
    return state
