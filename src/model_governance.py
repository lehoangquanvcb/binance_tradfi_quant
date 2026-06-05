"""Model governance and inventory utilities."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import pandas as pd

GOV_PATH = Path("outputs/model_governance.jsonl")


def register_model(model_name: str, version: str, purpose: str, owner: str = "Le Hoang Quan", metrics: dict | None = None, status: str = "Champion") -> dict:
    GOV_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "version": version,
        "purpose": purpose,
        "owner": owner,
        "metrics": metrics or {},
        "status": status,
    }
    with GOV_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_inventory() -> pd.DataFrame:
    if not GOV_PATH.exists():
        return pd.DataFrame(columns=["timestamp","model_name","version","purpose","owner","metrics","status"])
    return pd.read_json(GOV_PATH, lines=True)


def validation_check(backtest_metrics: dict, thresholds: dict | None = None) -> dict:
    thresholds = thresholds or {"min_sharpe": 0.3, "max_drawdown": 0.25, "min_hit_rate": 0.48}
    result = {
        "sharpe_ok": backtest_metrics.get("sharpe", 0) >= thresholds["min_sharpe"],
        "drawdown_ok": abs(backtest_metrics.get("max_drawdown", 1)) <= thresholds["max_drawdown"],
        "hit_rate_ok": backtest_metrics.get("hit_rate", 0) >= thresholds["min_hit_rate"],
    }
    result["approved_for_paper"] = all(result.values())
    result["approved_for_live"] = False  # Keep manual approval mandatory.
    return result
