"""Model governance and inventory utilities for CIO/PM decision-support workflows."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

GOV_PATH = Path("outputs/model_governance.jsonl")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def validation_check(backtest_metrics: dict, thresholds: dict | None = None) -> dict:
    """Validate model quality without falsely labeling weak models as Champion.

    Champion is reserved for models that pass statistical quality AND explicit live approval.
    Candidate is suitable for paper/research use.
    Watch means the model is informative but not yet approved for portfolio use.
    """
    thresholds = thresholds or {
        "min_auc": 0.58,
        "min_accuracy": 0.54,
        "min_hit_rate": 0.53,
        "min_sharpe": 0.30,
        "max_drawdown": 0.25,
    }

    auc = _to_float(backtest_metrics.get("auc"), 0.0)
    accuracy = _to_float(backtest_metrics.get("accuracy"), 0.0)
    hit_rate = _to_float(backtest_metrics.get("hit_rate"), accuracy)
    sharpe = _to_float(backtest_metrics.get("sharpe"), 0.0)
    max_drawdown = abs(_to_float(backtest_metrics.get("max_drawdown"), 1.0))

    result = {
        "auc": auc,
        "accuracy": accuracy,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "auc_ok": auc >= thresholds["min_auc"],
        "accuracy_ok": accuracy >= thresholds["min_accuracy"],
        "hit_rate_ok": hit_rate >= thresholds["min_hit_rate"],
        "sharpe_ok": sharpe >= thresholds["min_sharpe"],
        "drawdown_ok": max_drawdown <= thresholds["max_drawdown"],
    }

    # Paper approval can be based on predictive performance when full P&L backtest
    # metrics are not available. Live approval remains manual and intentionally false.
    result["approved_for_paper"] = bool(result["auc_ok"] and result["accuracy_ok"] and result["hit_rate_ok"])
    result["approved_for_live"] = bool(
        result["approved_for_paper"] and result["sharpe_ok"] and result["drawdown_ok"] and False
    )

    if result["approved_for_live"]:
        result["model_status"] = "Champion"
    elif result["approved_for_paper"]:
        result["model_status"] = "Candidate"
    elif auc >= 0.55 or accuracy >= 0.53:
        result["model_status"] = "Watch"
    else:
        result["model_status"] = "Research"

    return result


def register_model(
    model_name: str,
    version: str,
    purpose: str,
    owner: str = "Le Hoang Quan",
    metrics: dict | None = None,
    status: str | None = None,
) -> dict:
    GOV_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}
    if status is None:
        status = validation_check(metrics).get("model_status", "Watch")
    rec = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "version": version,
        "purpose": purpose,
        "owner": owner,
        "metrics": metrics,
        "status": status,
    }
    with GOV_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_inventory() -> pd.DataFrame:
    if not GOV_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "model_name", "version", "purpose", "owner", "metrics", "status"])
    return pd.read_json(GOV_PATH, lines=True)
