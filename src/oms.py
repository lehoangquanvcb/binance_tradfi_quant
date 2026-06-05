"""Order Management System (OMS) with safe defaults."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import pandas as pd

OMS_PATH = Path("outputs/oms_orders.jsonl")
VALID_STATUS = {"PENDING_APPROVAL", "APPROVED", "SUBMITTED", "PARTIALLY_FILLED", "FILLED", "CANCELLED", "REJECTED"}

@dataclass
class OrderTicket:
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    limit_price: float | None = None
    status: str = "PENDING_APPROVAL"
    reason: str = ""
    live_mode: bool = False
    created_at: str = datetime.utcnow().isoformat()
    updated_at: str = datetime.utcnow().isoformat()


def create_order(ticket: OrderTicket, path: Path = OMS_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = asdict(ticket)
    rec["order_id"] = f"ORD-{int(datetime.utcnow().timestamp()*1000)}"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_orders(path: Path = OMS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def approval_required(nav: float, notional: float, threshold_pct: float = 0.02) -> bool:
    return notional >= nav * threshold_pct
