"""V11 Regime Probability Forecast 1M/3M/6M."""
from __future__ import annotations
import pandas as pd
import numpy as np

def build_regime_forecast(regime_df: pd.DataFrame, macro_credit: pd.DataFrame | None=None) -> pd.DataFrame:
    if regime_df is None or regime_df.empty:
        return pd.DataFrame([{"horizon":"1M","risk_on_prob":0.4,"neutral_prob":0.4,"risk_off_prob":0.2,"expected_regime":"NEUTRAL"}])
    df=regime_df.copy()
    score=float(pd.to_numeric(df.tail(1).iloc[0].get("regime_score",50), errors="coerce"))
    cur=str(df.tail(1).iloc[0].get("market_regime", "NEUTRAL")).upper()
    # Smooth transition probabilities; longer horizon mean reverts toward neutral.
    horizons=[("1M",1.0),("3M",0.75),("6M",0.55)]
    rows=[]
    for h, strength in horizons:
        risk_on=np.clip(0.20 + (score/100)*0.65*strength,0.05,0.85)
        risk_off=np.clip(0.20 + ((100-score)/100)*0.55*strength,0.05,0.75)
        neutral=max(0.05,1-risk_on-risk_off)
        total=risk_on+risk_off+neutral
        risk_on,neutral,risk_off = risk_on/total, neutral/total, risk_off/total
        expected=max([("RISK_ON",risk_on),("NEUTRAL",neutral),("RISK_OFF",risk_off)], key=lambda x:x[1])[0]
        rows.append({"horizon":h,"current_regime":cur,"risk_on_prob":round(risk_on,4),"neutral_prob":round(neutral,4),"risk_off_prob":round(risk_off,4),"expected_regime":expected})
    return pd.DataFrame(rows)
