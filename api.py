"""
FORESIGHT scoring API.

Run locally:
    uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

app = FastAPI(
    title="FORESIGHT Scoring API",
    description="8-week demand forecast and inventory risk for a SKU.",
    version="1.0.0",
)

_forecast_df = None
_risk_df = None


def _load():
    global _forecast_df, _risk_df
    if _forecast_df is None:
        fpath = PROCESSED / "forecast_output.csv"
        rpath = PROCESSED / "risk_scores.csv"
        if not fpath.exists() or not rpath.exists():
            raise RuntimeError(
                "Model outputs not found. Run the FORESIGHT pipeline first."
            )
        _forecast_df = pd.read_csv(
            fpath, parse_dates=["week_start"]
        )
        _risk_df = pd.read_csv(rpath)
    return _forecast_df, _risk_df


class BatchRequest(BaseModel):
    sku_ids: List[str]


@app.get("/")
def root():
    return {
        "service": "FORESIGHT Scoring API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    required = [
        PROCESSED / "forecast_output.csv",
        PROCESSED / "risk_scores.csv",
    ]
    return {
        "status": "ok" if all(p.exists() for p in required) else "degraded",
        "artifacts_ready": all(p.exists() for p in required),
    }


@app.get("/forecast/{sku_id}")
def get_forecast(sku_id: str):
    forecast_df, risk_df = _load()
    sku = sku_id.strip().upper()

    fc = forecast_df[forecast_df["sku_id"] == sku].sort_values("week_start")
    rk = risk_df[risk_df["sku_id"] == sku]

    if fc.empty or rk.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found.")

    row = rk.iloc[0]
    return {
        "sku_id": sku,
        "category": row["category"],
        "risk": {
            "quadrant": row["risk_quadrant"],
            "stockout_risk": round(float(row["stockout_risk"]), 3),
            "overstock_risk": round(float(row["overstock_risk"]), 3),
            "recommended_action": row["recommended_action"],
            "stockout_value_at_risk": float(row["stockout_value_at_risk"]),
            "overstock_capital_locked": float(row["overstock_capital_locked"]),
            "rupee_value_at_stake": float(row["rupee_value_at_stake"]),
        },
        "inventory": {
            "on_hand_units": float(row["on_hand_units"]),
            "on_order_units": float(row["on_order_units"]),
            "lead_time_days": float(row["lead_time_days"]),
            "reorder_point": float(row["reorder_point"]),
        },
        "forecast": [
            {
                "week_start": r["week_start"].strftime("%Y-%m-%d"),
                "forecast_units": float(r["forecast"]),
                "low_80": float(r["forecast_low_80"]),
                "high_80": float(r["forecast_high_80"]),
            }
            for _, r in fc.iterrows()
        ],
    }


@app.post("/forecast/batch")
def get_forecast_batch(req: BatchRequest):
    results, not_found = [], []
    for sku_id in req.sku_ids:
        try:
            results.append(get_forecast(sku_id))
        except HTTPException:
            not_found.append(sku_id)
    return {"results": results, "not_found": not_found}
