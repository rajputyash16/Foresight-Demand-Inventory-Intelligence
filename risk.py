"""
risk.py
-------
D4 deliverable: turns the forecast + current inventory position into a
transparent, explainable risk score and recommended action per SKU.

Logic (matches Section 08 of the brief):
  - Stockout risk: compare forecast demand over the lead time against
    on-hand + on-order stock. If projected coverage falls below a safety
    margin, the SKU is at risk of stocking out.
  - Overstock risk: compare on-hand stock against forecast demand over a
    forward window (the full horizon). Holding far more than you will
    sell flags overstock.
  - Every SKU gets a 0-1 risk score on both axes, a quadrant (Reorder Now /
    Markdown-Clear / Watch-Volatile / Healthy), a recommended action, and
    the rupee value at stake so the ops team can prioritise.

Output:
    data/processed/risk_scores.csv

Run:
    python src/risk.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

SAFETY_MARGIN = 1.2  # cover lead-time demand with a 20% buffer to count as "safe"
OVERSTOCK_MULTIPLE = 2.5  # holding > 2.5x the horizon's expected sales = overstocked


def score_risk():
    fc = pd.read_csv(PROCESSED / "forecast_output.csv", parse_dates=["week_start"])

    # aggregate the 8-week forecast per SKU (what the risk layer needs)
    agg = fc.groupby("sku_id").agg(
        category=("category", "first"),
        total_horizon_demand=("forecast", "sum"),
        avg_weekly_demand=("forecast", "mean"),
        on_hand_units=("on_hand_units", "first"),
        on_order_units=("on_order_units", "first"),
        lead_time_days=("lead_time_days", "first"),
        reorder_point=("reorder_point", "first"),
        unit_cost=("unit_cost", "first"),
        avg_price=("avg_price", "first"),
    ).reset_index()

    agg["lead_time_weeks"] = agg["lead_time_days"] / 7.0
    agg["demand_over_lead_time"] = agg["avg_weekly_demand"] * agg["lead_time_weeks"]
    agg["available_stock"] = agg["on_hand_units"].fillna(0) + agg["on_order_units"].fillna(0)

    # --- stockout risk score (0-1): how far short of safe coverage are we? ---
    required = agg["demand_over_lead_time"] * SAFETY_MARGIN
    shortfall = (required - agg["available_stock"]).clip(lower=0)
    agg["stockout_risk"] = (shortfall / required.replace(0, np.nan)).clip(0, 1).fillna(0)

    # --- overstock risk score (0-1): how far over the safe multiple are we? ---
    safe_stock = agg["total_horizon_demand"] * 1.0  # 1x horizon demand = fully healthy
    overstock_ceiling = agg["total_horizon_demand"] * OVERSTOCK_MULTIPLE
    excess = (agg["on_hand_units"].fillna(0) - safe_stock).clip(lower=0)
    denom = (overstock_ceiling - safe_stock).replace(0, np.nan)
    agg["overstock_risk"] = (excess / denom).clip(0, 1).fillna(0)

    # --- rupee value at stake ---
    agg["stockout_value_at_risk"] = (
        agg["stockout_risk"] * agg["demand_over_lead_time"] * agg["avg_price"]
    ).round(0)
    agg["overstock_capital_locked"] = (
        (agg["on_hand_units"].fillna(0) - safe_stock).clip(lower=0) * agg["unit_cost"]
    ).round(0)
    agg["rupee_value_at_stake"] = (
        agg["stockout_value_at_risk"] + agg["overstock_capital_locked"]
    )

    # --- quadrant + recommended action (transparent thresholds, not a black box) ---
    def quadrant(row):
        so, ov = row["stockout_risk"], row["overstock_risk"]
        if so >= 0.5 and ov >= 0.5:
            return "Watch / Volatile", "Investigate — demand is erratic; review manually."
        if so >= 0.5:
            return "Reorder Now", "Raise a replenishment order before stock runs out."
        if ov >= 0.5:
            return "Markdown / Clear", "Promote or discount to free up capital."
        return "Healthy", "No action needed; leave as is."

    quad_action = agg.apply(quadrant, axis=1, result_type="expand")
    agg["risk_quadrant"] = quad_action[0]
    agg["recommended_action"] = quad_action[1]

    agg = agg.sort_values("rupee_value_at_stake", ascending=False)
    agg.to_csv(PROCESSED / "risk_scores.csv", index=False)
    return agg


def main():
    scored = score_risk()
    print(f"Scored {len(scored)} SKUs.\n")
    print(scored["risk_quadrant"].value_counts().to_string())
    print(f"\nTotal rupee value at stake across all SKUs: "
          f"Rs {scored['rupee_value_at_stake'].sum():,.0f}")
    print(f"  - Sales at risk from stockouts:  Rs {scored['stockout_value_at_risk'].sum():,.0f}")
    print(f"  - Capital locked in overstock:   Rs {scored['overstock_capital_locked'].sum():,.0f}")
    print(f"\nWritten to {PROCESSED/'risk_scores.csv'}")


if __name__ == "__main__":
    main()
