"""
generate_data.py
-----------------
Creates the four RAW extracts NorthBay Living would hand over:
  data/raw/sales_daily.csv
  data/raw/sku_master.csv
  data/raw/calendar.csv
  data/raw/inventory_snapshots.csv

The data is synthetic but deliberately messy (missing values, duplicates,
inconsistent labels) to mimic a real client extract, per the brief.

Run:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)  # fixed seed -> reproducible

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_SKUS = 200
START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")

CATEGORIES = {
    "Furniture": ["Chairs", "Tables", "Sofas", "Shelving"],
    "Decor": ["Wall Art", "Rugs", "Lighting", "Vases"],
    "Small Appliances": ["Kettles", "Blenders", "Heaters", "Fans"],
    "Bedding": ["Sheets", "Pillows", "Comforters"],
    "Kitchen": ["Cookware", "Storage", "Cutlery"],
}


# --------------------------------------------------------------------------
# 1. sku_master
# --------------------------------------------------------------------------
def build_sku_master():
    rows = []
    cats = list(CATEGORIES.keys())
    for i in range(1, N_SKUS + 1):
        sku_id = f"SKU{i:04d}"
        cat = RNG.choice(cats)
        sub = RNG.choice(CATEGORIES[cat])
        launch_offset = RNG.integers(0, (END_DATE - START_DATE).days - 30)
        launch_date = START_DATE + pd.Timedelta(days=int(launch_offset))
        unit_cost = round(RNG.uniform(200, 4000), 2)
        margin = RNG.uniform(1.4, 2.6)
        list_price = round(unit_cost * margin, 2)
        rows.append([sku_id, cat, sub, launch_date.date(), unit_cost, list_price])

    df = pd.DataFrame(
        rows,
        columns=["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"],
    )

    # --- inject messiness ---
    # inconsistent category labels for a handful of rows
    messy_idx = RNG.choice(df.index, size=6, replace=False)
    swap = {"Furniture": "furniture ", "Decor": "DECOR", "Kitchen": "kitchen"}
    for idx in messy_idx:
        c = df.loc[idx, "category"]
        if c in swap:
            df.loc[idx, "category"] = swap[c]

    # a few duplicated sku rows (client export glitch)
    dup_rows = df.sample(3, random_state=7)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # a few missing unit_cost
    missing_idx = RNG.choice(df.index, size=4, replace=False)
    df.loc[missing_idx, "unit_cost"] = np.nan

    return df


# --------------------------------------------------------------------------
# 2. calendar
# --------------------------------------------------------------------------
INDIAN_HOLIDAYS_2024_25 = [
    "2024-01-26", "2024-03-25", "2024-08-15", "2024-10-02", "2024-10-31",
    "2024-11-01", "2024-12-25", "2025-01-26", "2025-03-14", "2025-08-15",
    "2025-10-02", "2025-10-20", "2025-12-25",
]

PROMO_EVENTS = {
    "2024-01-01": "New Year Sale", "2024-08-01": "Independence Week Sale",
    "2024-10-15": "Festive Season Sale", "2024-11-25": "Black Friday",
    "2024-12-15": "Year End Clearance", "2025-01-01": "New Year Sale",
    "2025-08-01": "Independence Week Sale", "2025-10-15": "Festive Season Sale",
    "2025-11-25": "Black Friday",
}


def build_calendar():
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    df = pd.DataFrame({"date": dates})
    df["week"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(
        lambda m: "Winter" if m in (12, 1, 2) else "Summer" if m in (3, 4, 5, 6)
        else "Monsoon" if m in (7, 8, 9) else "Autumn"
    )
    df["is_holiday"] = df["date"].dt.strftime("%Y-%m-%d").isin(INDIAN_HOLIDAYS_2024_25).astype(int)

    # promo events run for ~10 days from the event start
    df["promo_event"] = ""
    for start, name in PROMO_EVENTS.items():
        start = pd.Timestamp(start)
        mask = (df["date"] >= start) & (df["date"] < start + pd.Timedelta(days=10))
        df.loc[mask, "promo_event"] = name

    return df


# --------------------------------------------------------------------------
# 3. sales_daily  (depends on sku_master + calendar for seasonality/promo lift)
# --------------------------------------------------------------------------
def build_sales_daily(sku_master, calendar):
    cal = calendar.set_index("date")
    all_rows = []

    for _, sku in sku_master.drop_duplicates("sku_id").iterrows():
        sku_id = sku["sku_id"]
        launch = pd.Timestamp(sku["launch_date"])
        base_demand = RNG.uniform(2, 40)  # avg units/day for this SKU
        trend = RNG.uniform(-0.0003, 0.0006)  # slow drift
        noise_scale = base_demand * 0.35

        # ~8% of SKUs are "dead stock": demand collapses after a while
        is_dying = RNG.random() < 0.08
        die_point = launch + pd.Timedelta(days=int(RNG.integers(120, 500)))

        # ~10% of SKUs are volatile bestsellers
        is_volatile = RNG.random() < 0.10

        price = sku["list_price"]
        active_dates = pd.date_range(max(launch, START_DATE), END_DATE, freq="D")

        for t, date in enumerate(active_dates):
            row_cal = cal.loc[date]
            season_mult = {"Winter": 1.15, "Summer": 0.9, "Monsoon": 0.85, "Autumn": 1.1}[row_cal["season"]]
            dow_mult = 1.25 if date.dayofweek >= 5 else 1.0  # weekend lift
            holiday_mult = 1.4 if row_cal["is_holiday"] else 1.0
            promo_mult = 1.8 if row_cal["promo_event"] != "" else 1.0

            demand = (base_demand + trend * t) * season_mult * dow_mult * holiday_mult * promo_mult
            demand += RNG.normal(0, noise_scale * (1.8 if is_volatile else 1.0))

            if is_dying and date > die_point:
                decay_days = (date - die_point).days
                demand *= max(0.02, np.exp(-decay_days / 60))

            units = max(0, int(round(demand)))

            unit_price = price * (0.8 if row_cal["promo_event"] != "" else 1.0)
            revenue = round(units * unit_price, 2)

            all_rows.append([date, sku_id, units, revenue, round(unit_price, 2),
                              1 if row_cal["promo_event"] != "" else 0])

    df = pd.DataFrame(all_rows, columns=["date", "sku_id", "units_sold", "revenue",
                                          "unit_price", "promo_flag"])

    # --- inject messiness ---
    # a handful of missing units_sold
    miss_idx = RNG.choice(df.index, size=250, replace=False)
    df.loc[miss_idx, "units_sold"] = np.nan

    # a handful of exact duplicate rows
    dup_rows = df.sample(120, random_state=11)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # a few negative units (data entry error, e.g. returns miscoded)
    neg_idx = RNG.choice(df.index, size=15, replace=False)
    df.loc[neg_idx, "units_sold"] = -RNG.integers(1, 5, size=15)

    return df


# --------------------------------------------------------------------------
# 4. inventory_snapshots (weekly snapshot per sku)
# --------------------------------------------------------------------------
def build_inventory_snapshots(sku_master, sales_daily):
    """
    Simulate a realistic (s, S)-style inventory policy day by day, driven by
    the SKU's ACTUAL simulated sales, then take a weekly snapshot. This keeps
    stock levels bounded and correlated with real demand, instead of an
    unconstrained random walk that can drift to unrealistic levels.
    """
    sales_clean = sales_daily.copy()
    sales_clean["units_sold"] = sales_clean["units_sold"].clip(lower=0)
    avg_daily = sales_clean.groupby("sku_id")["units_sold"].mean().fillna(1.0)
    daily_by_sku = {
        sku_id: g.groupby("date")["units_sold"].sum()
        for sku_id, g in sales_clean.groupby("sku_id")
    }

    all_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    snap_dates = set(pd.date_range(START_DATE, END_DATE, freq="7D"))
    rows = []

    for _, sku in sku_master.drop_duplicates("sku_id").iterrows():
        sku_id = sku["sku_id"]
        adu = max(avg_daily.get(sku_id, 1.0), 0.5)
        lead_time = int(RNG.integers(7, 30))
        # ops_quality varies by SKU: some are planned leanly (real stockout risk),
        # some are over-bought "just in case" (real overstock risk) - this is what
        # makes the risk-scoring layer meaningful instead of everything being fine.
        ops_quality = RNG.uniform(0.35, 1.8)
        ops_reliability = RNG.uniform(0.25, 0.95)  # chance the team reorders on time each day
        reorder_point = round(adu * lead_time * 0.9 * ops_quality)
        max_level = round(adu * (lead_time + 10) * ops_quality)

        on_hand = max(1, round(max_level * RNG.uniform(0.5, 1.0)))
        pending_orders = []  # list of (arrival_date, qty)
        daily_sales = daily_by_sku.get(sku_id, pd.Series(dtype=float))

        for d in all_dates:
            # receive any orders arriving today
            arrived = sum(q for (a, q) in pending_orders if a == d)
            if arrived:
                on_hand += arrived
                pending_orders = [(a, q) for (a, q) in pending_orders if a != d]

            # consume today's actual demand
            sold_today = daily_sales.get(d, adu)
            on_hand = max(0, on_hand - sold_today)

            on_order_total = sum(q for (_, q) in pending_orders)

            # reorder if position (on_hand + on_order) falls below reorder point
            if on_hand + on_order_total < reorder_point and RNG.random() < ops_reliability:
                order_qty = max(1, round(max_level - (on_hand + on_order_total)))
                arrival = d + pd.Timedelta(days=lead_time)
                pending_orders.append((arrival, order_qty))
                on_order_total += order_qty

            if d in snap_dates:
                rows.append([d, sku_id, int(round(on_hand)), int(round(on_order_total)),
                             lead_time, int(reorder_point)])

    df = pd.DataFrame(rows, columns=["date", "sku_id", "on_hand_units", "on_order_units",
                                      "lead_time_days", "reorder_point"])

    # inject a few missing lead times
    miss_idx = RNG.choice(df.index, size=10, replace=False)
    df.loc[miss_idx, "lead_time_days"] = np.nan

    return df


def main():
    print("Generating sku_master...")
    sku_master = build_sku_master()
    print("Generating calendar...")
    calendar = build_calendar()
    print("Generating sales_daily (this takes a few seconds)...")
    sales_daily = build_sales_daily(sku_master, calendar)
    print("Generating inventory_snapshots...")
    inventory = build_inventory_snapshots(sku_master, sales_daily)

    sku_master.to_csv(RAW_DIR / "sku_master.csv", index=False)
    calendar.to_csv(RAW_DIR / "calendar.csv", index=False)
    sales_daily.to_csv(RAW_DIR / "sales_daily.csv", index=False)
    inventory.to_csv(RAW_DIR / "inventory_snapshots.csv", index=False)

    print(f"\nDone. Files written to {RAW_DIR}")
    print(f"  sku_master.csv           {len(sku_master):>7,} rows")
    print(f"  calendar.csv             {len(calendar):>7,} rows")
    print(f"  sales_daily.csv          {len(sales_daily):>7,} rows")
    print(f"  inventory_snapshots.csv  {len(inventory):>7,} rows")


if __name__ == "__main__":
    main()
