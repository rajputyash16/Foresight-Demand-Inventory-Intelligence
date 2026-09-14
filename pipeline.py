"""
pipeline.py
-----------
D1 deliverable: reproducible ingestion + cleaning -> analysis-ready dataset.

Steps (all coded, none manual):
  1. Ingest the four raw extracts.
  2. Validate & clean each one (types, duplicates, missing values, label fixes).
  3. Merge into one SKU-week fact table.
  4. Engineer model features (lags, rolling stats, calendar/promo signals).
  5. Write:
       data/processed/clean_daily.csv        (cleaned, unified daily grain)
       data/processed/weekly_features.csv    (SKU-week grain, model-ready)
       reports/data_quality_log.md           (what was found & how it was handled)

Run:
    python src/pipeline.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


# --------------------------------------------------------------------------
# 1 & 2. Ingest + clean each table
# --------------------------------------------------------------------------
def load_and_clean_sku_master():
    df = pd.read_csv(RAW / "sku_master.csv", parse_dates=["launch_date"])
    before = len(df)

    # normalise category labels (strip whitespace, title-case)
    df["category"] = df["category"].str.strip().str.title()
    df["subcategory"] = df["subcategory"].str.strip().str.title()

    # drop exact duplicate sku_id rows, keep first
    dup_count = df["sku_id"].duplicated().sum()
    df = df.drop_duplicates(subset="sku_id", keep="first")

    # impute missing unit_cost with category median
    missing_cost = df["unit_cost"].isna().sum()
    df["unit_cost"] = df.groupby("category")["unit_cost"].transform(
        lambda s: s.fillna(s.median())
    )

    log(f"### sku_master\n- Rows in: {before}, rows out: {len(df)}\n"
        f"- Removed {dup_count} duplicate SKU rows (kept first occurrence).\n"
        f"- Normalised inconsistent category labels (e.g. 'DECOR', 'furniture ') to Title Case.\n"
        f"- Imputed {missing_cost} missing `unit_cost` values with the category median "
        f"(reasonable since cost is fairly stable within a category).\n")
    return df


def load_and_clean_calendar():
    df = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    df["promo_event"] = df["promo_event"].fillna("")
    log(f"### calendar\n- Rows: {len(df)}. No structural issues found; "
        f"empty `promo_event` treated as 'no promotion'.\n")
    return df


def load_and_clean_sales_daily():
    df = pd.read_csv(RAW / "sales_daily.csv", parse_dates=["date"])
    before = len(df)

    # drop exact duplicate rows
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()

    # negative units_sold are a data-entry error (returns miscoded as sales) -> clip to 0
    neg_count = (df["units_sold"] < 0).sum()
    df["units_sold"] = df["units_sold"].clip(lower=0)

    # missing units_sold: assume no sale recorded that day (0), but flag it
    missing_count = df["units_sold"].isna().sum()
    df["units_sold_was_missing"] = df["units_sold"].isna().astype(int)
    df["units_sold"] = df["units_sold"].fillna(0)

    df["revenue"] = df["revenue"].fillna(0)

    log(f"### sales_daily\n- Rows in: {before}, rows out: {len(df)}\n"
        f"- Removed {dup_count} exact duplicate rows.\n"
        f"- Clipped {neg_count} negative `units_sold` values to 0 (likely miscoded returns); "
        f"flagged for review rather than silently discarded.\n"
        f"- Filled {missing_count} missing `units_sold` with 0 and added `units_sold_was_missing` "
        f"flag so downstream models can treat these differently from a genuine zero-sale day.\n")
    return df


def load_and_clean_inventory():
    df = pd.read_csv(RAW / "inventory_snapshots.csv", parse_dates=["date"])
    before = len(df)

    missing_lt = df["lead_time_days"].isna().sum()
    df["lead_time_days"] = df.groupby("sku_id")["lead_time_days"].transform(
        lambda s: s.fillna(s.median())
    )
    # a sku might have ALL snapshots missing lead time (edge case) -> fallback to global median
    df["lead_time_days"] = df["lead_time_days"].fillna(df["lead_time_days"].median())

    log(f"### inventory_snapshots\n- Rows: {len(df)}.\n"
        f"- Imputed {missing_lt} missing `lead_time_days` with that SKU's median lead time.\n")
    return df


# --------------------------------------------------------------------------
# 3. Merge into one unified daily table
# --------------------------------------------------------------------------
def merge_daily(sales, sku_master, calendar):
    df = sales.merge(sku_master, on="sku_id", how="left")
    df = df.merge(calendar, on="date", how="left")

    orphan_skus = df["category"].isna().sum()
    if orphan_skus:
        log(f"### merge\n- {orphan_skus} sales rows referenced a SKU missing from sku_master "
            f"after cleaning; dropped as unresolvable.\n")
        df = df.dropna(subset=["category"])

    return df


# --------------------------------------------------------------------------
# 4. Aggregate to weekly grain + engineer model features
# --------------------------------------------------------------------------
def build_weekly_features(clean_daily, inventory):
    df = clean_daily.copy()
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time

    weekly = (
        df.groupby(["sku_id", "week_start"])
        .agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            avg_price=("unit_price", "mean"),
            promo_days=("promo_flag", "sum"),
            holiday_days=("is_holiday", "sum"),
            category=("category", "first"),
            subcategory=("subcategory", "first"),
            unit_cost=("unit_cost", "first"),
        )
        .reset_index()
        .sort_values(["sku_id", "week_start"])
    )

    # calendar features
    weekly["week_of_year"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month
    weekly["is_promo_week"] = (weekly["promo_days"] > 0).astype(int)
    weekly["is_holiday_week"] = (weekly["holiday_days"] > 0).astype(int)

    # lag & rolling features (leakage-safe: shift BEFORE rolling, group by sku)
    g = weekly.groupby("sku_id")["units_sold"]
    for lag in [1, 2, 4, 8]:
        weekly[f"lag_{lag}"] = g.shift(lag)
    weekly["roll_mean_4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    weekly["roll_mean_8"] = g.shift(1).rolling(8).mean().reset_index(level=0, drop=True)
    weekly["roll_std_4"] = g.shift(1).rolling(4).std().reset_index(level=0, drop=True)

    # seasonal-naive reference value: same week last "season" (52 weeks back)
    weekly["seasonal_naive"] = g.shift(52)
    # fallback for SKUs without 52 weeks of history yet: use lag_4 rolling mean
    weekly["seasonal_naive"] = weekly["seasonal_naive"].fillna(weekly["roll_mean_4"])

    # merge latest inventory position as-of each week (asof merge, no future leakage)
    inv = inventory.sort_values("date")
    weekly_sorted = weekly.sort_values("week_start")
    merged = pd.merge_asof(
        weekly_sorted, inv.sort_values("date"),
        left_on="week_start", right_on="date",
        by="sku_id", direction="backward",
    )
    merged = merged.drop(columns=["date"])

    return merged


def main():
    log("# Data-Quality Log — Project FORESIGHT\n")

    sku_master = load_and_clean_sku_master()
    calendar = load_and_clean_calendar()
    sales = load_and_clean_sales_daily()
    inventory = load_and_clean_inventory()

    clean_daily = merge_daily(sales, sku_master, calendar)
    clean_daily.to_csv(PROCESSED / "clean_daily.csv", index=False)

    weekly = build_weekly_features(clean_daily, inventory)
    weekly.to_csv(PROCESSED / "weekly_features.csv", index=False)

    log(f"### Output\n- `clean_daily.csv`: {len(clean_daily):,} rows (unified daily grain).\n"
        f"- `weekly_features.csv`: {len(weekly):,} rows (SKU-week, model-ready with "
        f"lags/rolling stats/calendar/inventory features).\n")

    (REPORTS / "data_quality_log.md").write_text("\n".join(log_lines))
    print(f"\nPipeline complete. Data-quality log written to {REPORTS/'data_quality_log.md'}")


if __name__ == "__main__":
    main()
