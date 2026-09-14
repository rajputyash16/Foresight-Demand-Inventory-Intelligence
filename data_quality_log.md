# Data-Quality Log — Project FORESIGHT

### sku_master
- Rows in: 203, rows out: 200
- Removed 3 duplicate SKU rows (kept first occurrence).
- Normalised inconsistent category labels (e.g. 'DECOR', 'furniture ') to Title Case.
- Imputed 4 missing `unit_cost` values with the category median (reasonable since cost is fairly stable within a category).

### calendar
- Rows: 731. No structural issues found; empty `promo_event` treated as 'no promotion'.

### sales_daily
- Rows in: 73141, rows out: 73021
- Removed 120 exact duplicate rows.
- Clipped 15 negative `units_sold` values to 0 (likely miscoded returns); flagged for review rather than silently discarded.
- Filled 250 missing `units_sold` with 0 and added `units_sold_was_missing` flag so downstream models can treat these differently from a genuine zero-sale day.

### inventory_snapshots
- Rows: 21000.
- Imputed 10 missing `lead_time_days` with that SKU's median lead time.

### Output
- `clean_daily.csv`: 73,021 rows (unified daily grain).
- `weekly_features.csv`: 10,669 rows (SKU-week, model-ready with lags/rolling stats/calendar/inventory features).
