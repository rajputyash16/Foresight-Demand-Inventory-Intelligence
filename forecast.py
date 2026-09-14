"""
forecast.py
-----------
D3 deliverable: weekly SKU-level demand forecast, backtested against a
seasonal-naive baseline using rolling-origin cross-validation.

Non-negotiable rule from the brief: no data leakage. Every feature used at
week t is built only from data available before week t (see pipeline.py:
lags/rolling stats are shifted before rolling; seasonal_naive uses week
t-52). This script's backtest never lets a fold train on its own test period.

Outputs:
    data/processed/backtest_results.csv   (per-fold WAPE, baseline vs model)
    data/processed/forecast_output.csv    (final forecast + 80% interval per SKU)
    models/forecast_model.pkl

Run:
    python src/forecast.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
import pickle

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = PROCESSED

FEATURES = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_mean_4", "roll_mean_8", "roll_std_4",
    "week_of_year", "month", "is_promo_week", "is_holiday_week",
    "avg_price", "unit_cost",
]
TARGET = "units_sold"
HORIZON_WEEKS = 8  # forecast horizon required by the brief (6-8 weeks)


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error - robust to low-volume SKUs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom


def bias(y_true, y_pred):
    return np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))


def load_data():
    df = pd.read_csv(PROCESSED / "weekly_features.csv", parse_dates=["week_start"])
    df = df.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    return df


def rolling_origin_folds(df, n_folds=4, horizon=HORIZON_WEEKS):
    """
    Rolling-origin CV: each fold's train set is everything BEFORE the fold's
    cutoff week; the test set is the `horizon` weeks immediately after it.
    Never a random split for time series.
    """
    weeks = sorted(df["week_start"].unique())
    fold_cutoffs = []
    # space folds out across the back half of history
    usable = weeks[len(weeks) // 2:]
    step = max(1, (len(usable) - horizon) // n_folds)
    for i in range(n_folds):
        idx = i * step
        if idx + horizon >= len(usable):
            break
        fold_cutoffs.append(usable[idx])
    return fold_cutoffs


def train_predict(train_df, test_df):
    train = train_df.dropna(subset=FEATURES + [TARGET])
    if len(train) < 50:
        return None  # not enough history to train this fold

    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=10, random_state=42, verbosity=-1,
    )
    model.fit(train[FEATURES], train[TARGET])

    test = test_df.copy()
    test_valid = test.dropna(subset=FEATURES)
    preds = pd.Series(index=test.index, dtype=float)
    if len(test_valid):
        preds.loc[test_valid.index] = model.predict(test_valid[FEATURES])
    preds = preds.fillna(test["roll_mean_4"]).fillna(0).clip(lower=0)
    return model, preds


def run_backtest(df):
    cutoffs = rolling_origin_folds(df)
    results = []

    for fold_i, cutoff in enumerate(cutoffs, start=1):
        train_df = df[df["week_start"] < cutoff]
        test_mask = (df["week_start"] >= cutoff) & (
            df["week_start"] < cutoff + pd.Timedelta(weeks=HORIZON_WEEKS)
        )
        test_df = df[test_mask]
        if train_df.empty or test_df.empty:
            continue

        out = train_predict(train_df, test_df)
        if out is None:
            continue
        _, preds = out

        y_true = test_df[TARGET].values
        model_wape = wape(y_true, preds.values)
        baseline_wape = wape(y_true, test_df["seasonal_naive"].fillna(0).values)
        model_bias = bias(y_true, preds.values)

        results.append({
            "fold": fold_i, "cutoff_week": cutoff.date(),
            "n_rows": len(test_df),
            "baseline_wape": round(baseline_wape, 4),
            "model_wape": round(model_wape, 4),
            "model_bias": round(model_bias, 3),
            "model_beats_baseline": bool(model_wape < baseline_wape),
        })

    return pd.DataFrame(results)


def train_final_model(df):
    """Train on ALL available history to produce the live forward forecast."""
    train = df.dropna(subset=FEATURES + [TARGET])
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=10, random_state=42, verbosity=-1,
    )
    model.fit(train[FEATURES], train[TARGET])
    return model


def forecast_forward(df, model, horizon=HORIZON_WEEKS):
    """
    Iteratively forecast `horizon` weeks ahead per SKU, feeding each
    prediction back in as a lag for the next step (no future truth used).
    """
    last_week = df["week_start"].max()
    all_forecasts = []

    for sku_id, g in df.groupby("sku_id"):
        g = g.sort_values("week_start").reset_index(drop=True)
        history = g["units_sold"].tolist()
        static_row = g.iloc[-1]

        for h in range(1, horizon + 1):
            future_week = last_week + pd.Timedelta(weeks=h)
            lag_1 = history[-1] if len(history) >= 1 else 0
            lag_2 = history[-2] if len(history) >= 2 else 0
            lag_4 = history[-4] if len(history) >= 4 else 0
            lag_8 = history[-8] if len(history) >= 8 else 0
            roll_mean_4 = np.mean(history[-4:]) if len(history) >= 1 else 0
            roll_mean_8 = np.mean(history[-8:]) if len(history) >= 1 else 0
            roll_std_4 = np.std(history[-4:]) if len(history) >= 2 else 0

            feat_row = pd.DataFrame([{
                "lag_1": lag_1, "lag_2": lag_2, "lag_4": lag_4, "lag_8": lag_8,
                "roll_mean_4": roll_mean_4, "roll_mean_8": roll_mean_8,
                "roll_std_4": roll_std_4,
                "week_of_year": future_week.isocalendar()[1],
                "month": future_week.month,
                "is_promo_week": 0, "is_holiday_week": 0,
                "avg_price": static_row["avg_price"],
                "unit_cost": static_row["unit_cost"],
            }])
            point = max(0, float(model.predict(feat_row[FEATURES])[0]))
            # simple, honest uncertainty band from recent volatility
            spread = max(roll_std_4, point * 0.15, 1.0)
            history.append(point)

            all_forecasts.append({
                "sku_id": sku_id, "category": static_row["category"],
                "week_start": future_week, "horizon_week": h,
                "forecast": round(point, 1),
                "forecast_low_80": round(max(0, point - 1.28 * spread), 1),
                "forecast_high_80": round(point + 1.28 * spread, 1),
                "on_hand_units": static_row.get("on_hand_units", np.nan),
                "on_order_units": static_row.get("on_order_units", np.nan),
                "lead_time_days": static_row.get("lead_time_days", np.nan),
                "reorder_point": static_row.get("reorder_point", np.nan),
                "unit_cost": static_row["unit_cost"],
                "avg_price": static_row["avg_price"],
            })

    return pd.DataFrame(all_forecasts)


def main():
    df = load_data()
    print(f"Loaded {len(df):,} SKU-week rows across {df['sku_id'].nunique()} SKUs.")

    print("\nRunning rolling-origin backtest (model vs seasonal-naive baseline)...")
    backtest = run_backtest(df)
    backtest.to_csv(PROCESSED / "backtest_results.csv", index=False)
    print(backtest.to_string(index=False))

    overall_model = backtest["model_wape"].mean()
    overall_base = backtest["baseline_wape"].mean()
    win = overall_model < overall_base
    print(f"\nAverage WAPE — model: {overall_model:.3f} | seasonal-naive baseline: {overall_base:.3f}")
    print("RESULT: model beats the baseline." if win else
          "RESULT: model does NOT beat the baseline on backtest — "
          "per the brief, this is reported honestly, not hidden.")

    print("\nTraining final model on full history and forecasting forward 8 weeks...")
    model = train_final_model(df)
    forecast_df = forecast_forward(df, model)
    forecast_df.to_csv(PROCESSED / "forecast_output.csv", index=False)

    with open(MODELS / "forecast_model.pkl", "wb") as f:
        pickle.dump(model, f)

    summary = {
        "avg_model_wape": round(overall_model, 4),
        "avg_baseline_wape": round(overall_base, 4),
        "model_beats_baseline": bool(win),
    }
    pd.Series(summary).to_csv(PROCESSED / "backtest_summary.csv")
    print(f"\nForecast written: {len(forecast_df):,} rows -> {PROCESSED/'forecast_output.csv'}")


if __name__ == "__main__":
    main()
