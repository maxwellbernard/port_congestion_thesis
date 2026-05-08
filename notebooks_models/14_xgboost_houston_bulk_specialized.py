
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

notebook_start = time.perf_counter()

ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "final_dataframes" / "bulk_carrier.csv"
MACRO_DIR    = ROOT / "raw_macro_data"
TARGET_PORT  = "HOUSTON"

TARGETS = [
    "target_p95_waiting_1w",
    "target_p95_waiting_2w",
]

TRAIN_END = "2024-06-30"
VAL_END   = "2024-12-31"
GAP_DAYS  = 14

RANDOM_STATE = 42

FEAT_M1 = [
    "feat_p95_waiting_hours_lag1w",
    "feat_p95_waiting_hours_lag2w",
    "feat_p95_waiting_hours_lag4w",
    "feat_p95_waiting_hours_roll4w_mean",
    "feat_p95_waiting_hours_roll4w_std",
    "feat_p95_waiting_hours_roll8w_mean",
    "feat_p95_waiting_hours_roll8w_std",
    "feat_p95_waiting_change_1w",
]

FEAT_M2_HOUSTON = [
    "macro_bdi_weekly",
    "macro_ism_pmi",
    "macro_us_exports_goods",
    "macro_corn_futures",
    "macro_wti_crude",
    "macro_wti_production",
    "macro_wti_exports",
]


df_full = pl.read_csv(DATA_PATH, try_parse_dates=True)
df      = df_full.filter(pl.col("port_name") == TARGET_PORT)

print(f"Houston bulk carrier rows: {df.shape[0]}")
print(f"Date range: {df['ref_date'].min()} → {df['ref_date'].max()}")


def _load_commodity_csv(
    path: Path,
    col_name: str,
    lag_days: int,
    comma_thousands: bool = False,
) -> pl.DataFrame:
    """Load a Bloomberg commodity CSV and apply a publication lag.

    Args:
        path: Path to the CSV file (columns: Date, Index).
        col_name: Output macro column name.
        lag_days: Publication lag in days to add to each date.
        comma_thousands: Set True if Index uses comma as thousands separator
            (e.g. "3,440.00").

    Returns:
        DataFrame with columns [date, col_name], sorted by date.
    """
    index_expr = (
        pl.col("Index").str.replace_all(",", "").cast(pl.Float64)
        if comma_thousands
        else pl.col("Index").cast(pl.Float64)
    )
    return (
        pl.read_csv(path, schema_overrides={"Index": pl.Utf8})
        .filter(pl.col("Date").is_not_null() & (pl.col("Date").str.strip_chars() != ""))
        .with_columns(
            pl.col("Date").str.to_date("%m/%d/%Y").alias("date"),
            index_expr.alias(col_name),
        )
        .select("date", col_name)
        .sort("date")
        .with_columns(
            (pl.col("date") + pl.duration(days=lag_days)).cast(pl.Date).alias("date")
        )
    )


corn_df     = _load_commodity_csv(
    MACRO_DIR / "Corn futures daily_Bloomberg_C 1 COMB Comdty.csv",
    "macro_corn_futures", lag_days=1,
)
wti_df      = _load_commodity_csv(
    MACRO_DIR / "WTI Crude Oil daily_Bloomberg_CL1 COMB Comdty.csv",
    "macro_wti_crude", lag_days=1,
)
wti_prod_df = _load_commodity_csv(
    MACRO_DIR / "US WTI Crude Oil Total Production weekly_Bloomberg_DOETCRUD13657.csv",
    "macro_wti_production", lag_days=5,
)
wti_exp_df  = _load_commodity_csv(
    MACRO_DIR / "US WTI Crude Oil Total Exports weekly_Bloomberg_DOEBCEXP3322.csv",
    "macro_wti_exports", lag_days=5, comma_thousands=True,
)

df = df.sort("ref_date")
for commodity_df in [corn_df, wti_df, wti_prod_df, wti_exp_df]:
    col = [c for c in commodity_df.columns if c != "date"][0]
    df = df.join_asof(commodity_df, left_on="ref_date", right_on="date", strategy="backward")
    if "date" in df.columns:
        df = df.drop("date")
    n_nulls = df[col].is_null().sum()
    print(f"  Joined {col}: {n_nulls} null rows (pre-series start)")

print(f"\nHouston rows after commodity join: {df.shape[0]}")

feat_all = [c for c in df.columns if c.startswith("feat_")]
macro_available = [c for c in df.columns if c.startswith("macro_")]

missing = [f for f in FEAT_M2_HOUSTON if f not in df.columns]
if missing:
    raise ValueError(f"Missing Houston macro features: {missing}")

FEAT_M3 = feat_all
FEAT_M4 = feat_all + FEAT_M2_HOUSTON

print(f"Feature counts — M1: {len(FEAT_M1)}, M2: {len(FEAT_M2_HOUSTON)}, "
      f"M3: {len(FEAT_M3)}, M4: {len(FEAT_M4)}")
print(f"All macro available: {macro_available}")
print(f"Houston macro selected: {FEAT_M2_HOUSTON}")


_train_end = pl.lit(TRAIN_END).str.to_date("%Y-%m-%d")
_val_end   = pl.lit(VAL_END).str.to_date("%Y-%m-%d")

df_train_all = df.filter(pl.col("ref_date") <= _train_end)
df_val_all   = df.filter(
    (pl.col("ref_date") > _train_end.dt.offset_by(f"{GAP_DAYS}d"))
    & (pl.col("ref_date") <= _val_end)
)
df_test_all  = df.filter(
    pl.col("ref_date") > _val_end.dt.offset_by(f"{GAP_DAYS}d")
)

print(f"Train: {df_train_all.shape[0]} | Val: {df_val_all.shape[0]} | Test: {df_test_all.shape[0]}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n": 0}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}


def print_results(results: list[dict[str, object]]) -> None:
    """Pretty-print results."""
    header = f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        r2_val = r["r2"]
        r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
        print(f"{r['model']:<22} {r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}")


def seasonal_naive_predict(
    df_train: pl.DataFrame, df_eval: pl.DataFrame, target: str
) -> np.ndarray:
    """Week-of-year median from training set."""
    seasonal = (
        df_train.filter(pl.col(target).is_not_null())
        .group_by("feat_week_of_year")
        .agg(pl.col(target).median().alias("m0_pred"))
    )
    overall_median = float(df_train[target].drop_nulls().median())
    preds = (
        df_eval
        .join(seasonal, on="feat_week_of_year", how="left")
        .with_columns(
            pl.col("m0_pred").fill_null(overall_median).clip(lower_bound=0).alias("pred")
        )
    )
    return preds["pred"].to_numpy()


def train_xgboost(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    target: str,
    feature_cols: list[str],
    model_name: str,
) -> tuple[dict[str, object], np.ndarray, xgb.XGBRegressor]:
    """Fit XGBoost on Houston-only data, evaluate on test set.

    Args:
        df_train: Houston training data.
        df_val: Houston validation data for early stopping.
        df_test: Houston test data.
        target: Target column name.
        feature_cols: Features to use.
        model_name: Label for output.

    Returns:
        Tuple of (metrics dict, test predictions, fitted model).
    """
    t0 = time.perf_counter()

    X_train = df_train.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    y_train = df_train[target].fill_null(0).to_numpy().astype(np.float32)
    X_val   = df_val.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    y_val   = df_val[target].fill_null(0).to_numpy().astype(np.float32)
    X_test  = df_test.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        early_stopping_rounds=50,
        verbosity=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = np.clip(model.predict(X_test), 0, None)

    elapsed = time.perf_counter() - t0
    metrics = {**evaluate(df_test[target].to_numpy(), preds), "model": model_name}
    print(f"  best_iter={model.best_iteration} | {elapsed:.1f}s")
    return metrics, preds, model


def print_feature_importance(
    model: xgb.XGBRegressor,
    feature_cols: list[str],
    top_n: int = 20,
    label: str = "",
) -> None:
    """Print top-N features by XGBoost gain importance."""
    imp = pl.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_.tolist(),
    }).sort("importance", descending=True)
    print(f"\nTop {top_n} features — {label}")
    for row in imp.head(top_n).iter_rows(named=True):
        bar = "█" * int(row["importance"] * 300)
        print(f"  {row['feature']:<50} {row['importance']:.4f}  {bar}")


grand_results: list[dict[str, object]] = []

for target in TARGETS:
    horizon_label = target.replace("target_p95_waiting_", "")

    df_train = df_train_all.filter(pl.col(target).is_not_null())
    df_val   = df_val_all.filter(pl.col(target).is_not_null())
    df_test  = df_test_all.filter(pl.col(target).is_not_null())

    print("\n" + "#" * 70)
    print(f"# HORIZON: {horizon_label}  |  N_test={df_test.shape[0]}")
    print("#" * 70)

    results: list[dict[str, object]] = []

    m0_preds = seasonal_naive_predict(df_train, df_test, target)
    m0_met = {**evaluate(df_test[target].to_numpy(), m0_preds), "model": "M0-Naive"}
    results.append(m0_met)

    print("\nM1 — lag-only:")
    m1_met, _, _ = train_xgboost(df_train, df_val, df_test, target, FEAT_M1, "M1-XGB-Houston")
    results.append(m1_met)

    print("\nM2 — Houston macro-only (BDI, ISM_PMI, US_exports, corn, WTI):")
    m2_met, _, _ = train_xgboost(df_train, df_val, df_test, target, FEAT_M2_HOUSTON, "M2-XGB-Houston")
    results.append(m2_met)

    print("\nM3 — AIS-only:")
    m3_met, _, _ = train_xgboost(df_train, df_val, df_test, target, FEAT_M3, "M3-XGB-Houston")
    results.append(m3_met)

    print("\nM4 — AIS + Houston macro:")
    m4_met, m4_preds, m4_model = train_xgboost(df_train, df_val, df_test, target, FEAT_M4, "M4-XGB-Houston")
    results.append(m4_met)

    print("\n--- Results ---")
    print_results(results)

    print_feature_importance(m4_model, FEAT_M4, top_n=20, label=f"M4-Houston {horizon_label}")

    for r in results:
        grand_results.append({**r, "horizon": horizon_label})


print("\n" + "=" * 70)
print("HOUSTON BULK CARRIER — GRAND SUMMARY")
print("=" * 70)
header = f"{'Horizon':<8} {'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
print(header)
print("-" * len(header))
for r in grand_results:
    r2_val = r["r2"]
    r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
    print(
        f"{r['horizon']:<8} {r['model']:<22} "
        f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
    )

print("\n" + "=" * 70)
print("SPECIALIZATION PROGRESSION — Houston Bulk Carrier M4 (2w horizon)")
print("=" * 70)
print("  Level 1 — General (combined):        see 00_xgboost_combined.py Houston/bulk")
print("  Level 2 — Vessel-specialized (bulk): see 04_xgboost_bulk_carrier.py HOUSTON")
print("  Level 3 — Port-specialized (this):   M4-XGB-Houston above")
print("\nMacro features by level:")
print("  Level 1: all 7 macro (undifferentiated)")
print("  Level 2: BDI, ISM_PMI, US_exports, US_trade_balance  (4 bulk indicators)")
print(f"  Level 3: {', '.join(FEAT_M2_HOUSTON)}  (7 Houston-specific)")

total_time = time.perf_counter() - notebook_start
print(f"\nTotal runtime: {total_time:.1f}s")
