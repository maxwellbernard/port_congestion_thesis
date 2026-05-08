
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

notebook_start = time.perf_counter()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "final_dataframes" / "bulk_carrier.csv"
DATASET_NAME = "BULK CARRIER"

TARGETS = [
    "target_p95_waiting_1w",
    "target_p95_waiting_2w",
]

TRAIN_END = "2024-06-30"
VAL_END = "2024-12-31"
GAP_DAYS = 14

RANDOM_STATE = 42

PORTS_TO_KEEP = [
    "HOUSTON",
    "LOS_ANGELES_LONG_BEACH",
    "NEW_YORK_NEW_JERSEY",
    "PORT_OF_VIRGINIA",
]


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

FEAT_M2 = [
    "macro_bdi_weekly",
    "macro_ism_pmi",
    "macro_us_exports_goods",
    "macro_us_trade_balance",
]

PORT_COL = "port_encoded"


def load_and_preprocess(
    path: Path, ports_to_keep: list[str]
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Load CSV, parse dates, filter ports, encode port."""
    df = pl.read_csv(path, try_parse_dates=True)
    df = df.filter(pl.col("port_name").is_in(ports_to_keep))
    df = df.sort(["port_name", "ref_date"])

    port_map = {name: i for i, name in enumerate(sorted(df["port_name"].unique().to_list()))}
    df = df.with_columns(
        pl.col("port_name").replace_strict(port_map).cast(pl.Int32).alias(PORT_COL)
    )

    return df, port_map


df, port_map = load_and_preprocess(DATA_PATH, PORTS_TO_KEEP)
ports = sorted(port_map.keys())
print(f"[{DATASET_NAME}] Loaded {df.shape[0]} rows")
print(f"Port encoding: {port_map}")

feat_all = [c for c in df.columns if c.startswith("feat_")]
macro_all = [c for c in df.columns if c.startswith("macro_")]

FEAT_M3 = feat_all + [PORT_COL]
FEAT_M4 = feat_all + FEAT_M2 + [PORT_COL]
FEAT_M1_FULL = FEAT_M1 + [PORT_COL]
FEAT_M2_FULL = FEAT_M2 + [PORT_COL]

print(f"Feature counts — M1: {len(FEAT_M1_FULL)}, M2: {len(FEAT_M2_FULL)}, "
      f"M3: {len(FEAT_M3)}, M4: {len(FEAT_M4)}")


_train_end = pl.lit(TRAIN_END).str.to_date("%Y-%m-%d")
_val_end = pl.lit(VAL_END).str.to_date("%Y-%m-%d")

df_train_all = df.filter(pl.col("ref_date") <= _train_end)
df_val_all = df.filter(
    (pl.col("ref_date") > _train_end.dt.offset_by(f"{GAP_DAYS}d"))
    & (pl.col("ref_date") <= _val_end)
)
df_test_all = df.filter(pl.col("ref_date") > _val_end.dt.offset_by(f"{GAP_DAYS}d"))

print(f"Train: {df_train_all.shape[0]} | Val: {df_val_all.shape[0]} | Test: {df_test_all.shape[0]}")
print(f"(Gap: {GAP_DAYS} days excluded between train/val and val/test)")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for severity forecasts."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "n": 0}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}


def evaluate_per_port(
    df_test: pl.DataFrame,
    y_pred: np.ndarray,
    target: str,
    model_name: str,
    ports: list[str],
) -> list[dict[str, object]]:
    """Compute global + per-port metrics."""
    y_true = df_test[target].to_numpy()
    results: list[dict[str, object]] = []

    metrics = evaluate(y_true, y_pred)
    results.append({"model": model_name, "port": "ALL", **metrics})

    port_arr = df_test["port_name"].to_numpy()
    for port in ports:
        mask = port_arr == port
        if mask.sum() == 0:
            continue
        metrics = evaluate(y_true[mask], y_pred[mask])
        results.append({"model": model_name, "port": port, **metrics})

    return results


def print_results(results: list[dict[str, object]]) -> None:
    """Pretty-print a results table."""
    header = f"{'Model':<14} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        r2_val = r["r2"]
        r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
        print(
            f"{r['model']:<14} {r['port']:<30} "
            f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
        )


def seasonal_naive_predict(
    df_train: pl.DataFrame, df_eval: pl.DataFrame, target: str
) -> np.ndarray:
    """Predict using week-of-year median from training set per port."""
    seasonal_medians = (
        df_train.filter(pl.col(target).is_not_null())
        .group_by(["port_name", "feat_week_of_year"])
        .agg(pl.col(target).median().alias("m0_pred"))
    )
    port_medians = (
        df_train.filter(pl.col(target).is_not_null())
        .group_by("port_name")
        .agg(pl.col(target).median().alias("m0_fallback"))
    )
    preds = (
        df_eval.join(seasonal_medians, on=["port_name", "feat_week_of_year"], how="left")
        .join(port_medians, on="port_name", how="left")
        .with_columns(
            pl.coalesce(["m0_pred", "m0_fallback"]).clip(lower_bound=0).alias("pred")
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
    ports: list[str],
) -> tuple[list[dict[str, object]], np.ndarray, xgb.XGBRegressor]:
    """Train XGBoost on train, early-stop on val, evaluate on test."""
    t0 = time.perf_counter()

    X_train = df_train.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    y_train = df_train[target].fill_null(0).to_numpy().astype(np.float32)
    X_val = df_val.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    y_val = df_val[target].fill_null(0).to_numpy().astype(np.float32)
    X_test = df_test.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        early_stopping_rounds=50,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)

    elapsed = time.perf_counter() - t0
    results = evaluate_per_port(df_test, preds, target, model_name, ports)
    print(f"Best iteration: {model.best_iteration} | Time: {elapsed:.1f}s")
    return results, preds, model


grand_results: list[dict[str, object]] = []

for target in TARGETS:
    horizon_start = time.perf_counter()
    horizon_label = target.replace("target_p95_waiting_", "")

    df_train = df_train_all.filter(pl.col(target).is_not_null())
    df_val = df_val_all.filter(pl.col(target).is_not_null())
    df_test = df_test_all.filter(pl.col(target).is_not_null())

    print("\n" + "#" * 80)
    print(f"# [{DATASET_NAME}] HORIZON: {horizon_label} — target = {target}")
    print(f"# Train: {df_train.shape[0]} | Val: {df_val.shape[0]} | Test: {df_test.shape[0]}")
    print("#" * 80)

    m0_preds = seasonal_naive_predict(df_train, df_test, target)
    m0_results = evaluate_per_port(df_test, m0_preds, target, "M0-Naive", ports)
    print("\n=== M0 Seasonal Naive ===")
    print_results(m0_results)

    print("\n=== M1: XGBoost — Lag-only ===")
    m1_results, _, _ = train_xgboost(
        df_train, df_val, df_test, target, FEAT_M1_FULL, "M1-XGB", ports
    )
    print_results(m1_results)

    print("\n=== M2: XGBoost — Macro-only ===")
    m2_results, _, _ = train_xgboost(
        df_train, df_val, df_test, target, FEAT_M2_FULL, "M2-XGB", ports
    )
    print_results(m2_results)

    print("\n=== M3: XGBoost — AIS-only ===")
    m3_results, _, _ = train_xgboost(
        df_train, df_val, df_test, target, FEAT_M3, "M3-XGB", ports
    )
    print_results(m3_results)

    print("\n=== M4: XGBoost — AIS + Macro ===")
    m4_results, _, m4_model = train_xgboost(
        df_train, df_val, df_test, target, FEAT_M4, "M4-XGB", ports
    )
    print_results(m4_results)

    importances = m4_model.feature_importances_
    importance_df = pl.DataFrame({
        "feature": FEAT_M4,
        "importance": importances.tolist(),
    }).sort("importance", descending=True)

    print(f"\n--- Top 15 Features (M4 XGBoost, {horizon_label}) ---")
    for row in importance_df.head(15).iter_rows(named=True):
        print(f"  {row['feature']:<45} {row['importance']:.4f}")

    horizon_results = m0_results + m1_results + m2_results + m3_results + m4_results
    print(f"\n{'=' * 80}")
    print(f"[{DATASET_NAME}] XGBOOST SUMMARY — {horizon_label}")
    print(f"{'=' * 80}")
    print_results(horizon_results)
    print(f"Horizon time: {time.perf_counter() - horizon_start:.1f}s")

    for r in horizon_results:
        grand_results.append({**r, "horizon": horizon_label})


print("\n" + "=" * 92)
print(f"[{DATASET_NAME}] XGBOOST GRAND SUMMARY — ALL HORIZONS")
print("=" * 92)
header = f"{'Horizon':<8} {'Model':<14} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
print(header)
print("-" * len(header))
for r in grand_results:
    r2_val = r["r2"]
    r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
    print(
        f"{r['horizon']:<8} {r['model']:<14} {r['port']:<30} "
        f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
    )

total_time = time.perf_counter() - notebook_start
print(f"\nTotal notebook runtime: {total_time:.1f}s ({total_time / 60:.1f}m)")
