
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

notebook_start = time.perf_counter()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "final_dataframes" / "container.csv"
DATASET_LABEL = "container"

TARGETS = [
    "target_p95_waiting_1w",
    "target_p95_waiting_2w",
]

TRAIN_END = "2024-06-30"
VAL_END   = "2024-12-31"
GAP_DAYS  = 14

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
    "macro_gscpi",
    "macro_ism_pmi",
    "macro_us_imports_goods",
    "macro_us_trade_balance",
]


df = pl.read_csv(DATA_PATH, try_parse_dates=True)
df = df.filter(pl.col("port_name").is_in(PORTS_TO_KEEP)).sort(["port_name", "ref_date"])
ports = sorted(df["port_name"].unique().to_list())

feat_all  = [c for c in df.columns if c.startswith("feat_")]
macro_all = [c for c in df.columns if c.startswith("macro_")]

FEAT_M3 = feat_all
FEAT_M4 = feat_all + FEAT_M2

print(f"Dataset: {DATASET_LABEL} — {df.shape[0]} rows")
print(f"Ports: {ports}")
print(f"Feature counts — M1:{len(FEAT_M1)}  M2:{len(FEAT_M2)}  M3:{len(FEAT_M3)}  M4:{len(FEAT_M4)}")


_train_end = pl.lit(TRAIN_END).str.to_date("%Y-%m-%d")
_val_end   = pl.lit(VAL_END).str.to_date("%Y-%m-%d")

df_train_all = df.filter(pl.col("ref_date") <= _train_end)
df_val_all   = df.filter(
    (pl.col("ref_date") > _train_end.dt.offset_by(f"{GAP_DAYS}d"))
    & (pl.col("ref_date") <= _val_end)
)
df_test_all  = df.filter(pl.col("ref_date") > _val_end.dt.offset_by(f"{GAP_DAYS}d"))

print(f"Train:{df_train_all.shape[0]}  Val:{df_val_all.shape[0]}  Test:{df_test_all.shape[0]}")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, R² — nan-safe."""
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
    """Pretty-print results table."""
    header = f"{'Model':<16} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        r2_val = r["r2"]
        r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
        print(
            f"{r['model']:<16} {r['port']:<30} "
            f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
        )


def train_xgb_port(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    target: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, int]:
    """Train XGBoost on a single port's data.

    Args:
        df_train: Port training rows.
        df_val: Port validation rows.
        df_test: Port test rows.
        target: Target column name.
        feature_cols: Feature column names (no port encoding).

    Returns:
        Tuple of (test predictions clipped to ≥0, best_iteration).
    """
    X_tr = df_train.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    y_tr = df_train[target].fill_null(0).to_numpy().astype(np.float32)
    X_va = df_val.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    y_va = df_val[target].fill_null(0).to_numpy().astype(np.float32)
    X_te = df_test.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)

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
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds = np.clip(model.predict(X_te), 0, None)
    return preds, model.best_iteration


def seasonal_naive_predict(
    df_train: pl.DataFrame, df_eval: pl.DataFrame, target: str
) -> np.ndarray:
    """Week-of-year median per port from training set."""
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
    return (
        df_eval
        .join(seasonal_medians, on=["port_name", "feat_week_of_year"], how="left")
        .join(port_medians, on="port_name", how="left")
        .with_columns(pl.coalesce(["m0_pred", "m0_fallback"]).clip(lower_bound=0).alias("pred"))
    )["pred"].to_numpy()


grand_results: list[dict[str, object]] = []

for target in TARGETS:
    horizon_start = time.perf_counter()
    hlabel = target.replace("target_p95_waiting_", "")

    df_train = df_train_all.filter(pl.col(target).is_not_null())
    df_val   = df_val_all.filter(pl.col(target).is_not_null())
    df_test  = df_test_all.filter(pl.col(target).is_not_null())

    print("\n" + "#" * 80)
    print(f"# HORIZON: {hlabel} — {target}")
    print(f"# Train:{df_train.shape[0]}  Val:{df_val.shape[0]}  Test:{df_test.shape[0]}")
    print("#" * 80)

    specs: dict[str, list[str]] = {
        "M1-PP-XGB": FEAT_M1,
        "M2-PP-XGB": FEAT_M2,
        "M3-PP-XGB": FEAT_M3,
        "M4-PP-XGB": FEAT_M4,
    }

    m0_preds = seasonal_naive_predict(df_train, df_test, target)
    m0_global = evaluate(df_test[target].to_numpy(), m0_preds)
    m0_results = [{"model": "M0-Naive", "port": "ALL", **m0_global}]
    for port in ports:
        mask = df_test["port_name"].to_numpy() == port
        if mask.sum() == 0:
            continue
        m0_results.append({
            "model": "M0-Naive",
            "port": port,
            **evaluate(df_test[target].to_numpy()[mask], m0_preds[mask]),
        })

    print("\n=== M0 Seasonal Naive ===")
    print_results(m0_results)
    for r in m0_results:
        grand_results.append({**r, "horizon": hlabel})

    for spec_name, feat_cols in specs.items():
        missing = [f for f in feat_cols if f not in df.columns]
        if missing:
            print(f"WARNING: {spec_name} skipped — missing columns: {missing[:3]}")
            continue

        print(f"\n=== {spec_name} — per-port training ===")
        all_test_true:  list[np.ndarray] = []
        all_test_pred:  list[np.ndarray] = []
        all_test_ports: list[np.ndarray] = []
        spec_results: list[dict[str, object]] = []

        for port in ports:
            df_tr_p  = df_train.filter(pl.col("port_name") == port)
            df_va_p  = df_val.filter(pl.col("port_name") == port)
            df_te_p  = df_test.filter(pl.col("port_name") == port)

            if df_tr_p.shape[0] < 10 or df_va_p.shape[0] < 5 or df_te_p.shape[0] < 5:
                print(f"  {port}: too few rows — skipping")
                continue

            preds, best_iter = train_xgb_port(df_tr_p, df_va_p, df_te_p, target, feat_cols)
            y_true = df_te_p[target].to_numpy()

            metrics = evaluate(y_true, preds)
            spec_results.append({"model": spec_name, "port": port, **metrics})
            print(f"  {port:<30} best_iter={best_iter:>4}  RMSE={metrics['rmse']:>7.2f}  R²={metrics.get('r2', float('nan')):>6.3f}")

            all_test_true.append(y_true)
            all_test_pred.append(preds)
            all_test_ports.append(np.full(len(y_true), port))

        if all_test_true:
            global_metrics = evaluate(
                np.concatenate(all_test_true), np.concatenate(all_test_pred)
            )
            spec_results.insert(0, {"model": spec_name, "port": "ALL", **global_metrics})

        print_results(spec_results)
        for r in spec_results:
            grand_results.append({**r, "horizon": hlabel})

    print(f"Horizon time: {time.perf_counter() - horizon_start:.1f}s")


print("\n" + "=" * 92)
print(f"PER-PORT XGBOOST GRAND SUMMARY — {DATASET_LABEL.upper()} — ALL HORIZONS")
print("=" * 92)
header = f"{'Horizon':<8} {'Model':<16} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
print(header)
print("-" * len(header))
for r in grand_results:
    r2_val = r["r2"]
    r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
    print(
        f"{r['horizon']:<8} {r['model']:<16} {r['port']:<30} "
        f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
    )

total_time = time.perf_counter() - notebook_start
print(f"\nTotal notebook runtime: {total_time:.1f}s ({total_time / 60:.1f}m)")
