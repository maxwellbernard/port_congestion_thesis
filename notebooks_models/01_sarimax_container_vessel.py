
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

notebook_start = time.perf_counter()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "final_dataframes" / "container.csv"

TARGETS = [
    "target_p95_waiting_1w",
    "target_p95_waiting_2w",
]

TRAIN_END = "2024-06-30"
VAL_END = "2024-12-31"
GAP_DAYS = 14

PORTS_TO_KEEP = [
    "HOUSTON",
    "LOS_ANGELES_LONG_BEACH",
    "NEW_YORK_NEW_JERSEY",
    "PORT_OF_VIRGINIA",
]

ORDER = (1, 0, 0)
SEASONAL_ORDER = (1, 0, 0, 52)

FEAT_AIS_EXOG = [
    "feat_mean_queue_length",
    "feat_service_rate",
    "feat_arrival_count",
    "feat_mean_berth_count",
    "feat_pct_waiting",
]

FEAT_MACRO = [
    "macro_gscpi",
    "macro_bdi_weekly",
    "macro_ism_pmi",
    "macro_us_imports_goods",
    "macro_us_exports_goods",
    "macro_us_trade_balance",
]

FEAT_M4_EXOG = FEAT_AIS_EXOG + FEAT_MACRO


def load_data(path: Path, ports_to_keep: list[str]) -> pl.DataFrame:
    """Load container CSV, parse dates, filter to active ports."""
    df = pl.read_csv(path, try_parse_dates=True)
    df = df.filter(pl.col("port_name").is_in(ports_to_keep))
    df = df.sort(["port_name", "ref_date"])
    return df


df = load_data(DATA_PATH, PORTS_TO_KEEP)
ports = df["port_name"].unique().sort().to_list()
print(f"Loaded {df.shape[0]} rows, {len(ports)} ports: {ports}")
print(f"Date range: {df['ref_date'].min()} → {df['ref_date'].max()}")


_train_end = pl.lit(TRAIN_END).str.to_date("%Y-%m-%d")
_val_end = pl.lit(VAL_END).str.to_date("%Y-%m-%d")

df_train = df.filter(pl.col("ref_date") <= _train_end)
df_val = df.filter(
    (pl.col("ref_date") > _train_end.dt.offset_by(f"{GAP_DAYS}d"))
    & (pl.col("ref_date") <= _val_end)
)
df_test = df.filter(pl.col("ref_date") > _val_end.dt.offset_by(f"{GAP_DAYS}d"))

print(f"Train: {df_train.shape[0]} rows ({df_train['ref_date'].min()} → {df_train['ref_date'].max()})")
print(f"Val:   {df_val.shape[0]} rows ({df_val['ref_date'].min()} → {df_val['ref_date'].max()})")
print(f"Test:  {df_test.shape[0]} rows ({df_test['ref_date'].min()} → {df_test['ref_date'].max()})")
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


def print_results(results: list[dict[str, object]]) -> None:
    """Pretty-print a results table."""
    header = f"{'Model':<12} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        r2_val = r["r2"]
        r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
        print(
            f"{r['model']:<12} {r['port']:<30} "
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
        df_eval.join(
            seasonal_medians, on=["port_name", "feat_week_of_year"], how="left"
        )
        .join(port_medians, on="port_name", how="left")
        .with_columns(
            pl.coalesce(["m0_pred", "m0_fallback"]).clip(lower_bound=0).alias("pred")
        )
    )
    return preds["pred"].to_numpy()


def evaluate_m0(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    target: str,
    ports: list[str],
) -> list[dict[str, object]]:
    """Run M0 seasonal naive and return per-port results."""
    m0_preds = seasonal_naive_predict(df_train, df_test, target)
    m0_actual = df_test[target].to_numpy()

    results: list[dict[str, object]] = []
    metrics = evaluate(m0_actual, m0_preds)
    results.append({"model": "M0-Naive", "port": "ALL", **metrics})

    for port in ports:
        mask = df_test["port_name"].to_numpy() == port
        if mask.sum() == 0:
            continue
        metrics = evaluate(m0_actual[mask], m0_preds[mask])
        results.append({"model": "M0-Naive", "port": port, **metrics})

    return results


def fit_sarimax_per_port(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    target: str,
    exog_cols: list[str] | None,
    model_name: str,
    ports: list[str],
) -> list[dict[str, object]]:
    """Fit SARIMAX per port and return results on test set.

    Args:
        df_train: Training data.
        df_val: Validation data (combined with train for final fit).
        df_test: Test data for evaluation.
        target: Target column name.
        exog_cols: Exogenous feature columns, or None for M1.
        model_name: Label for results table.
        ports: List of port names.

    Returns:
        Results list with global + per-port metrics.
    """
    results: list[dict[str, object]] = []
    all_preds = np.full(df_test.shape[0], np.nan)
    test_port_names = df_test["port_name"].to_numpy()

    for port in ports:
        t_port = time.perf_counter()

        train_port = df_train.filter(pl.col("port_name") == port).sort("ref_date")
        val_port = df_val.filter(pl.col("port_name") == port).sort("ref_date")
        test_port = df_test.filter(pl.col("port_name") == port).sort("ref_date")

        trainval = pl.concat([train_port, val_port])

        y_trainval = trainval[target].fill_null(0).to_numpy().astype(float)
        y_test = test_port[target].to_numpy().astype(float)
        n_test = len(y_test)

        exog_train = None
        exog_test = None
        if exog_cols:
            exog_train = (
                trainval.select(exog_cols)
                .fill_null(strategy="forward")
                .fill_null(0)
                .to_numpy()
                .astype(float)
            )
            exog_test = (
                test_port.select(exog_cols)
                .fill_null(strategy="forward")
                .fill_null(0)
                .to_numpy()
                .astype(float)
            )

        if np.std(y_trainval) < 0.01:
            preds = np.zeros(n_test)
            print(f"  {port}: near-zero variance, predicting 0 ({time.perf_counter() - t_port:.1f}s)")
        else:
            try:
                model = SARIMAX(
                    y_trainval,
                    exog=exog_train,
                    order=ORDER,
                    seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False, maxiter=200)
                preds = fit.forecast(steps=n_test, exog=exog_test)
                preds = np.clip(preds, 0, None)
                print(f"  {port}: fitted ({time.perf_counter() - t_port:.1f}s)")
            except Exception as e:
                print(f"  {port}: SARIMAX failed ({e}), using zeros ({time.perf_counter() - t_port:.1f}s)")
                preds = np.zeros(n_test)

        port_mask = test_port_names == port
        all_preds[port_mask] = preds

        metrics = evaluate(y_test, preds)
        results.append({"model": model_name, "port": port, **metrics})

    y_all = df_test[target].to_numpy()
    global_metrics = evaluate(y_all, all_preds)
    results.insert(0, {"model": model_name, "port": "ALL", **global_metrics})

    return results


grand_results: list[dict[str, object]] = []

for target in TARGETS:
    horizon_start = time.perf_counter()
    horizon_label = target.replace("target_p95_waiting_", "")
    print("\n" + "#" * 80)
    print(f"# HORIZON: {horizon_label} — target = {target}")
    print("#" * 80)

    t0 = time.perf_counter()
    m0_results = evaluate_m0(df_train, df_test, target, ports)
    print(f"\n=== M0 Seasonal Naive ({time.perf_counter() - t0:.1f}s) ===")
    print_results(m0_results)

    t0 = time.perf_counter()
    print(f"\n=== M1: SARIMAX — AR only ===")
    m1_results = fit_sarimax_per_port(
        df_train, df_val, df_test, target,
        exog_cols=None, model_name="M1-SARIMAX", ports=ports,
    )
    print(f"Total M1 time: {time.perf_counter() - t0:.1f}s")
    print_results(m1_results)

    t0 = time.perf_counter()
    print(f"\n=== M3: SARIMAX + AIS exogenous ===")
    m3_results = fit_sarimax_per_port(
        df_train, df_val, df_test, target,
        exog_cols=FEAT_AIS_EXOG, model_name="M3-SARIMAX", ports=ports,
    )
    print(f"Total M3 time: {time.perf_counter() - t0:.1f}s")
    print_results(m3_results)

    t0 = time.perf_counter()
    print(f"\n=== M4: SARIMAX + AIS + Macro exogenous ===")
    m4_results = fit_sarimax_per_port(
        df_train, df_val, df_test, target,
        exog_cols=FEAT_M4_EXOG, model_name="M4-SARIMAX", ports=ports,
    )
    print(f"Total M4 time: {time.perf_counter() - t0:.1f}s")
    print_results(m4_results)

    horizon_results = m0_results + m1_results + m3_results + m4_results
    print(f"\n{'=' * 80}")
    print(f"SARIMAX SUMMARY — {horizon_label}")
    print(f"{'=' * 80}")
    print_results(horizon_results)
    print(f"Horizon time: {time.perf_counter() - horizon_start:.1f}s")

    for r in horizon_results:
        grand_results.append({**r, "horizon": horizon_label})


print("\n" + "=" * 90)
print("SARIMAX GRAND SUMMARY — ALL HORIZONS")
print("=" * 90)
header = f"{'Horizon':<8} {'Model':<12} {'Port':<30} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}"
print(header)
print("-" * len(header))
for r in grand_results:
    r2_val = r["r2"]
    r2_str = f"{r2_val:>8.3f}" if isinstance(r2_val, float) and not np.isnan(r2_val) else "     N/A"
    print(
        f"{r['horizon']:<8} {r['model']:<12} {r['port']:<30} "
        f"{r['rmse']:>8.2f} {r['mae']:>8.2f} {r2_str} {r['n']:>6}"
    )

total_time = time.perf_counter() - notebook_start
print(f"\nTotal notebook runtime: {total_time:.1f}s ({total_time / 60:.1f}m)")
