
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

notebook_start = time.perf_counter()

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TRAIN_END = "2024-06-30"
VAL_END = "2024-12-31"
GAP_DAYS = 14
RANDOM_STATE = 42
PORT_COL = "port_encoded"

PORTS_TO_KEEP = [
    "HOUSTON",
    "LOS_ANGELES_LONG_BEACH",
    "NEW_YORK_NEW_JERSEY",
    "PORT_OF_VIRGINIA",
]

DATASETS: list[dict] = [
    {
        "label": "container",
        "data_path": ROOT / "final_dataframes" / "container.csv",
        "threshold": 48.0,
        "primary_ports": ["HOUSTON", "LOS_ANGELES_LONG_BEACH"],
    },
    {
        "label": "bulk_carrier",
        "data_path": ROOT / "final_dataframes" / "bulk_carrier.csv",
        "threshold": 120.0,
        "primary_ports": ["HOUSTON", "LOS_ANGELES_LONG_BEACH", "PORT_OF_VIRGINIA"],
    },
]

TARGETS = [
    "target_p95_waiting_1w",
    "target_p95_waiting_2w",
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
    "macro_bdi_weekly",
    "macro_ism_pmi",
    "macro_us_imports_goods",
    "macro_us_exports_goods",
    "macro_us_trade_balance",
]

XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 50,
    "verbosity": 0,
}


def encode_ports(df: pl.DataFrame) -> pl.DataFrame:
    """Label-encode port_name as a deterministic integer (alphabetical order).

    Args:
        df: DataFrame containing a port_name column.

    Returns:
        DataFrame with an additional port_encoded integer column.
    """
    ports_sorted = sorted(df["port_name"].unique().to_list())
    return df.with_columns(
        pl.col("port_name")
        .cast(pl.Enum(ports_sorted))
        .to_physical()
        .cast(pl.Int32)
        .alias(PORT_COL)
    )


def seasonal_naive_predict(
    df_train: pl.DataFrame,
    df_pred: pl.DataFrame,
    target: str,
) -> np.ndarray:
    """Predict using seasonal naive: week-of-year median from training set.

    Falls back to global training median for unseen week numbers.

    Args:
        df_train: Training DataFrame containing target and feat_week_of_year.
        df_pred: DataFrame to predict for, must contain feat_week_of_year.
        target: Target column name.

    Returns:
        Numpy array of predictions aligned to df_pred rows.
    """
    week_col = "feat_week_of_year"
    medians = (
        df_train.filter(pl.col(target).is_not_null())
        .group_by([week_col, "port_name"])
        .agg(pl.col(target).median().alias("median_val"))
    )
    _gm = df_train.filter(pl.col(target).is_not_null())[target].median()
    global_median: float = float(_gm) if _gm is not None else 0.0

    preds = []
    for row in df_pred.iter_rows(named=True):
        match = medians.filter(
            (pl.col(week_col) == row[week_col])
            & (pl.col("port_name") == row["port_name"])
        )
        if len(match) > 0:
            _v = match["median_val"][0]
            preds.append(float(_v) if _v is not None else 0.0)
        else:
            _gp = df_train.filter(
                pl.col(target).is_not_null()
                & (pl.col("port_name") == row["port_name"])
            )[target].median()
            preds.append(float(_gp) if _gp is not None else global_median)

    return np.clip(np.array(preds, dtype=np.float32), 0, None)


def risk_metrics(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    label: str,
) -> dict[str, object]:
    """Compute ROC-AUC, Brier Score, and Brier Skill Score.

    Brier Skill Score (BSS) measures improvement over a climatological forecast
    (always predicting the observed positive rate). BSS > 0 means the model beats
    climatology; BSS = 0 means no skill; BSS < 0 means worse than climatology.

    Args:
        y_true_bin: Binary ground truth array (1 = congested, 0 = not congested).
        y_prob: Predicted probabilities in [0, 1].
        label: Row label string for display.

    Returns:
        Dictionary of evaluation metrics.
    """
    n = len(y_true_bin)
    n_pos = int(y_true_bin.sum())
    prev = n_pos / n if n > 0 else 0.0

    if n_pos == 0 or n_pos == n:
        return {
            "label": label,
            "n": n,
            "prev_pct": round(prev * 100, 1),
            "auc": np.nan,
            "brier": np.nan,
            "bss": np.nan,
        }

    auc = float(roc_auc_score(y_true_bin, y_prob))
    brier = float(brier_score_loss(y_true_bin, y_prob))
    brier_ref = prev * (1.0 - prev)
    bss = float(1.0 - brier / brier_ref) if brier_ref > 0 else np.nan

    return {
        "label": label,
        "n": n,
        "prev_pct": round(prev * 100, 1),
        "auc": auc,
        "brier": brier,
        "bss": bss,
    }


def _fmt(val: object, spec: str = ">8.3f") -> str:
    """Format a metric value, returning N/A for NaN."""
    if isinstance(val, float) and np.isnan(val):
        return "     N/A"
    return format(val, spec)


def print_risk_table(rows: list[dict[str, object]]) -> None:
    """Pretty-print risk evaluation metrics table.

    Args:
        rows: List of metric dictionaries from risk_metrics().
    """
    header = (
        f"{'Port / Scope':<32} {'N':>6} {'Prev%':>7} {'AUC':>8} {'Brier':>8} {'BSS':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<32} {r['n']:>6} {r['prev_pct']:>7.1f} "
            f"{_fmt(r['auc'])} {_fmt(r['brier'])} {_fmt(r['bss'])}"
        )


def print_spec_comparison(
    spec_results: dict[str, dict[str, object]],
) -> None:
    """Print a compact spec-vs-spec comparison table for global AUC/Brier/BSS.

    Args:
        spec_results: Mapping of spec_name → global metric dict from risk_metrics().
    """
    header = (
        f"{'Spec':<12} {'N':>6} {'Prev%':>7} {'AUC':>8} {'Brier':>8} {'BSS':>8}"
    )
    print(header)
    print("-" * len(header))
    for spec_name, r in spec_results.items():
        print(
            f"{spec_name:<12} {r['n']:>6} {r['prev_pct']:>7.1f} "
            f"{_fmt(r['auc'])} {_fmt(r['brier'])} {_fmt(r['bss'])}"
        )


def plot_calibration_figure(
    dataset_label: str,
    horizon_label: str,
    threshold: float,
    spec_data: dict[str, tuple[np.ndarray, np.ndarray]],
    y_test_bin: np.ndarray,
    port_names: np.ndarray,
    primary_ports: list[str],
) -> None:
    """Save reliability diagram for all-ports combined and each primary port.

    Shows one curve per spec on each subplot for cross-spec comparison.

    Args:
        dataset_label: Dataset name string (e.g. 'container').
        horizon_label: Horizon string (e.g. '1w').
        threshold: Exceedance threshold in hours.
        spec_data: Mapping spec_name → (y_test_bin, risk_probs) tuples.
        y_test_bin: Binary outcomes for all test rows (from M4 spec).
        port_names: Port name for each test row.
        primary_ports: Ports to show in individual subplots.
    """
    SPEC_COLORS = {
        "M0-Naive": "grey",
        "M1-XGB": "green",
        "M2-XGB": "orange",
        "M3-XGB": "steelblue",
        "M4-XGB": "crimson",
    }

    n_cols = 1 + len(primary_ports)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(4 * n_cols, 4), constrained_layout=True
    )
    if n_cols == 1:
        axes = [axes]

    def _draw(
        ax: Any,
        y_bin: np.ndarray,
        specs: dict[str, np.ndarray],
        title: str,
    ) -> None:
        n_pos = int(y_bin.sum())
        if n_pos == 0 or n_pos == len(y_bin):
            ax.text(
                0.5,
                0.5,
                "No class variation\n(all one outcome)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="grey",
            )
            ax.set_title(title, fontsize=10)
            return

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        for spec_name, y_p in specs.items():
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_bin, y_p, n_bins=5, strategy="quantile"
                )
                ax.plot(
                    mean_pred,
                    frac_pos,
                    "o-",
                    color=SPEC_COLORS.get(spec_name, "black"),
                    linewidth=1.5,
                    markersize=4,
                    label=spec_name,
                )
            except ValueError:
                pass

        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    all_specs = {name: probs for name, (_, probs) in spec_data.items()}
    _draw(axes[0], y_test_bin, all_specs, f"ALL ports\n(p95 > {threshold:.0f}h)")

    for i, port in enumerate(primary_ports, start=1):
        mask = port_names == port
        port_specs = {name: probs[mask] for name, (_, probs) in spec_data.items()}
        _draw(
            axes[i],
            y_test_bin[mask],
            port_specs,
            port.replace("_", "\n").replace("LOS\nANGELES\nLONG\nBEACH", "LA/LB"),
        )

    fig.suptitle(
        f"{dataset_label.upper()} — {horizon_label} Congestion Risk Calibration"
        f" (threshold: p95 > {threshold:.0f}h)",
        fontsize=12,
        fontweight="bold",
    )
    out_path = FIGURES_DIR / f"calibration_{dataset_label}_{horizon_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path.relative_to(Path(__file__).resolve().parent)}")


grand_results: list[dict[str, object]] = []

for ds in DATASETS:
    ds_start = time.perf_counter()
    label = ds["label"]
    threshold = ds["threshold"]
    primary = ds["primary_ports"]

    print(f"\n{'=' * 80}")
    print(f"DATASET: {label.upper()}   threshold: p95 > {threshold:.0f}h")
    print(f"{'=' * 80}")

    df = pl.read_csv(ds["data_path"], try_parse_dates=True)
    df = (
        df.filter(pl.col("port_name").is_in(PORTS_TO_KEEP))
        .sort(["port_name", "ref_date"])
        .pipe(encode_ports)
    )

    feat_all = [c for c in df.columns if c.startswith("feat_")]
    macro_all = [c for c in df.columns if c.startswith("macro_")]

    feat_m1 = [f for f in FEAT_M1 if f in df.columns]
    feat_m2 = [f for f in FEAT_M2 if f in df.columns]

    _train_end = pl.lit(TRAIN_END).str.to_date("%Y-%m-%d")
    _val_end = pl.lit(VAL_END).str.to_date("%Y-%m-%d")

    df_train_all = df.filter(pl.col("ref_date") <= _train_end)
    df_val_all = df.filter(
        (pl.col("ref_date") > _train_end.dt.offset_by(f"{GAP_DAYS}d"))
        & (pl.col("ref_date") <= _val_end)
    )
    df_test_all = df.filter(pl.col("ref_date") > _val_end.dt.offset_by(f"{GAP_DAYS}d"))

    for target in TARGETS:
        hlabel = target.replace("target_p95_waiting_", "")
        t_start = time.perf_counter()

        df_train = df_train_all.filter(pl.col(target).is_not_null())
        df_val = df_val_all.filter(pl.col(target).is_not_null())
        df_test = df_test_all.filter(pl.col(target).is_not_null())

        print(
            f"\n--- Horizon: {hlabel}  |  Train:{df_train.shape[0]}  "
            f"Val:{df_val.shape[0]}  Test:{df_test.shape[0]}"
        )

        y_va = df_val[target].to_numpy().astype(np.float32)
        y_te = df_test[target].to_numpy().astype(np.float32)
        y_val_bin = (y_va > threshold).astype(int)
        y_test_bin = (y_te > threshold).astype(int)

        port_names = df_test["port_name"].to_numpy()

        SPECS: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        val_naive = seasonal_naive_predict(df_train, df_val, target)
        test_naive = seasonal_naive_predict(df_train, df_test, target)
        SPECS["M0-Naive"] = (val_naive, test_naive)

        xgb_specs: list[tuple[str, list[str]]] = [
            ("M1-XGB", feat_m1 + [PORT_COL]),
            ("M2-XGB", feat_m2 + [PORT_COL]),
            ("M3-XGB", feat_all + [PORT_COL]),
            ("M4-XGB", feat_all + macro_all + [PORT_COL]),
        ]

        y_tr = df_train[target].fill_null(0).to_numpy().astype(np.float32)

        for spec_name, features in xgb_specs:
            features_avail = [f for f in features if f in df.columns]
            if len(features_avail) < 2:
                print(f"  SKIP {spec_name}: insufficient features available")
                continue

            X_tr = (
                df_train.select(features_avail).fill_null(0).to_numpy().astype(np.float32)
            )
            X_va = (
                df_val.select(features_avail).fill_null(0).to_numpy().astype(np.float32)
            )
            X_te = (
                df_test.select(features_avail).fill_null(0).to_numpy().astype(np.float32)
            )

            model = xgb.XGBRegressor(**XGB_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

            val_p = np.clip(model.predict(X_va), 0, None)
            test_p = np.clip(model.predict(X_te), 0, None)
            SPECS[spec_name] = (val_p, test_p)

        spec_global_results: dict[str, dict[str, object]] = {}
        spec_risk_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        print(f"\n  SPEC COMPARISON (global, horizon={hlabel}):")
        rows_spec: list[dict[str, object]] = []

        for spec_name, (val_preds, test_preds) in SPECS.items():
            if len(np.unique(y_val_bin)) < 2:
                print(
                    f"  WARNING: val labels all one class at threshold {threshold:.0f}h"
                    f" — skipping {hlabel}"
                )
                break

            calibrator = LogisticRegression(
                C=1.0, random_state=RANDOM_STATE, max_iter=1000
            )
            calibrator.fit(val_preds.reshape(-1, 1), y_val_bin)
            risk_probs = calibrator.predict_proba(test_preds.reshape(-1, 1))[:, 1]

            m = risk_metrics(y_test_bin, risk_probs, spec_name)
            spec_global_results[spec_name] = m
            spec_risk_probs[spec_name] = (y_test_bin, risk_probs)
            rows_spec.append(m)

        if rows_spec:
            print_spec_comparison(
                {r["label"]: r for r in rows_spec if isinstance(r["label"], str)}
            )

        if "M4-XGB" in spec_risk_probs:
            _, m4_risk_probs = spec_risk_probs["M4-XGB"]
            print(f"\n  M4-XGB PER-PORT DETAIL (horizon={hlabel}):")
            rows_port: list[dict[str, object]] = [
                risk_metrics(y_test_bin, m4_risk_probs, "ALL")
            ]
            for port in PORTS_TO_KEEP:
                mask = port_names == port
                if mask.sum() > 0:
                    rows_port.append(
                        risk_metrics(y_test_bin[mask], m4_risk_probs[mask], port)
                    )
            print_risk_table(rows_port)

        if hlabel == "1w" and spec_risk_probs:
            plot_calibration_figure(
                label,
                hlabel,
                threshold,
                spec_risk_probs,
                y_test_bin,
                port_names,
                primary,
            )

        for spec_name, m in spec_global_results.items():
            grand_results.append(
                {
                    **m,
                    "dataset": label,
                    "horizon": hlabel,
                    "threshold": threshold,
                    "spec": spec_name,
                }
            )

        print(f"  Horizon time: {time.perf_counter() - t_start:.1f}s")

    print(f"Dataset time: {time.perf_counter() - ds_start:.1f}s")


print(f"\n{'=' * 110}")
print("RISK PROBABILITY GRAND SUMMARY — ALL DATASETS × ALL HORIZONS × ALL SPECS")
print(f"{'=' * 110}")
header = (
    f"{'Dataset':<15} {'Horizon':<8} {'Threshold':>11} {'Spec':<12} "
    f"{'N':>6} {'Prev%':>7} {'AUC':>8} {'Brier':>8} {'BSS':>8}"
)
print(header)
print("-" * len(header))
for r in grand_results:
    print(
        f"{r['dataset']:<15} {r['horizon']:<8} {r['threshold']:>10.0f}h {r['spec']:<12} "
        f"{r['n']:>6} {r['prev_pct']:>7.1f} "
        f"{_fmt(r['auc'])} {_fmt(r['brier'])} {_fmt(r['bss'])}"
    )

total_time = time.perf_counter() - notebook_start
print(f"\nTotal runtime: {total_time:.1f}s ({total_time / 60:.1f}m)")
