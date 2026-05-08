
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "final_dataframes"
OUT_DIR = ROOT / "images" / "congestion_eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_1W = "target_p95_waiting_1w"

DATASETS: dict[str, str] = {
    "Container": "container.csv",
    "Bulk Carrier": "bulk_carrier.csv",
}

PORTS = [
    "HOUSTON",
    "LOS_ANGELES_LONG_BEACH",
    "NEW_YORK_NEW_JERSEY",
    "PORT_OF_VIRGINIA",
]

PORT_LABELS = {
    "HOUSTON": "Houston",
    "LOS_ANGELES_LONG_BEACH": "LA / Long Beach",
    "NEW_YORK_NEW_JERSEY": "NY / NJ",
    "PORT_OF_VIRGINIA": "Port of Virginia",
}

PORT_COLORS = {
    "HOUSTON": "#E63946",
    "LOS_ANGELES_LONG_BEACH": "#457B9D",
    "NEW_YORK_NEW_JERSEY": "#2A9D8F",
    "PORT_OF_VIRGINIA": "#F4A261",
}

DATASET_COLORS = {
    "Container": "#264653",
    "Bulk Carrier": "#E9C46A",
}

HIGH_CONGESTION_THRESHOLD = 48.0

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8F8F8",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


dfs: dict[str, pl.DataFrame] = {}

for name, fname in DATASETS.items():
    path = DATA_DIR / fname
    df = pl.read_csv(path, try_parse_dates=True)
    df = df.filter(pl.col("port_name").is_in(PORTS))
    dfs[name] = df
    print(f"  {name}: {df.shape[0]:,} rows, {df['ref_date'].min()} → {df['ref_date'].max()}")


print("\n" + "=" * 100)
print(f"  {'Dataset':<14} {'Port':<26} {'N':>6} {'Zeros%':>7} {'Mean':>8} {'Median':>8} "
      f"{'p75':>8} {'p95':>8} {'Max':>8}")
print("=" * 100)

for name, df in dfs.items():
    col = df[TARGET_1W]
    n = col.drop_nulls().len()
    zeros = (col.drop_nulls() == 0).sum()
    arr = col.drop_nulls().to_numpy()
    print(f"  {name:<14} {'ALL':<26} {n:>6} {zeros/n*100:>6.1f}% "
          f"{arr.mean():>8.1f} {np.median(arr):>8.1f} "
          f"{np.percentile(arr, 75):>8.1f} {np.percentile(arr, 95):>8.1f} "
          f"{arr.max():>8.1f}")

    for port in PORTS:
        port_col = df.filter(pl.col("port_name") == port)[TARGET_1W].drop_nulls()
        pn = port_col.len()
        if pn == 0:
            continue
        pz = (port_col == 0).sum()
        parr = port_col.to_numpy()
        label = PORT_LABELS[port]
        print(f"  {'':<14} {label:<26} {pn:>6} {pz/pn*100:>6.1f}% "
              f"{parr.mean():>8.1f} {np.median(parr):>8.1f} "
              f"{np.percentile(parr, 75):>8.1f} {np.percentile(parr, 95):>8.1f} "
              f"{parr.max():>8.1f}")
    print("-" * 100)


def _weekly_resample(df: pl.DataFrame, port: str, target: str) -> tuple[list, list]:
    """Return (dates, weekly-median values) for a port time series."""
    port_df = (
        df.filter(pl.col("port_name") == port)
        .select(["ref_date", target])
        .drop_nulls()
        .sort("ref_date")
        .with_columns(
            pl.col("ref_date").dt.truncate("1w").alias("week")
        )
        .group_by("week")
        .agg(pl.col(target).median().alias("val"))
        .sort("week")
    )
    return port_df["week"].to_list(), port_df["val"].to_numpy()


for name, df in dfs.items():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Weekly p95 Waiting Time — {name}", fontsize=13, fontweight="bold")

    for ax, port in zip(axes.flat, PORTS):
        dates, vals = _weekly_resample(df, port, TARGET_1W)
        color = PORT_COLORS[port]

        ax.fill_between(dates, vals, alpha=0.25, color=color)
        ax.plot(dates, vals, color=color, linewidth=1.2)
        ax.axhline(HIGH_CONGESTION_THRESHOLD, color="red", linestyle="--",
                   linewidth=0.8, alpha=0.7, label=f"{HIGH_CONGESTION_THRESHOLD:.0f}h threshold")

        ax.set_title(PORT_LABELS[port])
        ax.set_ylabel("p95 wait (hrs)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        arr = np.array(vals)
        above = arr >= HIGH_CONGESTION_THRESHOLD
        if above.any():
            ax.fill_between(dates, vals, HIGH_CONGESTION_THRESHOLD,
                            where=above, alpha=0.35, color="red", label="High congestion")

        zero_pct = (arr == 0).mean() * 100
        ax.text(0.02, 0.96, f"{zero_pct:.0f}% zero", transform=ax.transAxes,
                fontsize=8, va="top", color="gray")

    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fname = f"timeseries_{name.lower().replace(' ', '_')}.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / fname}")


for port in PORTS:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_title(f"p95 Waiting Time by Vessel Type — {PORT_LABELS[port]}", fontweight="bold")

    for name, df in dfs.items():
        dates, vals = _weekly_resample(df, port, TARGET_1W)
        if len(dates) == 0:
            continue
        color = DATASET_COLORS[name]
        ax.plot(dates, vals, color=color, linewidth=1.3, label=name, alpha=0.85)

    ax.axhline(HIGH_CONGESTION_THRESHOLD, color="red", linestyle="--",
               linewidth=0.8, alpha=0.6, label=f"{HIGH_CONGESTION_THRESHOLD:.0f}h threshold")
    ax.set_ylabel("p95 wait (hrs)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=9)
    fig.tight_layout()

    fname = f"cross_dataset_{port.lower()}.png"
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / fname}")


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle("Seasonal Pattern — Median p95 Waiting Time by Month", fontsize=12, fontweight="bold")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

for ax, (name, df) in zip(axes, dfs.items()):
    monthly = (
        df.filter(pl.col("port_name").is_in(PORTS))
        .select(["port_name", "feat_month", TARGET_1W])
        .drop_nulls()
        .group_by(["port_name", "feat_month"])
        .agg(pl.col(TARGET_1W).median().alias("median_wait"))
        .sort(["port_name", "feat_month"])
    )

    for port in PORTS:
        port_monthly = monthly.filter(pl.col("port_name") == port)
        if port_monthly.is_empty():
            continue
        months = port_monthly["feat_month"].to_list()
        vals = port_monthly["median_wait"].to_numpy()
        ax.plot(months, vals, marker="o", markersize=4, linewidth=1.5,
                color=PORT_COLORS[port], label=PORT_LABELS[port])

    ax.set_title(name)
    ax.set_xlabel("Month")
    ax.set_ylabel("Median p95 wait (hrs)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUT_DIR / "seasonal_monthly.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'seasonal_monthly.png'}")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Distribution of p95 Waiting Time (non-zero only)", fontsize=12, fontweight="bold")

for ax, (name, df) in zip(axes, dfs.items()):
    for port in PORTS:
        vals = df.filter(pl.col("port_name") == port)[TARGET_1W].drop_nulls()
        arr = vals.to_numpy()
        nonzero = arr[arr > 0]
        if len(nonzero) < 10:
            continue
        ax.hist(nonzero, bins=40, alpha=0.5, density=True,
                color=PORT_COLORS[port], label=PORT_LABELS[port])

    ax.set_title(name)
    ax.set_xlabel("p95 wait (hrs)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUT_DIR / "distribution_nonzero.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'distribution_nonzero.png'}")


def _congestion_regimes(
    df: pl.DataFrame,
    port: str,
    target: str,
    threshold: float,
    min_consecutive_days: int = 7,
) -> list[tuple]:
    """Return list of (start, end, max_wait) for sustained congestion periods."""
    port_df = (
        df.filter(pl.col("port_name") == port)
        .select(["ref_date", target])
        .drop_nulls()
        .sort("ref_date")
    )
    dates = port_df["ref_date"].to_list()
    vals = port_df[target].to_numpy()

    regimes: list[tuple] = []
    in_regime = False
    start_idx = 0

    for i, v in enumerate(vals):
        if v >= threshold and not in_regime:
            in_regime = True
            start_idx = i
        elif v < threshold and in_regime:
            length = i - start_idx
            if length >= min_consecutive_days:
                regimes.append((
                    dates[start_idx],
                    dates[i - 1],
                    float(vals[start_idx:i].max()),
                    length,
                ))
            in_regime = False

    if in_regime:
        length = len(vals) - start_idx
        if length >= min_consecutive_days:
            regimes.append((
                dates[start_idx],
                dates[-1],
                float(vals[start_idx:].max()),
                length,
            ))

    return regimes


print("\n" + "=" * 90)
print(f"  CONGESTION REGIMES (p95 ≥ {HIGH_CONGESTION_THRESHOLD:.0f}h for ≥7 consecutive days)")
print("=" * 90)

for name, df in dfs.items():
    print(f"\n  {name.upper()}")
    print(f"  {'Port':<26} {'Start':<12} {'End':<12} {'Days':>5} {'Peak (hrs)':>12}")
    print(f"  {'-' * 70}")
    for port in PORTS:
        regimes = _congestion_regimes(df, port, TARGET_1W, HIGH_CONGESTION_THRESHOLD)
        if not regimes:
            print(f"  {PORT_LABELS[port]:<26} — no sustained congestion regimes")
        for start, end, peak, days in sorted(regimes, key=lambda x: x[0]):
            print(f"  {PORT_LABELS[port]:<26} {str(start):<12} {str(end):<12} "
                  f"{days:>5} {peak:>12.1f}")


container_df = dfs["Container"]

pivot = (
    container_df.select(["ref_date", "port_name", TARGET_1W])
    .drop_nulls()
    .pivot(on="port_name", index="ref_date", values=TARGET_1W, aggregate_function="mean")
    .sort("ref_date")
)

port_cols = [p for p in PORTS if p in pivot.columns]
corr_matrix = np.corrcoef(
    np.array([pivot[p].fill_null(0).to_numpy() for p in port_cols])
)

fig, ax = plt.subplots(figsize=(6, 5))
labels = [PORT_LABELS[p] for p in port_cols]
im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.04)
ax.set_title("Cross-Port Correlation of p95 Waiting Time\n(Container)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "cross_port_correlation_container.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {OUT_DIR / 'cross_port_correlation_container.png'}")

bulk_df = dfs["Bulk Carrier"]
pivot_bulk = (
    bulk_df.select(["ref_date", "port_name", TARGET_1W])
    .drop_nulls()
    .pivot(on="port_name", index="ref_date", values=TARGET_1W, aggregate_function="mean")
    .sort("ref_date")
)

port_cols_bulk = [p for p in PORTS if p in pivot_bulk.columns]
corr_bulk = np.corrcoef(
    np.array([pivot_bulk[p].fill_null(0).to_numpy() for p in port_cols_bulk])
)

fig, ax = plt.subplots(figsize=(6, 5))
labels_bulk = [PORT_LABELS[p] for p in port_cols_bulk]
im = ax.imshow(corr_bulk, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(labels_bulk)))
ax.set_yticks(range(len(labels_bulk)))
ax.set_xticklabels(labels_bulk, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(labels_bulk, fontsize=9)
for i in range(len(labels_bulk)):
    for j in range(len(labels_bulk)):
        ax.text(j, i, f"{corr_bulk[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.04)
ax.set_title("Cross-Port Correlation of p95 Waiting Time\n(Bulk Carrier)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "cross_port_correlation_bulk.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'cross_port_correlation_bulk.png'}")


saved = sorted(OUT_DIR.glob("*.png"))
print(f"\n{len(saved)} plots saved to {OUT_DIR}:")
for p in saved:
    print(f"  {p.name}")
