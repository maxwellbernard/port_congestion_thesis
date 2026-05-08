
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

FEATURES_DIR = Path("../ais_data/features")
IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"

SPLIT_FILES: dict[str, str] = {
    "bulk_carrier": "daily_features_bulk_carrier.parquet",
    "container_ship": "daily_features_container.parquet",
}

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
    }
)

PORT_COLORS = {
    "LOS_ANGELES_LONG_BEACH": "#FF6B6B",
    "NEW_YORK_NEW_JERSEY": "#4ECDC4",
    "HOUSTON": "#FFE66D",
    "PORT_OF_VIRGINIA": "#96CEB4",
}


datasets: dict[str, pl.DataFrame] = {}
for split_name, filename in SPLIT_FILES.items():
    path = FEATURES_DIR / filename
    if path.exists():
        datasets[split_name] = pl.read_parquet(path)
        print(f"Loaded {split_name}: {datasets[split_name].shape}")
    else:
        print(f"WARNING: {path} not found, skipping {split_name}")


for split_name, df in datasets.items():
    print(f"\n{'=' * 60}")
    print(f"  {split_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape}")
    print(f"Rows: {df.height:,}")
    print(f"Columns: {df.width}")

    id_cols = [c for c in df.columns if not c.startswith(("feat_", "target_"))]
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    print(f"ID columns ({len(id_cols)}): {id_cols}")
    print(f"Feature columns: {len(feat_cols)}")
    print(f"Target columns: {len(target_cols)}")

    null_counts = df.null_count()
    has_nulls = False
    for col in df.columns:
        n = null_counts[col][0]
        if n > 0:
            has_nulls = True
            print(f"  NULL: {col}: {n} ({n / df.height * 100:.1f}%)")
    if not has_nulls:
        print("  No nulls in any column.")

    print("\nRows per port:")
    print(
        df.group_by("port_name")
        .agg(
            pl.len().alias("days"),
            pl.col("ref_date").min().alias("first_date"),
            pl.col("ref_date").max().alias("last_date"),
        )
        .sort("port_name")
    )


for split_name, df in datasets.items():
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    if not target_cols:
        continue

    fig, axes = plt.subplots(1, len(target_cols), figsize=(6 * len(target_cols), 4))
    if len(target_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, target_cols):
        vals = df[col].drop_nulls().to_numpy()
        ax.hist(vals, bins=40, color="#4ECDC4", edgecolor="white", alpha=0.8)
        ax.axvline(
            np.median(vals),
            color="red",
            linestyle="--",
            label=f"median={np.median(vals):.1f}",
        )
        ax.axvline(
            np.mean(vals),
            color="orange",
            linestyle="--",
            label=f"mean={np.mean(vals):.1f}",
        )
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Target Distributions — {split_name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_target_distributions.png", dpi=150)
    plt.show()

for split_name, df in datasets.items():
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    print(f"\n--- {split_name} — Target summary statistics ---")
    for col in target_cols:
        series = df[col].drop_nulls()
        print(f"\n{col}:")
        print(f"  count:  {series.len()}")
        print(f"  mean:   {series.mean():.2f}")
        print(f"  std:    {series.std():.2f}")
        print(f"  min:    {series.min():.2f}")
        print(f"  p25:    {series.quantile(0.25):.2f}")
        print(f"  p50:    {series.quantile(0.50):.2f}")
        print(f"  p75:    {series.quantile(0.75):.2f}")
        print(f"  p95:    {series.quantile(0.95):.2f}")
        print(f"  max:    {series.max():.2f}")

for split_name, df in datasets.items():
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    for col in target_cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        ports = df["port_name"].unique().sort().to_list()
        for port in ports:
            vals = df.filter(pl.col("port_name") == port)[col].drop_nulls().to_numpy()
            if len(vals) > 0:
                ax.hist(
                    vals,
                    bins=30,
                    alpha=0.5,
                    label=port.replace("_", " "),
                    color=PORT_COLORS.get(port, "gray"),
                )
        ax.set_title(
            f"{col} — by Port [{split_name}]", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Hours")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        plt.tight_layout()
        split_images = IMAGES_DIR / split_name
        split_images.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            split_images / f"eda_target_by_port_{col}.png", dpi=150
        )
        plt.show()


for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    base_feats = [
        c
        for c in feat_cols
        if not any(s in c for s in ["_lag", "_roll", "_change_1w"])
        and c
        not in [
            "feat_week_of_year",
            "feat_month",
            "feat_quarter",
            "feat_consecutive_high_days",
            "feat_high_pct_4w",
            "feat_waiting_trend_4w",
        ]
    ]

    n_feats = len(base_feats)
    if n_feats == 0:
        continue
    n_cols_grid = 4
    n_rows_grid = (n_feats + n_cols_grid - 1) // n_cols_grid

    fig, axes = plt.subplots(
        n_rows_grid, n_cols_grid, figsize=(4 * n_cols_grid, 3 * n_rows_grid)
    )
    axes = axes.flatten()

    for i, col in enumerate(base_feats):
        ax = axes[i]
        vals = df[col].drop_nulls().to_numpy()
        ax.hist(vals, bins=40, color="#45B7D1", edgecolor="white", alpha=0.8)
        ax.set_title(col.replace("feat_", ""), fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Base Feature Distributions — {split_name}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_base_feature_distributions.png", dpi=150)
    plt.show()

for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    regime_temporal = [
        "feat_week_of_year",
        "feat_month",
        "feat_quarter",
        "feat_consecutive_high_days",
        "feat_high_pct_4w",
        "feat_waiting_trend_4w",
    ]
    regime_temporal = [c for c in regime_temporal if c in feat_cols]
    if not regime_temporal:
        continue

    fig, axes = plt.subplots(
        1, len(regime_temporal), figsize=(4 * len(regime_temporal), 3)
    )
    if len(regime_temporal) == 1:
        axes = [axes]

    for ax, col in zip(axes, regime_temporal):
        vals = df[col].drop_nulls().to_numpy()
        ax.hist(vals, bins=30, color="#96CEB4", edgecolor="white", alpha=0.8)
        ax.set_title(col.replace("feat_", ""), fontsize=9, fontweight="bold")

    plt.suptitle(
        f"Regime & Temporal Features — {split_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_regime_temporal_features.png", dpi=150)
    plt.show()

for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    print(f"\n--- {split_name} — Feature summary statistics ---")
    stats_rows = []
    for col in feat_cols:
        s = df[col].drop_nulls()
        if s.len() == 0:
            continue
        stats_rows.append(
            {
                "feature": col,
                "count": s.len(),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "p50": round(float(s.quantile(0.50)), 3),
                "max": round(float(s.max()), 3),
            }
        )

    stats_df = pl.DataFrame(stats_rows)
    with pl.Config(tbl_rows=len(stats_df)):
        print(stats_df)


for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    comparison_feats = [
        "feat_p95_waiting_hours",
        "feat_mean_waiting_hours",
        "feat_mean_queue_length",
        "feat_mean_berth_count",
        "feat_n_visits",
        "feat_arrival_count",
        "feat_service_rate",
        "feat_pct_waiting",
        "feat_queue_to_berth_ratio",
    ]
    comparison_feats = [c for c in comparison_feats if c in feat_cols]
    if not comparison_feats:
        continue

    df_pd = df.select(["port_name"] + comparison_feats).to_pandas()

    n_plots = len(comparison_feats)
    n_cols_box = 3
    n_rows_box = (n_plots + n_cols_box - 1) // n_cols_box

    fig, axes = plt.subplots(n_rows_box, n_cols_box, figsize=(15, 4 * n_rows_box))
    axes = axes.flatten()

    for i, col in enumerate(comparison_feats):
        ax = axes[i]
        sns.boxplot(
            data=df_pd,
            x="port_name",
            y=col,
            hue="port_name",
            ax=ax,
            palette=PORT_COLORS,
            legend=False,
            showfliers=True,
            fliersize=2,
        )
        ax.set_title(col.replace("feat_", ""), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_xticks(range(df_pd["port_name"].nunique()))
        ax.set_xticklabels(
            [p.replace("_", "\n") for p in sorted(df_pd["port_name"].unique())],
            fontsize=7,
        )

    for j in range(len(comparison_feats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        f"Feature Distributions by Port — {split_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_feature_boxplots_by_port.png", dpi=150)
    plt.show()


for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    ts_feats = [
        "feat_p95_waiting_hours",
        "feat_mean_queue_length",
        "feat_mean_berth_count",
        "feat_n_visits",
        "feat_arrival_count",
        "feat_service_rate",
        "feat_pct_waiting",
        "feat_queue_to_berth_ratio",
    ]
    ts_feats = [c for c in ts_feats if c in feat_cols]
    if not ts_feats:
        continue

    n_ts = len(ts_feats)
    fig, axes = plt.subplots(n_ts, 1, figsize=(14, 3.5 * n_ts), sharex=True)
    if n_ts == 1:
        axes = [axes]

    ports = df["port_name"].unique().sort().to_list()

    for ax, col in zip(axes, ts_feats):
        for port in ports:
            port_data = df.filter(pl.col("port_name") == port).sort("ref_date")
            ax.plot(
                port_data["ref_date"].to_list(),
                port_data[col].to_list(),
                label=port.replace("_", " "),
                color=PORT_COLORS.get(port, "gray"),
                linewidth=1.2,
                alpha=0.8,
            )
        ax.set_ylabel(col.replace("feat_", ""), fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[0].legend(loc="upper right", fontsize=8, ncol=len(ports))
    axes[-1].set_xlabel("Date")
    plt.suptitle(
        f"Daily Feature Time Series — {split_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_feature_time_series.png", dpi=150)
    plt.show()

for split_name, df in datasets.items():
    target_cols = sorted([c for c in df.columns if c.startswith("target_")])
    if not target_cols:
        continue

    fig, axes = plt.subplots(
        len(target_cols), 1, figsize=(14, 3.5 * len(target_cols)), sharex=True
    )
    if len(target_cols) == 1:
        axes = [axes]

    ports = df["port_name"].unique().sort().to_list()

    for ax, col in zip(axes, target_cols):
        for port in ports:
            port_data = df.filter(pl.col("port_name") == port).sort("ref_date")
            ax.plot(
                port_data["ref_date"].to_list(),
                port_data[col].to_list(),
                label=port.replace("_", " "),
                color=PORT_COLORS.get(port, "gray"),
                linewidth=1.2,
                alpha=0.8,
            )
        ax.set_ylabel(col.replace("target_", ""), fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[0].legend(loc="upper right", fontsize=8, ncol=len(ports))
    axes[-1].set_xlabel("Date")
    plt.suptitle(
        f"Daily Target Time Series — {split_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_target_time_series.png", dpi=150)
    plt.show()


primary_target = "target_p95_waiting_1w"

for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    if primary_target not in df.columns:
        continue

    print(f"\n--- {split_name} — Feature correlations with {primary_target} ---")

    corrs = []
    for col in feat_cols:
        valid = df.select(col, primary_target).drop_nulls()
        if valid.height < 10:
            continue
        r = np.corrcoef(valid[col].to_numpy(), valid[primary_target].to_numpy())[0, 1]
        if not np.isnan(r):
            corrs.append({"feature": col, "corr": round(float(r), 3)})

    corr_df = pl.DataFrame(corrs).sort("corr", descending=True)

    print("\nTop 15 positive:")
    print(corr_df.head(15))
    print("\nTop 10 negative:")
    print(corr_df.tail(10))

    corr_df_abs = corr_df.with_columns(
        pl.col("corr").abs().alias("abs_corr")
    ).sort("abs_corr", descending=True)

    top_n = min(25, corr_df_abs.height)
    top = corr_df_abs.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#FF6B6B" if v > 0 else "#4ECDC4" for v in top["corr"].to_list()]
    y_pos = range(top_n)

    ax.barh(
        y_pos,
        top["corr"].to_list(),
        color=colors,
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f.replace("feat_", "") for f in top["feature"].to_list()],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Pearson Correlation")
    ax.set_title(
        f"Top {top_n} Features — {split_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_correlation_bar_chart.png", dpi=150)
    plt.show()

for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    base_feats = [
        c
        for c in feat_cols
        if not any(s in c for s in ["_lag", "_roll", "_change_1w"])
        and c
        not in [
            "feat_week_of_year",
            "feat_month",
            "feat_quarter",
            "feat_consecutive_high_days",
            "feat_high_pct_4w",
            "feat_waiting_trend_4w",
        ]
    ]
    heatmap_cols = [c for c in base_feats if c in feat_cols]
    if primary_target in df.columns:
        heatmap_cols = heatmap_cols + [primary_target]

    if len(heatmap_cols) < 2:
        continue

    corr_matrix = df.select(heatmap_cols).drop_nulls().to_pandas().corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        xticklabels=[
            c.replace("feat_", "").replace("target_", "T:") for c in heatmap_cols
        ],
        yticklabels=[
            c.replace("feat_", "").replace("target_", "T:") for c in heatmap_cols
        ],
        annot_kws={"size": 7},
    )
    ax.set_title(
        f"Correlation Matrix — {split_name}", fontsize=14, fontweight="bold"
    )
    plt.xticks(fontsize=8, rotation=45, ha="right")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    split_images = IMAGES_DIR / split_name
    split_images.mkdir(parents=True, exist_ok=True)
    plt.savefig(split_images / "eda_correlation_heatmap.png", dpi=150)
    plt.show()


for split_name, df in datasets.items():
    print(f"\n--- {split_name} — Lag feature validation (first port) ---")
    first_port = df["port_name"].unique().sort().to_list()[0]
    port_df = df.filter(pl.col("port_name") == first_port).sort("ref_date")

    for lag_name, n_days in [("lag1w", 7), ("lag2w", 14), ("lag4w", 28)]:
        col = f"feat_p95_waiting_hours_{lag_name}"
        if col not in port_df.columns:
            continue
        actual_lag = port_df[col]
        expected_lag = port_df["feat_p95_waiting_hours"].shift(n_days)
        valid_mask = actual_lag.is_not_null() & expected_lag.is_not_null()
        n_valid = valid_mask.sum()
        if n_valid > 0:
            matches = (
                (actual_lag.filter(valid_mask) - expected_lag.filter(valid_mask)).abs()
                < 1e-6
            ).sum()
            print(
                f"  p95_waiting_hours_{lag_name}: "
                f"{matches}/{n_valid} rows match shift({n_days})"
            )
        else:
            print(f"  {col}: not enough non-null rows")

    if "feat_p95_waiting_hours_roll4w_mean" in port_df.columns:
        aligned = port_df.select(
            pl.col("feat_p95_waiting_hours").alias("current"),
            pl.col("feat_p95_waiting_hours").shift(1).alias("prev"),
            pl.col("feat_p95_waiting_hours_roll4w_mean").alias("roll4"),
        ).drop_nulls()

        corr_same = np.corrcoef(
            aligned["current"].to_numpy(), aligned["roll4"].to_numpy()
        )[0, 1]
        corr_prev = np.corrcoef(
            aligned["prev"].to_numpy(), aligned["roll4"].to_numpy()
        )[0, 1]
        print(f"\n  roll4w_mean correlation with current week: {corr_same:.3f}")
        print(f"  roll4w_mean correlation with previous week: {corr_prev:.3f}")
        print("  (Previous week should be >= current week if shift(1) is correct)")

for split_name, df in datasets.items():
    feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
    print(f"\n--- {split_name} — Data leakage check ---")
    if primary_target not in df.columns:
        print("  No target column found, skipping.")
        continue

    suspicious = []
    for col in feat_cols:
        valid = df.select(col, primary_target).drop_nulls()
        if valid.height < 10:
            continue
        r = np.corrcoef(valid[col].to_numpy(), valid[primary_target].to_numpy())[0, 1]
        if not np.isnan(r) and abs(r) > 0.99:
            suspicious.append((col, float(r)))

    if suspicious:
        print("  WARNING: Features with |correlation| > 0.99 with target:")
        for col, r in suspicious:
            print(f"    {col}: {r:.4f}")
    else:
        print("  No features with suspicious (>0.99) correlation. Looks clean.")

