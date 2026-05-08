"""Pre-model data quality report for daily AIS feature datasets.

Usage:
    python src/daily_features_pre_model_report.py
    python src/daily_features_pre_model_report.py \
        --csv ais_data/features/daily_features_container.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class DatasetContext:
    """Container for inferred dataset metadata.

    Attributes:
        date_col: Date column name when found.
        port_col: Port/entity grouping column name when found.
        feature_cols: Predictor feature columns (prefixed by ``feat_``).
        target_cols: Forecast target columns (prefixed by ``target_``).
        numeric_cols: Numeric columns in the dataset.
    """

    date_col: str | None
    port_col: str | None
    feature_cols: list[str]
    target_cols: list[str]
    numeric_cols: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Print a pre-model data quality report for daily features CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("ais_data/features/daily_features_container.csv"),
        help="Path to daily features CSV",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Top K rows to show for large summary tables",
    )
    return parser.parse_args()


def infer_context(df: pl.DataFrame) -> DatasetContext:
    """Infer key dataset structure used by diagnostics.

    Args:
        df: Input dataframe.

    Returns:
        Inferred dataset context.
    """
    cols = df.columns
    date_candidates = ["ref_date", "date", "ds", "timestamp", "datetime"]
    port_candidates = ["port_name", "port", "port_code", "entity", "location"]

    date_col = next((c for c in date_candidates if c in cols), None)
    port_col = next((c for c in port_candidates if c in cols), None)

    feature_cols = [c for c in cols if c.startswith("feat_")]
    target_cols = [c for c in cols if c.startswith("target_")]

    numeric_dtypes = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    numeric_cols = [
        c for c, dtype in zip(cols, df.dtypes, strict=False) if dtype in numeric_dtypes
    ]

    return DatasetContext(
        date_col=date_col,
        port_col=port_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
        numeric_cols=numeric_cols,
    )


def load_dataset(csv_path: Path, date_col: str | None = "ref_date") -> pl.DataFrame:
    """Load dataset CSV.

    Args:
        csv_path: CSV path.
        date_col: Optional date column to parse.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    del date_col
    return pl.read_csv(csv_path, try_parse_dates=True)


def to_float_or_none(value: Any) -> float | None:
    """Safely convert numeric-like values to float.

    Args:
        value: Value to convert.

    Returns:
        Float value when conversion is safe, else None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_overview_table(df: pl.DataFrame, ctx: DatasetContext) -> pl.DataFrame:
    """Create high-level dataset overview statistics.

    Args:
        df: Input dataframe.
        ctx: Inferred dataset context.

    Returns:
        One-row summary table.
    """
    row_count = df.height
    duplicate_rows = row_count - df.unique().height

    data: dict[str, int | float] = {
        "rows": row_count,
        "columns": df.width,
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": round(
            (duplicate_rows / row_count * 100) if row_count else 0.0, 2
        ),
        "n_feature_cols": len(ctx.feature_cols),
        "n_target_cols": len(ctx.target_cols),
        "n_numeric_cols": len(ctx.numeric_cols),
    }

    if ctx.port_col and ctx.port_col in df.columns:
        data["n_ports"] = int(df.select(pl.col(ctx.port_col).n_unique()).item())

    if ctx.date_col and ctx.date_col in df.columns:
        data["n_dates"] = int(df.select(pl.col(ctx.date_col).n_unique()).item())

    return pl.DataFrame([data])


def build_missingness_table(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-column missingness and cardinality metrics.

    Args:
        df: Input dataframe.

    Returns:
        Per-column diagnostics sorted by missingness.
    """
    rows = df.height
    records: list[dict[str, Any]] = []

    for col in df.columns:
        s = df.get_column(col)
        null_count = int(s.null_count())
        pct_missing = round((null_count / rows * 100) if rows else 0.0, 2)
        unique_count = int(s.n_unique())
        is_numeric = s.dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }
        zero_count = int((s == 0).sum()) if is_numeric else None
        pct_zero = (
            round((int(zero_count) / rows * 100), 2)
            if (is_numeric and rows and zero_count is not None)
            else None
        )

        records.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing_count": null_count,
                "pct_missing": pct_missing,
                "n_unique": unique_count,
                "zero_count": zero_count,
                "pct_zero": pct_zero,
            }
        )

    return pl.DataFrame(records).sort(
        ["pct_missing", "column"], descending=[True, False]
    )


def build_constant_columns_table(df: pl.DataFrame) -> pl.DataFrame:
    """Find columns with no variation.

    Args:
        df: Input dataframe.

    Returns:
        Table of constant or near-constant columns.
    """
    records: list[dict[str, Any]] = []
    for col in df.columns:
        s = df.get_column(col)
        n_unique = int(s.n_unique())
        if n_unique <= 1:
            value = s.drop_nulls().head(1).to_list()
            records.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "constant_value": value[0] if value else None,
                }
            )
    return (
        pl.DataFrame(records)
        if records
        else pl.DataFrame({"column": [], "n_unique": [], "constant_value": []})
    )


def build_numeric_summary_table(
    df: pl.DataFrame, numeric_cols: list[str]
) -> pl.DataFrame:
    """Compute numeric summary statistics per numeric column.

    Args:
        df: Input dataframe.
        numeric_cols: Numeric columns.

    Returns:
        Numeric summary table.
    """
    records: list[dict[str, Any]] = []
    for col in numeric_cols:
        s = df.get_column(col)
        non_null = s.drop_nulls()

        if non_null.len() == 0:
            records.append(
                {
                    "column": col,
                    "count_non_null": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "max": None,
                }
            )
            continue

        records.append(
            {
                "column": col,
                "count_non_null": int(non_null.len()),
                "mean": to_float_or_none(non_null.mean()),
                "std": (
                    to_float_or_none(non_null.std()) if non_null.len() > 1 else 0.0
                ),
                "min": to_float_or_none(non_null.min()),
                "p25": to_float_or_none(
                    non_null.quantile(0.25, interpolation="nearest")
                ),
                "p50": to_float_or_none(
                    non_null.quantile(0.50, interpolation="nearest")
                ),
                "p75": to_float_or_none(
                    non_null.quantile(0.75, interpolation="nearest")
                ),
                "max": to_float_or_none(non_null.max()),
            }
        )

    return pl.DataFrame(records).sort("column")


def build_granularity_and_gap_tables(
    df: pl.DataFrame, ctx: DatasetContext
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Check key granularity and date continuity by port.

    Args:
        df: Input dataframe.
        ctx: Inferred dataset context.

    Returns:
        Tuple of:
            1) key-level duplicate overview
            2) date-gap summary by port
    """
    if not ctx.port_col or not ctx.date_col:
        empty = pl.DataFrame({"info": ["No port/date key columns detected"]})
        return empty, empty

    keyed = (
        df.select([ctx.port_col, ctx.date_col])
        .with_columns(
            pl.col(ctx.date_col).cast(pl.Date, strict=False).alias(ctx.date_col)
        )
        .drop_nulls([ctx.port_col, ctx.date_col])
    )

    key_rows = keyed.height
    unique_key_rows = keyed.unique().height
    duplicate_key_rows = key_rows - unique_key_rows

    key_table = pl.DataFrame(
        [
            {
                "key": f"({ctx.port_col}, {ctx.date_col})",
                "rows_with_key": key_rows,
                "unique_key_rows": unique_key_rows,
                "duplicate_key_rows": duplicate_key_rows,
                "duplicate_key_rows_pct": round(
                    (duplicate_key_rows / key_rows * 100) if key_rows else 0.0, 2
                ),
            }
        ]
    )

    day_step = (
        keyed.unique()
        .sort([ctx.port_col, ctx.date_col])
        .with_columns(
            pl.col(ctx.date_col)
            .diff()
            .over(ctx.port_col)
            .dt.total_days()
            .alias("day_diff")
        )
        .with_columns((pl.col("day_diff") - 1).alias("missing_days_between_rows"))
    )

    gap_table = (
        day_step.filter(pl.col("missing_days_between_rows") > 0)
        .group_by(ctx.port_col)
        .agg(
            [
                pl.len().alias("n_gap_intervals"),
                pl.col("missing_days_between_rows").sum().alias("missing_days_total"),
            ]
        )
        .sort("missing_days_total", descending=True)
    )

    if gap_table.height == 0:
        gap_table = pl.DataFrame(
            [{ctx.port_col: "(none)", "n_gap_intervals": 0, "missing_days_total": 0}]
        )

    return key_table, gap_table


def build_target_coverage_table(
    df: pl.DataFrame, target_cols: list[str]
) -> pl.DataFrame:
    """Summarize coverage and range for target columns.

    Args:
        df: Input dataframe.
        target_cols: Target column names.

    Returns:
        Target diagnostics.
    """
    rows = df.height
    records: list[dict[str, Any]] = []

    for col in target_cols:
        s = df.get_column(col)
        non_null = s.drop_nulls()
        missing = int(s.null_count())
        records.append(
            {
                "target": col,
                "missing_count": missing,
                "pct_missing": round((missing / rows * 100) if rows else 0.0, 2),
                "min": to_float_or_none(non_null.min()) if non_null.len() else None,
                "p50": to_float_or_none(
                    non_null.quantile(0.50, interpolation="nearest")
                )
                if non_null.len()
                else None,
                "max": to_float_or_none(non_null.max()) if non_null.len() else None,
            }
        )

    return pl.DataFrame(records).sort("target")


def build_feature_target_corr_table(
    df: pl.DataFrame, feature_cols: list[str], target_cols: list[str], top_k: int
) -> pl.DataFrame:
    """Compute top absolute feature-target Pearson correlations.

    Args:
        df: Input dataframe.
        feature_cols: Predictor feature columns.
        target_cols: Target columns.
        top_k: Number of top rows to return.

    Returns:
        Top absolute correlations.
    """
    records: list[dict[str, str | float]] = []

    for target_col in target_cols:
        for feat_col in feature_cols:
            corr = df.select(pl.corr(feat_col, target_col)).item()
            if corr is None:
                continue
            records.append(
                {
                    "target": target_col,
                    "feature": feat_col,
                    "pearson_corr": float(corr),
                    "abs_corr": abs(float(corr)),
                }
            )

    if not records:
        return pl.DataFrame(
            {"target": [], "feature": [], "pearson_corr": [], "abs_corr": []}
        )

    return pl.DataFrame(records).sort("abs_corr", descending=True).head(top_k)


def build_leakage_warning_table(
    df: pl.DataFrame, target_cols: list[str]
) -> pl.DataFrame:
    """Flag columns with names that may indicate leakage risk.

    Args:
        df: Input dataframe.
        target_cols: Known target columns.

    Returns:
        Table of suspicious columns with reason text.
    """
    suspicious_patterns = re.compile(r"(lead|future|ahead|t\+|fwd|next)", re.IGNORECASE)
    records: list[dict[str, str]] = []

    for col in df.columns:
        if col in target_cols:
            continue
        if col.startswith("target_"):
            records.append(
                {
                    "column": col,
                    "reason": "Column starts with target_ but is not in declared targets.",
                }
            )
        elif suspicious_patterns.search(col):
            records.append(
                {
                    "column": col,
                    "reason": "Column name suggests future information. Verify construction timing.",
                }
            )

    return (
        pl.DataFrame(records) if records else pl.DataFrame({"column": [], "reason": []})
    )


def print_section(title: str, table: pl.DataFrame, top_k: int | None = None) -> None:
    """Print a titled section table.

    Args:
        title: Section heading.
        table: Table to print.
        top_k: Optional row limit for display only.
    """
    print(f"\n{title}")
    print("-" * len(title))
    if top_k is not None and table.height > top_k:
        print(table.head(top_k))
        print(f"... showing top {top_k} of {table.height} rows")
    else:
        print(table)


def main() -> None:
    """Run the pre-model dataset report."""
    args = parse_args()

    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(180)
    pl.Config.set_fmt_str_lengths(60)

    df = load_dataset(args.csv)
    ctx = infer_context(df)

    overview = build_overview_table(df, ctx)
    missingness = build_missingness_table(df)
    constant_cols = build_constant_columns_table(df)
    numeric_summary = build_numeric_summary_table(df, ctx.numeric_cols)
    key_table, gap_table = build_granularity_and_gap_tables(df, ctx)
    target_coverage = build_target_coverage_table(df, ctx.target_cols)
    corr_table = build_feature_target_corr_table(
        df, ctx.feature_cols, ctx.target_cols, args.top_k
    )
    leakage_warnings = build_leakage_warning_table(df, ctx.target_cols)

    high_missing = missingness.filter(pl.col("pct_missing") >= 20)

    print(f"\nPre-model report for: {args.csv}\n")
    print_section("Dataset overview", overview)
    print_section(
        "Per-column missingness and cardinality", missingness, top_k=args.top_k
    )
    print_section("High-missing columns (>=20%)", high_missing)
    print_section("Constant columns", constant_cols)
    print_section("Key granularity check", key_table)
    print_section("Date continuity gaps by port", gap_table, top_k=args.top_k)
    print_section("Target coverage summary", target_coverage)
    print_section("Top feature-target correlations", corr_table)
    print_section("Potential leakage-name warnings", leakage_warnings)
    print_section("Numeric summary", numeric_summary, top_k=args.top_k)

    print("\nQuick interpretation guide")
    print("-------------------------")
    print(
        "1. Review high-missing columns before modeling (drop, impute, or re-engineer)."
    )
    print(
        "2. Duplicate key rows indicate potential aggregation bugs and can bias models."
    )
    print("3. Date gaps by port may break rolling or lag assumptions if not handled.")
    print(
        "4. Leakage warnings are name-based only; verify feature construction timeline in pipeline."
    )
    print("5. Correlation is a quick filter, not causal evidence.")


if __name__ == "__main__":
    main()
