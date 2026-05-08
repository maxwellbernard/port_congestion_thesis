
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
MACRO_DIR = REPO_ROOT / "raw_macro_data"
FEATURES_DIR = REPO_ROOT / "ais_data" / "features"
OUTPUT_DIR = REPO_ROOT / "final_dataframes"

OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_FILES: dict[str, str] = {
    "bulk_carrier": "daily_features_bulk_carrier.parquet",
    "container": "daily_features_container.parquet",
}

PUBLICATION_LAG_DAYS: dict[str, int] = {
    "macro_gscpi": 45,
    "macro_bdi_weekly": 2,
    "macro_ism_pmi": 1,
    "macro_us_imports_goods": 35,
    "macro_us_exports_goods": 35,
    "macro_us_trade_balance": 35,
}


def _parse_eu_decimal(col: str) -> pl.Expr:
    """Parse a string column that uses comma as decimal separator."""
    return pl.col(col).str.replace(",", ".").cast(pl.Float64)


def _parse_date_mdy(col: str) -> pl.Expr:
    """Parse MM/DD/YYYY date strings."""
    return pl.col(col).str.to_date("%m/%d/%Y")


def _parse_date_dmy(col: str) -> pl.Expr:
    """Parse DD.MM.YYYY date strings."""
    return pl.col(col).str.to_date("%d.%m.%Y")


def _apply_publication_lag(df: pl.DataFrame, lag_days: int) -> pl.DataFrame:
    """Shift the date column forward by the publication lag.

    Args:
        df: DataFrame with a `date` column.
        lag_days: Number of days to add (publication delay).

    Returns:
        DataFrame with `date` shifted forward.
    """
    return df.with_columns(
        (pl.col("date") + pl.duration(days=lag_days)).cast(pl.Date).alias("date")
    )


def load_gscpi() -> pl.DataFrame:
    """Load Global Supply Chain Pressure Index (monthly).

    Returns:
        DataFrame with columns [date, macro_gscpi].
    """
    df = (
        pl.read_csv(
            MACRO_DIR / "Global Supply Chain Pressure Index monthly_Bloomberg.csv",
            separator=";",
            schema_overrides={"Index": pl.Utf8},
        )
        .filter(pl.col("Date").is_not_null() & (pl.col("Date") != ""))
        .with_columns(
            _parse_date_mdy("Date").alias("date"),
            _parse_eu_decimal("Index").alias("macro_gscpi"),
        )
        .select("date", "macro_gscpi")
        .sort("date")
    )
    return _apply_publication_lag(df, PUBLICATION_LAG_DAYS["macro_gscpi"])


def load_bdi_weekly() -> pl.DataFrame:
    """Load Baltic Dry Index (weekly).

    Returns:
        DataFrame with columns [date, macro_bdi_weekly].
    """
    df = (
        pl.read_csv(
            MACRO_DIR / "Baltic Dry Index Weekly_Bloomberg.csv",
            separator=";",
            schema_overrides={"Index": pl.Utf8},
        )
        .filter(pl.col("Date").is_not_null() & (pl.col("Date") != ""))
        .with_columns(
            _parse_date_mdy("Date").alias("date"),
            pl.col("Index").cast(pl.Float64).alias("macro_bdi_weekly"),
        )
        .select("date", "macro_bdi_weekly")
        .sort("date")
    )
    return _apply_publication_lag(df, PUBLICATION_LAG_DAYS["macro_bdi_weekly"])


def load_ism_pmi() -> pl.DataFrame:
    """Load ISM Manufacturing PMI (monthly).

    Returns:
        DataFrame with columns [date, macro_ism_pmi].
    """
    df = (
        pl.read_csv(
            MACRO_DIR / "ISM Manufacturing PMI monthly_ Bloomberg.csv",
            separator=";",
            schema_overrides={"Index": pl.Utf8},
            truncate_ragged_lines=True,
        )
        .filter(pl.col("Date").is_not_null() & (pl.col("Date") != ""))
        .with_columns(
            _parse_date_dmy("Date").alias("date"),
            _parse_eu_decimal("Index").alias("macro_ism_pmi"),
        )
        .select("date", "macro_ism_pmi")
        .sort("date")
    )
    return _apply_publication_lag(df, PUBLICATION_LAG_DAYS["macro_ism_pmi"])


def _load_census_trade_csv(filename: str, value_col: str, macro_col: str) -> pl.DataFrame:
    """Load a US Census Bureau trade CSV (monthly, millions USD).

    All three Census trade files share the same format: observation_date + one
    numeric value column.

    Args:
        filename: CSV filename in MACRO_DIR.
        value_col: Name of the value column in the CSV (e.g. "BOPGIMP").
        macro_col: Output column name (e.g. "macro_us_imports_goods").

    Returns:
        DataFrame with columns [date, macro_col], publication lag applied.
    """
    df = (
        pl.read_csv(MACRO_DIR / filename)
        .with_columns(
            pl.col("observation_date").str.to_date("%Y-%m-%d").alias("date"),
            pl.col(value_col).cast(pl.Float64).alias(macro_col),
        )
        .select("date", macro_col)
        .sort("date")
    )
    return _apply_publication_lag(df, PUBLICATION_LAG_DAYS[macro_col])


def load_us_imports_goods() -> pl.DataFrame:
    """Load US Imports of Goods (monthly, millions USD, SA).

    Returns:
        DataFrame with columns [date, macro_us_imports_goods].
    """
    return _load_census_trade_csv(
        "US Imports of Goods monthly BOPGIMP_U.S. Census Bureau.csv",
        "BOPGIMP",
        "macro_us_imports_goods",
    )


def load_us_exports_goods() -> pl.DataFrame:
    """Load US Exports of Goods (monthly, millions USD, SA).

    Returns:
        DataFrame with columns [date, macro_us_exports_goods].
    """
    return _load_census_trade_csv(
        "US Exports of Goods monthly BOPGEXP_U.S. Census Bureau.csv",
        "BOPGEXP",
        "macro_us_exports_goods",
    )


def load_us_trade_balance() -> pl.DataFrame:
    """Load US Trade Balance of Goods (monthly, millions USD, SA).

    Returns:
        DataFrame with columns [date, macro_us_trade_balance].
    """
    return _load_census_trade_csv(
        "US Trade Balance of Goods monthly BOPGTB_U.S. Census Bureau.csv",
        "BOPGTB",
        "macro_us_trade_balance",
    )


def load_all_macro() -> list[pl.DataFrame]:
    """Load all macro indicator DataFrames with publication lag applied.

    Returns:
        List of DataFrames, each with a lag-adjusted `date` column
        and one or more macro value columns.
    """
    loaders = [
        load_gscpi,
        load_bdi_weekly,
        load_ism_pmi,
        load_us_imports_goods,
        load_us_exports_goods,
        load_us_trade_balance,
    ]
    dfs: list[pl.DataFrame] = []
    for loader in loaders:
        df = loader()
        macro_col = [c for c in df.columns if c != "date"][0]
        lag = PUBLICATION_LAG_DAYS[macro_col]
        print(
            f"  Loaded {loader.__name__}: {df.shape[0]} rows, "
            f"lag-adjusted date range {df['date'].min()} → {df['date'].max()} "
            f"(+{lag}d pub lag)"
        )
        dfs.append(df)
    return dfs


def join_macro_to_features(
    features: pl.DataFrame, macro_dfs: list[pl.DataFrame]
) -> pl.DataFrame:
    """Join all macro indicators to AIS features using asof join.

    Each macro DataFrame is joined on `ref_date` >= macro `date` (backward),
    so only the most recent lag-adjusted macro value is used — no lookahead.

    After joining, ISM PMI nulls at the start of the series are backfilled
    with the earliest available value.

    Args:
        features: AIS daily features with `ref_date` column (Date type).
        macro_dfs: List of macro DataFrames, each with `date` column.

    Returns:
        Features DataFrame with macro columns appended.
    """
    result = features.sort("ref_date")

    for macro_df in macro_dfs:
        macro_cols = [c for c in macro_df.columns if c != "date"]
        result = result.join_asof(
            macro_df.sort("date"),
            left_on="ref_date",
            right_on="date",
            strategy="backward",
        )
        if "date" in result.columns:
            result = result.drop("date")

        n_nulls = result.select(pl.col(macro_cols[0]).is_null().sum()).item()
        print(
            f"  Joined {macro_cols}: {n_nulls} null rows "
            f"(pre-series start, will backfill if applicable)"
        )

    if "macro_ism_pmi" in result.columns:
        n_before = result.select(pl.col("macro_ism_pmi").is_null().sum()).item()
        result = result.with_columns(
            pl.col("macro_ism_pmi")
            .fill_null(strategy="forward")
            .over("port_name")
            .alias("macro_ism_pmi")
        )
        result = result.with_columns(
            pl.col("macro_ism_pmi")
            .fill_null(strategy="backward")
            .over("port_name")
            .alias("macro_ism_pmi")
        )
        n_after = result.select(pl.col("macro_ism_pmi").is_null().sum()).item()
        print(
            f"  Backfilled macro_ism_pmi: {n_before} → {n_after} nulls "
            f"(earliest value carried backward)"
        )

    return result


def main() -> None:
    """Run the full macro join pipeline and export to Excel."""
    print("Loading macro indicators...")
    macro_dfs = load_all_macro()

    for label, filename in FEATURE_FILES.items():
        print(f"\nProcessing {label}...")
        features = pl.read_parquet(FEATURES_DIR / filename)

        if features["ref_date"].dtype == pl.Utf8:
            features = features.with_columns(
                pl.col("ref_date").str.to_date("%Y-%m-%d")
            )
        elif features["ref_date"].dtype == pl.Datetime:
            features = features.with_columns(pl.col("ref_date").cast(pl.Date))

        joined = join_macro_to_features(features, macro_dfs)

        xlsx_path = OUTPUT_DIR / f"{label}.xlsx"
        csv_path = OUTPUT_DIR / f"{label}.csv"
        joined.write_excel(xlsx_path)
        joined.write_csv(csv_path)
        print(
            f"  Written {xlsx_path} and {csv_path} — "
            f"{joined.shape[0]} rows x {joined.shape[1]} cols"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
