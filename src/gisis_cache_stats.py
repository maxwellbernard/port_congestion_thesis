"""Print quick summary tables for a GISIS cache CSV.

Usage:
    python src/gisis_cache_stats.py
    python src/gisis_cache_stats.py --csv ais_data/cleaned/gisis_cache.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Show basic stats for a GISIS cache CSV in table format."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("ais_data/cleaned/gisis_cache.csv"),
        help="Path to gisis_cache.csv",
    )
    return parser.parse_args()


def load_gisis_cache(csv_path: Path) -> pl.DataFrame:
    """Load the GISIS cache CSV and normalize key fields.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A Polars DataFrame with normalized columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pl.read_csv(csv_path)
    return df.with_columns(
        [
            pl.col("ship_type_detailed")
            .cast(pl.String)
            .str.strip_chars()
            .replace("", None)
            .fill_null("(missing)")
            .alias("ship_type_detailed"),
            pl.col("gross_tonnage")
            .cast(pl.String)
            .str.replace_all(",", "")
            .replace("", None)
            .cast(pl.Float64, strict=False)
            .alias("gross_tonnage_num"),
            pl.col("imo_number")
            .cast(pl.String)
            .str.strip_chars()
            .replace("", None)
            .alias("imo_number"),
        ]
    )


def build_found_table(df: pl.DataFrame) -> pl.DataFrame:
    """Build percentage breakdown of the `found` column.

    Args:
        df: Input GISIS cache dataframe.

    Returns:
        Table with found value, count, and percentage of rows.
    """
    total_rows = df.height

    return (
        df.group_by("found")
        .agg(pl.len().alias("rows"))
        .with_columns((pl.col("rows") / total_rows * 100).round(2).alias("pct_rows"))
        .sort("found")
    )


def build_ship_type_table(df: pl.DataFrame) -> pl.DataFrame:
    """Build ship-type summary with IMO counts, share, and avg gross tonnage.

    Args:
        df: Input GISIS cache dataframe.

    Returns:
        Table with one row per ship_type_detailed.
    """
    total_unique_imo = df.select(
        pl.col("imo_number").drop_nulls().n_unique().alias("n")
    ).item()

    return (
        df.group_by("ship_type_detailed")
        .agg(
            [
                pl.col("imo_number").drop_nulls().n_unique().alias("unique_imo_count"),
                pl.col("gross_tonnage_num").mean().round(2).alias("avg_gross_tonnage"),
            ]
        )
        .with_columns(
            (
                pl.when(total_unique_imo > 0)
                .then(pl.col("unique_imo_count") / total_unique_imo * 100)
                .otherwise(0.0)
                .round(2)
            ).alias("pct_of_total_unique_imo")
        )
        .sort("unique_imo_count", descending=True)
    )


def main() -> None:
    """Run the GISIS cache summary script."""
    args = parse_args()
    df = load_gisis_cache(args.csv)

    found_table = build_found_table(df)
    ship_type_table = build_ship_type_table(df)

    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(140)

    print(f"\nGISIS cache stats for: {args.csv}\n")
    print("Found breakdown (% of rows):")
    print(found_table)
    print("\nShip type breakdown:")
    print(ship_type_table)


if __name__ == "__main__":
    main()
