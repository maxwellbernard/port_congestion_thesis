"""
Production runner for standardization_pipeline — calls run_standardization_pipeline()
without executing the interactive inspection cells.
Run from the notebooks/ directory.
"""

import logging
import os
from pathlib import Path

import polars as pl

os.environ.setdefault("MPLBACKEND", "Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_PATH = Path("../ais_data/cleaned/ais_cleaned.parquet")
OUTPUT_PATH = Path("../ais_data/standardized/ais_standardized.parquet")
SEGMENT_GAP_SECONDS: int = 2 * 60 * 60
MIN_SEGMENT_PINGS: int = 60


def create_segments(lf: pl.LazyFrame, gap_seconds: int = SEGMENT_GAP_SECONDS) -> pl.LazyFrame:
    """Assign a segment_id to each AIS ping.

    A segment is a continuous observation sequence for one vessel at one port.
    A new segment begins when MMSI changes, port_name changes, or the time gap
    between consecutive pings exceeds gap_seconds.

    Segment IDs are globally unique integers (cumulative sum of reset flags).

    Args:
        lf: Cleaned LazyFrame with mmsi, timestamp_utc, port_name columns.
        gap_seconds: Time gap in seconds that triggers a new segment.

    Returns:
        LazyFrame with segment_id column added.
    """
    return (
        lf
        .sort(["port_name", "mmsi", "timestamp_utc"])
        .with_columns(
            pl.col("timestamp_utc").shift(1).over(["port_name", "mmsi"]).alias("_prev_ts"),
        )
        .with_columns(
            (
                pl.col("_prev_ts").is_null()
                | (pl.col("timestamp_utc") - pl.col("_prev_ts"))
                    .dt.total_seconds()
                    .gt(gap_seconds)
            ).cast(pl.Int32).alias("_segment_reset")
        )
        .with_columns(
            pl.col("_segment_reset").cum_sum().alias("segment_id")
        )
        .drop(["_prev_ts", "_segment_reset"])
    )


logger.info("Starting standardization pipeline")
logger.info(f"  Input:  {INPUT_PATH}")
logger.info(f"  Output: {OUTPUT_PATH}")

lf_prod = pl.scan_parquet(INPUT_PATH)
lf_prod = create_segments(lf_prod, SEGMENT_GAP_SECONDS)

valid_segments = (
    lf_prod.group_by("segment_id")
           .agg(pl.len().alias("pings"))
           .filter(pl.col("pings") >= MIN_SEGMENT_PINGS)
           .select("segment_id")
)
lf_prod = lf_prod.join(valid_segments, on="segment_id", how="inner")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
lf_prod.sink_parquet(OUTPUT_PATH, compression="zstd")

rows = pl.scan_parquet(OUTPUT_PATH).select(pl.len()).collect().item()
logger.info(f"Standardized pipeline complete — {rows:,} rows → {OUTPUT_PATH}")
