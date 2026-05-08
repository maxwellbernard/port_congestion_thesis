
import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


INPUT_PATH = Path("../ais_data/cleaned/ais_cleaned.parquet")
OUTPUT_PATH = Path("../ais_data/standardized/ais_standardized.parquet")

SEGMENT_GAP_SECONDS: int = 2 * 60 * 60

MIN_SEGMENT_PINGS: int = 60


lf = pl.scan_parquet(INPUT_PATH)

print("Schema:")
for col, dtype in lf.collect_schema().items():
    print(f"  {col}: {dtype}")

total_rows = lf.select(pl.len()).collect().item()
print(f"\nTotal rows: {total_rows:,}")


def create_segments(lf: pl.LazyFrame, gap_seconds: int = SEGMENT_GAP_SECONDS) -> pl.LazyFrame:
    """
    Assign a segment_id to each AIS ping.

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

lf = create_segments(lf)

seg_stats = (
    lf.group_by(["mmsi", "port_name", "segment_id"])
      .agg(
          pl.len().alias("pings"),
          pl.col("timestamp_utc").min().alias("seg_start"),
          pl.col("timestamp_utc").max().alias("seg_end"),
      )
      .with_columns(
          ((pl.col("seg_end") - pl.col("seg_start")).dt.total_seconds() / 3600)
          .alias("duration_hours")
      )
      .collect()
)

n_segments = seg_stats.height
n_vessels = seg_stats["mmsi"].n_unique()

print(f"Total segments:       {n_segments:,}")
print(f"Unique vessels (MMSI): {n_vessels:,}")
print("\nPings per segment:")
print(seg_stats["pings"].describe())
print("\nSegment duration (hours):")
print(seg_stats["duration_hours"].describe())

print("Segments per port:")
print(
    seg_stats.group_by("port_name")
             .agg(pl.len().alias("segments"))
             .sort("segments", descending=True)
)

rows_after = lf.select(pl.len()).collect().item()
print(f"Rows in:  {total_rows:,}")
print(f"Rows out: {rows_after:,}")
print(f"Difference: {total_rows - rows_after:,}  (should be 0)")

for threshold in [2, 10, 30, 60, 120]:
    n = seg_stats.filter(pl.col("pings") < threshold).height
    pct = n / n_segments * 100
    print(f"  Segments with < {threshold:>4} pings: {n:>6,}  ({pct:5.1f}%)")

multi_seg_mmsi = (
    seg_stats.group_by("mmsi")
             .agg(pl.len().alias("n_segments"))
             .filter(pl.col("n_segments") >= 3)
             .sort("n_segments", descending=False)
             .head(1)["mmsi"]
             .item()
)
vessel_track = (
    lf.filter(pl.col("mmsi") == multi_seg_mmsi)
      .select(["mmsi", "timestamp_utc", "sog", "port_name", "segment_id"])
      .sort("timestamp_utc")
      .collect()
)
boundaries = []
segment_ids = vessel_track["segment_id"].unique().sort().to_list()
for seg in segment_ids:
    seg_rows = vessel_track.filter(pl.col("segment_id") == seg)
    boundaries.append(seg_rows.tail(1))
    next_rows = vessel_track.filter(pl.col("segment_id") == seg + 1)
    if next_rows.height > 0:
        boundaries.append(next_rows.head(1))

print(f"Vessel MMSI {multi_seg_mmsi} — segment boundaries (last/first ping per boundary):")
print(pl.concat(boundaries))


valid_segments = (
    lf.group_by("segment_id")
      .agg(pl.len().alias("pings"))
      .filter(pl.col("pings") >= MIN_SEGMENT_PINGS)
      .select("segment_id")
)
lf = lf.join(valid_segments, on="segment_id", how="inner")

rows_after_filter = lf.select(pl.len()).collect().item()
segs_after_filter = lf.select(pl.col("segment_id").n_unique()).collect().item()
print(f"Rows before filter:    {rows_after:,}")
print(f"Rows after filter:     {rows_after_filter:,}")
print(f"Rows dropped:          {rows_after - rows_after_filter:,}  ({(rows_after - rows_after_filter) / rows_after * 100:.1f}%)")
print(f"Segments before:       {n_segments:,}")
print(f"Segments after:        {segs_after_filter:,}")
print(f"Segments dropped:      {n_segments - segs_after_filter:,}  ({(n_segments - segs_after_filter) / n_segments * 100:.1f}%)")

lf.head(10).collect()


def run_standardization_pipeline(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    gap_seconds: int = SEGMENT_GAP_SECONDS,
    min_pings: int = MIN_SEGMENT_PINGS,
) -> None:
    """
    Execute the full standardization pipeline and write output parquet.

    Args:
        input_path: Path to the cleaned parquet (output of cleaning pipeline).
        output_path: Destination path for the standardized parquet.
        gap_seconds: Time gap in seconds that triggers a new segment.
        min_pings: Minimum pings a segment must have to be retained.
    """
    logger.info("Starting standardization pipeline")

    lf_prod = pl.scan_parquet(input_path)
    lf_prod = create_segments(lf_prod, gap_seconds)

    valid_segments = (
        lf_prod.group_by("segment_id")
               .agg(pl.len().alias("pings"))
               .filter(pl.col("pings") >= min_pings)
               .select("segment_id")
    )
    lf_prod = lf_prod.join(valid_segments, on="segment_id", how="inner")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lf_prod.sink_parquet(output_path, compression="zstd")

    logger.info(f"Standardized data written to {output_path}")


run_standardization_pipeline()
