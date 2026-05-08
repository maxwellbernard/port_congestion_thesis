"""
Production runner for cleaning_pipeline — skips all inspection cells.
Run from the notebooks/ directory or via run_pipeline.py.
"""

import logging
import math
import os
from pathlib import Path

import polars as pl

os.environ.setdefault("MPLBACKEND", "Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("../ais_data/filtered_parquet")
MERGED_PATH = Path("../ais_data/merged/ais_raw.parquet")
OUTPUT_PATH = Path("../ais_data/cleaned/ais_cleaned.parquet")
GISIS_CACHE_PATH = Path("../ais_data/cleaned/gisis_cache.csv")

PORTS_TO_KEEP: list[str] = [
    "LOS_ANGELES_LONG_BEACH",
    "NEW_YORK_NEW_JERSEY",
    "HOUSTON",
    "PORT_OF_VIRGINIA",
]

SHIP_TYPE_GROUPS: dict[str, str] = {
    "Bulk Carrier": "Bulk Carrier",
    "Container Ship (Fully Cellular)": "Container Ship",
    "Container Ship (Fully Cellular/Refrigerated)": "Container Ship",
}

KEEP_COLUMNS: list[str] = [
    "mmsi", "imo", "timestamp_utc", "latitude", "longitude",
    "sog", "cog", "heading", "vessel_type", "nav_status", "port_name",
]

SOG_SUSPICIOUS: float = 30.0
SPEED_JUMP_MS: float = 15.5

COLUMN_RENAME: dict[str, str] = {
    "base_date_time": "timestamp_utc",
    "status": "nav_status",
}


def _normalize_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Lowercase column names, apply rename map, cast heading to Float64."""
    schema = lf.collect_schema()
    lowercase_map = {col: col.lower() for col in schema.names() if col != col.lower()}
    if lowercase_map:
        lf = lf.rename(lowercase_map)
        schema = lf.collect_schema()

    applicable = {k: v for k, v in COLUMN_RENAME.items() if k in schema.names()}
    if applicable:
        lf = lf.rename(applicable)
        schema = lf.collect_schema()

    if schema.get("heading") == pl.Int64:
        lf = lf.with_columns(pl.col("heading").cast(pl.Float64))

    return lf


if not MERGED_PATH.exists():
    logger.info("Stage 1: Merging yearly parquets...")
    frames: list[pl.LazyFrame] = []
    for year_dir in sorted(DATA_DIR.iterdir()):
        files = sorted(year_dir.glob("*.parquet"))
        if not files:
            continue
        lf = pl.scan_parquet(str(year_dir / "*.parquet"))
        lf = _normalize_schema(lf)
        frames.append(lf)
        logger.info(f"  Queued {year_dir.name}: {len(files)} files")
    MERGED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).sink_parquet(MERGED_PATH, compression="zstd")
    logger.info(f"Merged parquet written to {MERGED_PATH}")
else:
    logger.info(f"Merged parquet already exists at {MERGED_PATH}, skipping merge")

deg_to_rad = math.pi / 180.0

logger.info("Stage 2: Cleaning...")
lf = pl.scan_parquet(MERGED_PATH).select(KEEP_COLUMNS)
lf = lf.filter(pl.col("port_name").is_in(PORTS_TO_KEEP))

lf = lf.with_columns([
    pl.when(pl.col("heading") == 511.0).then(None).otherwise(pl.col("heading")).alias("heading"),
    pl.when(pl.col("cog") == 360.0).then(None).otherwise(pl.col("cog")).alias("cog"),
    pl.when(pl.col("sog") >= 102.2).then(None).otherwise(pl.col("sog")).alias("sog"),
])

lf = lf.filter(pl.col("sog").is_not_null() & (pl.col("sog") <= SOG_SUSPICIOUS))

lf = lf.unique(subset=["mmsi", "timestamp_utc"], keep="first")

lf = (
    lf.sort(["mmsi", "timestamp_utc"])
    .with_columns([
        pl.col("latitude").shift(1).over("mmsi").alias("_prev_lat"),
        pl.col("longitude").shift(1).over("mmsi").alias("_prev_lon"),
        pl.col("timestamp_utc").shift(1).over("mmsi").alias("_prev_ts"),
    ])
    .with_columns([
        (
            ((pl.col("latitude") - pl.col("_prev_lat")) * 111_320.0).pow(2)
            + (
                (pl.col("longitude") - pl.col("_prev_lon"))
                * 111_320.0
                * (pl.col("latitude") * deg_to_rad).cos()
            ).pow(2)
        ).sqrt().alias("_dist_m"),
        (pl.col("timestamp_utc") - pl.col("_prev_ts")).dt.total_seconds().alias("_time_diff_s"),
    ])
    .with_columns(
        pl.when(pl.col("_time_diff_s") > 0)
        .then(pl.col("_dist_m") / pl.col("_time_diff_s") > SPEED_JUMP_MS)
        .otherwise(False)
        .alias("flag_speed_jump")
    )
    .drop(["_prev_lat", "_prev_lon", "_prev_ts", "_dist_m", "_time_diff_s"])
    .filter(~pl.col("flag_speed_jump"))
    .drop("flag_speed_jump")
)

logger.info("Stage 3: Building MMSI→IMO mapping...")
lf_for_mapping = pl.scan_parquet(MERGED_PATH).select(["mmsi", "imo"])
mmsi_imo_map = (
    lf_for_mapping
    .filter(pl.col("imo").is_not_null())
    .group_by(["mmsi", "imo"])
    .agg(pl.len().alias("imo_broadcast_count"))
    .with_columns(
        pl.col("imo_broadcast_count")
        .sum()
        .over("mmsi")
        .alias("imo_total_non_null")
    )
    .with_columns(
        (pl.col("imo_broadcast_count") / pl.col("imo_total_non_null")).alias("imo_confidence")
    )
    .sort(["mmsi", "imo_broadcast_count"], descending=[False, True])
    .group_by("mmsi")
    .first()
    .filter(pl.col("imo_confidence") >= 0.9)
    .collect()
)
logger.info(f"  High-confidence MMSI→IMO pairs: {mmsi_imo_map.height:,}")

logger.info("Stage 4: GISIS enrichment...")
if not GISIS_CACHE_PATH.exists():
    raise FileNotFoundError(f"GISIS cache not found: {GISIS_CACHE_PATH}")

gisis = (
    pl.read_csv(GISIS_CACHE_PATH)
    .filter(pl.col("found"))
    .select([
        pl.col("imo_number").cast(pl.String).alias("imo_join"),
        pl.col("ship_type_detailed").cast(pl.String).str.strip_chars().replace("", None).alias("ship_type_detailed"),
        pl.col("gross_tonnage").cast(pl.String).str.replace_all(",", "").replace("", None).cast(pl.Float64, strict=False).alias("gross_tonnage"),
    ])
)

vessel_lookup = (
    mmsi_imo_map.select(["mmsi", "imo"])
    .with_columns(pl.col("imo").str.replace("^IMO", "").alias("imo_join"))
    .join(gisis, on="imo_join", how="left")
    .drop("imo_join")
    .with_columns(
        pl.col("ship_type_detailed")
        .replace_strict(SHIP_TYPE_GROUPS, default="Other Cargo")
        .alias("ship_type_group")
    )
)
logger.info(f"  Vessel lookup: {vessel_lookup.height:,} vessels")

logger.info("Stage 5: Writing cleaned parquet...")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
lf.sink_parquet(OUTPUT_PATH, compression="zstd")
logger.info(f"  Cleaned parquet written: {OUTPUT_PATH}")

logger.info("Stage 6: Enriching with ship type...")
join_cols = vessel_lookup.select(["mmsi", "ship_type_detailed", "gross_tonnage", "ship_type_group"])
lf_enriched = pl.scan_parquet(OUTPUT_PATH).join(join_cols.lazy(), on="mmsi", how="inner")
tmp_path = OUTPUT_PATH.with_suffix(".tmp.parquet")
lf_enriched.sink_parquet(tmp_path, compression="zstd")
tmp_path.replace(OUTPUT_PATH)

rows = pl.scan_parquet(OUTPUT_PATH).select(pl.len()).collect().item()
logger.info(f"Cleaned pipeline complete — {rows:,} rows → {OUTPUT_PATH}")
