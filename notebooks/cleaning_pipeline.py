
import logging
import math
from pathlib import Path

import polars as pl

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

HEADING_PLACEHOLDER: float = 511.0
COG_NOT_AVAILABLE: float = 360.0
SOG_NOT_AVAILABLE: float = 102.2

IMO_CONFIDENCE_THRESHOLD: float = 0.90

SOG_SUSPICIOUS: float = 30.0
SPEED_JUMP_MS: float = 15.5

KEEP_COLUMNS: list[str] = [
    "mmsi",
    "imo",
    "timestamp_utc",
    "latitude",
    "longitude",
    "sog",
    "cog",
    "heading",
    "vessel_type",
    "nav_status",
    "port_name",
]

COLUMN_RENAME: dict[str, str] = {
    "base_date_time": "timestamp_utc",
    "status": "nav_status",
}


def normalize_schema(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Normalise raw parquet column names to canonical project names.

    Lowercases all column names first (fixes MMSI→mmsi, SOG→sog, etc.),
    then applies the canonical rename map, then casts heading to Float64
    if needed (2025 files store it as Int64).

    Args:
        lf: Raw LazyFrame from scan_parquet.

    Returns:
        LazyFrame with canonical column names and consistent dtypes.
    """
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


def merge_yearly_parquets(data_dir: Path, output_path: Path) -> None:
    """
    Concatenate all yearly parquet files into one raw merged parquet.

    Each year directory is scanned separately so schema differences are
    resolved before concatenation. Output uses canonical column names.

    Args:
        data_dir: Root directory containing year subdirectories.
        output_path: Destination path for the merged raw parquet.

    Raises:
        FileNotFoundError: If data_dir does not exist.
        ValueError: If no parquet files are found under data_dir.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    frames: list[pl.LazyFrame] = []
    for year_dir in sorted(data_dir.iterdir()):
        parquet_files = sorted(year_dir.glob("*.parquet"))
        print(f"Found {len(parquet_files)} parquet files in {year_dir}")
        if not parquet_files:
            logger.warning(f"No parquet files in {year_dir}, skipping")
            continue
        lf = pl.scan_parquet(str(year_dir / "*.parquet"))
        lf = normalize_schema(lf)
        frames.append(lf)
        logger.info(f"Queued {year_dir.name}: {len(parquet_files)} files")

    if not frames:
        raise ValueError(f"No parquet files found under {data_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).sink_parquet(output_path, compression="zstd")
    logger.info(f"Merged parquet written to {output_path}")


if not MERGED_PATH.exists():
    merge_yearly_parquets(DATA_DIR, MERGED_PATH)
else:
    logger.info(f"Merged parquet already exists at {MERGED_PATH}, skipping merge")


lf = pl.scan_parquet(MERGED_PATH).select(KEEP_COLUMNS)
lf = lf.filter(pl.col("port_name").is_in(PORTS_TO_KEEP))

print("Schema:")
for col, dtype in lf.collect_schema().items():
    print(f"  {col}: {dtype}")

total_rows = lf.select(pl.len()).collect().item()
print(f"Total rows: {total_rows:,}")

print("\nRows per year:")
print(
    lf.with_columns(pl.col("timestamp_utc").dt.year().alias("year"))
    .group_by("year")
    .agg(pl.len().alias("rows"))
    .sort("year")
    .collect()
)

print("\nRows per port:")
print(
    lf.group_by("port_name")
    .agg(pl.len().alias("rows"))
    .sort("rows", descending=True)
    .collect()
)

NAV_STATUS_LABELS: dict[int, str] = {
    0: "Underway engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted manoeuvrability",
    4: "Constrained by draught",
    5: "Moored",
    6: "Aground",
    7: "Fishing",
    8: "Underway sailing",
    15: "Unknown/default",
}
nav_status_counts = (
    lf.group_by("nav_status")
    .agg(pl.len().alias("count"))
    .with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("pct"))
    .sort("count", descending=True)
    .collect()
)
print(f"\nnav_status distribution (total rows: {total_rows:,}):")
for row in nav_status_counts.iter_rows(named=True):
    status_code = row["nav_status"]
    code_display = "NA" if status_code is None else str(int(status_code))
    label = (
        "Missing"
        if status_code is None
        else NAV_STATUS_LABELS.get(status_code, f"code {code_display}")
    )
    print(f"  {code_display:>3}  {label:<32}  {row['count']:>12,}  ({row['pct']:.1f}%)")

sog_stopped_rows = lf.filter(pl.col("sog") < 0.5)
sog_stopped_total = sog_stopped_rows.select(pl.len()).collect().item()
nav_status_stopped = (
    sog_stopped_rows.group_by("nav_status")
    .agg(pl.len().alias("count"))
    .with_columns((pl.col("count") / pl.col("count").sum() * 100).alias("pct"))
    .sort("count", descending=True)
    .collect()
)
print(
    f"\nnav_status distribution among SOG < 0.5 kn rows ({sog_stopped_total:,} rows):"
)
for row in nav_status_stopped.iter_rows(named=True):
    status_code = row["nav_status"]
    code_display = "NA" if status_code is None else str(int(status_code))
    label = (
        "Missing"
        if status_code is None
        else NAV_STATUS_LABELS.get(status_code, f"code {code_display}")
    )
    print(f"  {code_display:>3}  {label:<32}  {row['count']:>12,}  ({row['pct']:.1f}%)")

unique_counts = lf.select(
    pl.col("mmsi").n_unique().alias("unique_mmsi"),
    pl.col("imo").n_unique().alias("unique_imo"),
    pl.col("imo").is_null().sum().alias("null_imo"),
).collect()
null_mmsi = lf.select(pl.col("mmsi").is_null().sum()).collect().item()

print(f"\nUnique vessel identifiers (total rows: {total_rows:,}):")
print(
    f"  Unique MMSI: {unique_counts['unique_mmsi'][0]:,}  "
    f"({null_mmsi:,} null, {null_mmsi / total_rows * 100:.1f}% missing)"
)
print(
    f"  Unique IMO:  {unique_counts['unique_imo'][0]:,}  "
    f"({unique_counts['null_imo'][0]:,} null, "
    f"{unique_counts['null_imo'][0] / total_rows * 100:.1f}% missing)"
)


def build_mmsi_imo_mapping(
    lf: pl.LazyFrame,
    confidence_threshold: float = IMO_CONFIDENCE_THRESHOLD,
) -> pl.DataFrame:
    """Build a one-row-per-MMSI lookup table mapping MMSI → best IMO.

    For each MMSI, selects the most frequently broadcast non-null IMO.
    Flags mappings where the mode IMO accounts for less than
    `confidence_threshold` of that MMSI's non-null IMO broadcasts,
    indicating potential MMSI reassignment across vessels.

    Args:
        lf: LazyFrame with mmsi and imo columns.
        confidence_threshold: Minimum fraction of non-null IMO broadcasts
            that must agree on the mode IMO (default 0.90).

    Returns:
        DataFrame with columns: mmsi, imo, imo_broadcast_count,
        imo_total_non_null, imo_confidence, imo_confident.
    """
    pair_counts = (
        lf.filter(pl.col("imo").is_not_null())
        .group_by(["mmsi", "imo"])
        .agg(pl.len().alias("imo_broadcast_count"))
        .collect()
    )

    if pair_counts.height == 0:
        return pl.DataFrame(
            schema={
                "mmsi": pl.Int64,
                "imo": pl.Utf8,
                "imo_broadcast_count": pl.UInt32,
                "imo_total_non_null": pl.UInt32,
                "imo_confidence": pl.Float64,
                "imo_confident": pl.Boolean,
            }
        )

    mapping = (
        pair_counts.with_columns(
            pl.col("imo_broadcast_count")
            .sum()
            .over("mmsi")
            .alias("imo_total_non_null"),
        )
        .with_columns(
            (pl.col("imo_broadcast_count") / pl.col("imo_total_non_null")).alias(
                "imo_confidence"
            ),
        )
        .sort(["mmsi", "imo_broadcast_count"], descending=[False, True])
        .group_by("mmsi")
        .first()
        .with_columns(
            (pl.col("imo_confidence") >= confidence_threshold).alias("imo_confident"),
        )
        .sort("mmsi")
    )
    return mapping


mmsi_imo_map = build_mmsi_imo_mapping(lf)
n_total_mmsi = unique_counts["unique_mmsi"][0]
n_with_imo = mmsi_imo_map.height
n_no_imo = n_total_mmsi - n_with_imo
n_confident = mmsi_imo_map.filter(pl.col("imo_confident")).height
n_low_conf = n_with_imo - n_confident

print(f"\nMMSI → IMO mapping (confidence threshold: {IMO_CONFIDENCE_THRESHOLD:.0%}):")
print(f"  Total unique MMSI:      {n_total_mmsi:>6,}")
print(f"  MMSI with IMO match:    {n_with_imo:>6,}  ({n_with_imo / n_total_mmsi * 100:.1f}%)")
print(f"  MMSI without any IMO:   {n_no_imo:>6,}  ({n_no_imo / n_total_mmsi * 100:.1f}%)")
print(f"  High-confidence (≥90%): {n_confident:>6,}  ({n_confident / n_total_mmsi * 100:.1f}%)")
print(f"  Low-confidence  (<90%): {n_low_conf:>6,}  ({n_low_conf / n_total_mmsi * 100:.1f}%)")

if n_low_conf > 0:
    print(f"\nLow-confidence mappings (potential MMSI reassignment):")
    low_conf = (
        mmsi_imo_map.filter(~pl.col("imo_confident"))
        .sort("imo_confidence")
        .select("mmsi", "imo", "imo_broadcast_count", "imo_total_non_null", "imo_confidence")
    )
    print(low_conf)

print(
    f"\n→ Keeping {n_confident:,} high-confidence mappings, "
    f"excluding {n_low_conf:,} low-confidence + {n_no_imo:,} no-IMO"
)

null_counts = lf.select(
    [pl.col(c).is_null().sum().alias(c) for c in lf.collect_schema().names()]
).collect()
print(f"Null counts (raw) — total rows: {total_rows:,}\n")
for col, count in zip(null_counts.columns, null_counts.row(0)):
    pct = count / total_rows * 100
    print(f"  {col:<22} {count:>10,}  ({pct:5.2f}%)")

placeholder_counts = lf.select(
    (pl.col("heading") == HEADING_PLACEHOLDER).sum().alias("heading == 511"),
    (pl.col("cog") == COG_NOT_AVAILABLE).sum().alias("cog == 360.0"),
    (pl.col("sog") >= SOG_NOT_AVAILABLE).sum().alias("sog >= 102.2"),
).collect()
print(f"AIS placeholder counts — total rows: {total_rows:,}\n")
for col, count in zip(placeholder_counts.columns, placeholder_counts.row(0)):
    pct = count / total_rows * 100
    print(f"  {col:<22} {count:>10,}  ({pct:5.2f}%)")


def clean_placeholder_values(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Replace known AIS "not available" sentinel values with null.

    Sentinels handled per AIS specification:
    - heading == 511        → not available
    - cog == 360.0          → not available
    - sog >= 102.2          → not available / capped
    Args:
        lf: Normalised LazyFrame.

    Returns:
        LazyFrame with sentinel values replaced by null.
    """
    return lf.with_columns(
        pl.when(pl.col("heading") == HEADING_PLACEHOLDER)
        .then(None)
        .otherwise(pl.col("heading"))
        .alias("heading"),
        pl.when(pl.col("cog") == COG_NOT_AVAILABLE)
        .then(None)
        .otherwise(pl.col("cog"))
        .alias("cog"),
        pl.when(pl.col("sog") >= SOG_NOT_AVAILABLE)
        .then(None)
        .otherwise(pl.col("sog"))
        .alias("sog"),
    )


lf = clean_placeholder_values(lf)

null_after = lf.select(
    [pl.col(c).is_null().sum().alias(c) for c in ["heading", "cog", "sog"]]
).collect()
print("Null counts after placeholder replacement:")
print(null_after)
print("(Compare to null counts from the raw inspection cell above)")


rows_before_sog = lf.select(pl.len()).collect().item()
high_sog_count = lf.select((pl.col("sog") > SOG_SUSPICIOUS).sum()).collect().item()
null_sog_count = lf.select(pl.col("sog").is_null().sum()).collect().item()
print(f"Rows with null SOG:          {null_sog_count:,}")
print(f"Rows with SOG > {SOG_SUSPICIOUS} knots:  {high_sog_count:,}")

lf = lf.filter(pl.col("sog").is_not_null() & (pl.col("sog") <= SOG_SUSPICIOUS))

rows_after_sog = lf.select(pl.len()).collect().item()
print(f"Rows before: {rows_before_sog:,}")
print(f"Rows after:  {rows_after_sog:,}")
print(f"Dropped:     {rows_before_sog - rows_after_sog:,}")


def remove_duplicates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove duplicate records on (mmsi, timestamp_utc).

    Keeps first occurrence. MarineCadastre downsamples to one observation per
    vessel per minute, so duplicates indicate data delivery overlaps across
    files (e.g. same day appearing in two downloads).

    Args:
        lf: LazyFrame.

    Returns:
        LazyFrame with duplicate (mmsi, timestamp_utc) pairs removed.
    """
    return lf.unique(
        subset=["mmsi", "timestamp_utc"], keep="first", maintain_order=False
    )


rows_before_dedup = lf.select(pl.len()).collect().item()
lf = remove_duplicates(lf)
rows_after_dedup = lf.select(pl.len()).collect().item()

print(f"Rows before deduplication: {rows_before_dedup:,}")
print(f"Rows after  deduplication: {rows_after_dedup:,}")
print(f"Duplicates removed:        {rows_before_dedup - rows_after_dedup:,}")


def flag_speed_jumps(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Flag records where implied speed from consecutive positions exceeds 30 knots.

    Uses flat-earth distance approximation. Sorts by (mmsi, timestamp_utc) and
    computes per-vessel position shifts with .over("mmsi"). Intermediate
    columns (prefixed _) are dropped before return.

    Args:
        lf: LazyFrame.

    Returns:
        LazyFrame with flag_speed_jump boolean column added.
    """
    deg_to_rad = math.pi / 180.0

    return (
        lf.sort(["mmsi", "timestamp_utc"])
        .with_columns(
            pl.col("latitude").shift(1).over("mmsi").alias("_prev_lat"),
            pl.col("longitude").shift(1).over("mmsi").alias("_prev_lon"),
            pl.col("timestamp_utc").shift(1).over("mmsi").alias("_prev_ts"),
        )
        .with_columns(
            (
                ((pl.col("latitude") - pl.col("_prev_lat")) * 111_320.0).pow(2)
                + (
                    (pl.col("longitude") - pl.col("_prev_lon"))
                    * 111_320.0
                    * (pl.col("latitude") * deg_to_rad).cos()
                ).pow(2)
            )
            .sqrt()
            .alias("_dist_m"),
            (pl.col("timestamp_utc") - pl.col("_prev_ts"))
            .dt.total_seconds()
            .alias("_time_diff_s"),
        )
        .with_columns(
            pl.when(pl.col("_time_diff_s") > 0)
            .then(pl.col("_dist_m") / pl.col("_time_diff_s") > SPEED_JUMP_MS)
            .otherwise(False)
            .alias("flag_speed_jump"),
        )
        .drop(["_prev_lat", "_prev_lon", "_prev_ts", "_dist_m", "_time_diff_s"])
    )


lf = flag_speed_jumps(lf)

speed_jump_count = lf.select(pl.col("flag_speed_jump").sum()).collect().item()
print(
    f"Speed jump flags: {speed_jump_count:,} ({speed_jump_count / rows_after_dedup * 100:.3f}% of rows)"
)

print("\nSample of speed-jump flagged records:")
print(
    lf.filter(pl.col("flag_speed_jump"))
    .select(["mmsi", "timestamp_utc", "latitude", "longitude", "sog", "port_name"])
    .head(10)
    .collect()
)

rows_before_jump_drop = lf.select(pl.len()).collect().item()
lf = lf.filter(~pl.col("flag_speed_jump")).drop("flag_speed_jump")
rows_after_jump_drop = lf.select(pl.len()).collect().item()
print(f"Rows before: {rows_before_jump_drop:,}")
print(f"Rows after:  {rows_after_jump_drop:,}")
print(f"Dropped:     {rows_before_jump_drop - rows_after_jump_drop:,}")

final_total = lf.select(pl.len()).collect().item()
final_nulls = lf.select(
    [pl.col(c).is_null().sum().alias(c) for c in lf.collect_schema().names()]
).collect()
print(f"Final null counts — total rows: {final_total:,}\n")
for col, count in zip(final_nulls.columns, final_nulls.row(0)):
    pct = count / final_total * 100
    print(f"  {col:<22} {count:>10,}  ({pct:5.2f}%)")


def build_gisis_vessel_lookup(
    mmsi_imo_map: pl.DataFrame,
    gisis_cache_path: Path = GISIS_CACHE_PATH,
) -> pl.DataFrame:
    """Build a vessel lookup table by joining MMSI→IMO mapping with GISIS data.

    Strips the "IMO" prefix from the mapping's imo column to match the
    gisis_cache imo_number format. Returns one row per MMSI with
    ship_type_detailed, gross_tonnage (numeric), and ship_type_group.

    Args:
        mmsi_imo_map: High-confidence MMSI→IMO mapping from build_mmsi_imo_mapping.
        gisis_cache_path: Path to gisis_cache.csv.

    Returns:
        DataFrame with columns: mmsi, imo, ship_type_detailed, gross_tonnage,
        ship_type_group.

    Raises:
        FileNotFoundError: If gisis_cache_path does not exist.
    """
    if not gisis_cache_path.exists():
        raise FileNotFoundError(f"GISIS cache not found: {gisis_cache_path}")

    gisis = (
        pl.read_csv(gisis_cache_path)
        .filter(pl.col("found"))
        .select(
            pl.col("imo_number").cast(pl.String).alias("imo_join"),
            pl.col("ship_type_detailed")
            .cast(pl.String)
            .str.strip_chars()
            .replace("", None)
            .alias("ship_type_detailed"),
            pl.col("gross_tonnage")
            .cast(pl.String)
            .str.replace_all(",", "")
            .replace("", None)
            .cast(pl.Float64, strict=False)
            .alias("gross_tonnage"),
        )
    )

    lookup = (
        mmsi_imo_map.select("mmsi", "imo")
        .with_columns(
            pl.col("imo").str.replace("^IMO", "").alias("imo_join"),
        )
        .join(gisis, on="imo_join", how="left")
        .drop("imo_join")
        .with_columns(
            pl.col("ship_type_detailed")
            .replace(SHIP_TYPE_GROUPS, default="Other Cargo")
            .alias("ship_type_group"),
        )
    )

    return lookup


_gisis_lookup = build_gisis_vessel_lookup(mmsi_imo_map)
print(f"Vessel lookup: {_gisis_lookup.height:,} vessels")
print(
    _gisis_lookup.group_by("ship_type_group")
    .agg(
        pl.len().alias("vessels"),
        pl.col("gross_tonnage").mean().round(0).alias("avg_gt"),
    )
    .sort("vessels", descending=True)
)


def run_cleaning_pipeline(
    merged_path: Path = MERGED_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """
    Execute the full cleaning pipeline end-to-end and write output parquet.

    Rebuilds the lazy query from scratch (does not reuse the interactive lf
    defined above). Use this for a clean production run.

    Args:
        merged_path: Path to the merged raw parquet (output of Stage 1).
        output_path: Destination path for the cleaned parquet file.
    """
    logger.info("Starting AIS cleaning pipeline")

    lf_prod = pl.scan_parquet(merged_path).select(KEEP_COLUMNS)
    lf_prod = lf_prod.filter(pl.col("port_name").is_in(PORTS_TO_KEEP))
    lf_prod = clean_placeholder_values(lf_prod)
    lf_prod = lf_prod.filter(
        pl.col("sog").is_not_null() & (pl.col("sog") <= SOG_SUSPICIOUS)
    )
    lf_prod = remove_duplicates(lf_prod)
    lf_prod = flag_speed_jumps(lf_prod)
    lf_prod = lf_prod.filter(~pl.col("flag_speed_jump")).drop("flag_speed_jump")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lf_prod.sink_parquet(output_path, compression="zstd")
    logger.info(f"Cleaned data written to {output_path}")

    lf_cleaned = pl.scan_parquet(output_path)
    mmsi_imo = build_mmsi_imo_mapping(lf_cleaned)

    n_total = lf_cleaned.select(pl.col("mmsi").n_unique()).collect().item()
    n_matched = mmsi_imo.height
    n_no_imo = n_total - n_matched
    n_confident = mmsi_imo.filter(pl.col("imo_confident")).height
    n_low_conf = n_matched - n_confident

    logger.info(
        f"MMSI→IMO mapping: {n_matched:,}/{n_total:,} matched "
        f"({n_confident:,} high-confidence, {n_low_conf:,} low-confidence, "
        f"{n_no_imo:,} no IMO)"
    )

    mmsi_imo_confident = mmsi_imo.filter(pl.col("imo_confident"))
    logger.info(
        f"Excluded {n_low_conf:,} low-confidence and {n_no_imo:,} no-IMO MMSIs — "
        f"keeping {mmsi_imo_confident.height:,} high-confidence mappings"
    )

    mapping_path = output_path.parent / "mmsi_imo_mapping.csv"
    mmsi_imo_confident.write_csv(mapping_path)
    logger.info(f"MMSI→IMO mapping written to {mapping_path}")

    vessel_lookup = build_gisis_vessel_lookup(mmsi_imo_confident)
    lookup_path = output_path.parent / "vessel_lookup.csv"
    vessel_lookup.write_csv(lookup_path)

    n_with_type = vessel_lookup.filter(
        pl.col("ship_type_detailed").is_not_null()
    ).height
    logger.info(
        f"Vessel lookup written to {lookup_path} — "
        f"{vessel_lookup.height:,} vessels, "
        f"{n_with_type:,} with GISIS ship type"
    )

    group_counts = (
        vessel_lookup.group_by("ship_type_group")
        .agg(pl.len().alias("vessels"))
        .sort("vessels", descending=True)
    )
    for row in group_counts.iter_rows(named=True):
        logger.info(
            f"  {row['ship_type_group']}: {row['vessels']:,} vessels"
        )

    join_cols = vessel_lookup.select(
        "mmsi", "ship_type_detailed", "gross_tonnage", "ship_type_group"
    )
    lf_enriched = (
        pl.scan_parquet(output_path)
        .join(join_cols.lazy(), on="mmsi", how="inner")
    )
    enriched_tmp = output_path.with_suffix(".tmp.parquet")
    lf_enriched.sink_parquet(enriched_tmp, compression="zstd")
    enriched_tmp.rename(output_path)

    rows_after_enrich = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logger.info(
        f"Cleaned parquet enriched with ship_type_detailed, gross_tonnage, "
        f"ship_type_group — {rows_after_enrich:,} rows retained (inner join "
        f"dropped vessels without GISIS verification)"
    )


run_cleaning_pipeline()
