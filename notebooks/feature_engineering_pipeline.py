
import logging
from pathlib import Path

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


INPUT_PATH = Path("../ais_data/standardized/ais_standardized.parquet")
OUTPUT_PATH = Path("../ais_data/features/port_daily_features.parquet")

SOG_STOPPED: float = 0.5
SOG_MANEUVERING: float = 2.0

MIN_SOG_TRANSITIONS: int = 2
STATIC_FILTER_MIN_HOURS: float = 24.0

MIN_STOP_DURATION_MINUTES: int = 10

NAV_STATUS_BERTH: int = 5
NAV_STATUS_ANCHORAGE: int = 1

ZONE_BERTH_DWELL_HOURS: dict[str, tuple[float, float]] = {
    "Bulk Carrier": (2.0, 168.0),
    "Container Ship": (1.0, 72.0),
    "_default": (2.0, 168.0),
}
ZONE_ANCHORAGE_DWELL_HOURS: dict[str, tuple[float, float]] = {
    "Bulk Carrier": (1.0, 168.0),
    "Container Ship": (1.0, 120.0),
    "_default": (1.0, 168.0),
}

ZONE_SPREAD_THRESHOLD_KM: float = 0.02

MAX_WAITING_HOURS: float = 14 * 24
MAX_SERVICE_HOURS: float = 30 * 24

ROLLING_WINDOW_DAYS: int = 7
FORECAST_HORIZONS: list[int] = [1, 2]
CONGESTION_THRESHOLD_HOURS_CONTAINER: float = 48.0
CONGESTION_THRESHOLD_HOURS_BULK: float = 120.0

PORT_BBOXES: dict[str, dict[str, float]] = {
    "LOS_ANGELES_LONG_BEACH": {
        "lon_min": -118.340177,
        "lat_min": 33.593389,
        "lon_max": -118.040626,
        "lat_max": 33.830902,
    },
    "NEW_YORK_NEW_JERSEY": {
        "lon_min": -74.330503,
        "lat_min": 40.345893,
        "lon_max": -73.9,
        "lat_max": 40.755294,
    },
    "HOUSTON": {
        "lon_min": -95.133091,
        "lat_min": 29.05,
        "lon_max": -94.381116,
        "lat_max": 29.848808,
    },
    "PORT_OF_VIRGINIA": {
        "lon_min": -76.5043,
        "lat_min": 36.792667,
        "lon_max": -75.974667,
        "lat_max": 37.325,
    },
}

REQUIRED_COLUMNS: list[str] = [
    "mmsi",
    "timestamp_utc",
    "latitude",
    "longitude",
    "sog",
    "nav_status",
    "port_name",
    "segment_id",
    "ship_type_group",
    "gross_tonnage",
]

SHIP_TYPE_SPLITS: dict[str, str | None] = {
    "bulk_carrier": "Bulk Carrier",
    "container_ship": "Container Ship",
}


def filter_static_infrastructure(
    lf: pl.LazyFrame,
    min_transitions: int = MIN_SOG_TRANSITIONS,
    min_hours: float = STATIC_FILTER_MIN_HOURS,
) -> pl.LazyFrame:
    """Remove long-duration segments with no SOG state transitions.

    Permanently moored infrastructure (barges, floating docks, nav aids
    misclassified as vessel_type 70-79) reports constant SOG ≈ 0.0 for
    days, months, or years. These are detected by checking whether a
    segment ever transitions between stopped and moving.

    Only segments longer than `min_hours` are subject to this filter.
    Short segments (< 24h by default) with few transitions are kept —
    they are likely real vessel visits (e.g. direct-to-berth arrivals
    that stop once and stay stopped until departure).

    Args:
        lf: LazyFrame with sog, segment_id, timestamp_utc columns.
        min_transitions: Minimum SOG transitions for long segments.
        min_hours: Segments shorter than this are always kept.

    Returns:
        LazyFrame with static-infrastructure segments removed.
    """
    seg_stats = (
        lf.sort(["segment_id", "timestamp_utc"])
        .select(["segment_id", "sog", "timestamp_utc"])
        .with_columns(
            (pl.col("sog") < SOG_STOPPED).alias("_is_stopped"),
        )
        .with_columns(
            (pl.col("_is_stopped") != pl.col("_is_stopped").shift(1).over("segment_id"))
            .fill_null(False)
            .cast(pl.Int32)
            .alias("_is_transition"),
        )
        .group_by("segment_id")
        .agg(
            pl.col("_is_transition").sum().alias("n_transitions"),
            (
                (pl.col("timestamp_utc").max() - pl.col("timestamp_utc").min())
                .dt.total_seconds()
                / 3600
            ).alias("duration_hours"),
        )
    )

    valid_segments = seg_stats.filter(
        (pl.col("duration_hours") < min_hours)
        | (pl.col("n_transitions") >= min_transitions)
    ).select("segment_id")

    return lf.join(valid_segments, on="segment_id", how="semi")


lf = pl.scan_parquet(INPUT_PATH).select(REQUIRED_COLUMNS)

print("Schema:")
for col, dtype in lf.collect_schema().items():
    print(f"  {col}: {dtype}")

total_rows = lf.select(pl.len()).collect().item()
print(f"\nTotal rows: {total_rows:,}")

lf = filter_static_infrastructure(lf)

rows_after_filter = lf.select(pl.len()).collect().item()
removed = total_rows - rows_after_filter
print(f"Rows after behavioral filter: {rows_after_filter:,}")
print(
    f"Rows removed (static infrastructure): {removed:,} "
    f"({removed / total_rows * 100:.1f}%)"
)
print(
    f"  Filter: drop segments > {STATIC_FILTER_MIN_HOURS}h with "
    f"< {MIN_SOG_TRANSITIONS} SOG transitions"
)

print("\nRows per port:")
print(
    lf.group_by("port_name")
    .agg(pl.len().alias("rows"))
    .sort("rows", descending=True)
    .collect()
)


def label_movement_state(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add movement_state column based on SOG thresholds.

    States (from literature — Belcore et al. 2025; Chen et al. 2023):
        STOPPED:    SOG < 0.5 knots  (anchored or berthed)
        SLOW:       0.5 <= SOG < 2.0 knots  (maneuvering)
        TRANSITING: SOG >= 2.0 knots

    Args:
        lf: LazyFrame with sog column.

    Returns:
        LazyFrame with movement_state column added.
    """
    return lf.with_columns(
        pl.when(pl.col("sog") < SOG_STOPPED)
        .then(pl.lit("STOPPED"))
        .when(pl.col("sog") < SOG_MANEUVERING)
        .then(pl.lit("SLOW"))
        .otherwise(pl.lit("TRANSITING"))
        .alias("movement_state"),
    )


def detect_stop_events(lf: pl.LazyFrame) -> pl.DataFrame:
    """Extract stop events from consecutive stopped pings.

    Uses run-length encoding: consecutive pings with the same movement
    state within a segment form a "run". Runs where state == STOPPED
    are aggregated into stop events with centroid positions and durations.

    This collects to a DataFrame because the result (stop events) is
    orders of magnitude smaller than the input (~25M pings → ~100-500k events).

    Args:
        lf: LazyFrame with movement_state, segment_id, timestamp_utc,
            latitude, longitude, mmsi, nav_status, port_name columns.

    Returns:
        DataFrame of stop events with centroid positions, durations,
        and majority nav_status during the stop.
    """
    return (
        lf.sort(["segment_id", "timestamp_utc"])
        .with_columns(
            (
                (
                    pl.col("movement_state")
                    != pl.col("movement_state").shift(1).over("segment_id")
                )
                .fill_null(True)
                .cum_sum()
                .over("segment_id")
            ).alias("run_group_id"),
        )
        .filter(pl.col("movement_state") == "STOPPED")
        .group_by(["segment_id", "run_group_id"])
        .agg(
            pl.col("mmsi").first(),
            pl.col("port_name").first(),
            pl.col("ship_type_group").first(),
            pl.col("gross_tonnage").first(),
            pl.col("timestamp_utc").min().alias("stop_start"),
            pl.col("timestamp_utc").max().alias("stop_end"),
            pl.col("latitude").mean().alias("mean_lat"),
            pl.col("longitude").mean().alias("mean_lon"),
            pl.len().alias("n_pings"),
            pl.col("nav_status").mode().first().alias("nav_status"),
            pl.col("latitude").std().alias("std_lat"),
            pl.col("longitude").std().alias("std_lon"),
        )
        .with_columns(
            (
                (pl.col("stop_end") - pl.col("stop_start")).dt.total_seconds() / 3600
            ).alias("duration_hours"),
            (
                (
                    (pl.col("std_lat").fill_null(0.0) * 111.0) ** 2
                    + (pl.col("std_lon").fill_null(0.0) * 85.0) ** 2
                ).sqrt()
            ).alias("spread_km"),
        )
        .filter(pl.col("duration_hours") >= MIN_STOP_DURATION_MINUTES / 60)
        .collect()
    )


lf = label_movement_state(lf)
stop_events = detect_stop_events(lf)

print(f"Stop events detected: {stop_events.height:,}")
print("\nStop events per port:")
print(
    stop_events.group_by("port_name")
    .agg(pl.len().alias("events"))
    .sort("events", descending=True)
)

print("\nStop event duration (hours):")
print(stop_events["duration_hours"].describe())

print("\nSample stop events:")
print(stop_events.head(10))

for status, label in [(5, "Moored"), (1, "At Anchor")]:
    subset = stop_events.filter(pl.col("nav_status") == status)
    if subset.height > 0:
        print(f"\nnav_status={status} ({label}) — spread_km distribution:")
        print(subset["spread_km"].describe())
        print(f"  p25: {subset['spread_km'].quantile(0.25):.4f}")
        print(f"  p75: {subset['spread_km'].quantile(0.75):.4f}")
        print(f"  p90: {subset['spread_km'].quantile(0.90):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, (status, label, color) in enumerate(
    [(5, "Moored (nav=5)", "#FF6B6B"), (1, "At Anchor (nav=1)", "#4ECDC4")]
):
    ax = axes[i]
    subset = stop_events.filter(pl.col("nav_status") == status)
    if subset.height > 0:
        vals = subset["spread_km"].drop_nulls().to_numpy()
        ax.hist(vals, bins=100, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(
            ZONE_SPREAD_THRESHOLD_KM,
            color="black",
            linestyle="--",
            label=f"threshold={ZONE_SPREAD_THRESHOLD_KM}",
        )
        ax.set_title(f"{label} (n={len(vals):,})")
        ax.set_xlabel("spread_km")
        ax.set_ylabel("count")
        ax.legend()
plt.suptitle("Position Spread by nav_status — Threshold Calibration", fontweight="bold")
plt.tight_layout()
plt.show()

for status, label in [(5, "Moored"), (1, "At Anchor")]:
    subset = stop_events.filter(pl.col("nav_status") == status)
    if subset.height > 0:
        print(f"\nnav_status={status} ({label}) — duration_hours distribution:")
        print(subset["duration_hours"].describe())
        print(f"  p1:  {subset['duration_hours'].quantile(0.01):.2f}")
        print(f"  p5:  {subset['duration_hours'].quantile(0.05):.2f}")
        print(f"  p95: {subset['duration_hours'].quantile(0.95):.2f}")
        print(f"  p99: {subset['duration_hours'].quantile(0.99):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, (status, label, color) in enumerate(
    [(5, "Moored (nav=5)", "#FF6B6B"), (1, "At Anchor (nav=1)", "#4ECDC4")]
):
    ax = axes[i]
    subset = stop_events.filter(pl.col("nav_status") == status)
    if subset.height > 0:
        vals = subset["duration_hours"].drop_nulls().to_numpy()
        vals_capped = vals[vals <= 200]
        ax.hist(vals_capped, bins=80, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(
            ZONE_BERTH_DWELL_HOURS["_default"][0],
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"berth_min={ZONE_BERTH_DWELL_HOURS['_default'][0]}h",
        )
        ax.axvline(
            ZONE_ANCHORAGE_DWELL_HOURS["_default"][0],
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"anch_min={ZONE_ANCHORAGE_DWELL_HOURS['_default'][0]}h",
        )
        ax.set_title(f"{label} (n={len(vals):,})")
        ax.set_xlabel("duration_hours (capped at 200)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
plt.suptitle("Dwell Duration by nav_status — Threshold Calibration", fontweight="bold")
plt.tight_layout()
plt.show()


def classify_stop_zones(stop_events: pl.DataFrame) -> pl.DataFrame:
    """Assign zone_type to each stop event using nav_status + position spread.

    All stop events have already passed SOG < 0.5 kn + persistence filtering
    in Stage 1. This function classifies them as BERTH, ANCHORAGE, or NOISE.

    Classification priority:
        1. nav_status == 5 (Moored)                          → BERTH
        2. nav_status == 1 (At Anchor)                       → ANCHORAGE
        3. Unreliable status + low spread + berth dwell      → BERTH
        4. Unreliable status + high spread + anchorage dwell → ANCHORAGE
        5. Everything else                                   → NOISE

    Rules 1-2 trust reliable nav_status without dwell constraints because
    when crews actively set status 5 or 1, it is a high-precision signal.
    Dwell constraints are only applied to rules 3-4 where we fall back to
    position spread for unreliable status codes (0, 15, null, etc.).

    Dwell bounds are type-specific (ship_type_group column):
        Bulk Carrier  — berth 2–168h (7d), anchorage 1–168h (7d)
        Container Ship — berth 1–72h (3d), anchorage 1–120h (5d)
        _default      — berth 2–168h,      anchorage 1–168h

    Args:
        stop_events: DataFrame with nav_status, duration_hours, spread_km,
            and ship_type_group.

    Returns:
        stop_events with zone_type column added.
    """
    _berth_min = (
        pl.when(pl.col("ship_type_group") == "Bulk Carrier")
        .then(pl.lit(ZONE_BERTH_DWELL_HOURS["Bulk Carrier"][0]))
        .when(pl.col("ship_type_group") == "Container Ship")
        .then(pl.lit(ZONE_BERTH_DWELL_HOURS["Container Ship"][0]))
        .otherwise(pl.lit(ZONE_BERTH_DWELL_HOURS["_default"][0]))
    )
    _berth_max = (
        pl.when(pl.col("ship_type_group") == "Bulk Carrier")
        .then(pl.lit(ZONE_BERTH_DWELL_HOURS["Bulk Carrier"][1]))
        .when(pl.col("ship_type_group") == "Container Ship")
        .then(pl.lit(ZONE_BERTH_DWELL_HOURS["Container Ship"][1]))
        .otherwise(pl.lit(ZONE_BERTH_DWELL_HOURS["_default"][1]))
    )
    berth_dwell = (pl.col("duration_hours") >= _berth_min) & (
        pl.col("duration_hours") <= _berth_max
    )
    _anchorage_min = (
        pl.when(pl.col("ship_type_group") == "Bulk Carrier")
        .then(pl.lit(ZONE_ANCHORAGE_DWELL_HOURS["Bulk Carrier"][0]))
        .when(pl.col("ship_type_group") == "Container Ship")
        .then(pl.lit(ZONE_ANCHORAGE_DWELL_HOURS["Container Ship"][0]))
        .otherwise(pl.lit(ZONE_ANCHORAGE_DWELL_HOURS["_default"][0]))
    )
    _anchorage_max = (
        pl.when(pl.col("ship_type_group") == "Bulk Carrier")
        .then(pl.lit(ZONE_ANCHORAGE_DWELL_HOURS["Bulk Carrier"][1]))
        .when(pl.col("ship_type_group") == "Container Ship")
        .then(pl.lit(ZONE_ANCHORAGE_DWELL_HOURS["Container Ship"][1]))
        .otherwise(pl.lit(ZONE_ANCHORAGE_DWELL_HOURS["_default"][1]))
    )
    anchorage_dwell = (pl.col("duration_hours") >= _anchorage_min) & (
        pl.col("duration_hours") <= _anchorage_max
    )
    nav_moored = pl.col("nav_status") == NAV_STATUS_BERTH
    nav_anchored = pl.col("nav_status") == NAV_STATUS_ANCHORAGE
    reliable_status = pl.col("nav_status").is_in([NAV_STATUS_BERTH, NAV_STATUS_ANCHORAGE])
    low_spread = pl.col("spread_km") <= ZONE_SPREAD_THRESHOLD_KM

    return stop_events.with_columns(
        pl.when(nav_moored)
        .then(pl.lit("BERTH"))
        .when(nav_anchored)
        .then(pl.lit("ANCHORAGE"))
        .when(~reliable_status & low_spread & berth_dwell)
        .then(pl.lit("BERTH"))
        .when(~reliable_status & ~low_spread & anchorage_dwell)
        .then(pl.lit("ANCHORAGE"))
        .otherwise(pl.lit("NOISE"))
        .alias("zone_type")
    )


logger.info(
    "STAGE 2: Zone Classification (nav_status primary, position spread fallback)"
)
stop_events = classify_stop_zones(stop_events)

print("\nZone classification:")
print(
    stop_events.group_by(["port_name", "zone_type"])
    .agg(pl.len().alias("stop_events"))
    .sort(["port_name", "zone_type"])
)

print("\nnav_status → zone_type breakdown:")
print(
    stop_events.group_by(["nav_status", "zone_type"])
    .agg(pl.len().alias("count"))
    .sort(["nav_status", "zone_type"])
)


def plot_zone_map(
    stop_events: pl.DataFrame,
    port_bboxes: dict[str, dict[str, float]],
    images_dir: Path | None = None,
) -> None:
    """Plot zone map per port on satellite basemap.

    Stop events are coloured by zone_type (BERTH / ANCHORAGE / NOISE).
    Convex hull polygons are omitted — they are misleading when anchorage
    stops are spatially dispersed across a large open-water area.
    """
    if images_dir is None:
        images_dir = Path(__file__).resolve().parent.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    zone_colors = {"BERTH": "#FF6B6B", "ANCHORAGE": "#4ECDC4", "NOISE": "#C0C0C0"}

    for port_name, bbox in port_bboxes.items():
        port_events = stop_events.filter(pl.col("port_name") == port_name)
        if port_events.height == 0:
            continue

        fig, ax = plt.subplots(figsize=(12, 10))

        for zone_type, color in zone_colors.items():
            zone_pts = port_events.filter(pl.col("zone_type") == zone_type)
            if zone_pts.height > 0:
                ax.scatter(
                    zone_pts["mean_lon"].to_numpy(),
                    zone_pts["mean_lat"].to_numpy(),
                    c=color,
                    s=8,
                    alpha=0.4,
                    label=f"{zone_type} stops",
                    edgecolors="none",
                )

        ax.plot(
            [
                bbox["lon_min"],
                bbox["lon_max"],
                bbox["lon_max"],
                bbox["lon_min"],
                bbox["lon_min"],
            ],
            [
                bbox["lat_min"],
                bbox["lat_min"],
                bbox["lat_max"],
                bbox["lat_max"],
                bbox["lat_min"],
            ],
            "k--",
            linewidth=1.5,
            label="AIS bbox",
        )

        ax.set_xlim(bbox["lon_min"] - 0.02, bbox["lon_max"] + 0.02)
        ax.set_ylim(bbox["lat_min"] - 0.02, bbox["lat_max"] + 0.02)

        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.Esri.WorldImagery,
                zoom=12,
            )
        except Exception:
            logger.warning(f"  Could not add basemap for {port_name}")

        ax.set_title(f"{port_name} — Operational Zones", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plot_path = images_dir / f"zones_{port_name.lower()}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {plot_path}")
        plt.close()


plot_zone_map(stop_events, PORT_BBOXES)


def detect_vessel_visits(stop_events: pl.DataFrame) -> pl.DataFrame:
    """Detect vessel visit lifecycles and compute per-visit waiting times.

    For each segment, examines the sequence of classified stop events:
    - WAIT_THEN_BERTH: anchorage stop followed by berth stop.
      waiting_time = first_berth_start - first_anchorage_start
    - DIRECT_TO_BERTH: berth stops only, no anchorage wait.
      waiting_time = 0
    - ANCHORAGE_ONLY: anchorage stops only, vessel never reached berth
      (censored observation, excluded from target computation).

    Args:
        stop_events: Stop events with zone_type column already assigned
            by classify_stop_zones().

    Returns:
        DataFrame of vessel visits with per-visit timing columns.
    """
    events = (
        stop_events.with_columns(pl.col("zone_type").fill_null("NOISE"))
        .filter(pl.col("zone_type").is_in(["ANCHORAGE", "BERTH"]))
        .sort(["segment_id", "stop_start"])
    )

    if events.height == 0:
        logger.warning("No stop events in ANCHORAGE or BERTH zones")
        return pl.DataFrame(
            schema={
                "segment_id": pl.Int64,
                "mmsi": pl.Int64,
                "port_name": pl.Utf8,
                "ship_type_group": pl.Utf8,
                "gross_tonnage": pl.Float64,
                "visit_start": pl.Datetime,
                "visit_end": pl.Datetime,
                "wait_start": pl.Datetime,
                "service_start": pl.Datetime,
                "service_end": pl.Datetime,
                "waiting_time_hours": pl.Float64,
                "service_time_hours": pl.Float64,
                "visit_type": pl.Utf8,
            }
        )

    visits = events.group_by("segment_id").agg(
        pl.col("mmsi").first(),
        pl.col("port_name").first(),
        pl.col("ship_type_group").first(),
        pl.col("gross_tonnage").first(),
        pl.col("stop_start")
        .filter(pl.col("zone_type") == "ANCHORAGE")
        .min()
        .alias("wait_start"),
        pl.col("stop_start")
        .filter(pl.col("zone_type") == "BERTH")
        .min()
        .alias("service_start"),
        pl.col("stop_end")
        .filter(pl.col("zone_type") == "BERTH")
        .max()
        .alias("service_end"),
        pl.col("stop_start").min().alias("visit_start"),
        pl.col("stop_end").max().alias("visit_end"),
        pl.col("zone_type")
        .filter(pl.col("zone_type") == "ANCHORAGE")
        .len()
        .alias("_n_anchorage"),
        pl.col("zone_type")
        .filter(pl.col("zone_type") == "BERTH")
        .len()
        .alias("_n_berth"),
    )

    visits = visits.with_columns(
        pl.when(
            (pl.col("_n_anchorage") > 0)
            & (pl.col("_n_berth") > 0)
            & (pl.col("wait_start") < pl.col("service_start"))
        )
        .then(pl.lit("WAIT_THEN_BERTH"))
        .when(pl.col("_n_berth") > 0)
        .then(pl.lit("DIRECT_TO_BERTH"))
        .when(pl.col("_n_anchorage") > 0)
        .then(pl.lit("ANCHORAGE_ONLY"))
        .otherwise(pl.lit("OTHER"))
        .alias("visit_type"),
        pl.when(
            (pl.col("_n_anchorage") > 0)
            & (pl.col("_n_berth") > 0)
            & (pl.col("wait_start") < pl.col("service_start"))
        )
        .then(
            (pl.col("service_start") - pl.col("wait_start")).dt.total_seconds() / 3600
        )
        .when(pl.col("_n_berth") > 0)
        .then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .alias("waiting_time_hours"),
        pl.when(pl.col("_n_berth") > 0)
        .then(
            (pl.col("service_end") - pl.col("service_start")).dt.total_seconds() / 3600
        )
        .otherwise(pl.lit(None))
        .alias("service_time_hours"),
    ).drop(["_n_anchorage", "_n_berth"])

    n_before = visits.height
    visits = visits.filter(
        (pl.col("waiting_time_hours").is_null())
        | (pl.col("waiting_time_hours") <= MAX_WAITING_HOURS)
    ).filter(
        (pl.col("service_time_hours").is_null())
        | (pl.col("service_time_hours") <= MAX_SERVICE_HOURS)
    )
    n_dropped = n_before - visits.height
    if n_dropped > 0:
        logger.info(
            f"  Outlier visits removed: {n_dropped} "
            f"(waiting > {MAX_WAITING_HOURS}h or service > {MAX_SERVICE_HOURS}h)"
        )

    return visits


logger.info("STAGE 4: Vessel Visit Detection")
visits = detect_vessel_visits(stop_events)

print(f"Total vessel visits: {visits.height:,}")
print("\nVisit type distribution (all ports):")
print(
    visits.group_by("visit_type")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)
print("\nVisit type distribution by port:")
print(
    visits.group_by(["port_name", "visit_type"])
    .agg(pl.len().alias("count"))
    .sort(["port_name", "visit_type"])
)

completed_visits = visits.filter(pl.col("waiting_time_hours").is_not_null())
print(f"\nCompleted visits (with waiting time): {completed_visits.height:,}")
print("\nWaiting time (hours):")
print(completed_visits["waiting_time_hours"].describe())
print("\nService time (hours):")
svc = visits.filter(pl.col("service_time_hours").is_not_null())
print(svc["service_time_hours"].describe())

print("\nWaiting time per port (completed visits):")
print(
    completed_visits.group_by("port_name")
    .agg(
        pl.len().alias("n_visits"),
        pl.col("waiting_time_hours").mean().alias("mean_wait_hrs"),
        pl.col("waiting_time_hours").median().alias("median_wait_hrs"),
        pl.col("waiting_time_hours").quantile(0.95).alias("p95_wait_hrs"),
    )
    .sort("p95_wait_hrs", descending=True)
)

neg_waits = completed_visits.filter(pl.col("waiting_time_hours") < 0).height
extreme_waits = completed_visits.filter(pl.col("waiting_time_hours") > 720).height
print("\nSanity checks:")
print(f"  Negative waiting times: {neg_waits} (should be 0)")
print(f"  Waiting times > 30 days: {extreme_waits}")


def plot_waiting_histograms(
    visits: pl.DataFrame,
    images_dir: Path | None = None,
) -> None:
    """Plot waiting time distribution per port."""
    if images_dir is None:
        images_dir = Path(__file__).resolve().parent.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    completed = visits.filter(
        pl.col("waiting_time_hours").is_not_null() & (pl.col("waiting_time_hours") >= 0)
    )
    ports = completed["port_name"].unique().sort().to_list()
    n_ports = len(ports)

    if n_ports == 0:
        logger.warning("No completed visits to plot")
        return

    fig, axes = plt.subplots(1, n_ports, figsize=(5 * n_ports, 4), squeeze=False)

    for i, port in enumerate(ports):
        ax = axes[0, i]
        port_data = completed.filter(pl.col("port_name") == port)
        wait_hours = port_data["waiting_time_hours"].to_numpy()
        p95 = float(np.percentile(wait_hours, 95))

        ax.hist(wait_hours, bins=50, color="#4ECDC4", edgecolor="white", alpha=0.8)
        ax.axvline(
            p95, color="#FF6B6B", linewidth=2, linestyle="--", label=f"p95 = {p95:.1f}h"
        )
        ax.set_title(port.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Waiting time (hours)")
        ax.set_ylabel("Vessel visits")
        ax.legend(fontsize=9)

    plt.suptitle("Waiting Time Distribution by Port", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = images_dir / "waiting_time_histograms.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {plot_path}")
    plt.close()


plot_waiting_histograms(visits)


def compute_rolling_visit_metrics(
    visits: pl.DataFrame,
    window_days: int = ROLLING_WINDOW_DAYS,
) -> pl.DataFrame:
    """Aggregate per-visit metrics over a trailing 7-day window for each date.

    For each ref_date, includes all completed visits whose visit_start falls
    in [ref_date - 6, ref_date]. Each visit is expanded to the 7 ref_dates
    it contributes to (avoids expensive cross-join).

    Features produced (literature support noted):
        feat_p95_waiting_hours — Peng et al. (2023), Chen et al. (2023)
        feat_p50_waiting_hours — Chen et al. (2023)
        feat_mean_waiting_hours — Zhang et al. (2024)
        feat_p95_service_hours — service time extreme
        feat_mean_service_hours — Chen et al. (2023), Zhang et al. (2024)
        feat_n_visits — arrival rate proxy — Zhang et al. (2024)
        feat_n_wait_visits — visits that waited at anchor
        feat_n_direct_visits — visits straight to berth
        feat_pct_waiting — novel: congestion probability

    Args:
        visits: DataFrame of vessel visits with timing columns.
        window_days: Trailing window size in days.

    Returns:
        Daily visit metrics per port.
    """
    usable = visits.filter(
        pl.col("visit_type").is_in(["WAIT_THEN_BERTH", "DIRECT_TO_BERTH"])
    )

    if usable.height == 0:
        return pl.DataFrame()

    usable = usable.with_columns(
        pl.col("visit_start").cast(pl.Date).alias("visit_date"),
    )
    expanded = (
        usable.with_columns(
            pl.date_ranges(
                pl.col("visit_date"),
                pl.col("visit_date").dt.offset_by(f"{window_days - 1}d"),
            ).alias("_ref_dates"),
        )
        .explode("_ref_dates")
        .with_columns(
            pl.col("_ref_dates").cast(pl.Date).alias("ref_date"),
        )
        .drop("_ref_dates")
    )

    return (
        expanded.group_by(["port_name", "ref_date"])
        .agg(
            pl.col("waiting_time_hours").quantile(0.95).alias("feat_p95_waiting_hours"),
            pl.col("waiting_time_hours").quantile(0.50).alias("feat_p50_waiting_hours"),
            pl.col("waiting_time_hours").mean().alias("feat_mean_waiting_hours"),
            pl.col("service_time_hours").quantile(0.95).alias("feat_p95_service_hours"),
            pl.col("service_time_hours").mean().alias("feat_mean_service_hours"),
            pl.len().alias("feat_n_visits"),
            (pl.col("visit_type") == "WAIT_THEN_BERTH")
            .sum()
            .alias("feat_n_wait_visits"),
            (pl.col("visit_type") == "DIRECT_TO_BERTH")
            .sum()
            .alias("feat_n_direct_visits"),
            pl.col("gross_tonnage").mean().alias("feat_mean_gross_tonnage"),
        )
        .with_columns(
            (pl.col("feat_n_wait_visits") / pl.col("feat_n_visits")).alias(
                "feat_pct_waiting"
            ),
        )
        .sort(["port_name", "ref_date"])
    )


def _expand_zone_events_to_daily(
    events_z: pl.DataFrame,
    zone: str,
) -> pl.DataFrame:
    """Expand stop events for one zone type to daily rows with overlap hours.

    For each stop event, generates one row per calendar date the event
    overlaps, with the hours of overlap on that specific date.

    Args:
        events_z: Stop events with zone_type column.
        zone: Zone type to filter ("ANCHORAGE" or "BERTH").

    Returns:
        DataFrame with (port_name, ref_date, daily_hours, mmsi) rows.
    """
    zone_events = events_z.filter(pl.col("zone_type") == zone)
    if zone_events.height == 0:
        return pl.DataFrame(
            schema={
                "port_name": pl.Utf8,
                "ref_date": pl.Date,
                "daily_hours": pl.Float64,
                "mmsi": pl.Int64,
            }
        )

    return (
        zone_events.with_columns(
            pl.date_ranges(
                pl.col("stop_start").cast(pl.Date),
                pl.col("stop_end").cast(pl.Date),
            ).alias("_dates"),
        )
        .explode("_dates")
        .with_columns(pl.col("_dates").cast(pl.Date).alias("ref_date"))
        .with_columns(
            (
                (
                    pl.min_horizontal(
                        "stop_end",
                        pl.col("ref_date").cast(pl.Datetime("us"))
                        + pl.duration(days=1),
                    )
                    - pl.max_horizontal(
                        "stop_start",
                        pl.col("ref_date").cast(pl.Datetime("us")),
                    )
                ).dt.total_seconds()
                / 3600
            )
            .clip(lower_bound=0.0)
            .alias("daily_hours"),
        )
        .drop("_dates")
    )


def compute_rolling_snapshot_metrics(
    stop_events: pl.DataFrame,
    lf: pl.LazyFrame,
    window_days: int = ROLLING_WINDOW_DAYS,
) -> pl.DataFrame:
    """Compute daily snapshot metrics with 7-day rolling aggregation.

    Strategy: expand stop events and segments to daily rows, compute
    per-day metrics, then apply rolling aggregation. This avoids expensive
    cross-joins with a large date spine.

    Features produced:
        feat_mean_queue_length — Chen et al. (2023), Zhang et al. (2024)
        feat_max_queue_length — peak daily anchorage count in the window
        feat_mean_berth_count — AbuAlhaol et al. (2018)
        feat_max_berth_count — peak daily berth count in the window
        feat_arrival_count — Zhang et al. (2024), Nakashima et al. (2024)
        feat_departure_count — departure rate
        feat_mean_port_vessels — time-weighted average vessels in port
        feat_service_rate — departures / arrivals ratio

    Args:
        stop_events: Stop events with zone_type already assigned by
            classify_stop_zones().
        lf: Full ping LazyFrame for segment boundary computation.
        window_days: Trailing window size in days.

    Returns:
        Daily snapshot metrics per port.
    """
    hours_per_window = window_days * 24

    events_z = stop_events.with_columns(pl.col("zone_type").fill_null("NOISE"))

    anchorage_daily = (
        _expand_zone_events_to_daily(events_z, "ANCHORAGE")
        .group_by(["port_name", "ref_date"])
        .agg(
            pl.col("daily_hours").sum().alias("daily_anch_hours"),
            pl.col("mmsi").n_unique().alias("daily_anch_vessels"),
        )
    )

    berth_daily = (
        _expand_zone_events_to_daily(events_z, "BERTH")
        .group_by(["port_name", "ref_date"])
        .agg(
            pl.col("daily_hours").sum().alias("daily_berth_hours"),
            pl.col("mmsi").n_unique().alias("daily_berth_vessels"),
        )
    )

    seg_bounds = (
        lf.group_by("segment_id")
        .agg(
            pl.col("mmsi").first(),
            pl.col("port_name").first(),
            pl.col("timestamp_utc").min().alias("seg_start"),
            pl.col("timestamp_utc").max().alias("seg_end"),
        )
        .collect()
    )

    daily_arrivals = (
        seg_bounds.with_columns(
            pl.col("seg_start").cast(pl.Date).alias("ref_date"),
        )
        .group_by(["port_name", "ref_date"])
        .agg(pl.len().alias("daily_arrivals"))
    )

    daily_departures = (
        seg_bounds.with_columns(
            pl.col("seg_end").cast(pl.Date).alias("ref_date"),
        )
        .group_by(["port_name", "ref_date"])
        .agg(pl.len().alias("daily_departures"))
    )

    daily_port = (
        seg_bounds.with_columns(
            pl.date_ranges(
                pl.col("seg_start").cast(pl.Date),
                pl.col("seg_end").cast(pl.Date),
            ).alias("_dates"),
        )
        .explode("_dates")
        .with_columns(pl.col("_dates").cast(pl.Date).alias("ref_date"))
        .with_columns(
            (
                (
                    pl.min_horizontal(
                        "seg_end",
                        pl.col("ref_date").cast(pl.Datetime("us"))
                        + pl.duration(days=1),
                    )
                    - pl.max_horizontal(
                        "seg_start",
                        pl.col("ref_date").cast(pl.Datetime("us")),
                    )
                ).dt.total_seconds()
                / 86400
            )
            .clip(lower_bound=0.0)
            .alias("daily_vessel_days"),
        )
        .group_by(["port_name", "ref_date"])
        .agg(
            pl.col("mmsi").n_unique().alias("daily_port_vessels"),
            pl.col("daily_vessel_days").sum().alias("daily_vessel_days"),
        )
    )

    all_dates = set()
    for src in [
        anchorage_daily,
        berth_daily,
        daily_arrivals,
        daily_departures,
        daily_port,
    ]:
        if src.height > 0 and "ref_date" in src.columns:
            all_dates.update(src["ref_date"].to_list())
    if not all_dates:
        return pl.DataFrame()

    date_spine = pl.DataFrame({"ref_date": sorted(all_dates)}).join(
        pl.DataFrame({"port_name": list(PORT_BBOXES.keys())}),
        how="cross",
    )

    daily = (
        date_spine.join(anchorage_daily, on=["port_name", "ref_date"], how="left")
        .join(berth_daily, on=["port_name", "ref_date"], how="left")
        .join(daily_arrivals, on=["port_name", "ref_date"], how="left")
        .join(daily_departures, on=["port_name", "ref_date"], how="left")
        .join(daily_port, on=["port_name", "ref_date"], how="left")
    )

    daily_fill = [
        "daily_anch_hours",
        "daily_anch_vessels",
        "daily_berth_hours",
        "daily_berth_vessels",
        "daily_arrivals",
        "daily_departures",
        "daily_port_vessels",
        "daily_vessel_days",
    ]
    daily = daily.with_columns(
        [pl.col(c).fill_null(0).alias(c) for c in daily_fill]
    ).sort(["port_name", "ref_date"])

    w = window_days
    daily = daily.with_columns(
        (
            pl.col("daily_anch_hours").rolling_sum(w, min_samples=1).over("port_name")
            / hours_per_window
        ).alias("feat_mean_queue_length"),
        pl.col("daily_anch_vessels")
        .rolling_max(w, min_samples=1)
        .over("port_name")
        .alias("feat_max_queue_length"),
        (
            pl.col("daily_berth_hours").rolling_sum(w, min_samples=1).over("port_name")
            / hours_per_window
        ).alias("feat_mean_berth_count"),
        pl.col("daily_berth_vessels")
        .rolling_max(w, min_samples=1)
        .over("port_name")
        .alias("feat_max_berth_count"),
        pl.col("daily_arrivals")
        .rolling_sum(w, min_samples=1)
        .over("port_name")
        .alias("feat_arrival_count"),
        pl.col("daily_departures")
        .rolling_sum(w, min_samples=1)
        .over("port_name")
        .alias("feat_departure_count"),
        (
            pl.col("daily_vessel_days").rolling_sum(w, min_samples=1).over("port_name")
            / float(w)
        ).alias("feat_mean_port_vessels"),
    )

    daily = daily.with_columns(
        pl.when(pl.col("feat_arrival_count") > 0)
        .then(pl.col("feat_departure_count") / pl.col("feat_arrival_count"))
        .otherwise(pl.lit(None))
        .alias("feat_service_rate"),
    )

    snapshot_cols = [
        "port_name",
        "ref_date",
        "feat_mean_queue_length",
        "feat_max_queue_length",
        "feat_mean_berth_count",
        "feat_max_berth_count",
        "feat_arrival_count",
        "feat_departure_count",
        "feat_mean_port_vessels",
        "feat_service_rate",
    ]
    return daily.select(snapshot_cols).sort(["port_name", "ref_date"])


def merge_daily_features(
    visit_metrics: pl.DataFrame,
    snapshot_metrics: pl.DataFrame,
) -> pl.DataFrame:
    """Join visit-based and snapshot-based daily metrics.

    Args:
        visit_metrics: Daily rolling visit aggregations (keyed on ref_date).
        snapshot_metrics: Daily rolling snapshot aggregations (keyed on ref_date).

    Returns:
        Combined daily feature DataFrame per port.
    """
    daily = snapshot_metrics.join(
        visit_metrics,
        on=["port_name", "ref_date"],
        how="left",
    )

    visit_cols = [
        c for c in visit_metrics.columns if c not in ("port_name", "ref_date")
    ]
    daily = daily.with_columns([pl.col(c).fill_null(0).alias(c) for c in visit_cols])

    daily = daily.with_columns(
        pl.when(pl.col("feat_mean_berth_count") > 0)
        .then(pl.col("feat_mean_queue_length") / pl.col("feat_mean_berth_count"))
        .otherwise(pl.lit(0.0))
        .alias("feat_queue_to_berth_ratio"),
    )

    return daily.sort(["port_name", "ref_date"])


def build_features_for_split(
    visits: pl.DataFrame,
    stop_events: pl.DataFrame,
    lf: pl.LazyFrame,
    split_name: str,
    ship_type_filter: str | None,
    threshold_hours: float = CONGESTION_THRESHOLD_HOURS_CONTAINER,
) -> pl.DataFrame:
    """Run stages 5-7 for a single ship-type split.

    Filters visits and stop_events by ship_type_group (or keeps all if
    filter is None), then aggregates, engineers features, and assembles
    the final dataset.

    Args:
        visits: Full visit DataFrame (all ship types).
        stop_events: Full stop events DataFrame (all ship types).
        lf: Full ping LazyFrame (for snapshot metrics).
        split_name: Label for logging (e.g. "bulk_carrier").
        ship_type_filter: Value to filter ship_type_group, or None for all.

    Returns:
        Final daily feature DataFrame for this split.
    """
    logger.info(f"--- Building features for split: {split_name} ---")

    if ship_type_filter is not None:
        split_visits = visits.filter(
            pl.col("ship_type_group") == ship_type_filter
        )
        split_stops = stop_events.filter(
            pl.col("ship_type_group") == ship_type_filter
        )
        split_lf = lf.filter(pl.col("ship_type_group") == ship_type_filter)
    else:
        split_visits = visits
        split_stops = stop_events
        split_lf = lf

    logger.info(
        f"  {split_name}: {split_visits.height:,} visits, "
        f"{split_stops.height:,} stop events"
    )

    visit_metrics = compute_rolling_visit_metrics(split_visits)
    snapshot_metrics = compute_rolling_snapshot_metrics(split_stops, split_lf)
    daily = merge_daily_features(visit_metrics, snapshot_metrics)
    logger.info(f"  {split_name}: {daily.height:,} daily rows")

    daily = daily.sort(["port_name", "ref_date"])
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)
    daily = add_rate_of_change(daily)
    daily = add_regime_indicators(daily, threshold_hours=threshold_hours)
    daily = add_temporal_features(daily)
    daily = define_targets(daily)

    daily = assemble_final_dataset(daily)
    logger.info(f"  {split_name}: final shape {daily.shape}")

    return daily


logger.info("STAGE 5-7: Building features for each ship-type split")

SPLIT_THRESHOLDS: dict[str, float] = {
    "bulk_carrier": CONGESTION_THRESHOLD_HOURS_BULK,
    "container_ship": CONGESTION_THRESHOLD_HOURS_CONTAINER,
}

split_results: dict[str, pl.DataFrame] = {}
for split_name, ship_type_filter in SHIP_TYPE_SPLITS.items():
    split_results[split_name] = build_features_for_split(
        visits, stop_events, lf, split_name, ship_type_filter,
        threshold_hours=SPLIT_THRESHOLDS[split_name],
    )

daily = split_results["container_ship"]

for split_name, split_df in split_results.items():
    print(f"\n{split_name}: {split_df.height:,} rows, {split_df.width} columns")
    print(
        split_df.group_by("port_name")
        .agg(
            pl.len().alias("days"),
            pl.col("ref_date").min().alias("first_date"),
            pl.col("ref_date").max().alias("last_date"),
        )
        .sort("port_name")
    )


def add_lag_features(
    daily: pl.DataFrame,
    lag_weeks: list[int] | None = None,
) -> pl.DataFrame:
    """Add lagged features per port.

    Lag offsets are specified in weeks but applied as day-shifts
    (lag_weeks * 7) since rows are daily.

    Literature: Peng et al. (2023), Zhang et al. (2024)

    Args:
        daily: Daily feature DataFrame sorted by port + ref_date.
        lag_weeks: Lag offsets in weeks. Defaults to [1, 2, 4].

    Returns:
        DataFrame with lag feature columns added.
    """
    if lag_weeks is None:
        lag_weeks = [1, 2, 4]

    lag_metrics = [
        "feat_p95_waiting_hours",
        "feat_mean_waiting_hours",
        "feat_mean_queue_length",
        "feat_n_visits",
        "feat_mean_berth_count",
        "feat_service_rate",
    ]

    new_cols = []
    for metric in lag_metrics:
        for lw in lag_weeks:
            new_cols.append(
                pl.col(metric)
                .shift(lw * 7)
                .over("port_name")
                .alias(f"{metric}_lag{lw}w")
            )

    return daily.with_columns(new_cols)


def add_rolling_features(
    daily: pl.DataFrame,
    window_weeks: list[int] | None = None,
) -> pl.DataFrame:
    """Add rolling window features per port.

    Rolling mean captures trend; rolling std captures volatility.
    All windows are shifted by 1 day to exclude the current observation.
    Window sizes are specified in weeks but applied as days.

    Literature: Peng et al. (2023), Zhang et al. (2024)
    Novel: rolling std as congestion stability signal.

    Args:
        daily: Daily feature DataFrame sorted by port + ref_date.
        window_weeks: Rolling window sizes in weeks. Defaults to [4, 8].

    Returns:
        DataFrame with rolling feature columns added.
    """
    if window_weeks is None:
        window_weeks = [4, 8]

    roll_metrics = [
        "feat_p95_waiting_hours",
        "feat_mean_queue_length",
        "feat_n_visits",
        "feat_service_rate",
    ]

    new_cols = []
    for metric in roll_metrics:
        for ww in window_weeks:
            wd = ww * 7
            new_cols.append(
                pl.col(metric)
                .rolling_mean(window_size=wd)
                .shift(1)
                .over("port_name")
                .alias(f"{metric}_roll{ww}w_mean")
            )
            new_cols.append(
                pl.col(metric)
                .rolling_std(window_size=wd)
                .shift(1)
                .over("port_name")
                .alias(f"{metric}_roll{ww}w_std")
            )

    return daily.with_columns(new_cols)


def add_rate_of_change(daily: pl.DataFrame) -> pl.DataFrame:
    """Add week-over-week rate of change features.

    Compares current value to 7 days ago (1 week).

    Literature: Zhang et al. (2024)

    Args:
        daily: Daily feature DataFrame.

    Returns:
        DataFrame with change features added.
    """
    return daily.with_columns(
        (pl.col("feat_p95_waiting_hours") - pl.col("feat_p95_waiting_hours").shift(7))
        .over("port_name")
        .alias("feat_p95_waiting_change_1w"),
        (pl.col("feat_mean_queue_length") - pl.col("feat_mean_queue_length").shift(7))
        .over("port_name")
        .alias("feat_queue_length_change_1w"),
    )


def add_regime_indicators(
    daily: pl.DataFrame,
    threshold_hours: float = CONGESTION_THRESHOLD_HOURS_CONTAINER,
) -> pl.DataFrame:
    """Add congestion regime persistence indicators.

    With daily rows, consecutive_high_days counts consecutive days
    where the rolling 7-day p95 exceeds the threshold.

    Literature: Peng et al. (2023), Nakashima et al. (2024)
    Novel: waiting_trend_4w as directional momentum signal.

    Args:
        daily: Daily feature DataFrame.
        threshold_hours: Operational threshold in hours.

    Returns:
        DataFrame with regime indicator columns.
    """
    daily = daily.with_columns(
        (pl.col("feat_p95_waiting_hours") > threshold_hours)
        .cast(pl.Int32)
        .alias("_is_high"),
    )

    daily = (
        daily.sort(["port_name", "ref_date"])
        .with_columns(
            (
                (pl.col("_is_high") != pl.col("_is_high").shift(1))
                .fill_null(True)
                .cum_sum()
                .over("port_name")
            ).alias("_streak_group"),
        )
        .with_columns(
            pl.when(pl.col("_is_high") == 1)
            .then(pl.col("_is_high").cum_sum().over(["port_name", "_streak_group"]))
            .otherwise(0)
            .alias("feat_consecutive_high_days"),
        )
    )

    daily = daily.with_columns(
        pl.col("_is_high")
        .rolling_sum(window_size=28, min_samples=1)
        .shift(1)
        .over("port_name")
        .cast(pl.Float64)
        .truediv(28.0)
        .alias("feat_high_pct_4w"),
    )

    daily = daily.with_columns(
        (
            (
                pl.col("feat_p95_waiting_hours")
                - pl.col("feat_p95_waiting_hours").shift(28).over("port_name")
            )
            / 4.0
        ).alias("feat_waiting_trend_4w"),
    )

    return daily.drop(["_is_high", "_streak_group"])


def add_temporal_features(daily: pl.DataFrame) -> pl.DataFrame:
    """Add calendar features from ref_date.

    Args:
        daily: Daily feature DataFrame.

    Returns:
        DataFrame with temporal feature columns.
    """
    return daily.with_columns(
        pl.col("ref_date").dt.week().alias("feat_week_of_year"),
        pl.col("ref_date").dt.month().alias("feat_month"),
        pl.col("ref_date").dt.quarter().alias("feat_quarter"),
        pl.col("ref_date").dt.weekday().alias("feat_day_of_week"),
    )


def define_targets(
    daily: pl.DataFrame,
    horizons: list[int] = FORECAST_HORIZONS,
) -> pl.DataFrame:
    """Define forward-looking target variables.

    CRITICAL: This is the ONLY place where future data is used.
    No feature column may reference future data.

    For horizon h (in weeks): the target is the rolling 7-day p95
    waiting time observed h*7 days into the future. Since rows are
    daily and feat_p95_waiting_hours at ref_date D+h*7 covers
    [D+h*7-6, D+h*7], shift(-h*7) gives the correct future window.

    Only the continuous p95 waiting time target is defined here.
    Risk probability (P(p95 > threshold)) is derived post-prediction
    by thresholding the severity forecast.

    Args:
        daily: Daily feature DataFrame.
        horizons: Forecast horizons in weeks.

    Returns:
        DataFrame with target columns added.
    """
    new_cols = []
    for h in horizons:
        new_cols.append(
            pl.col("feat_p95_waiting_hours")
            .shift(-h * 7)
            .over("port_name")
            .alias(f"target_p95_waiting_{h}w")
        )

    return daily.with_columns(new_cols)


def assemble_final_dataset(daily: pl.DataFrame) -> pl.DataFrame:
    """Drop rows with null targets, validate, and prepare for output.

    Args:
        daily: Full daily feature DataFrame with targets.

    Returns:
        Clean final training DataFrame.
    """
    target_col = "target_p95_waiting_1w"
    if target_col in daily.columns:
        final = daily.filter(pl.col(target_col).is_not_null())
    else:
        final = daily

    n_dropped = daily.height - final.height
    logger.info(
        f"Dropped {n_dropped} rows with null targets "
        f"({n_dropped / daily.height * 100:.1f}%)"
    )

    roll_cols = [c for c in final.columns if "_roll" in c]
    if roll_cols:
        warmup_mask = pl.any_horizontal([pl.col(c).is_null() for c in roll_cols])
        n_warmup = final.filter(warmup_mask).height
        final = final.filter(~warmup_mask)
        logger.info(
            f"  Warmup rows dropped (rolling window not yet populated): {n_warmup}"
        )

    feat_cols = [c for c in final.columns if c.startswith("feat_")]
    for col in feat_cols:
        n_null = final[col].null_count()
        if n_null > 0:
            logger.warning(f"  Feature {col} has {n_null} nulls — filling with 0")
            final = final.with_columns(pl.col(col).fill_null(0))

    return final


FEATURES_DIR = OUTPUT_PATH.parent

SPLIT_OUTPUT_NAMES: dict[str, str] = {
    "bulk_carrier": "daily_features_bulk_carrier",
    "container_ship": "daily_features_container",
}

for split_name, split_df in split_results.items():
    out_name = SPLIT_OUTPUT_NAMES[split_name]
    parquet_path = FEATURES_DIR / f"{out_name}.parquet"
    csv_path = FEATURES_DIR / f"{out_name}.csv"

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    split_df.write_parquet(parquet_path, compression="zstd")
    split_df.write_csv(csv_path)

    print(f"\n{split_name}: {split_df.shape}")
    print(f"  Parquet: {parquet_path}")
    print(f"  CSV:     {csv_path}")

logger.info(f"All splits written to {FEATURES_DIR}")

feat_cols = sorted([c for c in daily.columns if c.startswith("feat_")])
print(f"\nAll {len(feat_cols)} features:")
for c in feat_cols:
    print(f"  {c}")


def plot_daily_p95(
    daily: pl.DataFrame,
    images_dir: Path | None = None,
) -> None:
    """Plot rolling 7-day p95 waiting time per port over time."""
    if images_dir is None:
        images_dir = Path(__file__).resolve().parent.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    port_colors = {
        "LOS_ANGELES_LONG_BEACH": "#FF6B6B",
        "NEW_YORK_NEW_JERSEY": "#4ECDC4",
        "HOUSTON": "#FFE66D",
        "PORT_OF_VIRGINIA": "#96CEB4",
    }

    for port in daily["port_name"].unique().sort().to_list():
        port_data = daily.filter(pl.col("port_name") == port).sort("ref_date")
        color = port_colors.get(port, "gray")
        ax.plot(
            port_data["ref_date"].to_list(),
            port_data["feat_p95_waiting_hours"].to_list(),
            label=port.replace("_", " "),
            color=color,
            linewidth=1.0,
            alpha=0.7,
        )

    ax.axhline(
        CONGESTION_THRESHOLD_HOURS_CONTAINER,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"Threshold ({CONGESTION_THRESHOLD_HOURS_CONTAINER}h)",
    )

    ax.set_title(
        "Rolling 7-Day p95 Waiting Time by Port",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("p95 Waiting Time (hours)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = images_dir / "daily_p95_waiting_time.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {plot_path}")
    plt.close()


base_images_dir = Path(__file__).resolve().parent.parent / "images"
for split_name, split_df in split_results.items():
    split_images = base_images_dir / split_name
    plot_daily_p95(split_df, images_dir=split_images)
    print(f"  {split_name}: p95 plot saved")

for split_name, ship_type_filter in SHIP_TYPE_SPLITS.items():
    split_images = base_images_dir / split_name
    if ship_type_filter is not None:
        split_visits = visits.filter(
            pl.col("ship_type_group") == ship_type_filter
        )
        split_stops = stop_events.filter(
            pl.col("ship_type_group") == ship_type_filter
        )
    else:
        split_visits = visits
        split_stops = stop_events
    plot_waiting_histograms(split_visits, images_dir=split_images)
    plot_zone_map(split_stops, PORT_BBOXES, images_dir=split_images)
    print(f"  {split_name}: histogram + zone map saved")


def run_feature_pipeline(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Execute the full feature engineering pipeline end-to-end.

    Stages 0-4 run once on all data. Stages 5-7 run per ship-type split,
    producing separate output files and images for each.

    Args:
        input_path: Path to standardized AIS parquet.
        output_path: Base output path (split suffix appended automatically).
    """
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)

    logger.info("Stage 0: Loading and filtering...")
    lf_prod = pl.scan_parquet(input_path).select(REQUIRED_COLUMNS)
    lf_prod = filter_static_infrastructure(lf_prod)

    logger.info("Stage 1: Movement state & stop events...")
    lf_prod = label_movement_state(lf_prod)
    stop_events_prod = detect_stop_events(lf_prod)
    logger.info(f"  Stop events: {stop_events_prod.height:,}")

    logger.info("Stage 2: Zone classification...")
    stop_events_prod = classify_stop_zones(stop_events_prod)
    zone_counts = (
        stop_events_prod.group_by("zone_type")
        .agg(pl.len().alias("n"))
        .sort("zone_type")
    )
    logger.info(f"  Zone counts: {zone_counts.to_dicts()}")

    logger.info("Stage 4: Vessel visit detection...")
    visits_prod = detect_vessel_visits(stop_events_prod)
    logger.info(f"  Visits: {visits_prod.height:,}")

    features_dir = output_path.parent
    images_base = Path(input_path).resolve().parent.parent / "images"

    for split_name, ship_type_filter in SHIP_TYPE_SPLITS.items():
        split_final = build_features_for_split(
            visits_prod, stop_events_prod, lf_prod,
            split_name, ship_type_filter,
            threshold_hours=SPLIT_THRESHOLDS[split_name],
        )

        out_name = SPLIT_OUTPUT_NAMES[split_name]
        parquet_path = features_dir / f"{out_name}.parquet"
        csv_path = features_dir / f"{out_name}.csv"

        features_dir.mkdir(parents=True, exist_ok=True)
        split_final.write_parquet(parquet_path, compression="zstd")
        split_final.write_csv(csv_path)
        logger.info(f"  {split_name}: {parquet_path} ({split_final.shape})")

        split_images = images_base / split_name
        plot_daily_p95(split_final, images_dir=split_images)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


run_feature_pipeline()
