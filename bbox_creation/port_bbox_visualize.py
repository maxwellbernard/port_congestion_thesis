import csv
from pathlib import Path

import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pyproj import Transformer

from port_bbox import PORT_REGIONS, get_combined_port_bbox, union_bboxes


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "images" / "raw_bbox"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WGS84_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

AIS_BUFFER_DEG: float = 0.05

ANCHORAGE_COLOR = "#FF8C00"
PORT_COLOR = "#00BFFF"
RAW_BBOX_COLOR = "#FF2222"
AIS_BBOX_COLOR = "#00FF88"


def to_mercator(lon: float, lat: float) -> tuple[float, float]:
    """Convert (lon, lat) WGS84 to (x, y) Web Mercator."""
    x, y = WGS84_TO_MERCATOR.transform(lon, lat)
    return x, y


def bbox_to_mercator(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert (s, w, n, e) WGS84 bbox to (xmin, ymin, xmax, ymax) Mercator."""
    s, w, n, e = bbox
    xmin, ymin = to_mercator(w, s)
    xmax, ymax = to_mercator(e, n)
    return xmin, ymin, xmax, ymax


def add_filled_patch(
    ax: plt.Axes,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    color: str,
    fill_alpha: float = 0.20,
    edge_alpha: float = 1.0,
    linewidth: float = 1.2,
    linestyle: str = "solid",
    label: str | None = None,
) -> None:
    """Draw a filled rectangle with a distinct solid edge on ax.

    Args:
        ax: Matplotlib axes.
        xmin, ymin, xmax, ymax: Mercator coordinates.
        color: Hex color for both fill and edge.
        fill_alpha: Opacity of the fill (0–1).
        edge_alpha: Opacity of the edge (0–1).
        linewidth: Edge line width.
        linestyle: Edge line style.
        label: Legend label (only set on first patch per category).
    """
    ax.add_patch(mpatches.FancyBboxPatch(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        boxstyle="square,pad=0",
        linewidth=0,
        edgecolor="none",
        facecolor=color,
        alpha=fill_alpha,
        label=label,
    ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        boxstyle="square,pad=0",
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        alpha=edge_alpha,
        linestyle=linestyle,
    ))


def add_outline_patch(
    ax: plt.Axes,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    color: str,
    linewidth: float = 2.2,
    linestyle: str = "--",
    label: str | None = None,
) -> None:
    """Draw an outline-only rectangle on ax."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        boxstyle="square,pad=0",
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        label=label,
    ))


def apply_buffer(
    bbox: tuple[float, float, float, float],
    buffer_deg: float = AIS_BUFFER_DEG,
) -> tuple[float, float, float, float]:
    """Expand (s, w, n, e) bbox by buffer_deg on every side."""
    s, w, n, e = bbox
    return (s - buffer_deg, w - buffer_deg, n + buffer_deg, e + buffer_deg)


def truncate(text: str, max_len: int = 22) -> str:
    """Truncate label text for display."""
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def derive_sub_bboxes(
    data: dict,
    search_bbox: tuple[float, float, float, float],
    use_search_bbox_floor: bool = False,
) -> dict[str, tuple | None]:
    """
    Derive anchorage-only, berthing-only, and buffered AIS bboxes from features.

    Args:
        data: Output dict from get_combined_port_bbox.
        search_bbox: Original (s, w, n, e) search area passed to get_combined_port_bbox.
        use_search_bbox_floor: If True, union the feature bbox with the search area
            before buffering. Use for ports where NOAA/OSM return no offshore features
            but the search area deliberately covers an offshore queueing zone.
            For ports with good ENC coverage, leave False to avoid inflating the AIS bbox.

    Returns:
        Dict with anchorage_bbox, berthing_bbox, ais_bbox — each (s,w,n,e) or None.
    """
    anch_bboxes = [
        f["bbox"] for f in data["features"]
        if "NOAA" in f["source"] and f.get("bbox")
    ]
    berth_bboxes = [
        f["bbox"] for f in data["features"]
        if "OSM" in f["source"] and f.get("bbox")
    ]
    anchorage_bbox = union_bboxes(anch_bboxes) if anch_bboxes else None
    berthing_bbox = union_bboxes(berth_bboxes) if berth_bboxes else None

    candidates = [data["combined_bbox"]]
    if use_search_bbox_floor:
        candidates.append(search_bbox)
    candidates = [b for b in candidates if b is not None]
    ais_base = union_bboxes(candidates) if candidates else None
    ais_bbox = apply_buffer(ais_base) if ais_base else None

    return {
        "anchorage_bbox": anchorage_bbox,
        "berthing_bbox": berthing_bbox,
        "ais_bbox": ais_bbox,
    }


def visualise_port(
    region_name: str,
    data: dict,
    sub_bboxes: dict,
    padding_fraction: float = 0.15,
    dpi: int = 180,
) -> Path:
    """
    Plot satellite basemap with anchorage / port / AIS bboxes for one region.

    Args:
        region_name: Human-readable port region label.
        data: Output dict from get_combined_port_bbox.
        sub_bboxes: Output dict from derive_sub_bboxes.
        padding_fraction: Fractional padding added around the AIS bbox for display.
        dpi: Output image resolution.

    Returns:
        Path to the saved PNG file.

    Raises:
        ValueError: If no combined bbox was found.
    """
    ais_bbox = sub_bboxes["ais_bbox"]
    combined_bbox = data["combined_bbox"]
    if ais_bbox is None:
        raise ValueError(f"No features found for {region_name} — cannot plot.")

    xmin, ymin, xmax, ymax = bbox_to_mercator(ais_bbox)
    pad_x = (xmax - xmin) * padding_fraction
    pad_y = (ymax - ymin) * padding_fraction

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    ctx.add_basemap(
        ax,
        source=ctx.providers.USGS.USImageryTopo,
        attribution_size=7,
    )

    for feat in data["features"]:
        bbox = feat.get("bbox")
        if not bbox:
            continue
        is_noaa = "NOAA" in feat["source"]
        color = ANCHORAGE_COLOR if is_noaa else PORT_COLOR
        fx1, fy1, fx2, fy2 = bbox_to_mercator(bbox)
        add_outline_patch(ax, fx1, fy1, fx2, fy2, color=color, linewidth=0.6, linestyle="solid")

        name = feat.get("name", "")
        width_deg = combined_bbox[3] - combined_bbox[1]
        feat_width = bbox[3] - bbox[1]
        if name and name != "unnamed" and feat_width > (width_deg * 0.05):
            cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
            ax.text(
                cx, cy, truncate(name),
                ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none"),
            )

    if sub_bboxes["anchorage_bbox"]:
        ax1, ay1, ax2, ay2 = bbox_to_mercator(sub_bboxes["anchorage_bbox"])
        add_filled_patch(
            ax, ax1, ay1, ax2, ay2,
            color=ANCHORAGE_COLOR, fill_alpha=0.18, edge_alpha=1.0, linewidth=2.0,
            label="Anchorage zone (NOAA ENC)",
        )

    if sub_bboxes["berthing_bbox"]:
        bx1, by1, bx2, by2 = bbox_to_mercator(sub_bboxes["berthing_bbox"])
        add_filled_patch(
            ax, bx1, by1, bx2, by2,
            color=PORT_COLOR, fill_alpha=0.18, edge_alpha=1.0, linewidth=2.0,
            label="Berthing zone (OSM)",
        )

    rx1, ry1, rx2, ry2 = bbox_to_mercator(combined_bbox)
    add_outline_patch(
        ax, rx1, ry1, rx2, ry2,
        color=RAW_BBOX_COLOR, linewidth=1.8, linestyle="--",
        label="Raw feature bbox",
    )

    add_outline_patch(
        ax, xmin, ymin, xmax, ymax,
        color=AIS_BBOX_COLOR, linewidth=2.5, linestyle="--",
        label=f"AIS filter bbox (+{AIS_BUFFER_DEG}° buffer)",
    )

    ax.legend(
        loc="lower left", fontsize=8, framealpha=0.75,
        facecolor="#111111", labelcolor="white", edgecolor="#555555",
    )

    n_anch = sum(1 for f in data["features"] if "NOAA" in f["source"])
    n_osm = sum(1 for f in data["features"] if "OSM" in f["source"])
    s, w, n, e = ais_bbox

    ax.set_title(
        f"{region_name}  —  Port & Anchorage Coverage\n"
        f"OSM port features: {n_osm}   |   NOAA ENC anchorage features: {n_anch}\n"
        f"AIS filter bbox  (S {s:.4f}, W {w:.4f}, N {n:.4f}, E {e:.4f})",
        fontsize=10, color="white", pad=10,
    )
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_axis_off()
    plt.tight_layout(pad=0.5)

    safe_name = region_name.replace("/", "_").replace(" ", "_")
    out_path = OUTPUT_DIR / f"port_bbox_{safe_name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


def save_bbox_csv(records: list[dict], out_path: Path) -> None:
    """
    Write per-port bbox summary to CSV.

    Each bbox column is formatted as (xMIN, yMIN, xMAX, yMAX) i.e. (W, S, E, N).

    Args:
        records: List of dicts with port name and bbox fields.
        out_path: Destination CSV path.
    """
    fieldnames = [
        "port",
        "berthing_zone", "anchorage_zone", "ais_bbox", "adjusted_ais_bbox",
        "lon_min", "lat_min", "lon_max", "lat_max",
    ]

    def fmt(bbox: tuple | None) -> str:
        if bbox is None:
            return ""
        s, w, n, e = bbox
        return f"({round(w, 6)}, {round(s, 6)}, {round(e, 6)}, {round(n, 6)})"

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            ais = rec["ais_bbox"]
            s, w, n, e = ais if ais else (None, None, None, None)
            writer.writerow({
                "port":              rec["port"],
                "berthing_zone":     fmt(rec["berthing_bbox"]),
                "anchorage_zone":    fmt(rec["anchorage_bbox"]),
                "ais_bbox":          fmt(rec["raw_bbox"]),
                "adjusted_ais_bbox": fmt(ais),
                "lon_min": round(w, 6) if w is not None else "",
                "lat_min": round(s, 6) if s is not None else "",
                "lon_max": round(e, 6) if e is not None else "",
                "lat_max": round(n, 6) if n is not None else "",
            })

    print(f"  CSV saved → {out_path}")


if __name__ == "__main__":
    import time

    saved_images: list[Path] = []
    csv_records: list[dict] = []

    for region, search_bbox in PORT_REGIONS.items():
        print(f"\n{'=' * 60}")
        print(f"Fetching data for: {region}")
        time.sleep(5)
        data = get_combined_port_bbox(region, search_bbox)

        if data["combined_bbox"] is None:
            print("  Skipping — no features found.")
            continue

        use_floor = False
        sub = derive_sub_bboxes(data, search_bbox, use_search_bbox_floor=use_floor)
        path = visualise_port(region, data, sub)
        saved_images.append(path)

        csv_records.append({
            "port":           region,
            "anchorage_bbox": sub["anchorage_bbox"],
            "berthing_bbox":  sub["berthing_bbox"],
            "raw_bbox":       data["combined_bbox"],
            "ais_bbox":       sub["ais_bbox"],
        })

    csv_path = OUTPUT_DIR / "port_bboxes.csv"
    save_bbox_csv(csv_records, csv_path)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(saved_images)} image(s) + CSV saved to {OUTPUT_DIR}/")
