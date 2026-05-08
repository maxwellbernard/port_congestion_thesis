import requests
from typing import Optional

ENC_BASE_URL = "https://encdirect.noaa.gov/arcgis/rest/services/encdirect/enc_harbour/MapServer"

ANCHORAGE_AREA_LAYER = 186
ANCHOR_BERTH_LAYER = 187

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

PORT_REGIONS = {
    "LA/Long Beach":        (33.55, -118.40, 33.85, -118.00),
    "New York/New Jersey":  (40.30, -74.40,  40.70, -73.95),
    "Houston":              (29.08, -95.20,  29.90, -94.30),
    "Port of Virginia":     (36.70, -76.60,  37.20, -75.90),
}


def query_enc_layer(
    layer_id: int,
    bbox: tuple[float, float, float, float],
    timeout: int = 60,
) -> list[dict]:
    """
    Query a NOAA ENC MapServer layer for features within a bounding box.

    Args:
        layer_id: ArcGIS layer ID (e.g. 186 for Anchorage_Area).
        bbox: (south, west, north, east) in decimal degrees.
        timeout: Request timeout in seconds.

    Returns:
        List of GeoJSON feature dicts.
    """
    s, w, n, e = bbox
    params = {
        "f": "geojson",
        "geometry": f"{w},{s},{e},{n}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
    }
    url = f"{ENC_BASE_URL}/{layer_id}/query"
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("features", [])


def bbox_from_geojson_feature(feature: dict) -> Optional[tuple]:
    """
    Derive (south, west, north, east) bbox from a GeoJSON feature's geometry.

    Args:
        feature: GeoJSON feature dict with a geometry field.

    Returns:
        (south, west, north, east) tuple, or None if geometry is missing.
    """
    geom = feature.get("geometry")
    if not geom:
        return None

    def flatten_coords(coords):
        """Recursively flatten nested coordinate lists."""
        if isinstance(coords[0], (int, float)):
            yield coords
        else:
            for item in coords:
                yield from flatten_coords(item)

    coords = list(flatten_coords(geom["coordinates"]))
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lats), min(lons), max(lats), max(lons))


def get_noaa_anchorages(
    region_name: str,
    search_bbox: tuple[float, float, float, float],
) -> list[dict]:
    """
    Query NOAA ENC for anchorage area polygons within a port region.

    Args:
        region_name: Human-readable port region label.
        search_bbox: (south, west, north, east) search area.

    Returns:
        List of anchorage dicts with name and bbox.
    """
    print(f"\n  [NOAA ENC] Querying anchorages for {region_name}...")
    results = []

    for layer_id, layer_label in [
        (ANCHORAGE_AREA_LAYER, "Anchorage_Area"),
        (ANCHOR_BERTH_LAYER, "Anchor_Berth"),
    ]:
        try:
            features = query_enc_layer(layer_id, search_bbox)
            for feat in features:
                props = feat.get("properties", {})
                name = props.get("OBJNAM") or props.get("INFORM") or "unnamed"
                bbox = bbox_from_geojson_feature(feat)
                if bbox:
                    results.append({
                        "name": name,
                        "source": f"NOAA ENC {layer_label}",
                        "bbox": bbox,
                        "properties": props,
                    })
        except Exception as exc:
            print(f"    WARNING: layer {layer_id} query failed: {exc}")

    print(f"  Found {len(results)} anchorage features")
    return results


def build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    """
    Build Overpass QL query for port and anchorage polygons.

    Args:
        bbox: (south, west, north, east) in decimal degrees.
    """
    s, w, n, e = bbox
    return f"""
    [out:json][timeout:60];
    (
      way["landuse"="harbour"]({s},{w},{n},{e});
      way["industrial"="port"]({s},{w},{n},{e});
      way["seamark:type"="anchorage"]({s},{w},{n},{e});
      way["landuse"="anchorage"]({s},{w},{n},{e});
      relation["landuse"="harbour"]({s},{w},{n},{e});
      relation["industrial"="port"]({s},{w},{n},{e});
      relation["seamark:type"="anchorage"]({s},{w},{n},{e});
      relation["landuse"="anchorage"]({s},{w},{n},{e});
    );
    out geom;
    """


def get_osm_features(
    region_name: str,
    search_bbox: tuple[float, float, float, float],
    retries: int = 4,
    backoff: float = 15.0,
) -> list[dict]:
    """
    Query Overpass for OSM port/anchorage polygons within a region.

    Args:
        region_name: Human-readable port region label.
        search_bbox: (south, west, north, east) search area.
        retries: Number of retry attempts on 429/504 errors.
        backoff: Seconds to wait before each retry (doubles each attempt).

    Returns:
        List of feature dicts with name, source, and bbox.
    """
    import time

    print(f"\n  [OSM] Querying port/anchorage features for {region_name}...")
    query = build_overpass_query(search_bbox)

    wait = backoff
    for attempt in range(1, retries + 1):
        response = requests.post(OVERPASS_URL, data={"data": query}, timeout=90)
        if response.status_code in (429, 504) and attempt < retries:
            print(f"    Overpass {response.status_code} — retrying in {wait:.0f}s (attempt {attempt}/{retries})")
            time.sleep(wait)
            wait *= 2
            continue
        response.raise_for_status()
        break

    elements = response.json().get("elements", [])

    results = []
    for el in elements:
        if "bounds" not in el:
            continue
        b = el["bounds"]
        tags = el.get("tags", {})
        name = tags.get("name", tags.get("name:en", "unnamed"))
        results.append({
            "name": name,
            "source": f"OSM {el['type']}/{el['id']}",
            "bbox": (b["minlat"], b["minlon"], b["maxlat"], b["maxlon"]),
            "tags": tags,
        })

    print(f"  Found {len(results)} OSM features")
    return results


def union_bboxes(bboxes: list[tuple]) -> tuple:
    """
    Merge multiple (s, w, n, e) bboxes into a single covering bbox.

    Args:
        bboxes: List of (south, west, north, east) tuples.

    Returns:
        Single (south, west, north, east) covering all inputs.
    """
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


def get_combined_port_bbox(
    region_name: str,
    search_bbox: tuple[float, float, float, float],
) -> dict:
    """
    Combine OSM port polygons and NOAA ENC anchorage areas into a single bbox.

    Args:
        region_name: Human-readable port region label.
        search_bbox: (south, west, north, east) search area.

    Returns:
        Dict with all features and a single combined bbox.
    """
    print(f"\n{'=' * 60}")
    print(f"Port region: {region_name}")

    osm_features = get_osm_features(region_name, search_bbox)
    enc_features = get_noaa_anchorages(region_name, search_bbox)

    all_features = osm_features + enc_features
    all_bboxes = [f["bbox"] for f in all_features if f.get("bbox")]

    if not all_bboxes:
        print("  WARNING: No features found — widen search_bbox")
        return {"region": region_name, "features": [], "combined_bbox": None}

    combined = union_bboxes(all_bboxes)
    print(f"\n  --> Combined bbox (s,w,n,e): {combined}")
    return {
        "region": region_name,
        "features": all_features,
        "combined_bbox": combined,
    }


if __name__ == "__main__":
    results = {}
    for region, bbox in PORT_REGIONS.items():
        results[region] = get_combined_port_bbox(region, bbox)

    print(f"\n\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    for region, data in results.items():
        print(f"\n{region}:")
        print(f"  OSM features:  {sum(1 for f in data['features'] if 'OSM' in f['source'])}")
        print(f"  NOAA features: {sum(1 for f in data['features'] if 'NOAA' in f['source'])}")
        print(f"  Combined bbox: {data['combined_bbox']}")