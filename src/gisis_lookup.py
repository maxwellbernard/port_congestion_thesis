"""GISIS Ship and Company Particulars lookup pipeline.

Fetches vessel data from the IMO GISIS Ship search page by IMO number.
The GISIS search is an ASP.NET Web Forms page requiring hidden-field
round-tripping (__VIEWSTATE, __EVENTVALIDATION, etc.).

Authentication workflow:
    1. Log into GISIS manually in Chrome/Firefox/Edge
    2. Run the script — it extracts session cookies from your browser automatically
    3. Cookies are cached to ~/.gisis_cookies.json for reuse
    4. When cookies expire, just log in again in your browser and re-run

Usage:
    from src.gisis_lookup import lookup_many_imos, authenticate

    session = authenticate()  # pulls cookies from browser
    df = lookup_many_imos(
        imo_list=[9618446, 9321483],
        session=session,
        cache_path="ais_data/cleaned/gisis_cache.csv",
    )
"""

from __future__ import annotations

import html
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import browser_cookie3
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

GISIS_LIMIT = 0

GISIS_URL = "https://gisis.imo.org/Public/SHIPS/Default.aspx"
GISIS_LOGIN_URL = "https://gisis.imo.org/Public/Shared/Public/Login.aspx"
COOKIE_CACHE_PATH = Path.home() / ".gisis_cookies.json"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": GISIS_URL,
}

OUTPUT_COLUMNS = [
    "imo_number",
    "found",
    "name",
    "flag",
    "type",
    "ship_type_detailed",
    "gross_tonnage",
    "call_sign",
    "mmsi",
    "year_built",
    "source",
    "lookup_timestamp",
    "error_message",
]

REQUEST_TIMEOUT = 30

REAUTH_EVERY: int = 200


def _normalize_imo(imo: int | str) -> str:
    """Strip 'IMO' prefix and whitespace, return digits only."""
    return re.sub(r"^IMO", "", str(imo).strip(), flags=re.IGNORECASE)


def create_session(
    cookies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> requests.Session:
    """Create and configure a requests.Session for GISIS.

    Args:
        cookies: Dict of browser-authenticated cookies
            (e.g. ASP.NET_SessionId, .ASPXAUTH).
        headers: Optional extra headers; merged with defaults.

    Returns:
        Configured requests.Session.
    """
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    if headers:
        session.headers.update(headers)
    if cookies:
        session.cookies.update(cookies)
    return session


GISIS_DOMAIN = "gisis.imo.org"

_BROWSER_LOADERS = [
    ("Chrome", browser_cookie3.chrome),
    ("Firefox", browser_cookie3.firefox),
    ("Edge", browser_cookie3.edge),
    ("Safari", browser_cookie3.safari),
]


def _save_cookies(session: requests.Session, path: Path = COOKIE_CACHE_PATH) -> None:
    """Persist session cookies to a JSON file for reuse.

    Args:
        session: Session whose cookies to save.
        path: File path for the cookie cache.
    """
    cookies = dict(session.cookies)
    data = {
        "cookies": cookies,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2))
    path.chmod(0o600)
    logger.info(f"Cookies cached → {path}")


def _load_cached_cookies(path: Path = COOKIE_CACHE_PATH) -> dict[str, str] | None:
    """Load previously-cached cookies from disk.

    Args:
        path: File path for the cookie cache.

    Returns:
        Cookie dict, or None if cache missing/invalid.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cookies = data.get("cookies", {})
        saved_at = data.get("saved_at", "")
        logger.info(f"Loaded cached cookies (saved {saved_at})")
        return cookies
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Cookie cache at {path} is corrupt, ignoring")
        return None


def _extract_browser_cookies(
    browser: str | None = None,
) -> dict[str, str]:
    """Extract GISIS cookies from a local browser's cookie store.

    Requires that you are logged into GISIS in the browser.

    Args:
        browser: Browser name to try ("chrome", "firefox", "edge",
            "safari"). If None, tries all in order.

    Returns:
        Dict of cookie name → value for gisis.imo.org.

    Raises:
        RuntimeError: If no GISIS cookies found in any browser.
    """
    loaders = _BROWSER_LOADERS
    if browser:
        browser_lower = browser.lower()
        loaders = [(n, fn) for n, fn in _BROWSER_LOADERS if n.lower() == browser_lower]
        if not loaders:
            raise ValueError(
                f"Unknown browser '{browser}'. "
                f"Supported: {[n for n, _ in _BROWSER_LOADERS]}"
            )

    for name, loader_fn in loaders:
        try:
            cj = loader_fn(domain_name=".imo.org")
            cookies: dict[str, str] = {}
            for c in cj:
                domain = c.domain or ""
                if "imo.org" in domain:
                    logger.debug(f"  cookie: {c.name} (domain={domain})")
                    cookies[c.name] = c.value
            if cookies:
                logger.info(
                    f"Extracted {len(cookies)} cookies from {name}: "
                    f"{list(cookies.keys())}"
                )
                return cookies
            logger.debug(f"No imo.org cookies in {name}")
        except Exception as exc:
            logger.debug(f"Could not read {name} cookies: {exc}")

    raise RuntimeError(
        "No GISIS cookies found in any browser. "
        "Please log into https://gisis.imo.org in your browser first, "
        "then re-run the script."
    )


def authenticate(
    browser: str | None = None,
    cookie_cache: Path | str = COOKIE_CACHE_PATH,
    force_refresh: bool = False,
) -> requests.Session:
    """Build an authenticated session for GISIS.

    Resolution order:
        1. Try previously-cached cookies (fast, no browser access)
        2. Extract fresh cookies from browser cookie store
        3. Cache the working cookies for next time

    Args:
        browser: Specific browser to extract from ("chrome",
            "firefox", "edge", "safari"). None = try all.
        cookie_cache: Path to the cookie cache file.
        force_refresh: Skip cache, always re-extract from browser.

    Returns:
        Authenticated requests.Session.

    Raises:
        RuntimeError: If no valid cookies found anywhere.
    """
    cookie_cache = Path(cookie_cache)
    session = create_session()

    if not force_refresh:
        cached = _load_cached_cookies(cookie_cache)
        if cached:
            session.cookies.update(cached)
            try:
                fetch_search_page(session)
                logger.info("Cached cookies are valid")
                return session
            except PermissionError:
                logger.info("Cached cookies expired, extracting fresh from browser …")
                session = create_session()

    cookies = _extract_browser_cookies(browser=browser)
    session.cookies.update(cookies)

    try:
        fetch_search_page(session)
    except PermissionError:
        raise RuntimeError(
            "Extracted browser cookies but GISIS session is not authenticated. "
            "Your browser session may have expired — please log in again at "
            "https://gisis.imo.org and re-run."
        )

    logger.info("Browser cookies are valid")
    _save_cookies(session, cookie_cache)

    return session


def fetch_search_page(session: requests.Session) -> str:
    """GET the GISIS ship search page and validate authentication.

    Args:
        session: Authenticated requests.Session.

    Returns:
        HTML text of the search page.

    Raises:
        PermissionError: If redirected to login or session expired.
        requests.HTTPError: On non-200 status.
    """
    resp = session.get(GISIS_URL, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    resp.raise_for_status()

    if "Login.aspx" in resp.url or (
        "Log in" in resp.text
        and "ctl00_bodyPlaceHolder_Default_btnSearchShips" not in resp.text
    ):
        raise PermissionError(
            f"Session expired or unauthenticated — landed on {resp.url}"
        )

    if "Log in" in resp.text and "ctl00_bodyPlaceHolder_Default_btnSearchShips" not in resp.text:
        raise PermissionError(
            "Page loaded but appears unauthenticated (login prompt detected)."
        )

    return resp.text


def parse_hidden_fields(page_html: str) -> dict[str, str]:
    """Extract ASP.NET hidden form fields from page HTML.

    Args:
        page_html: Full HTML of the GISIS search page.

    Returns:
        Dict with __VIEWSTATE, __VIEWSTATEGENERATOR,
        __EVENTVALIDATION, __EVENTTARGET, __EVENTARGUMENT,
        and __VIEWSTATEENCRYPTED.

    Raises:
        ValueError: If required hidden fields are missing.
    """
    soup = BeautifulSoup(page_html, "lxml")

    fields: dict[str, str] = {}
    required = ["__VIEWSTATE", "__EVENTVALIDATION"]
    optional = [
        "__VIEWSTATEGENERATOR",
        "__VIEWSTATEENCRYPTED",
        "__EVENTTARGET",
        "__EVENTARGUMENT",
    ]

    for name in required + optional:
        tag = soup.find("input", attrs={"name": name})
        fields[name] = tag["value"] if tag else ""

    missing = [f for f in required if not fields.get(f)]
    if missing:
        raise ValueError(f"Missing required hidden fields: {missing}")

    return fields


def build_search_payload(
    hidden_fields: dict[str, str],
    imo_number: int | str,
) -> dict[str, str]:
    """Build the POST form payload for a ship-by-IMO search.

    Args:
        hidden_fields: Dict from parse_hidden_fields().
        imo_number: 7-digit IMO number (int or str).

    Returns:
        Dict ready to POST to GISIS_URL.
    """
    imo_str = _normalize_imo(imo_number)

    payload = {
        "__EVENTTARGET": hidden_fields.get("__EVENTTARGET", ""),
        "__EVENTARGUMENT": hidden_fields.get("__EVENTARGUMENT", ""),
        "__VIEWSTATE": hidden_fields["__VIEWSTATE"],
        "__VIEWSTATEGENERATOR": hidden_fields.get("__VIEWSTATEGENERATOR", ""),
        "__VIEWSTATEENCRYPTED": hidden_fields.get("__VIEWSTATEENCRYPTED", ""),
        "__EVENTVALIDATION": hidden_fields["__EVENTVALIDATION"],
        "ctl00$bodyPlaceHolder$Default$tbxShipImoNumber": imo_str,
        "ctl00$bodyPlaceHolder$Default$tbxShipName": "",
        "ctl00$bodyPlaceHolder$Default$cbxShipPreviousNames": "on",
        "ctl00$bodyPlaceHolder$Default$ddlShipFlags": "",
        "ctl00$bodyPlaceHolder$Default$tbxShipCallSign": "",
        "ctl00$bodyPlaceHolder$Default$tbxShipMMSI": "",
        "ctl00$bodyPlaceHolder$Default$btnSearchShips": "Search",
        "ctl00$bodyPlaceHolder$Default$tbxCompanyImoNumber": "",
        "ctl00$bodyPlaceHolder$Default$tbxCompanyName": "",
    }
    return payload


def submit_ship_search(
    session: requests.Session,
    payload: dict[str, str],
) -> str:
    """POST the search payload and return result HTML.

    Args:
        session: Authenticated requests.Session.
        payload: Form payload from build_search_payload().

    Returns:
        HTML text of the results page.

    Raises:
        PermissionError: If redirected to login.
        requests.HTTPError: On non-200 status.
    """
    resp = session.post(
        GISIS_URL,
        data=payload,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    resp.raise_for_status()

    if "Login.aspx" in resp.url:
        raise PermissionError(
            f"Session expired during POST — redirected to {resp.url}"
        )
    return resp.text


def _parse_tooltip(title_attr: str) -> dict[str, str | None]:
    """Parse the tooltip/title attribute from a result row.

    The title contains HTML like:
        Gross tonnage: 2,245<br/>Call sign: A6E2120<br/>...

    Args:
        title_attr: Raw title attribute string.

    Returns:
        Dict with parsed tooltip fields (snake_case keys).
    """
    unescaped = html.unescape(title_attr)

    label_map = {
        "gross tonnage": "gross_tonnage",
        "call sign": "call_sign",
        "mmsi": "mmsi",
        "ship type (detailed)": "ship_type_detailed",
        "year of build": "year_built",
    }

    result: dict[str, str | None] = {v: None for v in label_map.values()}

    parts = re.split(r"<br\s*/?>", unescaped, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip()
        if ":" not in part:
            continue
        label, _, value = part.partition(":")
        label_lower = label.strip().lower()
        value = value.strip()
        if label_lower in label_map:
            result[label_map[label_lower]] = value if value else None

    return result


def parse_ship_result(
    page_html: str,
    imo_number: int | str,
) -> dict[str, Any]:
    """Parse the ship search results table from the response HTML.

    Args:
        page_html: HTML text returned from submit_ship_search().
        imo_number: The IMO number that was searched.

    Returns:
        Dict with one row of output. If no result, found=False with
        null fields. If multiple rows, returns the first match.
    """
    imo_str = _normalize_imo(imo_number)
    base: dict[str, Any] = {
        "imo_number": imo_str,
        "found": False,
        "name": None,
        "flag": None,
        "type": None,
        "ship_type_detailed": None,
        "gross_tonnage": None,
        "call_sign": None,
        "mmsi": None,
        "year_built": None,
        "source": "GISIS",
        "lookup_timestamp": datetime.now(timezone.utc).isoformat(),
        "error_message": None,
    }

    soup = BeautifulSoup(page_html, "lxml")
    grid = soup.find("table", id="ctl00_bodyPlaceHolder_Default_gridShips")

    if not grid:
        base["error_message"] = "Results grid not found in HTML"
        return base

    rows = grid.find("tbody")
    if not rows:
        base["error_message"] = "No results"
        return base

    first_row = rows.find("tr")
    if not first_row:
        base["error_message"] = "No result rows"
        return base

    cells = first_row.find_all("td")
    if len(cells) >= 3:
        base["name"] = cells[0].get_text(strip=True) or None
        base["flag"] = cells[1].get_text(strip=True) or None
        base["type"] = cells[2].get_text(strip=True) or None
        base["found"] = True

    title_attr = first_row.get("title", "")
    if title_attr:
        tooltip = _parse_tooltip(title_attr)
        base.update({k: v for k, v in tooltip.items() if v is not None})

    return base


def lookup_one_imo(
    session: requests.Session,
    imo_number: int | str,
    hidden_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Full end-to-end lookup for a single IMO number.

    If hidden_fields is provided, skips the initial GET (saves one
    request when batching — the caller can reuse fields from the
    previous response).

    Args:
        session: Authenticated requests.Session.
        imo_number: 7-digit IMO number.
        hidden_fields: Pre-parsed hidden fields (optional).

    Returns:
        Normalized dict with one output row.
    """
    imo_str = _normalize_imo(imo_number)
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        if hidden_fields is None:
            page_html = fetch_search_page(session)
            hidden_fields = parse_hidden_fields(page_html)

        payload = build_search_payload(hidden_fields, imo_str)
        result_html = submit_ship_search(session, payload)
        result = parse_ship_result(result_html, imo_str)
        result["lookup_timestamp"] = timestamp

        try:
            new_fields = parse_hidden_fields(result_html)
        except ValueError:
            new_fields = None

        return result, new_fields

    except PermissionError:
        raise

    except Exception as exc:
        logger.warning(f"IMO {imo_str}: lookup failed — {exc}")
        return {
            "imo_number": imo_str,
            "found": False,
            "name": None,
            "flag": None,
            "type": None,
            "ship_type_detailed": None,
            "gross_tonnage": None,
            "call_sign": None,
            "mmsi": None,
            "year_built": None,
            "source": "GISIS",
            "lookup_timestamp": timestamp,
            "error_message": str(exc),
        }, None


def _load_cache(cache_path: Path) -> pd.DataFrame:
    """Load existing cache from CSV or parquet."""
    if not cache_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if cache_path.suffix == ".parquet":
        return pd.read_parquet(cache_path)

    return pd.read_csv(cache_path, dtype={"imo_number": str, "mmsi": str})


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Write cache dataframe to disk."""
    if cache_path.suffix == ".parquet":
        df.to_parquet(cache_path, index=False)
    else:
        df.to_csv(cache_path, index=False)
    logger.info(f"Cache saved → {cache_path} ({len(df)} rows)")


def lookup_many_imos(
    imo_list: list[int | str] | pd.Series,
    cookies: dict[str, str] | None = None,
    session: requests.Session | None = None,
    cache_path: str | Path | None = None,
    flush_every: int = 25,
    delay_range: tuple[float, float] = (3.0, 6.0),
    max_retries: int = 3,
) -> pd.DataFrame:
    """Batch-lookup many IMO numbers from GISIS.

    Args:
        imo_list: Iterable of IMO numbers.
        cookies: Browser cookies for authentication
            (ignored if session is provided).
        session: Pre-built requests.Session (optional).
        cache_path: Path to CSV/parquet cache file. Already-cached
            IMOs are skipped; new results are appended.
        flush_every: Flush cache to disk every N lookups.
        delay_range: (min, max) seconds of random delay between requests.
            Default 3–6s to be polite to GISIS servers.
        max_retries: Retries per IMO on transient failures.

    Returns:
        pandas DataFrame with one row per input IMO.
    """
    imos = [_normalize_imo(i) for i in imo_list]
    unique_imos = list(dict.fromkeys(imos))
    logger.info(f"Batch lookup: {len(unique_imos)} unique IMOs")

    if session is None:
        session = create_session(cookies=cookies)

    cache_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    cached_imos: set[str] = set()
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_df = _load_cache(cache_path)
        cached_imos = set(cache_df["imo_number"].astype(str))
        logger.info(f"Cache loaded: {len(cached_imos)} IMOs already cached")

    to_lookup = [i for i in unique_imos if i not in cached_imos]
    logger.info(f"IMOs to fetch: {len(to_lookup)} (skipping {len(unique_imos) - len(to_lookup)} cached)")

    if not to_lookup:
        return cache_df[cache_df["imo_number"].isin(unique_imos)].reset_index(drop=True)

    logger.info("Fetching initial search page …")
    page_html = fetch_search_page(session)
    hidden_fields = parse_hidden_fields(page_html)

    results: list[dict[str, Any]] = []

    for idx, imo in enumerate(to_lookup, 1):
        if idx > 1 and (idx - 1) % REAUTH_EVERY == 0:
            logger.info(
                f"Proactive re-auth after {idx - 1} lookups …"
            )
            try:
                session = authenticate(force_refresh=True)
                page_html = fetch_search_page(session)
                hidden_fields = parse_hidden_fields(page_html)
                logger.info("Re-auth successful, continuing")
            except (PermissionError, RuntimeError):
                logger.warning(
                    "Proactive re-auth failed — will retry on next lookup error"
                )

        logger.info(f"[{idx}/{len(to_lookup)}] Looking up IMO {imo}")

        for attempt in range(1, max_retries + 1):
            try:
                result, new_fields = lookup_one_imo(
                    session, imo, hidden_fields=hidden_fields
                )
                if new_fields is not None:
                    hidden_fields = new_fields
                results.append(result)
                break

            except PermissionError:
                logger.warning("Session expired mid-batch. Re-extracting cookies from browser …")
                try:
                    session = authenticate(force_refresh=True)
                    page_html = fetch_search_page(session)
                    hidden_fields = parse_hidden_fields(page_html)
                except (PermissionError, RuntimeError):
                    logger.error(
                        "Re-authentication failed. Log into GISIS in your "
                        "browser, then re-run. Cached results are safe."
                    )
                    raise

            except Exception as exc:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"IMO {imo} attempt {attempt} failed ({exc}), "
                        f"retrying in {wait}s …"
                    )
                    time.sleep(wait)
                    try:
                        page_html = fetch_search_page(session)
                        hidden_fields = parse_hidden_fields(page_html)
                    except Exception:
                        pass
                else:
                    logger.warning(f"IMO {imo}: all {max_retries} attempts failed")
                    results.append({
                        "imo_number": imo,
                        "found": False,
                        "name": None,
                        "flag": None,
                        "type": None,
                        "ship_type_detailed": None,
                        "gross_tonnage": None,
                        "call_sign": None,
                        "mmsi": None,
                        "year_built": None,
                        "source": "GISIS",
                        "lookup_timestamp": datetime.now(timezone.utc).isoformat(),
                        "error_message": str(exc),
                    })

        if cache_path and idx % flush_every == 0:
            batch_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
            merged = pd.concat([cache_df, batch_df], ignore_index=True)
            _save_cache(merged, cache_path)

        if idx < len(to_lookup):
            delay = random.uniform(*delay_range)
            time.sleep(delay)

    new_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    full_df = pd.concat([cache_df, new_df], ignore_index=True)

    if cache_path:
        _save_cache(full_df, cache_path)

    requested = set(unique_imos)
    return (
        full_df[full_df["imo_number"].isin(requested)]
        .drop_duplicates(subset="imo_number", keep="last")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GISIS Ship Lookup")
    parser.add_argument(
        "--input",
        default="ais_data/cleaned/unique_vessels.csv",
        help="Path to CSV with an 'imo' column",
    )
    parser.add_argument(
        "--cache",
        default="ais_data/cleaned/gisis_cache.csv",
        help="Path to result cache CSV",
    )
    parser.add_argument(
        "--browser",
        choices=["chrome", "firefox", "edge", "safari"],
        help="Browser to extract cookies from (default: try all)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Skip cached cookies, re-extract from browser",
    )
    args = parser.parse_args()

    vessels = pd.read_csv(args.input, dtype=str)
    imo_col = "imo" if "imo" in vessels.columns else "imo_number"
    imo_list = vessels[imo_col].dropna().unique().tolist()

    if GISIS_LIMIT > 0:
        imo_list = imo_list[:GISIS_LIMIT]
        logger.info(f"GISIS_LIMIT={GISIS_LIMIT}, using first {GISIS_LIMIT} IMOs")

    logger.info(f"Loaded {len(imo_list)} unique IMOs from {args.input}")

    session = authenticate(
        browser=args.browser,
        force_refresh=args.force_refresh,
    )

    df = lookup_many_imos(
        imo_list=imo_list,
        session=session,
        cache_path=args.cache,
    )

    output_path = Path("ais_data/cleaned/ship_details.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved → {output_path} ({len(df)} rows, {df['found'].sum()} found)")
    print(f"\nResults: {len(df)} rows, {df['found'].sum()} found")
    print(df.head(10).to_string(index=False))
