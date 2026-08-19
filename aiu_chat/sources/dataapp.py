"""EUROCONTROL Data App API client with deterministic resolvers.

The API keys metric data by a per-stakeholder, per-DAY "sync". Everything hangs
off a sync id, so answering a question is a matter of picking the right sync and
then reading the right metric/ranking off it. Three query shapes are supported
(the LLM chooses the shape + inputs; this module builds the calls so the model
can't hallucinate the API):

    entity   name/code -> entity id -> sync -> metric values (per-entity figure)
    network  date -> network-wide sync -> *_networks values (network figure)
    ranking  date/entity -> sync -> *_ranking_datas, ordered (top/bottom X)

All calls are GET, public, read-only. See docs/dataapp_api.md for the recipes.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date as _date

import requests

from aiu_chat import config

USER_AGENT = "aiu-chat/0.1"
TIMEOUT = 30
# The API silently caps page size at 100 regardless of what we ask for.
API_PAGE_SIZE = 100
# The Data App holds no syncs before this date — verified against the live API
# for countries, airports and aircraft operators alike (2022 and 2023 return
# zero syncs for every entity kind). Requests reaching further back are lifted
# to it and the answer says so, rather than reporting an empty or 5-day series.
DATAAPP_FIRST_DAY = "2024-01-01"

# Entity kind -> (dimension endpoint, syncs filter field, syncs dataType).
# NOTE: the dataType strings are the live API's, verified against the running
# service — they are NOT the tidy names you might guess (a country's syncs are
# "state-specific", not "country"; the network is "network-wide").
ENTITY_ENDPOINTS = {
    "country": ("/countries", "country.id", "state-specific"),
    "airport": ("/airports", "airport.id", "airport"),
    "ansp": ("/air_navigation_service_providers",
             "airNavigationServiceProvider.id", "air-navigation-service-provider"),
    "aircraft_operator": ("/aircraft_operators", "aircraftOperator.id", "aircraft-operator"),
}

NETWORK_DATATYPE = "network-wide"

# Metric -> (network endpoint, ranking-data endpoint, the nested prefix used in
# its sync filter). CO2 has no ranking endpoint (None) — the API doesn't expose
# co2 rankings.
METRIC_ENDPOINTS = {
    "traffic": ("/traffic_networks", "/traffic_ranking_datas", "traffic"),
    "delay": ("/delay_networks", "/delay_ranking_datas", "delay"),
    "co2": ("/co2_networks", None, "co2"),
    "punctuality": ("/punctualities_networks", "/punctualities_ranking_datas", "punctuality"),
}

# rankingCategory values the API accepts (from the OpenAPI Traffic-read enum).
# These select WHICH kind of thing a ranking ranks.
RANKING_CATEGORIES = {
    "states", "aircraft_operators", "airports", "airport_pairs",
    "area_control_center", "map_area_control_center", "network",
    "flight_breakdown", "market_segments", "breakdown",
}

# dateRange values on metric/ranking rows: the latest day (DY), 7-day window
# (WK), month (MM), year-to-date (Y2D). Y2D's avgValue is the daily average
# across the year so far — the figure the interactive Data App shows for "daily".
DATE_RANGES = {"DY", "WK", "MM", "Y2D"}


class DataAppError(RuntimeError):
    """API unreachable or a resolve step failed."""


@dataclass
class Entity:
    kind: str
    id: int
    name: str
    code: str


@dataclass
class DataAppResult:
    metric: str
    entity: Entity
    sync_id: int
    sync_date: str
    records: list[dict]  # the metric value rows (networkType/dateRange/value/...)


@dataclass
class TimeseriesResult:
    """A daily time series of one metric over a period, for one entity or the
    network. `rows` is tidy: one dict per day, e.g.
    [{"date": "2026-01-01", "value": 24864.0, "avgValue": ...}, ...] sorted by
    date. Ready to become a DataFrame for manipulation and charting."""
    metric: str
    entity: Entity
    start: str              # YYYY-MM-DD (inclusive)
    end: str                # YYYY-MM-DD (inclusive)
    rows: list[dict] = field(default_factory=list)
    # True if the requested START was lifted to DATAAPP_FIRST_DAY because the
    # API holds nothing earlier. The END is never trimmed.
    truncated: bool = False
    # The window the CALLER asked for, before any adjustment. `start`/`end` above
    # are what we actually queried; these two are what the user typed, so
    # coverage_note() can compare them and explain any shortfall.
    requested_start: str = ""
    requested_end: str = ""

    def coverage_note(self) -> str | None:
        """Explain a partial result, or None when the window is fully covered.

        A series can fall short at either end: before DATAAPP_FIRST_DAY (nothing
        exists yet) or after the latest available day (D-1, not yet reported).
        Returning fewer days than asked without saying so is the failure mode
        behind several logged conversations — the answer looked authoritative
        while quietly describing a different period than the question."""
        if not self.rows:
            return None
        req_start = self.requested_start or self.start
        req_end = self.requested_end or self.end
        have_start = self.rows[0].get("date", "")
        have_end = self.rows[-1].get("date", "")
        missing_head = bool(have_start and req_start and have_start > req_start)
        missing_tail = bool(have_end and req_end and have_end < req_end)

        # Days can also be missing from the MIDDLE while both ends line up (a
        # reporting gap). Ends matching is not proof the series is complete.
        span_days = 0
        try:
            span_days = (_date.fromisoformat(have_end)
                         - _date.fromisoformat(have_start)).days + 1
        except ValueError:
            pass
        # Allow a little slack: the API occasionally omits an odd day.
        has_gaps = bool(span_days and len(self.rows) < span_days * 0.95)

        if not (missing_head or missing_tail or has_gaps):
            return None

        note = (f"Only part of the requested period is available. "
                f"You asked for {req_start} to {req_end}; "
                f"the data covers {have_start} to {have_end}")
        why = []
        if missing_head:
            why.append(f"EUROCONTROL Data App coverage starts on {DATAAPP_FIRST_DAY}")
        if missing_tail:
            why.append(f"figures run only to the latest reported day ({have_end})")
        if has_gaps:
            why.append(f"only {len(self.rows)} of {span_days} days in that span "
                       "were reported")
        return note + " — " + "; ".join(why) + "."


@dataclass
class RankingResult:
    """A top/bottom-N ranking read off one sync for one metric+category."""
    metric: str
    category: str          # e.g. "airport_pairs", "aircraft_operators", "airports"
    scope: str             # what the ranking is scoped to ("network" or an entity name)
    sync_id: int
    sync_date: str
    date_range: str        # DY / WK / MM / Y2D
    rows: list[dict] = field(default_factory=list)  # [{name, value, avgValue, rankNumber, share}]


def _get(session: requests.Session, path: str, params: dict) -> dict:
    url = f"{config.DATAAPP_BASE}{path}"
    try:
        r = session.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        raise DataAppError(f"Data App API unreachable ({url}): {exc}") from exc
    if r.status_code != 200:
        raise DataAppError(f"Data App API {path} returned HTTP {r.status_code}.")
    return r.json()


_entity_cache: dict[tuple[str, str], tuple[float, Entity]] = {}


def clear_caches() -> None:
    """Drop memoised lookups (tests, and after a data refresh)."""
    _entity_cache.clear()


def resolve_entity(kind: str, query: str, session: requests.Session) -> Entity:
    """Resolve a name or code to an entity id, memoised for a short TTL.

    Entity ids are stable, so re-resolving "France" on every turn is pure
    latency. Sync ids are NOT cached here — they roll daily and a stale one
    would silently serve yesterday's figure as today's."""
    key = (kind, (query or "").strip().lower())
    hit = _entity_cache.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < config.DATAAPP_CACHE_TTL_S:
        return hit[1]
    entity = _resolve_entity_uncached(kind, query, session)
    _entity_cache[key] = (time.monotonic(), entity)
    return entity


def _resolve_entity_uncached(kind: str, query: str, session: requests.Session) -> Entity:
    """Resolve a name or code to an entity id via the dimension endpoint."""
    if kind not in ENTITY_ENDPOINTS:
        raise DataAppError(f"Unknown entity kind: {kind}")
    endpoint, _, _ = ENTITY_ENDPOINTS[kind]
    field_name = "iso2" if kind == "country" and len(query) == 2 else (
        "code" if (kind != "country" and len(query) <= 4 and query.isupper()) else "name"
    )
    data = _get(session, endpoint, {field_name: query, "itemsPerPage": 5}).get("data", [])
    if not data:
        # Retry by name if a code lookup missed.
        data = _get(session, endpoint, {"name": query, "itemsPerPage": 5}).get("data", [])
    if not data:
        raise DataAppError(f"No {kind} found for '{query}'.")
    top = data[0]
    return Entity(
        kind=kind, id=top["id"], name=top.get("name", query),
        code=top.get("iso2") or top.get("code") or "",
    )


def find_sync(
    session: requests.Session,
    *,
    entity: Entity | None = None,
    date: str | None = None,
) -> tuple[int, str]:
    """Find a sync id + date.

    - `entity=None` -> the network-wide sync (for whole-network figures/rankings).
    - `entity` given -> that entity's sync (dataType from its kind).
    A `date` (YYYY-MM-DD) pins to that single day (`syncDate[after]==[before]`);
    otherwise the newest sync is used (`order[syncDate]=desc`, one item).
    """
    if entity is not None:
        _, sync_field, dataType = ENTITY_ENDPOINTS[entity.kind]
        params: dict = {"dataType": dataType, "itemsPerPage": 1, sync_field: entity.id}
        label = f"{entity.kind} '{entity.name}'"
    else:
        params = {"dataType": NETWORK_DATATYPE, "itemsPerPage": 1}
        label = "network"
    if date:
        params["syncDate[after]"] = date
        params["syncDate[before]"] = date
    else:
        params["order[syncDate]"] = "desc"

    data = _get(session, "/syncs", params).get("data", [])
    if not data:
        when = f" on {date}" if date else ""
        raise DataAppError(f"No sync found for {label}{when}.")
    return data[0]["id"], data[0].get("syncDate", "")[:10]


def latest_sync(entity: Entity, session: requests.Session) -> tuple[int, str]:
    """Latest sync id + date for an entity (kept for the existing entity path)."""
    return find_sync(session, entity=entity, date=None)


def fetch_metric(
    metric: str, kind: str, query: str, *,
    date: str | None = None, session: requests.Session | None = None,
) -> DataAppResult:
    """Per-entity figure: resolve entity -> sync (optionally for `date`) -> values."""
    if metric not in METRIC_ENDPOINTS:
        raise DataAppError(f"Unknown metric: {metric}")
    own = session is None
    session = session or requests.Session()
    try:
        entity = resolve_entity(kind, query, session)
        sync_id, sync_date = find_sync(session, entity=entity, date=date)
        endpoint, _, prefix = METRIC_ENDPOINTS[metric]
        data = _get(
            session, endpoint,
            {f"{prefix}.sync.id": sync_id, "itemsPerPage": 30},
        ).get("data", [])
        # Strip the heavy nested entity object from each record; keep the values.
        records = [{k: v for k, v in r.items() if k != prefix} for r in data]
    finally:
        if own:
            session.close()
    return DataAppResult(
        metric=metric, entity=entity, sync_id=sync_id, sync_date=sync_date, records=records
    )


def fetch_network(
    metric: str, *, date: str | None = None, session: requests.Session | None = None
) -> DataAppResult:
    """Whole-network figure for a metric (optionally on a specific date).

    Answers "how many flights on the network on <date>", "network ATFM delay",
    etc. Uses the network-wide sync + the `*_networks` endpoint directly.
    """
    if metric not in METRIC_ENDPOINTS:
        raise DataAppError(f"Unknown metric: {metric}")
    own = session is None
    session = session or requests.Session()
    try:
        sync_id, sync_date = find_sync(session, date=date)
        endpoint, _, prefix = METRIC_ENDPOINTS[metric]
        data = _get(
            session, endpoint,
            {f"{prefix}.sync.id": sync_id, "itemsPerPage": 30},
        ).get("data", [])
        records = [{k: v for k, v in r.items() if k != prefix} for r in data]
    finally:
        if own:
            session.close()
    net = Entity(kind="network", id=0, name="Network", code="")
    return DataAppResult(
        metric=metric, entity=net, sync_id=sync_id, sync_date=sync_date, records=records
    )


def _clamp_period(start: str, end: str) -> tuple[str, str, bool]:
    """Order start<=end and lift the start to the first day the API has data for.

    Returns (start, end, adjusted). The span itself is NOT capped: pagination
    walks arbitrarily long windows and the page ceiling bounds how hard we hit
    the API, so the user's requested end date is always honoured.

    The old behaviour trimmed the END to a fixed number of days, which combined
    disastrously with the data floor — a 2023->2026 request was cut to a 370-day
    window landing almost entirely in the pre-2024 dead zone, so the app narrated
    5 real days and blamed "limits" (logged turns 27/28/29)."""
    s = _date.fromisoformat(start)
    e = _date.fromisoformat(end)
    if e < s:
        s, e = e, s
    adjusted = False
    floor = _date.fromisoformat(DATAAPP_FIRST_DAY)
    if s < floor:
        s = floor
        adjusted = True
    if e < s:
        # The whole window predates the floor; return an empty-but-honest range.
        e = s
    return s.isoformat(), e.isoformat(), adjusted


def find_syncs_in_range(
    session: requests.Session,
    *,
    start: str,
    end: str,
    entity: Entity | None = None,
) -> list[tuple[int, str]]:
    """All (sync_id, sync_date) in [start, end] for an entity (or the network).

    ONE /syncs call using the syncDate range filter — not a per-day loop. Sorted
    ascending by date; deduped to one sync per day (the API keys one sync per
    entity per day)."""
    if entity is not None:
        _, sync_field, dataType = ENTITY_ENDPOINTS[entity.kind]
        params: dict = {"dataType": dataType, sync_field: entity.id}
    else:
        params = {"dataType": NETWORK_DATATYPE}
    params["syncDate[after]"] = start
    params["syncDate[before]"] = end
    params["order[syncDate]"] = "asc"
    params["itemsPerPage"] = API_PAGE_SIZE

    # Walk the window with a syncDate cursor. Offset paging does NOT work here:
    # the API ignores `page` and re-serves the same first 100 rows, so we instead
    # advance `syncDate[after]` past the last day we saw on each pass.
    by_day: dict[str, int] = {}
    cursor = start
    for _ in range(max(1, config.DATAAPP_MAX_PAGES)):
        page_params = dict(params, **{"syncDate[after]": cursor})
        data = _get(session, "/syncs", page_params).get("data", [])
        if not data:
            break
        last_seen = cursor
        for row in data:
            d = (row.get("syncDate") or "")[:10]
            if d and d not in by_day:
                by_day[d] = row["id"]
            if d > last_seen:
                last_seen = d
        # No forward progress (all rows on/before the cursor) -> window exhausted.
        if last_seen <= cursor or len(data) < API_PAGE_SIZE:
            break
        cursor = last_seen
        if config.DATAAPP_THROTTLE_S:
            time.sleep(config.DATAAPP_THROTTLE_S)

    return [(sid, d) for d, sid in sorted(by_day.items())]


def fetch_timeseries(
    metric: str,
    *,
    start: str,
    end: str,
    kind: str | None = None,
    query: str | None = None,
    session: requests.Session | None = None,
) -> TimeseriesResult:
    """A daily time series of `metric` over [start, end] for one entity or the
    network.

    One /syncs range call gets every daily sync in the window; then each sync's
    DY (single-day) metric value is read. Returns tidy per-day rows ready to be
    turned into a DataFrame, manipulated (resampled / divided) and charted.

    `kind`+`query` name an entity (country/airport/ansp/aircraft_operator);
    omit both for the whole network."""
    if metric not in METRIC_ENDPOINTS:
        raise DataAppError(f"Unknown metric: {metric}")
    # Keep what the caller asked for: coverage_note() compares it to what we
    # actually got, so a short series can explain itself.
    requested_start, requested_end = start, end
    start, end, truncated = _clamp_period(start, end)

    own = session is None
    session = session or requests.Session()
    try:
        if kind and query:
            entity = resolve_entity(kind, query, session)
        else:
            entity = Entity(kind="network", id=0, name="Network", code="")
        ent_arg = entity if entity.kind != "network" else None
        syncs = find_syncs_in_range(session, start=start, end=end, entity=ent_arg)
        if not syncs:
            raise DataAppError(
                f"No {metric} syncs for {entity.name} between {start} and {end}.")

        endpoint, _, prefix = METRIC_ENDPOINTS[metric]

        def _read_day(item: tuple[int, str]) -> dict | None:
            sync_id, sync_date = item
            data = _get(
                session, endpoint,
                {f"{prefix}.sync.id": sync_id, "itemsPerPage": 30},
            ).get("data", [])
            # Prefer the single-day (DY) total row; fall back to the first row.
            recs = [{k: v for k, v in r.items() if k != prefix} for r in data]
            dy = _pick_day_record(recs)
            if dy is None:
                return None
            return {
                "date": sync_date,
                "value": dy.get("value"),
                "avgValue": dy.get("avgValue"),
            }

        # One call per day is unavoidable (the API has no bulk endpoint), but they
        # are independent, so run a bounded pool instead of ~200 serial requests.
        rows: list[dict] = []
        workers = max(1, min(config.DATAAPP_CONCURRENCY, len(syncs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_read_day, syncs):
                if row is not None:
                    rows.append(row)
    finally:
        if own:
            session.close()

    rows.sort(key=lambda r: r["date"])
    return TimeseriesResult(
        metric=metric, entity=entity, start=start, end=end,
        rows=rows, truncated=truncated,
        requested_start=requested_start, requested_end=requested_end,
    )


def _pick_day_record(records: list[dict]) -> dict | None:
    """From a sync's metric rows, pick the single-DAY (DY) total row — the daily
    figure. Falls back to any DY row, then the first row."""
    dy = [r for r in records if r.get("dateRange") == "DY"]
    if dy:
        total = [r for r in dy if r.get("networkType") == "total"]
        return total[0] if total else dy[0]
    return records[0] if records else None


def fetch_ranking(
    metric: str,
    category: str,
    *,
    scope_kind: str | None = None,
    scope_query: str | None = None,
    date: str | None = None,
    date_range: str = "DY",
    ascending: bool = False,
    limit: int = 15,
    session: requests.Session | None = None,
) -> RankingResult:
    """A top/bottom-N ranking off one sync.

    - `metric`: traffic | delay | punctuality (co2 has no rankings).
    - `category`: what to rank (states, airports, airport_pairs, aircraft_operators…).
    - `scope_kind`/`scope_query`: rank WITHIN an entity's sync (e.g. airport_pairs
      for a specific airline, or aircraft_operators for a country). None -> the
      network-wide sync (e.g. "which country/airport is highest across the network").
    - `date`: pin to a day; else newest sync.
    - `date_range`: DY (that day) / WK / MM / Y2D (yearly, uses avgValue).
    - `ascending`: True for "lowest" (bottom-N), False for "highest" (top-N).
    """
    if metric not in METRIC_ENDPOINTS:
        raise DataAppError(f"Unknown metric: {metric}")
    endpoint, ranking_endpoint, prefix = METRIC_ENDPOINTS[metric]
    if ranking_endpoint is None:
        raise DataAppError(f"No rankings available for metric '{metric}'.")
    if category not in RANKING_CATEGORIES:
        raise DataAppError(f"Unknown ranking category '{category}'.")
    if date_range not in DATE_RANGES:
        raise DataAppError(f"Unknown date range '{date_range}'.")

    own = session is None
    session = session or requests.Session()
    try:
        if scope_kind and scope_query:
            entity = resolve_entity(scope_kind, scope_query, session)
            sync_id, sync_date = find_sync(session, entity=entity, date=date)
            scope = entity.name
        else:
            sync_id, sync_date = find_sync(session, date=date)
            scope = "network"

        # The ranking-data filter chain is nested one level deeper than the
        # metric filter: <prefix>Ranking.<prefix>.sync.id / .rankingCategory.
        rk = f"{prefix}Ranking.{prefix}"
        # Y2D rankings compare daily averages -> order by avgValue; DY/WK/MM by value.
        order_field = "avgValue" if date_range == "Y2D" else "value"
        params = {
            f"{rk}.sync.id": sync_id,
            f"{rk}.rankingCategory": category,
            "dateRange": date_range,
            f"order[{order_field}]": "asc" if ascending else "desc",
            "itemsPerPage": max(limit, 30),
        }
        data = _get(session, ranking_endpoint, params).get("data", [])
    finally:
        if own:
            session.close()

    rows = []
    for r in data[:limit]:
        rows.append({
            "name": r.get("name"),
            "value": r.get("value"),
            "avgValue": r.get("avgValue"),
            "rankNumber": r.get("rankNumber"),
            "share": r.get("share"),
            "dateRange": r.get("dateRange"),
        })
    return RankingResult(
        metric=metric, category=category, scope=scope,
        sync_id=sync_id, sync_date=sync_date, date_range=date_range, rows=rows,
    )
