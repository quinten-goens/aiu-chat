"""EUROCONTROL Data App API client with a deterministic 3-hop resolver.

The API keys metric data by a per-stakeholder, per-date "sync". Resolving a named
entity to data takes up to three calls (see docs/dataapp_api.md):

    1. dimension endpoint: name/code -> entity id
    2. syncs: entity id -> latest sync id (+ dataType)
    3. metric endpoint: filter by sync id -> values

This module hard-codes that workflow so the LLM never has to (and can't
hallucinate it). The LLM only chooses the metric and the entity (see the
dataapp answer path).
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from aiu_chat import config

USER_AGENT = "aiu-chat/0.1"
TIMEOUT = 30

# Entity kind -> (dimension endpoint, sync filter field).
ENTITY_ENDPOINTS = {
    "country": ("/countries", "country.id"),
    "airport": ("/airports", "airport.id"),
    "ansp": ("/air_navigation_service_providers", "airNavigationServiceProvider.id"),
    "aircraft_operator": ("/aircraft_operators", "aircraftOperator.id"),
}

# Metric -> (network endpoint, the nested prefix used in its sync filter).
METRIC_ENDPOINTS = {
    "traffic": ("/traffic_networks", "traffic"),
    "delay": ("/delay_networks", "delay"),
    "co2": ("/co2_networks", "co2"),
    "punctuality": ("/punctualities_networks", "punctuality"),
}


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


def _get(session: requests.Session, path: str, params: dict) -> dict:
    url = f"{config.DATAAPP_BASE}{path}"
    try:
        r = session.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    except requests.RequestException as exc:
        raise DataAppError(f"Data App API unreachable ({url}): {exc}") from exc
    if r.status_code != 200:
        raise DataAppError(f"Data App API {path} returned HTTP {r.status_code}.")
    return r.json()


def resolve_entity(kind: str, query: str, session: requests.Session) -> Entity:
    """Resolve a name or code to an entity id via the dimension endpoint."""
    if kind not in ENTITY_ENDPOINTS:
        raise DataAppError(f"Unknown entity kind: {kind}")
    endpoint, _ = ENTITY_ENDPOINTS[kind]
    field = "iso2" if kind == "country" and len(query) == 2 else (
        "code" if (kind != "country" and len(query) <= 4 and query.isupper()) else "name"
    )
    data = _get(session, endpoint, {field: query, "itemsPerPage": 5}).get("data", [])
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


def latest_sync(entity: Entity, session: requests.Session) -> tuple[int, str]:
    """Latest sync id + date for an entity (dataType resolved from the data)."""
    _, sync_field = ENTITY_ENDPOINTS[entity.kind]
    data = _get(
        session, "/syncs",
        {sync_field: entity.id, "order[syncDate]": "desc", "itemsPerPage": 1},
    ).get("data", [])
    if not data:
        raise DataAppError(f"No sync data for {entity.kind} '{entity.name}'.")
    return data[0]["id"], data[0].get("syncDate", "")[:10]


def fetch_metric(
    metric: str, kind: str, query: str, *, session: requests.Session | None = None
) -> DataAppResult:
    """Full 3-hop: resolve entity -> latest sync -> metric values."""
    if metric not in METRIC_ENDPOINTS:
        raise DataAppError(f"Unknown metric: {metric}")
    own = session is None
    session = session or requests.Session()
    try:
        entity = resolve_entity(kind, query, session)
        sync_id, sync_date = latest_sync(entity, session)
        endpoint, prefix = METRIC_ENDPOINTS[metric]
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
