"""NM live source: the EUROCONTROL Network Manager real-time API.

This is the API behind https://www.eurocontrol.int/performance/live.html. Unlike
the Data App API (which is D-1, yesterday's daily figures), this is genuinely
LIVE — current airborne traffic, network delay right now, and active ATFM
regulations. Public, no auth.

We use the two compact, chat-friendly endpoints:
  * /statistics  — network-wide counts + top delayed ACCs (small JSON)
  * /regulations — active ATFM regulations (we summarise, not the full geometry)

/flights (live positions) is a ~1.8MB GeoJSON and is intentionally not used for
text Q&A; /statistics already carries the airborne count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import requests

from aiu_chat import config

USER_AGENT = "aiu-chat/0.1"
TIMEOUT = 30


class NmLiveError(RuntimeError):
    """NM live API unreachable or returned an error."""


@dataclass
class NmRegulation:
    id: str
    location: str
    reason: str
    delay_min: int
    impacted_flights: int


@dataclass
class NmLiveSnapshot:
    airborne: int | None
    landed: int | None
    planned: int | None
    total: int | None
    total_delay_min: int | None
    top_delays: list[dict] = field(default_factory=list)       # [{displayName, delay, averageDelay}]
    regulations: list[NmRegulation] = field(default_factory=list)


def _get(session: requests.Session, path: str) -> dict:
    url = f"{config.NM_LIVE_BASE}{path}"
    try:
        r = session.get(
            url, timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
    except requests.RequestException as exc:
        raise NmLiveError(f"NM live API unreachable ({url}): {exc}") from exc
    if r.status_code != 200:
        raise NmLiveError(f"NM live API {path} returned HTTP {r.status_code}.")
    return r.json()


def fetch_snapshot(
    *, include_regulations: bool = True, session: requests.Session | None = None
) -> NmLiveSnapshot:
    """Fetch the current network statistics (+ active regulations)."""
    own = session is None
    session = session or requests.Session()
    try:
        stats = _get(session, "/statistics")
        regs: list[NmRegulation] = []
        if include_regulations:
            data = _get(session, "/regulations")
            for f in data.get("features", []):
                p = f.get("properties", {})
                regs.append(
                    NmRegulation(
                        id=str(f.get("id", "")),
                        location=p.get("locationName", ""),
                        reason=p.get("reason", ""),
                        delay_min=int(p.get("delay", 0) or 0),
                        impacted_flights=int(p.get("nrImpactedFlights", 0) or 0),
                    )
                )
            # Most impactful first.
            regs.sort(key=lambda r: r.delay_min, reverse=True)
    finally:
        if own:
            session.close()

    return NmLiveSnapshot(
        airborne=stats.get("nrAirborneTraffic"),
        landed=stats.get("nrLandedTraffic"),
        planned=stats.get("nrPlannedTraffic"),
        total=stats.get("nrTotalTraffic"),
        total_delay_min=stats.get("nrMinutesDelay"),
        top_delays=stats.get("topAccumulatedDelays", []) or [],
        regulations=regs,
    )
