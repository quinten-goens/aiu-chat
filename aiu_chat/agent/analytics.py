"""Aggregate logged turns/sessions into a usage summary for the viewer.

Pure functions over the slim record dicts the chatlog client returns (no
network here), so they are cheap to unit-test. The viewer fetches the rows and
hands them in.

The OpenAI-request estimate mirrors the real per-question call fan-out: a data
question fires ~4-5 requests (route + clarify + SQL + narrate + optional chart),
a concept ~3 (route + embed + answer). We estimate per-route so "how many
questions did N requests cost?" can be answered from real traffic.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Estimated OpenAI requests per answered turn, by route. Clarifying turns and
# out-of-scope ("none") turns are cheaper; these are averages, not exact.
REQUESTS_PER_ROUTE = {
    "data": 4.5,      # route + clarify + SQL + narrate + (chart when chartable)
    "both": 5.5,      # data path + concept embed + concept answer
    "concept": 3.0,   # route + embed + answer
    "nop": 3.0,       # route + clarify + answer
    "dataapp": 4.0,   # route + clarify + extract + narrate
    "nm_live": 3.0,   # route + clarify + answer
    "catalog": 1.0,   # answered from the catalogue, one LLM-free lookup
    "none": 1.0,      # route only
}
_DEFAULT_REQUESTS = 4.0


def _parse_dt(value: str):
    """Parse a PocketBase timestamp to a date string (YYYY-MM-DD), or None."""
    if not value:
        return None
    v = value.replace("T", " ").replace("Z", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(v[:26], fmt).date().isoformat()
        except ValueError:
            continue
    return v[:10] or None


@dataclass
class Analytics:
    total_sessions: int = 0
    total_turns: int = 0
    turns_per_session: float = 0.0
    route_counts: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    error_rate: float = 0.0
    clarification_count: int = 0
    avg_latency_ms: float | None = None
    median_latency_ms: float | None = None
    per_day: list[tuple[str, int]] = field(default_factory=list)  # (date, turns)
    top_questions: list[tuple[str, int]] = field(default_factory=list)
    estimated_openai_requests: int = 0
    requests_per_day: list[tuple[str, int]] = field(default_factory=list)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _estimate_requests(route: str) -> float:
    return REQUESTS_PER_ROUTE.get(route or "", _DEFAULT_REQUESTS)


def compute(turns: list[dict], *, total_sessions: int,
            top_n_questions: int = 15) -> Analytics:
    """Build an Analytics summary from slim turn records.

    Each turn dict may carry: route, error, needs_clarification, latency_ms,
    created, question. Missing keys are tolerated.
    """
    a = Analytics()
    a.total_sessions = total_sessions
    a.total_turns = len(turns)
    if total_sessions:
        a.turns_per_session = round(a.total_turns / total_sessions, 2)

    routes = Counter()
    latencies: list[float] = []
    per_day = Counter()
    req_per_day: dict[str, float] = {}
    questions = Counter()
    est_requests = 0.0

    for t in turns:
        route = (t.get("route") or "").strip()
        routes[route or "unknown"] += 1

        if t.get("error"):
            a.error_count += 1
        if t.get("needs_clarification"):
            a.clarification_count += 1

        lat = t.get("latency_ms")
        if isinstance(lat, (int, float)) and lat > 0:
            latencies.append(float(lat))

        day = _parse_dt(t.get("created") or t.get("created_at") or "")
        if day:
            per_day[day] += 1
            req_per_day[day] = req_per_day.get(day, 0.0) + _estimate_requests(route)

        est_requests += _estimate_requests(route)

        q = (t.get("question") or "").strip()
        if q:
            questions[q] += 1

    a.route_counts = dict(routes.most_common())
    if a.total_turns:
        a.error_rate = round(a.error_count / a.total_turns, 3)
    if latencies:
        a.avg_latency_ms = round(sum(latencies) / len(latencies))
        med = _median(latencies)
        a.median_latency_ms = round(med) if med is not None else None
    a.per_day = sorted(per_day.items())
    a.requests_per_day = sorted(
        ((d, round(v)) for d, v in req_per_day.items()), key=lambda x: x[0]
    )
    a.top_questions = questions.most_common(top_n_questions)
    a.estimated_openai_requests = round(est_requests)
    return a


def now_utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()
