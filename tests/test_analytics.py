"""Tests for the usage-analytics aggregation (aiu_chat.agent.analytics).

Pure logic over slim turn dicts — no network. Covers counts, route breakdown,
error/clarification rates, latency stats, per-day activity, top questions, and
the OpenAI-request estimate (per-route fan-out).
"""
from __future__ import annotations

from aiu_chat.agent import analytics


def _turns():
    return [
        {"route": "data", "latency_ms": 4000, "created": "2026-07-01 10:00:00.000Z",
         "question": "CO2 in 2024?"},
        {"route": "data", "latency_ms": 2000, "created": "2026-07-01 11:00:00.000Z",
         "question": "CO2 in 2024?"},  # repeated question
        {"route": "concept", "latency_ms": 3000, "created": "2026-07-02 09:00:00.000Z",
         "question": "What is ASMA?"},
        {"route": "data", "error": "boom", "latency_ms": 500,
         "created": "2026-07-02 09:30:00.000Z", "question": "broken"},
        {"route": "dataapp", "needs_clarification": True, "latency_ms": 800,
         "created": "2026-07-02 10:00:00.000Z", "question": "latest?"},
    ]


def test_headline_counts():
    a = analytics.compute(_turns(), total_sessions=2)
    assert a.total_turns == 5
    assert a.total_sessions == 2
    assert a.turns_per_session == 2.5


def test_route_breakdown_sorted():
    a = analytics.compute(_turns(), total_sessions=2)
    assert a.route_counts["data"] == 3
    assert a.route_counts["concept"] == 1
    assert a.route_counts["dataapp"] == 1
    # most_common ordering => data first
    assert list(a.route_counts)[0] == "data"


def test_error_and_clarification_rates():
    a = analytics.compute(_turns(), total_sessions=2)
    assert a.error_count == 1
    assert a.error_rate == round(1 / 5, 3)
    assert a.clarification_count == 1


def test_latency_stats():
    a = analytics.compute(_turns(), total_sessions=2)
    # latencies: 4000, 2000, 3000, 500, 800 -> mean 2060, median 2000
    assert a.avg_latency_ms == 2060
    assert a.median_latency_ms == 2000


def test_per_day_activity():
    a = analytics.compute(_turns(), total_sessions=2)
    assert a.per_day == [("2026-07-01", 2), ("2026-07-02", 3)]


def test_top_questions():
    a = analytics.compute(_turns(), total_sessions=2)
    # "CO2 in 2024?" asked twice -> first
    assert a.top_questions[0] == ("CO2 in 2024?", 2)


def test_openai_request_estimate():
    a = analytics.compute(_turns(), total_sessions=2)
    # 3 data (4.5) + 1 concept (3.0) + 1 dataapp (4.0) = 13.5 + 3 + 4 = 20.5 -> 20/21
    expected = round(3 * 4.5 + 3.0 + 4.0)
    assert a.estimated_openai_requests == expected


def test_empty_input():
    a = analytics.compute([], total_sessions=0)
    assert a.total_turns == 0
    assert a.route_counts == {}
    assert a.avg_latency_ms is None
    assert a.estimated_openai_requests == 0
    assert a.top_questions == []


def test_tolerates_missing_fields():
    # A turn with almost nothing set must not crash aggregation.
    a = analytics.compute([{"route": "data"}, {}], total_sessions=1)
    assert a.total_turns == 2
    assert a.route_counts.get("unknown") == 1
