"""Pagination tests for the Data App sync lookup.

The live API caps itemsPerPage at 100 and ignores `page`; the only way to walk a
longer window is a syncDate cursor. These tests pin that behaviour with a fake
API that reproduces the cap exactly.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from aiu_chat.sources import dataapp


class FakeAPI:
    """Mimics the real API: returns at most 100 rows, honours syncDate[after],
    and IGNORES `page` (exactly like the live endpoint)."""

    def __init__(self, first_day: str, n_days: int):
        self.days = [
            (date.fromisoformat(first_day) + timedelta(days=i)).isoformat()
            for i in range(n_days)
        ]
        self.calls = 0

    def __call__(self, session, path, params):
        self.calls += 1
        after = params.get("syncDate[after]")
        before = params.get("syncDate[before]")
        sel = [d for d in self.days if (not after or d >= after) and (not before or d <= before)]
        page = sel[:100]  # hard cap, `page` param deliberately ignored
        return {
            "meta": {"itemsPerPage": 100, "totalItems": len(sel)},
            "data": [
                {"id": 1000 + self.days.index(d), "syncDate": f"{d}T00:00:00+00:00"}
                for d in page
            ],
        }


def test_short_window_single_call(monkeypatch):
    fake = FakeAPI("2026-01-01", 30)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2026-01-01", end="2026-01-30")
    assert len(out) == 30
    assert fake.calls == 1


def test_long_window_is_complete(monkeypatch):
    """226 days must all come back — the bug was stopping silently at 100."""
    fake = FakeAPI("2026-01-01", 226)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2026-01-01", end="2026-08-14")
    assert len(out) == 226
    assert out[0][1] == "2026-01-01"
    assert out[-1][1] == "2026-08-14"
    assert fake.calls > 1


def test_results_sorted_and_deduped(monkeypatch):
    fake = FakeAPI("2026-01-01", 150)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2026-01-01", end="2026-05-30")
    dates = [d for _, d in out]
    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))


def test_page_ceiling_is_respected(monkeypatch):
    """A pathological window must not fire unbounded calls."""
    monkeypatch.setattr(dataapp.config, "DATAAPP_MAX_PAGES", 3)
    fake = FakeAPI("2020-01-01", 2000)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2020-01-01", end="2025-06-23")
    assert fake.calls <= 3
    assert len(out) <= 300


def test_timeseries_reads_every_day(monkeypatch):
    """Every sync day must appear as a row, in date order, with no gaps."""
    days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(150)]
    syncs = [(1000 + i, d) for i, d in enumerate(days)]
    monkeypatch.setattr(dataapp, "find_syncs_in_range", lambda *a, **k: syncs)

    def fake_get(session, path, params):
        sid = params.get("traffic.sync.id")
        return {"data": [{
            "dateRange": "DY", "networkType": "total",
            "value": float(sid), "avgValue": float(sid),
        }]}

    monkeypatch.setattr(dataapp, "_get", fake_get)
    res = dataapp.fetch_timeseries("traffic", start="2026-01-01", end="2026-05-30")
    assert len(res.rows) == 150
    assert [r["date"] for r in res.rows] == days
    assert res.rows[0]["value"] == 1000.0
    assert res.rows[-1]["value"] == 1149.0


def test_entity_resolution_is_cached(monkeypatch):
    calls = {"n": 0}

    def counting_get(session, path, params):
        calls["n"] += 1
        return {"data": [{"id": 7, "name": "France", "code": "FR"}]}

    dataapp.clear_caches()
    monkeypatch.setattr(dataapp, "_get", counting_get)
    a = dataapp.resolve_entity("country", "France", None)
    b = dataapp.resolve_entity("country", "France", None)
    assert a.id == b.id == 7
    assert calls["n"] == 1, "second lookup should hit the cache"


def test_entity_cache_expires(monkeypatch):
    """A stale entry must not outlive the TTL."""
    calls = {"n": 0}

    def counting_get(session, path, params):
        calls["n"] += 1
        return {"data": [{"id": 7, "name": "France", "code": "FR"}]}

    dataapp.clear_caches()
    monkeypatch.setattr(dataapp, "_get", counting_get)
    monkeypatch.setattr(dataapp.config, "DATAAPP_CACHE_TTL_S", 0)
    dataapp.resolve_entity("country", "France", None)
    dataapp.resolve_entity("country", "France", None)
    assert calls["n"] == 2


def test_entity_cache_distinguishes_kind_and_query(monkeypatch):
    seen = []

    def counting_get(session, path, params):
        seen.append(params)
        return {"data": [{"id": len(seen), "name": "X", "code": "X"}]}

    dataapp.clear_caches()
    monkeypatch.setattr(dataapp, "_get", counting_get)
    a = dataapp.resolve_entity("country", "France", None)
    b = dataapp.resolve_entity("country", "Spain", None)
    c = dataapp.resolve_entity("airport", "France", None)
    assert a.id != b.id != c.id
