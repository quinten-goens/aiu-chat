"""Tests for the Data App network + ranking query shapes.

Three layers:
  * unit    — spec helpers and the source's URL/param building (fake session).
  * agent   — the network/ranking branches of answer_dataapp_question (fakes).
  * live    — end-to-end against the real API, asserting the actual quiz answers
              (marked `live`; skipped when the API is unreachable / offline).

The quiz these mirror (verified answers in parentheses):
  - flights on the network on 2026-03-10 ................. 24,864
  - highest arrival punctuality on 2025-03-10 ........... Yerevan
  - most ATFM delay (country) on 2025-03-10 ............. Portugal
  - busiest airline in Estonia (2025) .................. airBaltic
  - busiest airport pair for British Airways Group ..... Glasgow/Edinburgh ⟷ LHR
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiu_chat.agent import dataapp_answer as da
from aiu_chat.sources import dataapp as d


# --- spec helpers ----------------------------------------------------------

def test_period_defaults_and_validates():
    assert da._period({}) == "DY"
    assert da._period({"period": "y2d"}) == "Y2D"
    assert da._period({"period": "bogus"}) == "DY"


def test_date_accepts_iso_only():
    assert da._date({"date": "2026-03-10"}) == "2026-03-10"
    assert da._date({"date": "March 10 2026"}) is None
    assert da._date({"date": None}) is None
    assert da._date({}) is None


# --- source param building (fake session, no network) ----------------------

def _fake_session(payload):
    """A session whose .get(...) returns one JSON payload and records the call."""
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    session.get.return_value = resp
    return session


def test_find_sync_network_pins_date():
    session = _fake_session({"data": [{"id": 42, "syncDate": "2026-03-10T00:00:00+00:00"}]})
    sid, sdate = d.find_sync(session, date="2026-03-10")
    assert sid == 42 and sdate == "2026-03-10"
    params = session.get.call_args.kwargs["params"]
    assert params["dataType"] == "network-wide"
    assert params["syncDate[after]"] == "2026-03-10"
    assert params["syncDate[before]"] == "2026-03-10"


def test_find_sync_entity_uses_kind_datatype_and_fk():
    session = _fake_session({"data": [{"id": 7, "syncDate": "2025-12-31T00:00:00+00:00"}]})
    ent = d.Entity(kind="country", id=41, name="Estonia", code="EE")
    d.find_sync(session, entity=ent, date="2025-12-31")
    params = session.get.call_args.kwargs["params"]
    assert params["dataType"] == "state-specific"   # NOT "country"
    assert params["country.id"] == 41


def test_find_sync_no_data_raises():
    session = _fake_session({"data": []})
    with pytest.raises(d.DataAppError, match="No sync"):
        d.find_sync(session, date="1990-01-01")


def test_fetch_ranking_builds_nested_filter_and_orders():
    # Two calls: /syncs then the ranking endpoint. Same payload works for both
    # (we only assert the *ranking* call's params).
    session = _fake_session({"data": [
        {"id": 100, "syncDate": "2025-03-10T00:00:00+00:00"},
    ]})
    # First .get -> sync; give the ranking call its own rows on the 2nd call.
    sync_resp = MagicMock(status_code=200)
    sync_resp.json.return_value = {"data": [{"id": 100, "syncDate": "2025-03-10T00:00:00+00:00"}]}
    rank_resp = MagicMock(status_code=200)
    rank_resp.json.return_value = {"data": [
        {"name": "Portugal", "value": 5573.0, "rankNumber": 1, "dateRange": "DY"},
        {"name": "United Kingdom", "value": 3558.0, "rankNumber": 2, "dateRange": "DY"},
    ]}
    session.get.side_effect = [sync_resp, rank_resp]

    res = d.fetch_ranking("delay", "states", date="2025-03-10",
                          date_range="DY", ascending=False, session=session)
    assert res.rows[0]["name"] == "Portugal"
    # The ranking call is the 2nd get; check its nested filter chain + order.
    params = session.get.call_args_list[1].kwargs["params"]
    assert params["delayRanking.delay.sync.id"] == 100
    assert params["delayRanking.delay.rankingCategory"] == "states"
    assert params["dateRange"] == "DY"
    assert params["order[value]"] == "desc"


def test_fetch_ranking_y2d_orders_by_avgvalue():
    # 3 gets: resolve entity, sync, ranking.
    resolve_resp = MagicMock(status_code=200)
    resolve_resp.json.return_value = {"data": [{"id": 41, "name": "Estonia", "iso2": "EE"}]}
    sync_resp = MagicMock(status_code=200)
    sync_resp.json.return_value = {"data": [{"id": 9, "syncDate": "2025-12-31T00:00:00+00:00"}]}
    rank_resp = MagicMock(status_code=200)
    rank_resp.json.return_value = {"data": [{"name": "airBaltic", "avgValue": 26.7}]}
    session = MagicMock()
    session.get.side_effect = [resolve_resp, sync_resp, rank_resp]
    d.fetch_ranking("traffic", "aircraft_operators", scope_kind="country",
                    scope_query="Estonia", date_range="Y2D", session=session)
    params = session.get.call_args_list[-1].kwargs["params"]
    assert params["order[avgValue]"] == "desc"       # Y2D orders by daily avg


def test_fetch_ranking_ascending_for_lowest():
    sync_resp = MagicMock(status_code=200)
    sync_resp.json.return_value = {"data": [{"id": 1, "syncDate": "2026-03-31"}]}
    rank_resp = MagicMock(status_code=200)
    rank_resp.json.return_value = {"data": [{"name": "Paris Le Bourget", "value": 0.4}]}
    session = MagicMock()
    session.get.side_effect = [sync_resp, rank_resp]
    d.fetch_ranking("punctuality", "airports", date="2026-03-31",
                    ascending=True, session=session)
    params = session.get.call_args_list[-1].kwargs["params"]
    assert params["order[value]"] == "asc"


def test_fetch_ranking_rejects_co2():
    with pytest.raises(d.DataAppError, match="No rankings"):
        d.fetch_ranking("co2", "states", session=MagicMock())


def test_fetch_ranking_rejects_bad_category():
    with pytest.raises(d.DataAppError, match="ranking category"):
        d.fetch_ranking("traffic", "unicorns", session=MagicMock())


# --- agent branches (fakes, no network/model) ------------------------------

class _FakeClient:
    def __init__(self, spec, text="ANSWER"):
        self._spec = spec
        self._text = text

    def chat_json(self, messages, temperature=0.0):
        return self._spec

    def chat(self, messages, temperature=0.0, json_mode=False):
        return self._text


def test_network_branch_narrates_network_figure():
    spec = {"query_kind": "network", "metric": "traffic", "date": "2026-03-10"}
    result = d.DataAppResult(
        metric="traffic", entity=d.Entity("network", 0, "Network", ""),
        sync_id=1, sync_date="2026-03-10",
        records=[{"networkType": "total", "dateRange": "DY", "value": 24864.0}],
    )
    ans = da.answer_dataapp_question(
        "flights on the network on 10 March 2026?",
        client=_FakeClient(spec, "24,864 flights."),
        fetch_net=lambda metric, date=None: result,
    )
    assert ans.ok
    assert ans.network is result
    assert ans.ranking is None
    assert "24,864" in ans.answer


def test_ranking_branch_narrates_top_entry():
    spec = {"query_kind": "ranking", "metric": "punctuality",
            "ranking_category": "airports", "date": "2025-03-10", "order": "highest"}
    ranking = d.RankingResult(
        metric="punctuality", category="airports", scope="network",
        sync_id=1, sync_date="2025-03-10", date_range="DY",
        rows=[{"name": "Yerevan", "value": 1.0, "rankNumber": 1}],
    )
    ans = da.answer_dataapp_question(
        "which airport had the highest punctuality on 10 March 2025?",
        client=_FakeClient(spec, "Yerevan, at 100%."),
        fetch_rank=lambda *a, **k: ranking,
    )
    assert ans.ok
    assert ans.ranking is ranking
    assert "Yerevan" in ans.answer


def test_ranking_branch_passes_scope_and_order_through():
    spec = {"query_kind": "ranking", "metric": "traffic",
            "ranking_category": "aircraft_operators", "date": "2025-12-31",
            "period": "Y2D", "scope_kind": "country", "scope": "Estonia",
            "order": "highest"}
    captured = {}

    def _fetch_rank(metric, category, *, scope_kind=None, scope_query=None,
                    date=None, date_range="DY", ascending=False, limit=15):
        captured.update(dict(metric=metric, category=category, scope_kind=scope_kind,
                             scope_query=scope_query, date=date, date_range=date_range,
                             ascending=ascending))
        return d.RankingResult("traffic", "aircraft_operators", "Estonia",
                               1, "2025-12-31", "Y2D",
                               rows=[{"name": "airBaltic", "avgValue": 26.7}])

    ans = da.answer_dataapp_question(
        "busiest airline in Estonia in 2025?",
        client=_FakeClient(spec, "airBaltic."), fetch_rank=_fetch_rank,
    )
    assert ans.ok
    assert captured["category"] == "aircraft_operators"
    assert captured["scope_kind"] == "country"
    assert captured["scope_query"] == "Estonia"
    assert captured["date_range"] == "Y2D"
    assert captured["ascending"] is False


def test_ranking_branch_rejects_co2():
    spec = {"query_kind": "ranking", "metric": "co2",
            "ranking_category": "states", "order": "highest"}
    ans = da.answer_dataapp_question(
        "which country had the most CO2 today?", client=_FakeClient(spec),
    )
    assert not ans.ok
    assert "co2" in ans.answer.lower()


def test_ranking_branch_rejects_missing_category():
    spec = {"query_kind": "ranking", "metric": "traffic",
            "ranking_category": None, "order": "highest"}
    ans = da.answer_dataapp_question("what is the busiest thing?", client=_FakeClient(spec))
    assert not ans.ok


def test_ranking_both_extremes_fetches_full_list_and_labels_ends():
    # "highest AND lowest" -> fetch a large window (descending) and hand the
    # narration only the head (highest) and tail (lowest), labelled.
    spec = {"query_kind": "ranking", "metric": "punctuality",
            "ranking_category": "airports", "date": "2026-03-31",
            "period": "Y2D", "order": "both"}
    captured = {}

    def _fetch_rank(metric, category, *, scope_kind=None, scope_query=None,
                    date=None, date_range="DY", ascending=False, limit=15):
        captured["limit"] = limit
        captured["ascending"] = ascending
        # Sorted highest-first (as the real fetch returns for ascending=False).
        rows = [
            {"name": "Ibiza", "avgValue": 0.908},
            {"name": "Bilbao", "avgValue": 0.869},
            {"name": "Paris Le Bourget", "avgValue": 0.603},
        ]
        return d.RankingResult("punctuality", "airports", "network",
                               1, "2026-03-31", "Y2D", rows=rows)

    captured_rows = {}

    class _CapClient(_FakeClient):
        def chat(self, messages, temperature=0.0, json_mode=False):
            # The user prompt carries the rows JSON; stash it to assert both ends.
            captured_rows["prompt"] = messages[-1].content
            return "Ibiza highest, Paris Le Bourget lowest."

    ans = da.answer_dataapp_question(
        "airports with the highest and lowest punctuality in Q1/2026?",
        client=_CapClient(spec), fetch_rank=_fetch_rank,
    )
    assert ans.ok
    assert captured["limit"] >= 100        # full window, not a top-N slice
    assert captured["ascending"] is False  # sorted highest-first
    # Both extremes reach the narration, not the middle row.
    assert "Ibiza" in captured_rows["prompt"]
    assert "Paris Le Bourget" in captured_rows["prompt"]
    assert "Bilbao" not in captured_rows["prompt"]


@pytest.mark.live
def test_live_hamburg_busiest_destination_2024_is_munich():
    # "destination FROM an airport" uses category 'airports' on the airport's sync
    # (NOT airport_pairs, which the airport sync doesn't carry).
    res = _live_or_skip(d.fetch_ranking, "traffic", "airports",
                        scope_kind="airport", scope_query="Hamburg",
                        date="2024-12-31", date_range="Y2D", ascending=False)
    assert res.rows
    assert res.rows[0]["name"] == "Munich"


@pytest.mark.live
def test_live_q1_2026_punctuality_extremes_are_ibiza_and_le_bourget():
    res = _live_or_skip(d.fetch_ranking, "punctuality", "airports",
                        date="2026-03-31", date_range="Y2D", ascending=False, limit=200)
    assert len(res.rows) >= 2
    assert res.rows[0]["name"] == "Ibiza"           # highest
    assert res.rows[-1]["name"] == "Paris Le Bourget"  # lowest


# --- live integration (real API; asserts the actual quiz numbers) ----------

def _live_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except d.DataAppError as exc:
        pytest.skip(f"Data App API unavailable: {exc}")


@pytest.mark.live
def test_live_network_flights_on_2026_03_10():
    res = _live_or_skip(d.fetch_network, "traffic", date="2026-03-10")
    dy = [r for r in res.records
          if r.get("dateRange") == "DY" and r.get("networkType") == "total"]
    assert dy, "no DY total row"
    assert dy[0]["value"] == 24864.0


@pytest.mark.live
def test_live_highest_punctuality_airport_2025_03_10_is_yerevan():
    res = _live_or_skip(d.fetch_ranking, "punctuality", "airports",
                        date="2025-03-10", date_range="DY", ascending=False)
    assert res.rows, "no ranking rows"
    # Yerevan tops (value 1.0); assert it's the top-valued airport.
    top_value = res.rows[0]["value"]
    names_at_top = [r["name"] for r in res.rows if r["value"] == top_value]
    assert "Yerevan" in names_at_top


@pytest.mark.live
def test_live_most_atfm_delay_country_2025_03_10_is_portugal():
    res = _live_or_skip(d.fetch_ranking, "delay", "states",
                        date="2025-03-10", date_range="DY", ascending=False)
    assert res.rows
    assert res.rows[0]["name"] == "Portugal"


@pytest.mark.live
def test_live_busiest_operator_in_estonia_is_airbaltic():
    res = _live_or_skip(d.fetch_ranking, "traffic", "aircraft_operators",
                        scope_kind="country", scope_query="Estonia",
                        date_range="Y2D", ascending=False)
    assert res.rows
    assert res.rows[0]["name"] == "airBaltic"


@pytest.mark.live
def test_live_ba_busiest_pairs_are_heathrow_shuttles():
    # Vintage-sensitive exact daily avg; assert the shape instead: the top pair(s)
    # for British Airways are London Heathrow shuttles (Glasgow/Edinburgh ⟷ LHR).
    res = _live_or_skip(d.fetch_ranking, "traffic", "airport_pairs",
                        scope_kind="aircraft_operator", scope_query="British Airways Group",
                        date="2025-12-31", date_range="Y2D", ascending=False)
    assert res.rows
    top = res.rows[0]["name"] or ""
    assert "London Heathrow" in top
    assert ("Glasgow" in top) or ("Edinburgh" in top)
