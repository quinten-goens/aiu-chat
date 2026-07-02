"""Tests for the Data App period time-series feature (#6): range fetch, the
manipulator (aggregate/derive with validated SQL), and the agent branch that
ties fetch -> manipulate -> visualise together.

Layers:
  * unit   — _clamp_period / _pick_day_record / range param building (fake session)
  * manip  — run_transform over frames (resample, per-flight ratio, safety)
  * agent  — _answer_timeseries branch with fakes (single & multi-metric)
  * live   — end-to-end against the real API (marked `live`; skipped if offline)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiu_chat import config
from aiu_chat.agent import agg_tool
from aiu_chat.agent import dataapp_answer as da
from aiu_chat.agent.sql_tool import UnsafeSQLError
from aiu_chat.sources import dataapp as d


# --- unit: period bounds + day-record picker -------------------------------

def test_clamp_period_orders_and_caps(monkeypatch):
    # Swapped bounds are ordered; nothing is truncated when within the cap.
    monkeypatch.setattr(config, "MAX_PERIOD_DAYS", 370)
    assert d._clamp_period("2026-05-01", "2026-01-01") == ("2026-01-01", "2026-05-01", False)
    # Over the cap -> end trimmed forward from the start, truncated flag set.
    monkeypatch.setattr(config, "MAX_PERIOD_DAYS", 10)
    s, e, trunc = d._clamp_period("2026-01-01", "2026-12-31")
    assert (s, e, trunc) == ("2026-01-01", "2026-01-10", True)


def test_pick_day_record_prefers_dy_total():
    recs = [
        {"dateRange": "Y2D", "networkType": "total", "value": 1},
        {"dateRange": "DY", "networkType": "avg", "value": 2},
        {"dateRange": "DY", "networkType": "total", "value": 3},
    ]
    assert d._pick_day_record(recs)["value"] == 3
    assert d._pick_day_record([{"dateRange": "MM", "value": 9}])["value"] == 9
    assert d._pick_day_record([]) is None


# --- unit: range /syncs param building (fake session) ----------------------

def _fake_session(payload):
    session = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = payload
    session.get.return_value = resp
    return session


def test_find_syncs_in_range_builds_range_filter_and_dedups():
    session = _fake_session({"data": [
        {"id": 1, "syncDate": "2026-01-01T00:00:00+00:00"},
        {"id": 2, "syncDate": "2026-01-02T00:00:00+00:00"},
        {"id": 3, "syncDate": "2026-01-02T06:00:00+00:00"},  # dup day -> dropped
    ]})
    out = d.find_syncs_in_range(session, start="2026-01-01", end="2026-01-31")
    assert out == [(1, "2026-01-01"), (2, "2026-01-02")]
    params = session.get.call_args.kwargs["params"]
    assert params["dataType"] == "network-wide"
    assert params["syncDate[after]"] == "2026-01-01"
    assert params["syncDate[before]"] == "2026-01-31"
    assert params["order[syncDate]"] == "asc"


def test_fetch_timeseries_reads_one_dy_value_per_day():
    session = MagicMock()
    syncs_resp = MagicMock(status_code=200)
    syncs_resp.json.return_value = {"data": [
        {"id": 10, "syncDate": "2026-01-01"},
        {"id": 11, "syncDate": "2026-01-02"},
    ]}
    day1 = MagicMock(status_code=200)
    day1.json.return_value = {"data": [{"dateRange": "DY", "networkType": "total", "value": 100.0}]}
    day2 = MagicMock(status_code=200)
    day2.json.return_value = {"data": [{"dateRange": "DY", "networkType": "total", "value": 120.0}]}
    session.get.side_effect = [syncs_resp, day1, day2]

    ts = d.fetch_timeseries("traffic", start="2026-01-01", end="2026-01-02", session=session)
    assert ts.metric == "traffic"
    assert ts.entity.name == "Network"
    assert [r["date"] for r in ts.rows] == ["2026-01-01", "2026-01-02"]
    assert [r["value"] for r in ts.rows] == [100.0, 120.0]
    assert ts.truncated is False


# --- manipulator: run_transform over frames --------------------------------

def _frames():
    delay = pd.DataFrame([
        {"date": "2026-03-01", "value": 100.0}, {"date": "2026-03-02", "value": 200.0},
        {"date": "2026-03-08", "value": 60.0},
    ])
    traffic = pd.DataFrame([
        {"date": "2026-03-01", "value": 10.0}, {"date": "2026-03-02", "value": 20.0},
        {"date": "2026-03-08", "value": 6.0},
    ])
    return {"delay_ts": delay, "traffic_ts": traffic}


def test_transform_per_flight_ratio_joins_two_metrics():
    sql = ("SELECT d.date, d.value * 1.0 / t.value AS delay_per_flight "
           "FROM delay_ts d JOIN traffic_ts t USING(date) ORDER BY d.date")
    out = agg_tool.run_transform(sql, _frames()).dataframe
    assert list(out["delay_per_flight"]) == [10.0, 10.0, 10.0]


def test_transform_weekly_resample():
    sql = ("SELECT date_trunc('week', CAST(date AS DATE)) AS week, SUM(value) AS total "
           "FROM delay_ts GROUP BY week ORDER BY week")
    out = agg_tool.run_transform(sql, _frames()).dataframe
    # ISO weeks start Monday: 2026-03-01 is a Sunday (its own week), while
    # 2026-03-02 (Mon) and 2026-03-08 (Sun) fall in the same Mon-Sun week.
    assert list(out["total"]) == [100.0, 260.0]
    assert len(out) == 2


def test_transform_rejects_non_select():
    for bad in ("DROP TABLE delay_ts", "SELECT * FROM delay_ts; DROP TABLE traffic_ts",
                "SELECT * FROM read_parquet('x.parquet')"):
        with pytest.raises(UnsafeSQLError):
            agg_tool.run_transform(bad, _frames())


def test_transform_rejects_unknown_table():
    with pytest.raises(UnsafeSQLError):
        agg_tool.run_transform("SELECT * FROM secrets", _frames())


# --- agent branch: _answer_timeseries --------------------------------------

class _TsClient:
    """chat_json returns queued dicts in order (extract spec, then chart spec);
    chat returns queued strings in order (manipulator SQL?, then narration)."""

    def __init__(self, json_seq, chat_seq):
        self._json = list(json_seq)
        self._chat = list(chat_seq)

    def chat_json(self, messages, temperature=0.0):
        return self._json.pop(0) if self._json else {}

    def chat(self, messages, temperature=0.0, json_mode=False):
        return self._chat.pop(0) if self._chat else "ANSWER"


def test_timeseries_single_metric_raw_series(monkeypatch):
    monkeypatch.setattr(config, "DATAAPP_TIMESERIES", True)
    spec = {"query_kind": "timeseries", "metric": "traffic", "metrics": ["traffic"],
            "start": "2026-01-01", "end": "2026-01-03", "transform": None, "entities": []}

    def fake_ts(metric, *, start, end, kind=None, query=None):
        rows = [{"date": "2026-01-01", "value": 100.0, "avgValue": None},
                {"date": "2026-01-02", "value": 110.0, "avgValue": None},
                {"date": "2026-01-03", "value": 120.0, "avgValue": None}]
        return d.TimeseriesResult("traffic", d.Entity("network", 0, "Network", ""),
                                  start, end, rows=rows)

    # chart spec (chat_json #2), then narration (chat #1). No manipulator (single
    # metric, no transform) -> chat is only called for narration.
    client = _TsClient(
        json_seq=[spec, {"show_chart": True, "chart_type": "line", "x": "date", "y": ["value"]}],
        chat_seq=["Traffic rose from 100 to 120 over 1-3 Jan 2026."],
    )
    ans = da.answer_dataapp_question("daily traffic 1-3 Jan 2026", client=client, fetch_ts=fake_ts)
    assert ans.ok
    assert ans.timeseries is not None
    ts = ans.timeseries
    assert ts.metric_line == "traffic"
    assert ts.transform_sql is None            # raw series, no manipulation
    assert len(ts.dataframe) == 3
    assert ts.chart_spec is not None           # a valid line chart over the frame
    assert "100" in ans.answer


def test_timeseries_multi_metric_runs_manipulator(monkeypatch):
    monkeypatch.setattr(config, "DATAAPP_TIMESERIES", True)
    spec = {"query_kind": "timeseries", "metric": "delay",
            "metrics": ["delay", "traffic"], "start": "2026-03-01", "end": "2026-03-02",
            "transform": "delay minutes divided by number of flights per day",
            "entities": []}

    def fake_ts(metric, *, start, end, kind=None, query=None):
        vals = {"delay": [100.0, 200.0], "traffic": [10.0, 20.0]}[metric]
        rows = [{"date": "2026-03-01", "value": vals[0], "avgValue": None},
                {"date": "2026-03-02", "value": vals[1], "avgValue": None}]
        return d.TimeseriesResult(metric, d.Entity("network", 0, "Network", ""),
                                  start, end, rows=rows)

    manip_sql = ("SELECT d.date, d.value*1.0/t.value AS delay_per_flight "
                 "FROM delay_ts d JOIN traffic_ts t USING(date) ORDER BY d.date")
    # chat #1 = manipulator SQL, chat #2 = narration.
    # chat_json #1 = extract spec, #2 = chart spec.
    client = _TsClient(
        json_seq=[spec, {"show_chart": True, "chart_type": "line", "x": "date",
                         "y": ["delay_per_flight"]}],
        chat_seq=[manip_sql, "Delay per flight held at 10 minutes."],
    )
    ans = da.answer_dataapp_question(
        "daily delay per flight 1-2 Mar 2026", client=client, fetch_ts=fake_ts)
    assert ans.ok
    ts = ans.timeseries
    assert ts.metric_line == "delay, traffic"
    assert ts.transform_sql == manip_sql           # the manipulator ran
    assert "delay_per_flight" in ts.dataframe.columns
    assert list(ts.dataframe["delay_per_flight"]) == [10.0, 10.0]


def test_timeseries_rejects_bad_dates(monkeypatch):
    monkeypatch.setattr(config, "DATAAPP_TIMESERIES", True)
    spec = {"query_kind": "timeseries", "metric": "traffic", "metrics": ["traffic"],
            "start": "not-a-date", "end": None, "entities": []}
    client = _TsClient(json_seq=[spec], chat_seq=[])
    ans = da.answer_dataapp_question("daily traffic lately", client=client,
                                     fetch_ts=lambda *a, **k: None)
    assert not ans.ok
    assert "date range" in ans.answer.lower()


# --- live integration ------------------------------------------------------

def _live_or_skip(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except d.DataAppError as exc:
        pytest.skip(f"Data App API unavailable: {exc}")


@pytest.mark.live
def test_live_network_daily_series_jan_to_mar_2026():
    ts = _live_or_skip(d.fetch_timeseries, "traffic",
                       start="2026-01-01", end="2026-03-01")
    assert ts.rows, "no daily rows returned"
    # A contiguous-ish daily series: dates sorted, values positive.
    dates = [r["date"] for r in ts.rows]
    assert dates == sorted(dates)
    assert all((r["value"] or 0) > 0 for r in ts.rows)
    # The known 10 March-adjacent value sanity: 1 Jan should be present.
    assert dates[0] >= "2026-01-01"
