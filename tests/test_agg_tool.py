"""Tests for the cross-frame aggregation executor (aiu_chat.agent.agg_tool).

Hand-written aggregation SQL over sample frames — no model. Verifies the
arithmetic runs in DuckDB, the frame-view allow-set is enforced, and the shared
safety checks (single SELECT, no DDL/DML, no catalog tables) still apply.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiu_chat.agent import agg_tool
from aiu_chat.agent.sql_tool import UnsafeSQLError


def _frames():
    return {
        "dataapp": pd.DataFrame([
            {"entity": "France", "dateRange": "DY", "value": 3000},
            {"entity": "Germany", "dateRange": "DY", "value": 2500},
            {"entity": "Spain", "dateRange": "DY", "value": 2000},
        ])
    }


def test_combined_total():
    res = agg_tool.run_aggregation(
        "SELECT SUM(value) AS total FROM dataapp WHERE dateRange='DY'", _frames())
    assert res.dataframe.iloc[0]["total"] == 7500


def test_ranking():
    res = agg_tool.run_aggregation(
        "SELECT entity, value FROM dataapp ORDER BY value DESC", _frames())
    assert list(res.dataframe["entity"]) == ["France", "Germany", "Spain"]


def test_rejects_catalog_table():
    with pytest.raises(UnsafeSQLError, match="Unknown table"):
        agg_tool.run_aggregation("SELECT * FROM co2_emissions_by_state", _frames())


def test_rejects_ddl():
    with pytest.raises(UnsafeSQLError):
        agg_tool.run_aggregation("DROP TABLE dataapp", _frames())


def test_rejects_multiple_statements():
    with pytest.raises(UnsafeSQLError, match="one statement"):
        agg_tool.run_aggregation(
            "SELECT * FROM dataapp; SELECT 1", _frames())


def test_no_frames_raises():
    with pytest.raises(UnsafeSQLError, match="No frames"):
        agg_tool.run_aggregation("SELECT 1", {})


def test_two_frames_allowed():
    frames = {
        "data": pd.DataFrame([{"y": 2024, "co2": 100}]),
        "dataapp": pd.DataFrame([{"entity": "FR", "value": 5}]),
    }
    # A query touching both views is allowed (both are registered).
    res = agg_tool.run_aggregation(
        "SELECT (SELECT SUM(co2) FROM data) AS hist, "
        "(SELECT SUM(value) FROM dataapp) AS live", frames)
    assert res.dataframe.iloc[0]["hist"] == 100
    assert res.dataframe.iloc[0]["live"] == 5


def test_collect_frames_from_turn():
    from types import SimpleNamespace
    # A turn with a SQL dataframe and fan-out dataapp results.
    sql_df = pd.DataFrame([{"y": 2024, "co2": 100}])
    data = SimpleNamespace(result=SimpleNamespace(dataframe=sql_df))
    r1 = SimpleNamespace(entity=SimpleNamespace(name="France"),
                         records=[{"dateRange": "DY", "value": 3000}])
    dataapp = SimpleNamespace(results=[r1])
    turn = SimpleNamespace(data=data, dataapp=dataapp)

    frames = agg_tool.collect_frames(turn)
    assert set(frames) == {"data", "dataapp"}
    assert frames["dataapp"].iloc[0]["entity"] == "France"
    assert frames["data"].iloc[0]["co2"] == 100
