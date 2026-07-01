"""Tests for the turn -> PocketBase record serializer (aiu_chat.agent.turn_log).

Pure logic (no network): verifies every renderable part of a Turn is captured,
NaN/Inf are scrubbed (PocketBase rejects them), and missing/partial paths don't
crash the serializer — logging must be robust to any Turn shape.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd

from aiu_chat.agent.turn_log import build_turn_record


def _data_turn():
    df = pd.DataFrame({"state": ["FR", "DE"], "co2": [1.5, float("nan")]})
    result = SimpleNamespace(dataframe=df, row_count=2, truncated=False, sql="SELECT ...")
    data = SimpleNamespace(
        sql="SELECT state, co2 FROM t",
        chart_spec={"show_chart": True, "chart_type": "bar", "x": "state", "y": ["co2"]},
        result=result,
    )
    return SimpleNamespace(
        question="Which states emit most CO2?",
        standalone_question="Which states emit most CO2 in 2024?",
        route="data",
        needs_clarification=False,
        answer="France leads.",
        sources=[],
        data=data, concept=None, nop=None, dataapp=None, nm_live=None,
    )


def test_captures_core_fields():
    rec = build_turn_record(_data_turn(), turn_index=3, model_tier="gpt_max", latency_ms=1200)
    assert rec["turn_index"] == 3
    assert rec["route"] == "data"
    assert rec["question"].startswith("Which states")
    assert rec["standalone_question"] != rec["question"]
    assert rec["answer"] == "France leads."
    assert rec["model_tier"] == "gpt_max"
    assert rec["latency_ms"] == 1200
    assert rec["sql"] == "SELECT state, co2 FROM t"
    assert rec["row_count"] == 2
    assert rec["chart_spec"]["chart_type"] == "bar"
    assert "created_at" in rec


def test_nan_is_scrubbed_from_table():
    rec = build_turn_record(_data_turn(), turn_index=0, model_tier="x", latency_ms=1)
    table = rec["result_table"]
    assert table == [{"state": "FR", "co2": 1.5}, {"state": "DE", "co2": None}]
    # No NaN/Inf floats survive (PocketBase would 400 on them).
    for row in table:
        for v in row.values():
            assert not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def test_error_turn_without_data():
    rec = build_turn_record(
        SimpleNamespace(
            question="q", standalone_question="q", route="data",
            needs_clarification=False, answer="", sources=[],
            data=None, concept=None, nop=None, dataapp=None, nm_live=None,
        ),
        turn_index=0, model_tier="gpt_max", latency_ms=50, error="boom",
    )
    assert rec["error"] == "boom"
    assert "sql" not in rec  # no data path -> no SQL captured
    assert rec["result_table"] if "result_table" in rec else True  # optional


def test_live_payload_nop():
    nop = SimpleNamespace(messages=[
        SimpleNamespace(id="1", type="TACTICAL", published="2026-07-01", text="CB over EDDF")
    ])
    turn = SimpleNamespace(
        question="situation?", standalone_question="situation?", route="nop",
        needs_clarification=False, answer="Storms.", sources=[],
        data=None, concept=None, nop=nop, dataapp=None, nm_live=None,
    )
    rec = build_turn_record(turn, turn_index=0, model_tier="x", latency_ms=1)
    assert rec["live_payload"]["nop"][0]["type"] == "TACTICAL"


def test_sources_serialised():
    src = SimpleNamespace(source_title="ASMA methodology",
                          source_url="https://x/asma", text="text", score=0.9)
    turn = SimpleNamespace(
        question="what is asma?", standalone_question="what is asma?", route="concept",
        needs_clarification=False, answer="It's ...", sources=[src],
        data=None, concept=None, nop=None, dataapp=None, nm_live=None,
    )
    rec = build_turn_record(turn, turn_index=0, model_tier="x", latency_ms=1)
    assert rec["sources"][0]["title"] == "ASMA methodology"
    assert rec["sources"][0]["url"] == "https://x/asma"
