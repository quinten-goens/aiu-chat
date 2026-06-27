"""Tests for the conversation-bug fixes (chart intent, availability, sampling)."""
from __future__ import annotations

import json

import pandas as pd

from aiu_chat.agent.orchestrator import _is_availability_question
from aiu_chat.agent.text_to_sql import _rows_to_json, wants_chart


# --- #3 explicit chart intent ----------------------------------------------

def test_wants_chart_detects_requests():
    for q in [
        "make it a bar chart",
        "could you plot this",
        "show me a graph of traffic",
        "add two bars for arrivals and departures",
        "visualise the trend",
    ]:
        assert wants_chart(q), q


def test_wants_chart_false_for_plain_questions():
    for q in ["which state had the most CO2?", "total flights in 2024", "what is ATFM?"]:
        assert not wants_chart(q), q


# --- #5 availability questions go to the catalog ---------------------------

def test_availability_question_detected():
    for q in [
        "What data do you have?",
        "Which datasets are available?",
        "what data is available for Vertical Flight Efficiency",
        "what reports do you have access to",
    ]:
        assert _is_availability_question(q), q


def test_normal_questions_not_availability():
    for q in ["which state emitted the most CO2 in 2025?", "how is ASMA time calculated?"]:
        assert not _is_availability_question(q), q


# --- #2 large-result sampling sends head AND tail --------------------------

def test_small_result_sent_whole():
    df = pd.DataFrame({"y": list(range(2024, 2027))})
    out = json.loads(_rows_to_json(df, max_rows=100))
    assert isinstance(out, list)
    assert len(out) == 3


def test_large_result_includes_head_and_tail():
    df = pd.DataFrame({"YEAR": list(range(2008, 2027)), "v": list(range(19))})
    out = json.loads(_rows_to_json(df, max_rows=6))
    assert "_note" in out
    # tail must contain the latest year so the model can't claim data "ends" early
    tail_years = [r["YEAR"] for r in out["tail"]]
    assert 2026 in tail_years
    head_years = [r["YEAR"] for r in out["head"]]
    assert 2008 in head_years
