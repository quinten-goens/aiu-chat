"""Tests for the orchestrator: routing, follow-up rewriting, and combining.

The path functions (answer_data_question / answer_concept_question) are patched
so we test only the orchestration logic, no model needed.
"""
from __future__ import annotations

from unittest.mock import patch

from aiu_chat.agent import orchestrator as orch
from aiu_chat.agent.concept import ConceptAnswer
from aiu_chat.agent.orchestrator import Turn, answer, rewrite_followup, route_question
from aiu_chat.agent.text_to_sql import DataAnswer


class FakeClient:
    """chat() returns scripted text; chat_json() returns scripted dicts."""

    def __init__(self, *, chat_text="", json_obj=None):
        self.chat_text = chat_text
        self.json_obj = json_obj or {}
        self.json_calls = 0

    def chat(self, messages, temperature=0.0, json_mode=False):
        return self.chat_text

    def chat_json(self, messages, temperature=0.0):
        self.json_calls += 1
        return self.json_obj


# --- routing ---------------------------------------------------------------

def test_route_parses_valid_route():
    assert route_question("q", FakeClient(json_obj={"route": "concept"})) == "concept"


def test_route_defaults_to_data_on_garbage():
    assert route_question("q", FakeClient(json_obj={"route": "nonsense"})) == "data"


def test_route_accepts_new_sources():
    for r in ("nop", "dataapp"):
        assert route_question("q", FakeClient(json_obj={"route": r})) == r


# --- follow-up rewriting ---------------------------------------------------

def test_rewrite_noop_without_history():
    assert rewrite_followup("what about France?", [], FakeClient()) == "what about France?"


def test_rewrite_uses_history():
    history = [Turn("CO2 in Germany?", "CO2 in Germany?", "data", answer="...")]
    client = FakeClient(chat_text="What were CO2 emissions in France?")
    out = rewrite_followup("what about France?", history, client)
    assert out == "What were CO2 emissions in France?"


# --- dispatch + combine ----------------------------------------------------

def _data_answer(text="42 flights", ok=True):
    return DataAnswer(question="q", sql="SELECT 1", result=None, answer=text, ok=ok)


def _concept_answer(text="ASMA is ...", ok=True):
    from aiu_chat.agent.retriever import RetrievedChunk
    src = [RetrievedChunk("t", "u", "Acronyms", 0.9)]
    return ConceptAnswer(question="q", answer=text, sources=src, ok=ok)


def test_data_route_calls_only_data_path():
    client = FakeClient(json_obj={"route": "data"})
    with patch.object(orch, "answer_data_question", return_value=_data_answer()) as md, \
         patch.object(orch, "answer_concept_question") as mc:
        turn = answer("how many flights?", client=client, catalog=object())
    md.assert_called_once()
    mc.assert_not_called()
    assert turn.route == "data"
    assert "42 flights" in turn.answer


def test_both_route_combines_paths_and_sources():
    client = FakeClient(json_obj={"route": "both"})
    with patch.object(orch, "answer_data_question", return_value=_data_answer("100 tonnes")), \
         patch.object(orch, "answer_concept_question", return_value=_concept_answer("ASMA means X")):
        turn = answer("co2 and what is asma?", client=client, catalog=object())
    assert turn.route == "both"
    assert "ASMA means X" in turn.answer
    assert "100 tonnes" in turn.answer
    assert len(turn.sources) == 1  # concept source carried through


def test_concept_route_only_concept():
    client = FakeClient(json_obj={"route": "concept"})
    with patch.object(orch, "answer_data_question") as md, \
         patch.object(orch, "answer_concept_question", return_value=_concept_answer()):
        turn = answer("what is ATFM?", client=client, catalog=object())
    md.assert_not_called()
    assert "ASMA is" in turn.answer


def test_nop_route_dispatches_to_nop():
    from aiu_chat.agent.nop_answer import NopAnswer
    client = FakeClient(json_obj={"route": "nop"})
    with patch.object(orch, "answer_nop_question",
                      return_value=NopAnswer("q", "storms on the network", ok=True)) as mn, \
         patch.object(orch, "answer_data_question") as md:
        turn = answer("any weather warnings now?", client=client, catalog=object())
    mn.assert_called_once()
    md.assert_not_called()
    assert turn.route == "nop"
    assert "storms" in turn.answer


def test_dataapp_route_dispatches_to_dataapp():
    from aiu_chat.agent.dataapp_answer import DataAppAnswer
    client = FakeClient(json_obj={"route": "dataapp"})
    with patch.object(orch, "answer_dataapp_question",
                      return_value=DataAppAnswer("q", "11968 flights today", ok=True)) as mda, \
         patch.object(orch, "answer_data_question") as md:
        turn = answer("flights in France today?", client=client, catalog=object())
    mda.assert_called_once()
    md.assert_not_called()
    assert turn.route == "dataapp"
    assert "11968" in turn.answer
