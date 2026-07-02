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


# --- clarification ---------------------------------------------------------

class _SeqClient:
    """Returns queued chat_json dicts in order (route call, then clarify call)."""

    def __init__(self, json_seq, chat_text=""):
        self._seq = list(json_seq)
        self.chat_text = chat_text

    def chat(self, messages, temperature=0.0, json_mode=False):
        return self.chat_text

    def chat_json(self, messages, temperature=0.0):
        return self._seq.pop(0)


def test_needs_clarification_returns_question():
    from aiu_chat.agent.orchestrator import needs_clarification
    c = _SeqClient([{"needs_clarification": True, "question": "Which airport?"}])
    assert needs_clarification("show me the delays", "data", c) == "Which airport?"


def test_needs_clarification_false_proceeds():
    from aiu_chat.agent.orchestrator import needs_clarification
    c = _SeqClient([{"needs_clarification": False}])
    assert needs_clarification("CO2 in 2024", "data", c) is None


def test_needs_clarification_skipped_for_concept_route():
    from aiu_chat.agent.orchestrator import needs_clarification
    # concept route is not clarifiable; no LLM call needed
    assert needs_clarification("what is ATFM?", "concept", FakeClient()) is None


def test_clarification_turn_short_circuits_dispatch():
    # route=data, then clarify asks a question -> no data path called.
    client = _SeqClient([
        {"route": "data"},
        {"needs_clarification": True, "question": "Which airport?"},
    ])
    with patch.object(orch, "answer_data_question") as md:
        turn = answer("show me the delays", client=client, catalog=object())
    md.assert_not_called()
    assert turn.needs_clarification is True
    assert turn.answer == "Which airport?"


def test_route_accepts_new_sources():
    for r in ("nop", "dataapp", "nm_live", "none"):
        assert route_question("q", FakeClient(json_obj={"route": r})) == r


# --- multi-source planning -------------------------------------------------

def test_plan_routes_returns_multiple():
    from aiu_chat.agent.orchestrator import plan_routes
    c = FakeClient(json_obj={"routes": ["dataapp", "data", "concept"]})
    assert plan_routes("q", c, max_routes=3) == ["dataapp", "data", "concept"]


def test_plan_routes_caps_and_dedups():
    from aiu_chat.agent.orchestrator import plan_routes
    c = FakeClient(json_obj={"routes": ["data", "data", "concept", "nm_live"]})
    assert plan_routes("q", c, max_routes=2) == ["data", "concept"]


def test_plan_routes_none_is_terminal():
    from aiu_chat.agent.orchestrator import plan_routes
    c = FakeClient(json_obj={"routes": ["data", "none"]})
    assert plan_routes("q", c, max_routes=3) == ["none"]


def test_multi_source_dispatch_runs_each_and_synthesizes():
    # Router picks data + dataapp; both paths run, and the synthesis pass (via
    # chat_text) produces the combined answer.
    client = FakeClient(json_obj={"routes": ["data", "dataapp"]},
                        chat_text="Combined: 100 tonnes historically; 5 today.")
    with patch.object(orch, "answer_data_question", return_value=_data_answer("100 tonnes")), \
         patch.object(orch, "answer_dataapp_question",
                      return_value=_dataapp_answer("5 flights today")):
        turn = answer("history and today for X", client=client, catalog=object())
    assert turn.routes == ["data", "dataapp"]
    assert turn.data is not None and turn.dataapp is not None
    assert turn.answer == "Combined: 100 tonnes historically; 5 today."


def test_multi_source_falls_back_to_concat_when_synthesis_empty():
    # Empty chat_text -> synthesis returns None -> concatenation of sub-answers.
    client = FakeClient(json_obj={"routes": ["data", "dataapp"]}, chat_text="")
    with patch.object(orch, "answer_data_question", return_value=_data_answer("100 tonnes")), \
         patch.object(orch, "answer_dataapp_question",
                      return_value=_dataapp_answer("5 flights today")):
        turn = answer("history and today for X", client=client, catalog=object())
    assert "100 tonnes" in turn.answer and "5 flights today" in turn.answer


# --- cross-frame aggregation (#4) ------------------------------------------

def test_aggregation_runs_when_enabled(monkeypatch):
    import pandas as pd
    from aiu_chat import config
    from aiu_chat.agent.text_to_sql import DataAnswer
    from aiu_chat.agent.sql_tool import SqlResult
    monkeypatch.setattr(config, "AGGREGATION", True)

    # A data frame with 3 rows to aggregate.
    df = pd.DataFrame([{"state": "FR", "co2": 100}, {"state": "DE", "co2": 200},
                       {"state": "ES", "co2": 300}])
    data = DataAnswer(question="q", sql="SELECT ...",
                      result=SqlResult(sql="SELECT ...", dataframe=df, row_count=3, truncated=False),
                      answer="per-state figures", ok=True)

    # Client: route=data; then agg-SQL gen returns a SUM; then narration.
    class AggClient:
        def __init__(self):
            self.chat_calls = 0
        def chat_json(self, messages, temperature=0.0):
            return {"routes": ["data"]}
        def chat(self, messages, temperature=0.0, json_mode=False):
            self.chat_calls += 1
            # First chat() after dispatch is the agg SQL; second is narration.
            if self.chat_calls == 1:
                return "SELECT SUM(co2) AS total FROM data"
            return "The combined total is 600."

    with patch.object(orch, "answer_data_question", return_value=data):
        turn = answer("combined CO2 across these states", client=AggClient(), catalog=object())

    assert turn.aggregate is not None
    assert turn.aggregate.dataframe.iloc[0]["total"] == 600
    assert "combined total is 600" in turn.answer.lower()


def test_aggregation_skipped_when_disabled(monkeypatch):
    import pandas as pd
    from aiu_chat import config
    from aiu_chat.agent.text_to_sql import DataAnswer
    from aiu_chat.agent.sql_tool import SqlResult
    monkeypatch.setattr(config, "AGGREGATION", False)
    df = pd.DataFrame([{"state": "FR", "co2": 100}, {"state": "DE", "co2": 200}])
    data = DataAnswer(question="q", sql="s",
                      result=SqlResult(sql="s", dataframe=df, row_count=2, truncated=False),
                      answer="figures", ok=True)
    client = FakeClient(json_obj={"routes": ["data"]}, chat_text="x")
    with patch.object(orch, "answer_data_question", return_value=data):
        turn = answer("combined CO2", client=client, catalog=object())
    assert turn.aggregate is None


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


def _dataapp_answer(text="5 flights today", ok=True):
    from aiu_chat.agent.dataapp_answer import DataAppAnswer
    return DataAppAnswer(question="q", answer=text, result=None, ok=ok)


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
    # "both" expands to the data + concept route set; the primary label is data.
    client = FakeClient(json_obj={"route": "both"})
    with patch.object(orch, "answer_data_question", return_value=_data_answer("100 tonnes")), \
         patch.object(orch, "answer_concept_question", return_value=_concept_answer("ASMA means X")):
        turn = answer("co2 and what is asma?", client=client, catalog=object())
    assert turn.routes == ["data", "concept"]
    # FakeClient.chat() returns "" so synthesis is skipped -> concatenation.
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


def test_nm_live_route_dispatches_to_nm():
    from aiu_chat.agent.nm_answer import NmLiveAnswer
    client = FakeClient(json_obj={"route": "nm_live"})
    with patch.object(orch, "answer_nm_question",
                      return_value=NmLiveAnswer("q", "5113 airborne now", ok=True)) as mnm, \
         patch.object(orch, "answer_data_question") as md:
        turn = answer("how many airborne right now?", client=client, catalog=object())
    mnm.assert_called_once()
    md.assert_not_called()
    assert turn.route == "nm_live"
    assert "5113" in turn.answer


def test_none_route_declines_without_calling_paths():
    client = FakeClient(json_obj={"route": "none"})
    with patch.object(orch, "answer_data_question") as md, \
         patch.object(orch, "answer_nm_question") as mnm:
        turn = answer("what is the GDP of Japan?", client=client, catalog=object())
    md.assert_not_called()
    mnm.assert_not_called()
    assert turn.route == "none"
    assert "outside" in turn.answer.lower()
