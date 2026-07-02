"""Tests for Data App per-source fan-out (aiu_chat.agent.dataapp_answer).

The LLM extract and the API fetch are both faked, so these test only the
fan-out orchestration: multi-entity extraction, per-entity fetch loop, isolated
failures, and merged narration.
"""
from __future__ import annotations

import pytest

from aiu_chat.agent import dataapp_answer as da
from aiu_chat.sources.dataapp import DataAppError, DataAppResult, Entity


class FakeClient:
    def __init__(self, spec, chat_text="ANSWER"):
        self.spec = spec
        self.chat_text = chat_text

    def chat_json(self, messages, temperature=0.0):
        return self.spec

    def chat(self, messages, temperature=0.0, json_mode=False):
        return self.chat_text


def _result(name, value=100):
    return DataAppResult(
        metric="traffic",
        entity=Entity(kind="country", id=1, name=name, code=name[:2].upper()),
        sync_id=1, sync_date="2026-06-30",
        records=[{"networkType": "total", "dateRange": "DY", "value": value}],
    )


def _fetch_factory(mapping, fail=()):
    def _fetch(metric, kind, entity):
        if entity in fail:
            raise DataAppError(f"could not resolve {entity}")
        return mapping.get(entity, _result(entity))
    return _fetch


# --- extraction ------------------------------------------------------------

def test_extract_entities_list():
    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": "France"},
        {"entity_kind": "country", "entity": "Germany"},
    ]}
    assert da._extract_entities(spec) == [("country", "France"), ("country", "Germany")]


def test_extract_entities_backcompat_single():
    spec = {"metric": "traffic", "entity_kind": "airport", "entity": "EGLL"}
    assert da._extract_entities(spec) == [("airport", "EGLL")]


def test_extract_entities_drops_invalid_and_dedups():
    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": "France"},
        {"entity_kind": "bogus", "entity": "X"},
        {"entity_kind": "country", "entity": "France"},  # dup
        {"entity_kind": "country", "entity": ""},         # empty
    ]}
    assert da._extract_entities(spec) == [("country", "France")]


# --- fan-out ---------------------------------------------------------------

def test_fanout_fetches_each_entity(monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "FANOUT", True)
    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": "France"},
        {"entity_kind": "country", "entity": "Germany"},
        {"entity_kind": "country", "entity": "Spain"},
    ]}
    ans = da.answer_dataapp_question(
        "compare traffic for FR, DE, ES",
        client=FakeClient(spec),
        fetch=_fetch_factory({}),
    )
    assert ans.ok
    assert len(ans.results) == 3
    assert {r.entity.name for r in ans.results} == {"France", "Germany", "Spain"}
    assert ans.result is ans.results[0]  # back-compat single result


def test_fanout_isolates_a_failing_entity(monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "FANOUT", True)
    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": "France"},
        {"entity_kind": "country", "entity": "Atlantis"},  # fails
    ]}
    ans = da.answer_dataapp_question(
        "traffic for France and Atlantis",
        client=FakeClient(spec),
        fetch=_fetch_factory({}, fail={"Atlantis"}),
    )
    assert ans.ok                       # France still answered
    assert len(ans.results) == 1
    assert "Atlantis" in ans.answer     # failure reported


def test_fanout_all_fail_returns_not_ok(monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "FANOUT", True)
    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": "Nowhere"},
    ]}
    ans = da.answer_dataapp_question(
        "traffic for Nowhere",
        client=FakeClient(spec),
        fetch=_fetch_factory({}, fail={"Nowhere"}),
    )
    assert not ans.ok


def test_fanout_disabled_uses_first_entity_only(monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "FANOUT", False)
    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": "France"},
        {"entity_kind": "country", "entity": "Germany"},
    ]}
    ans = da.answer_dataapp_question(
        "traffic for France and Germany",
        client=FakeClient(spec),
        fetch=_fetch_factory({}),
    )
    assert len(ans.results) == 1
    assert ans.results[0].entity.name == "France"


def test_fanout_respects_max_cap(monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "FANOUT", True)
    monkeypatch.setattr(config, "MAX_FANOUT", 2)
    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": e}
        for e in ("France", "Germany", "Spain", "Italy")
    ]}
    ans = da.answer_dataapp_question(
        "traffic for four countries",
        client=FakeClient(spec),
        fetch=_fetch_factory({}),
    )
    assert len(ans.results) == 2


def test_merged_records_are_tagged_by_entity(monkeypatch):
    from aiu_chat import config
    monkeypatch.setattr(config, "FANOUT", True)
    captured = {}

    class CapClient(FakeClient):
        def chat(self, messages, temperature=0.0, json_mode=False):
            captured["user"] = messages[-1].content
            return "ANSWER"

    spec = {"metric": "traffic", "entities": [
        {"entity_kind": "country", "entity": "France"},
        {"entity_kind": "country", "entity": "Germany"},
    ]}
    da.answer_dataapp_question(
        "compare FR and DE", client=CapClient(spec), fetch=_fetch_factory({}),
    )
    # The merged records handed to narration carry the entity name.
    assert "France" in captured["user"] and "Germany" in captured["user"]
