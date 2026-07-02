"""Tests for the entity resolver (aiu_chat.agent.entities).

Runs against a small synthetic entities.json so it's fast and deterministic and
doesn't depend on the full built index.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiu_chat.agent import entities as ent


@pytest.fixture
def index(tmp_path: Path, monkeypatch):
    """Write a tiny entities.json and point the resolver at it."""
    data = {
        "entities": [
            {"entity_id": "state:GB", "kind": "state", "canonical_name": "United Kingdom",
             "icao": "EG", "iata": "", "state_id": "", "lat": None, "lon": None,
             "filter_values": {"co2_emissions_by_state": "UNITED KINGDOM",
                               "airport_traffic": "United Kingdom"}},
            {"entity_id": "apt:EGLL", "kind": "airport", "canonical_name": "London Heathrow Airport",
             "icao": "EGLL", "iata": "LHR", "state_id": "state:GB", "lat": 51.47, "lon": -0.46,
             "filter_values": {"airport_traffic": "EGLL"}},
            {"entity_id": "apt:LFPG", "kind": "airport", "canonical_name": "Charles de Gaulle",
             "icao": "LFPG", "iata": "CDG", "state_id": "state:FR", "lat": 49.0, "lon": 2.55,
             "filter_values": {"airport_traffic": "LFPG"}},
        ],
        "aliases": [
            {"alias": "gb", "entity_id": "state:GB"},
            {"alias": "uk", "entity_id": "state:GB"},
            {"alias": "united kingdom", "entity_id": "state:GB"},
            {"alias": "eg", "entity_id": "state:GB"},
            {"alias": "egll", "entity_id": "apt:EGLL"},
            {"alias": "lhr", "entity_id": "apt:EGLL"},
            {"alias": "heathrow", "entity_id": "apt:EGLL"},
            {"alias": "london heathrow", "entity_id": "apt:EGLL"},
            {"alias": "london", "entity_id": "apt:EGLL"},
            {"alias": "lfpg", "entity_id": "apt:LFPG"},
            {"alias": "cdg", "entity_id": "apt:LFPG"},
            {"alias": "charles de gaulle", "entity_id": "apt:LFPG"},
            {"alias": "paris charles de gaulle", "entity_id": "apt:LFPG"},
        ],
    }
    p = tmp_path / "entities.json"
    p.write_text(json.dumps(data))
    monkeypatch.setattr(ent.config, "DATA_DIR", tmp_path)
    ent.reload()
    yield
    ent.reload()


def test_resolve_alias_and_iata(index):
    assert ent.resolve("heathrow").entity_id == "apt:EGLL"
    assert ent.resolve("LHR").entity_id == "apt:EGLL"
    assert ent.resolve("egll").entity_id == "apt:EGLL"


def test_resolve_name_variant_normalised(index):
    assert ent.resolve("Charles-de-Gaulle").entity_id == "apt:LFPG"
    assert ent.resolve("Paris - Charles de Gaulle").entity_id == "apt:LFPG"


def test_resolve_state_by_multiple_aliases(index):
    for alias in ("UK", "United Kingdom", "GB"):
        assert ent.resolve(alias, kind="state").entity_id == "state:GB"


def test_filter_value_reconciles_mismatch(index):
    # The smoking gun: same entity, different literal per table.
    assert ent.filter_value("state:GB", "co2_emissions_by_state") == "UNITED KINGDOM"
    assert ent.filter_value("state:GB", "airport_traffic") == "United Kingdom"


def test_filter_value_unknown_returns_none(index):
    assert ent.filter_value("state:GB", "no_such_table") is None
    assert ent.filter_value("state:ZZ", "airport_traffic") is None


def test_find_in_question_prefers_specific(index):
    hits = ent.find_in_question("How busy is London Heathrow?")
    assert hits and hits[0].entity_id == "apt:EGLL"


def test_find_in_question_multiple_entities(index):
    hits = ent.find_in_question("Compare CDG and Heathrow traffic")
    ids = {h.entity_id for h in hits}
    assert "apt:LFPG" in ids and "apt:EGLL" in ids


def test_short_code_alias_ignored_in_freetext(index):
    # "eg" (UK ICAO prefix, 2 chars) must NOT fire inside free text at all.
    hits = ent.find_in_question("There is a beginning to everything")
    assert all(h.entity_id != "state:GB" for h in hits)


def test_alias_matches_on_word_boundary_only(index):
    # "cdg" must not fire inside a larger token, and a name fragment must not
    # match mid-word.
    assert ent.find_in_question("The abcdge code is irrelevant") == []
    # But a real whole-word mention resolves.
    hits = ent.find_in_question("How much CO2 for CDG?")
    assert any(h.entity_id == "apt:LFPG" for h in hits)


def test_sql_prompt_hint_airport(index):
    hint = ent.sql_prompt_hint("Show Heathrow traffic")
    assert "Resolved entities" in hint
    assert "APT_ICAO = 'EGLL'" in hint


def test_sql_prompt_hint_state_mismatch(index):
    hint = ent.sql_prompt_hint("CO2 and traffic for the United Kingdom")
    # Must surface the differing literals so SQL filters correctly per table.
    assert "UNITED KINGDOM" in hint
    assert "United Kingdom" in hint


def test_sql_prompt_hint_empty_when_no_entity(index):
    assert ent.sql_prompt_hint("What is ASMA additional time?") == ""


def test_unknown_entity_returns_none(index):
    assert ent.resolve("Atlantis International") is None
    assert ent.candidates("nowhere") == []


def test_missing_index_is_safe(tmp_path, monkeypatch):
    # No entities.json at all -> resolver degrades to empty, never raises.
    monkeypatch.setattr(ent.config, "DATA_DIR", tmp_path)
    ent.reload()
    try:
        assert ent.resolve("heathrow") is None
        assert ent.find_in_question("Heathrow traffic") == []
    finally:
        ent.reload()
