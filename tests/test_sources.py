"""Tests for the NOP and Data App sources (mocked — no network/model)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiu_chat.agent.dataapp_answer import answer_dataapp_question
from aiu_chat.agent.nop_answer import _keyword, answer_nop_question
from aiu_chat.sources.dataapp import DataAppResult, Entity, resolve_entity
from aiu_chat.sources.nop import NopMessage, _strip_html


# --- NOP --------------------------------------------------------------------

def test_strip_html_to_text():
    html = "<p><h2>Weather:</h2></p><p>ISOL CB over Alps.<br/>Moving NE.</p>"
    text = _strip_html(html)
    assert "Weather:" in text
    assert "ISOL CB over Alps." in text
    assert "<" not in text


def test_keyword_picks_salient_token():
    assert _keyword("Are there any weather messages about thunderstorms?") == "thunderstorms"
    # all-stopwords -> None (fetch latest unfiltered)
    assert _keyword("show me the latest") is None


class _FakeClient:
    def __init__(self, text="answer"):
        self.text = text

    def chat(self, messages, temperature=0.0, json_mode=False):
        return self.text

    def chat_json(self, messages, temperature=0.0):
        return self._json

    _json: dict = {}


def test_nop_answer_uses_fetched_messages():
    msgs = [NopMessage("1", "NOP_tactical_update", "2026-06-27", "CB over Alps.")]
    ans = answer_nop_question(
        "weather?", client=_FakeClient("There are storms."),
        fetch=lambda query=None, limit=5: msgs,
    )
    assert ans.ok
    assert ans.messages == msgs


def test_nop_answer_no_messages():
    ans = answer_nop_question(
        "weather?", client=_FakeClient(), fetch=lambda query=None, limit=5: [],
    )
    assert not ans.ok


# --- Data App ---------------------------------------------------------------

def test_resolve_entity_by_name():
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": 2, "name": "France", "iso2": "FR"}]}
    session.get.return_value = resp
    ent = resolve_entity("country", "France", session)
    assert ent.id == 2
    assert ent.code == "FR"


def test_dataapp_answer_rejects_unmappable():
    client = _FakeClient()
    client._json = {"metric": None}
    ans = answer_dataapp_question("what is the meaning of life?", client=client)
    assert not ans.ok


def test_dataapp_answer_happy_path():
    client = _FakeClient("France had 11968 flights today.")
    client._json = {"metric": "traffic", "entity_kind": "country", "entity": "France"}
    result = DataAppResult(
        metric="traffic", entity=Entity("country", 2, "France", "FR"),
        sync_id=999, sync_date="2026-06-26",
        records=[{"networkType": "total", "dateRange": "DY", "value": 11968.0}],
    )
    ans = answer_dataapp_question(
        "flights in France today?", client=client,
        fetch=lambda metric, kind, entity: result,
    )
    assert ans.ok
    assert ans.result.entity.name == "France"
    assert "11968" in ans.answer


# --- NM live ----------------------------------------------------------------

def test_nm_live_answer_uses_snapshot():
    from aiu_chat.agent.nm_answer import answer_nm_question
    from aiu_chat.sources.nm_live import NmLiveSnapshot, NmRegulation

    snap = NmLiveSnapshot(
        airborne=5113, landed=21804, planned=5881, total=32798, total_delay_min=186177,
        top_delays=[{"displayName": "LONDON ACC", "delay": 20324.0, "averageDelay": 0}],
        regulations=[NmRegulation("R1", "REIMS ACC", "WEATHER", 5554, 94)],
    )
    ans = answer_nm_question(
        "how many airborne now?", client=_FakeClient("5113 aircraft are airborne."),
        fetch=lambda: snap,
    )
    assert ans.ok
    assert ans.snapshot.airborne == 5113
    assert "5113" in ans.answer
