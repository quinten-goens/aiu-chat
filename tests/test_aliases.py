"""ACC/ATSU names are not entities in the AIU datasets.

Three logged conversations asked about "Athens ACC" and were refused, even
though Greece FIR carries the answer. A near-miss must produce a clarifying
question naming real candidates, never a silent substitution.
"""
from __future__ import annotations

import duckdb
import pytest

from aiu_chat.agent import aliases


def test_athens_acc_offers_greece():
    nm = aliases.resolve_near_miss("Athens ACC")
    assert nm is not None
    assert "Greece" in nm.candidates


def test_case_and_suffix_insensitive():
    for probe in ("athens acc", "ATHENS ACC", "Athens", "Athens Area Control Centre"):
        nm = aliases.resolve_near_miss(probe)
        assert nm is not None, probe
        assert "Greece" in nm.candidates, probe


def test_question_names_the_candidate_and_the_problem():
    q = aliases.resolve_near_miss("Athens ACC").question()
    assert "Athens" in q
    assert "Greece" in q
    assert "?" in q


def test_known_entity_is_not_a_near_miss():
    """A name that IS a real entity must pass straight through."""
    assert aliases.resolve_near_miss("Greece") is None
    assert aliases.resolve_near_miss("Maastricht UAC") is None


def test_unrelated_text_is_not_a_near_miss():
    assert aliases.resolve_near_miss("fuel prices") is None
    assert aliases.resolve_near_miss("") is None


def test_every_candidate_is_a_real_entity():
    """The whole point of a clarifying question is that the name it offers can
    actually be queried. Cross-check every candidate against the parquet."""
    con = duckdb.connect()
    real = set()
    for table in ("enroute_delay_fir", "enroute_delay_ansp"):
        rows = con.execute(
            f"SELECT DISTINCT ENTITY_NAME FROM read_parquet('data/parquet/{table}.parquet')"
        ).fetchall()
        real.update(r[0] for r in rows)

    bogus = {
        cand
        for cands in aliases._NEAR_MISS.values()
        for cand in cands
        if cand not in real
    }
    assert not bogus, f"alias map names entities that do not exist: {sorted(bogus)}"
