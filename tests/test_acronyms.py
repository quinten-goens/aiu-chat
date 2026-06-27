"""Tests for acronym parsing and structured lookup."""
from __future__ import annotations

import duckdb
import pytest

from aiu_chat.ingest.acronyms import ACRONYM_TABLE, lookup_acronyms, parse_acronyms

SAMPLE = """\
Acronyms
A reference page.
ATFM - Air Traffic Flow Management
—
ASMA - Arrival Sequencing and Metering Area
—
A-CDM - Airport Collaborative Decision-Making
—
notanentry
SESAR - Single European Sky ATM Research programme
"""


def test_parse_extracts_pairs():
    pairs = dict(parse_acronyms(SAMPLE))
    assert pairs["ATFM"] == "Air Traffic Flow Management"
    assert pairs["ASMA"] == "Arrival Sequencing and Metering Area"
    assert pairs["A-CDM"] == "Airport Collaborative Decision-Making"
    assert "notanentry" not in pairs


@pytest.fixture()
def con_with_acronyms():
    con = duckdb.connect()
    con.execute(f"CREATE TABLE {ACRONYM_TABLE} (code VARCHAR, definition VARCHAR, source_url VARCHAR)")
    con.executemany(
        f"INSERT INTO {ACRONYM_TABLE} VALUES (?, ?, ?)",
        [("ATFM", "Air Traffic Flow Management", "u"), ("ASMA", "Arrival Sequencing and Metering Area", "u")],
    )
    return con


def test_lookup_exact_match(con_with_acronyms):
    hits = lookup_acronyms("What does ATFM stand for?", con=con_with_acronyms)
    assert len(hits) == 1
    assert hits[0]["definition"] == "Air Traffic Flow Management"


def test_lookup_case_insensitive_token(con_with_acronyms):
    # The token must still be uppercase in the query to be treated as an acronym.
    assert lookup_acronyms("ATFM meaning", con=con_with_acronyms)[0]["code"] == "ATFM"


def test_lookup_no_acronym_token(con_with_acronyms):
    assert lookup_acronyms("how is taxi time computed", con=con_with_acronyms) == []


def test_lookup_unknown_acronym(con_with_acronyms):
    assert lookup_acronyms("What is XYZZY?", con=con_with_acronyms) == []
