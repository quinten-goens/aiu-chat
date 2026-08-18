"""Regressions for failures found by auditing the logged conversations.

Each test names the logged turn it came from. Network-touching tests are marked
`live` so the default suite stays fast and offline:
    pytest -m live      # run these
    pytest -m "not live"
"""
from __future__ import annotations

import pytest

from aiu_chat.agent import aliases


def test_athens_acc_no_longer_dead_ends():
    """Turns 12 and 14: refused outright; must now clarify toward Greece."""
    nm = aliases.resolve_near_miss("Athens ACC")
    assert nm is not None and "Greece" in nm.candidates


def test_transient_500_is_retried():
    """Turn 15: an unretried OpenAI HTTP 500 lost the whole turn."""
    from aiu_chat.agent import llm

    assert llm._is_retryable(llm.OpenAIError("OpenAI API returned HTTP 500: oops"))
    assert not llm._is_retryable(llm.OpenAIError("OpenAI API returned HTTP 400: bad"))


@pytest.mark.live
def test_long_series_covers_the_whole_window():
    """Turns 18/19/21/22: answers silently stopped at 1 March (100-row cap)."""
    import requests

    from aiu_chat.sources import dataapp

    s = requests.Session()
    res = dataapp.fetch_timeseries(
        "traffic", start="2026-01-01", end="2026-06-01",
        kind="country", query="Spain", session=s)
    assert len(res.rows) > 100, "still truncated at the API page cap"
    assert res.rows[-1]["date"] >= "2026-05-25", res.rows[-1]["date"]


@pytest.mark.live
def test_sync_lookup_spans_a_long_window():
    """The exact reproduction: 226 days must come back, not 100."""
    import requests

    from aiu_chat.sources import dataapp

    s = requests.Session()
    out = dataapp.find_syncs_in_range(s, start="2026-01-01", end="2026-08-15")
    assert len(out) > 200, f"only {len(out)} days returned"
    assert out[0][1] == "2026-01-01"
