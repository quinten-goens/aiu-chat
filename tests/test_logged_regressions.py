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


def test_narration_sample_spans_the_whole_window():
    """Turns 18/19/21/22, second cause: the model was handed head(60) of a
    226-day series, so it honestly reported a window ending in March."""
    import pandas as pd

    from aiu_chat.agent.dataapp_answer import NARRATION_ROWS, _narration_sample

    days = pd.date_range("2026-01-01", "2026-08-14", freq="D")
    df = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "value": range(len(days))})

    out = _narration_sample(df)
    assert len(out) <= NARRATION_ROWS
    # First and last row must survive, or the narrated period is wrong again.
    assert out.iloc[0]["date"] == "2026-01-01"
    assert out.iloc[-1]["date"] == "2026-08-14"


def test_narration_sample_passes_short_frames_through():
    import pandas as pd

    df = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "value": [1, 2]})
    assert len(_ns(df)) == 2


def _ns(df):
    from aiu_chat.agent.dataapp_answer import _narration_sample
    return _narration_sample(df)


# --- second audit round: multi-year requests (turns 24-29) ------------------

def test_long_span_is_no_longer_trimmed():
    """Turns 27/28/29: a 2023->2026 ask was cut to 370 days landing in the
    pre-2024 dead zone, so the app narrated 5 days and blamed 'limits'."""
    from aiu_chat.sources import dataapp

    s, e, adjusted = dataapp._clamp_period("2023-01-01", "2026-08-15")
    assert e == "2026-08-15", "the requested end date must be honoured"
    assert s == dataapp.DATAAPP_FIRST_DAY
    assert adjusted is True


def test_coverage_question_gets_a_real_answer():
    """Turn 26: 'what's the max range you can go back?' hit a generic
    'I couldn't read the date range' dead end."""
    from aiu_chat.agent.dataapp_answer import coverage_answer

    ans = coverage_answer("What's the max range you can go back?")
    assert ans is not None
    assert ans.ok is True
    assert "2024" in ans.answer


def test_coverage_answer_ignores_normal_questions():
    from aiu_chat.agent.dataapp_answer import coverage_answer

    assert coverage_answer("How many flights in France on 10 March 2026?") is None
