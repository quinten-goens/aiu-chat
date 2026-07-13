"""The verify gate: every numeral in a draft must trace back to an executed query.

This is the component that decides whether a draft is publishable, so it is
tested for both failure modes -- missing a hallucinated number (dangerous) and
flagging a legitimate one (makes the gate noise, which gets it ignored).
"""
import datetime as dt

import pytest

from aiu_chat.nsr import verify
from aiu_chat.nsr.facts import Fact, Week, WeekFacts


@pytest.fixture
def wf():
    week = Week(monday=dt.date(2026, 5, 25), sunday=dt.date(2026, 5, 31), iso_week=22)
    f = WeekFacts(week=week)
    f.add(Fact("traffic_daily", "33,890", 33890.0, "flights/day", "api", "avgValue"))
    f.add(Fact("traffic_wow", "1.9%", 1.9327, "%", "derived", "computed"))
    f.add(Fact("delay_per_flight", "2.6", 2.5529, "min/flight", "api", "avg WK"))
    f.add(Fact("punctuality", "77%", 77.2195, "%", "api", "total WK"))
    f.add(Fact("punctuality_wow", "-4pp", -3.8935, "pp", "derived", "computed"))
    return f


def test_published_style_prose_is_clean(wf):
    # The real W22 sentences, which are entirely built from the facts above.
    prose = (
        "There were 33,890 daily flights in W22, 1.9% more than in W21.\n"
        "ATFM delay was 2.6 min/flight on W22.\n"
        "In W22 arrival punctuality decreased to 77%, -4pp worse than in W21."
    )
    assert verify.check(prose, wf) == []


def test_units_do_not_make_facts_look_invented(wf):
    # Facts render as "1.9%" / "-4pp" but the scanner extracts "1.9" / "-4".
    # If units aren't stripped from both sides, every percentage is flagged and
    # the gate becomes noise.
    assert verify.check("punctuality fell to 77%, -4pp worse", wf) == []


def test_thousands_separator_is_ignored(wf):
    assert verify.check("There were 33890 daily flights", wf) == []


def test_hallucinated_number_is_caught(wf):
    findings = verify.check("There were 34,500 daily flights in W22.", wf)
    assert [f.literal for f in findings] == ["34,500"]


def test_unsigned_form_of_a_signed_fact_is_accepted(wf):
    # "4pp worse" carries the direction in the adjective; the minus is optional.
    assert verify.check("punctuality was 4pp worse than in W21", wf) == []


def test_week_labels_and_years_are_not_measurements(wf):
    # W22 / W21 / 2025 are references, not figures to trace.
    assert verify.check("In W22, this is higher than in 2025 and W21.", wf) == []


def test_dates_written_into_bullets_are_caught(wf):
    # House style never dates a bullet ("from 28 May through 30 May"); the model
    # drifted into it on the first run and the gate must see that.
    findings = verify.check("regulated from 28 May through 30 May", wf)
    assert {f.literal for f in findings} == {"28", "30"}


def test_finding_reports_its_context(wf):
    finding = verify.check("delay reached 9.9 min/flight", wf)[0]
    assert finding.literal == "9.9"
    assert "min/flight" in finding.context
