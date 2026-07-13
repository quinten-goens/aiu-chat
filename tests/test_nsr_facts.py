"""Week arithmetic + the fact-collection contract.

The live backtest against the published reports lives in tests/nsr/backtest.py;
these are the offline invariants.
"""
import datetime as dt

import pytest

from aiu_chat.nsr import facts


def test_week_of_snaps_to_monday():
    # Any day in the week resolves to the same Mon..Sun span.
    for day in range(25, 32):
        w = facts.week_of(dt.date(2026, 5, day))
        assert w.monday == dt.date(2026, 5, 25)
        assert w.sunday == dt.date(2026, 5, 31)
        assert w.iso_week == 22
        assert w.label == "W22"


def test_sync_date_is_the_sunday():
    # Verified against the published W22 report: the WK aggregates hang off the
    # Sunday sync, not the Monday and not the publication date.
    assert facts.week_of(dt.date(2026, 5, 25)).sync_date == "2026-05-31"


def test_span_text_matches_house_style():
    assert facts.week_of(dt.date(2026, 5, 25)).span_text() == "25 - 31 May"
    # W18 in the real archive reads "27 April - 3 May".
    assert facts.week_of(dt.date(2026, 4, 27)).span_text() == "27 April - 3 May"


def test_previous_week():
    w = facts.week_of(dt.date(2026, 5, 25))
    assert w.previous().label == "W21"
    assert w.previous().sync_date == "2026-05-24"


def test_last_complete_week_is_the_one_that_just_ended():
    # Clicking the button on Monday 2026-06-01 must report on W22 (25-31 May),
    # not the week in progress.
    assert facts.last_complete_week(dt.date(2026, 6, 1)).label == "W22"
    # Still true later in that same week.
    assert facts.last_complete_week(dt.date(2026, 6, 3)).label == "W22"


def test_record_disambiguates_the_shared_delay_rows():
    # The delay payload repeats (networkType, dateRange); the split rows carry a
    # `share`. Picking the network total must not accidentally take a split row.
    records = [
        {"networkType": "avg", "dateRange": "WK", "value": 2.55},
        {"networkType": "avg", "dateRange": "WK", "value": 16161.0, "share": 0.19},
    ]
    total = facts._record(records, network_type="avg", date_range="WK", has_share=False)
    assert total["value"] == 2.55


def test_val_falls_back_to_avgvalue():
    # Traffic WK carries avgValue; punctuality WK carries value.
    assert facts._val({"avgValue": 33890.0}) == 33890.0
    assert facts._val({"value": 77.2}) == 77.2
    assert facts._val(None) is None


@pytest.mark.parametrize("now,before,expected", [
    (33890.0, 33248.0, 1.93),
    (2.5529, 1.5758, 62.0),
])
def test_pct_change(now, before, expected):
    assert facts._pct_change(now, before) == pytest.approx(expected, abs=0.05)
