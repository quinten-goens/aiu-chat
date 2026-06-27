"""Tests for chart-spec validation and figure building (no model needed).

The contract: a valid, chart-worthy spec yields a figure; anything off yields
None (so the caller still shows prose + table). A bad spec never raises.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiu_chat.agent.chart import build_figure, make_chart, parse_spec

DF = pd.DataFrame(
    {
        "STATE_NAME": ["FRANCE", "GERMANY", "SPAIN"],
        "TOTAL": [10.0, 30.0, 20.0],
        "YEAR": [2024, 2024, 2024],
    }
)


def test_valid_bar_spec_parses():
    spec = {"show_chart": True, "chart_type": "bar", "x": "STATE_NAME", "y": ["TOTAL"]}
    parsed = parse_spec(spec, DF)
    assert parsed is not None
    assert parsed.chart_type == "bar"
    assert parsed.y == ["TOTAL"]


def test_y_as_string_is_accepted():
    # x must be one-per-row for a sensible line, so use a per-year series.
    df = pd.DataFrame({"YEAR": [2022, 2023, 2024], "TOTAL": [10.0, 30.0, 20.0]})
    spec = {"show_chart": True, "chart_type": "line", "x": "YEAR", "y": "TOTAL"}
    parsed = parse_spec(spec, df)
    assert parsed is not None
    assert parsed.y == ["TOTAL"]


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {},
        {"show_chart": False, "chart_type": "bar", "x": "STATE_NAME", "y": ["TOTAL"]},
        {"show_chart": True, "chart_type": "pie", "x": "STATE_NAME", "y": ["TOTAL"]},   # bad type
        {"show_chart": True, "chart_type": "bar", "x": "NOPE", "y": ["TOTAL"]},          # bad x
        {"show_chart": True, "chart_type": "bar", "x": "STATE_NAME", "y": ["NOPE"]},     # bad y
        {"show_chart": True, "chart_type": "bar", "x": "STATE_NAME", "y": ["STATE_NAME"]},  # non-numeric y
    ],
)
def test_invalid_specs_return_none(spec):
    assert parse_spec(spec, DF) is None


def test_too_few_rows_returns_none():
    one_row = DF.head(1)
    spec = {"show_chart": True, "chart_type": "bar", "x": "STATE_NAME", "y": ["TOTAL"]}
    assert parse_spec(spec, one_row) is None


def test_invalid_series_is_dropped_not_failed():
    spec = {
        "show_chart": True, "chart_type": "bar", "x": "STATE_NAME",
        "y": ["TOTAL"], "series": "NONEXISTENT",
    }
    parsed = parse_spec(spec, DF)
    assert parsed is not None
    assert parsed.series is None  # dropped, chart still valid


def test_build_figure_produces_plotly_object():
    spec = {"show_chart": True, "chart_type": "bar", "x": "STATE_NAME", "y": ["TOTAL"], "title": "t"}
    parsed = parse_spec(spec, DF)
    fig = build_figure(parsed, DF)
    assert fig is not None
    assert fig.layout.title.text == "t"


def test_make_chart_endtoend_none_on_bad_spec():
    assert make_chart({"show_chart": True, "chart_type": "bad"}, DF) is None


def test_multi_y_bar_is_grouped_with_one_trace_per_measure():
    # arrivals vs departures as two measures -> two grouped traces (not stacked,
    # not a time column abused as a legend).
    df = pd.DataFrame(
        {"YEAR": [2016, 2017, 2018], "DEPARTURES": [10, 11, 12], "ARRIVALS": [10, 11, 13]}
    )
    spec = {"show_chart": True, "chart_type": "bar", "x": "YEAR",
            "y": ["DEPARTURES", "ARRIVALS"]}
    fig = make_chart(spec, df)
    assert fig is not None
    assert fig.layout.barmode == "group"
    assert sorted(t.name for t in fig.data) == ["ARRIVALS", "DEPARTURES"]


def test_unaggregated_data_with_many_rows_per_x_is_rejected():
    # Daily rows charted against MONTH_MON would be spaghetti (100 rows per x).
    df = pd.DataFrame(
        {"MONTH_MON": ["JAN"] * 100 + ["FEB"] * 100, "FLT_TOT_1": list(range(200))}
    )
    spec = {"show_chart": True, "chart_type": "line", "x": "MONTH_MON", "y": ["FLT_TOT_1"]}
    assert make_chart(spec, df) is None  # guard: not chart-worthy -> table only


def test_one_row_per_x_is_allowed():
    df = pd.DataFrame({"YEAR": [2022, 2023, 2024], "total": [1, 2, 3]})
    spec = {"show_chart": True, "chart_type": "bar", "x": "YEAR", "y": ["total"]}
    assert make_chart(spec, df) is not None
