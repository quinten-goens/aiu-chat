"""Build a Plotly figure from a validated, LLM-emitted chart spec.

Charts are an extension of the data path: they render the SAME rows DuckDB
returned. The model never writes plotting code — it emits a small JSON spec that
we validate against the actual result columns and turn into a Plotly figure.

Fail-soft by design: an invalid spec yields no chart (the caller still shows
prose + table). A bad chart must never crash the answer. See CLAUDE.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ALLOWED_CHART_TYPES = {"line", "bar", "area", "scatter"}


@dataclass
class ChartSpec:
    chart_type: str
    x: str
    y: list[str]
    series: str | None = None
    title: str | None = None


def parse_spec(spec: dict | None, df: pd.DataFrame) -> ChartSpec | None:
    """Validate a raw spec dict against the result DataFrame.

    Returns a ChartSpec if valid and chart-worthy, else None (no chart).
    """
    if not spec or not isinstance(spec, dict):
        return None
    if not spec.get("show_chart", False):
        return None

    # Tiny results aren't worth charting.
    if df is None or df.empty or len(df) < 2:
        return None

    chart_type = str(spec.get("chart_type", "")).lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        return None

    columns = set(df.columns)

    x = spec.get("x")
    if x not in columns:
        return None

    # y may be a string or list; keep only columns that exist and are numeric.
    raw_y = spec.get("y")
    y_list = [raw_y] if isinstance(raw_y, str) else list(raw_y or [])
    y_list = [c for c in y_list if c in columns and pd.api.types.is_numeric_dtype(df[c])]
    if not y_list:
        return None

    series = spec.get("series")
    if series is not None and series not in columns:
        series = None  # drop an invalid series rather than failing the whole chart

    # Guard against charting un-aggregated data: if x (optionally split by series)
    # has many rows per value, a line/area/bar becomes spaghetti. Require x to be
    # mostly unique within each series group. The table is still shown.
    group_cols = [x] + ([series] if series else [])
    rows = len(df)
    groups = df.groupby(group_cols, dropna=False).ngroups if rows else 0
    if groups and rows / groups > 1.5:  # >1.5 rows per x point on average
        return None

    title = spec.get("title")
    return ChartSpec(chart_type=chart_type, x=x, y=y_list, series=series, title=title)


def build_figure(spec: ChartSpec, df: pd.DataFrame):
    """Build a Plotly figure from a validated spec. Imported lazily so non-UI
    code paths don't require plotly."""
    import plotly.express as px

    kwargs = {"x": spec.x, "title": spec.title}
    # With a series column we colour by it and use the first y; otherwise y can
    # be one or more columns.
    multi_y = False
    if spec.series:
        kwargs["y"] = spec.y[0]
        kwargs["color"] = spec.series
    else:
        multi_y = len(spec.y) > 1
        kwargs["y"] = spec.y if multi_y else spec.y[0]

    # Multiple measures or a series on a bar chart should sit side by side, not
    # stacked, so e.g. arrivals vs departures are comparable per category.
    if spec.chart_type == "bar" and (multi_y or spec.series):
        kwargs["barmode"] = "group"

    fn = {
        "line": px.line,
        "bar": px.bar,
        "area": px.area,
        "scatter": px.scatter,
    }[spec.chart_type]
    fig = fn(df, **kwargs)
    # A cleaner legend title when plotting multiple y columns.
    if multi_y:
        fig.update_layout(legend_title_text="")
    return fig


def make_chart(spec: dict | None, df: pd.DataFrame):
    """Convenience: parse + build in one step. Returns a figure or None."""
    parsed = parse_spec(spec, df)
    if parsed is None:
        return None
    return build_figure(parsed, df)
