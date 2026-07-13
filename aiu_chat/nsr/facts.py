"""Deterministic fact collection for the weekly NSR.

Everything the report asserts numerically is gathered here, from the Data App
API, and carries its own provenance (endpoint, sync, field, unrounded value).
The LLM is handed the rounded `text` and is forbidden to alter it; the drafter's
verify pass re-checks every digit in the prose against these facts.

Nothing in this module computes a number the API could have supplied, and no
number is ever produced by a model. Week-on-week deltas are the one derived
quantity -- the API has no such field (`startDateValue` is week-to-date, not
week-on-week) -- so they are differenced here, in Python, from two syncs.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import requests

from aiu_chat.sources import dataapp

# An NSR week is a plain ISO week: Monday..Sunday. The figures are read off the
# sync dated on the Sunday, with dateRange="WK" (verified against the published
# W22 report: sync 2026-05-31 / WK reproduces 33,890 flights, 2.55 min, 77.2%).
SYNC_DATE_RANGE = "WK"


@dataclass(frozen=True)
class Week:
    """One NSR reporting week."""

    monday: dt.date
    sunday: dt.date
    iso_week: int

    @property
    def label(self) -> str:
        return f"W{self.iso_week}"

    @property
    def sync_date(self) -> str:
        """The sync carrying this week's WK aggregates."""
        return self.sunday.isoformat()

    def span_text(self) -> str:
        """'25 - 31 May' / '27 April - 3 May', matching house style."""
        if self.monday.month == self.sunday.month:
            return f"{self.monday.day} - {self.sunday.day} {self.sunday:%B}"
        return f"{self.monday.day} {self.monday:%B} - {self.sunday.day} {self.sunday:%B}"

    def previous(self) -> "Week":
        return week_of(self.monday - dt.timedelta(days=7))


def week_of(day: dt.date) -> Week:
    """The NSR week containing `day`."""
    monday = day - dt.timedelta(days=day.weekday())
    return Week(monday=monday, sunday=monday + dt.timedelta(days=6),
                iso_week=monday.isocalendar()[1])


def last_complete_week(today: dt.date | None = None) -> Week:
    """The week the Monday button reports on: the one that just ended."""
    today = today or dt.date.today()
    return week_of(today - dt.timedelta(days=today.weekday() + 7))


@dataclass(frozen=True)
class Fact:
    """One number, and where it came from.

    `text` is what may appear in the prose; `value` is the unrounded figure the
    provenance panel shows. `source` is human-readable; `field` pins the exact
    API field so a reviewer can re-fetch it.
    """

    key: str
    text: str
    value: float
    unit: str
    source: str
    field: str
    derived_from: tuple[str, ...] = ()


@dataclass
class WeekFacts:
    week: Week
    sync_id: int | None = None
    facts: dict[str, Fact] = field(default_factory=dict)
    acc_ranking: list[dict] = field(default_factory=list)
    airport_ranking: list[dict] = field(default_factory=list)

    def add(self, fact: Fact) -> None:
        self.facts[fact.key] = fact

    def get(self, key: str) -> Fact | None:
        return self.facts.get(key)

    def numbers(self) -> list[str]:
        """Every literal the prose is allowed to contain."""
        return [f.text for f in self.facts.values()]


def _record(records: list[dict], *, network_type: str, date_range: str,
            has_share: bool | None = None) -> dict | None:
    """Pick one row out of a network result.

    The delay payload repeats (networkType, dateRange) pairs: the first block is
    the network total, later blocks carry a `share` and split it en-route vs
    airport. `has_share` disambiguates them.
    """
    for r in records:
        if r.get("networkType") != network_type or r.get("dateRange") != date_range:
            continue
        if has_share is None:
            return r
        if has_share is (r.get("share") is not None):
            return r
    return None


def _val(rec: dict | None) -> float | None:
    """WK rows carry the figure in `value` or `avgValue` depending on metric."""
    if rec is None:
        return None
    v = rec.get("value")
    if v is None:
        v = rec.get("avgValue")
    return float(v) if v is not None else None


def _split_row(records: list[dict], network_type: str) -> dict | None:
    """The en-route (`total`) or airport (`avg`) per-flight row of the WK split.

    Both the minute totals and the per-flight figures are tagged with a `share`;
    only the latter are min/flight, and they are the small ones (a network week
    never averages 20+ min/flight, but the minute totals run to five figures).
    """
    candidates = [
        r for r in records
        if r.get("dateRange") == "WK"
        and r.get("share") is not None
        and r.get("networkType") == network_type
        and (_val(r) or 0.0) < 20.0
    ]
    return candidates[0] if candidates else None


def _pct_change(now: float, before: float) -> float:
    return (now - before) / before * 100.0


def _pct_text(v: float) -> str:
    """Punctuality as published: plain rounding.

    Tempting to truncate -- a couple of weeks read like it (W17: 84.63 -> "84")
    -- but across the archive rounding wins 65/71 and truncation only 38/71, and
    the archive contradicts itself anyway (W17 rounds 84.63 down, W33 rounds
    73.49 up). Those few are analyst variance; don't fit the code to them.
    """
    return f"{round(v)}%"


def collect(week: Week, *, session: requests.Session | None = None) -> WeekFacts:
    """Gather every number the NSR for `week` will assert."""
    own = session is None
    session = session or requests.Session()
    try:
        wf = WeekFacts(week=week)
        prev = week.previous()

        # --- this week and last week, for the three headline metrics -----------
        cur: dict[str, list[dict]] = {}
        old: dict[str, list[dict]] = {}
        for metric in ("traffic", "delay", "punctuality"):
            res = dataapp.fetch_network(metric, date=week.sync_date, session=session)
            cur[metric] = res.records
            wf.sync_id = res.sync_id
            old[metric] = dataapp.fetch_network(
                metric, date=prev.sync_date, session=session
            ).records

        src = f"Data App /{{metric}}_networks, sync {week.sync_date} (dateRange=WK)"

        # --- traffic ----------------------------------------------------------
        t_now = _val(_record(cur["traffic"], network_type="total", date_range="WK"))
        t_prev = _val(_record(old["traffic"], network_type="total", date_range="WK"))
        t_row = _record(cur["traffic"], network_type="total", date_range="WK") or {}

        if t_now is not None:
            wf.add(Fact("traffic_daily", f"{round(t_now):,}", t_now, "flights/day",
                        src.format(metric="traffic"), "avgValue (WK)"))
        if t_now is not None and t_prev is not None:
            d = _pct_change(t_now, t_prev)
            wf.add(Fact("traffic_wow", f"{d:.1f}%", d, "% vs prev week",
                        f"derived: WK avgValue {week.sync_date} vs {prev.sync_date}",
                        "computed", ("traffic_daily",)))
        if t_row.get("prevDateValue") is not None:
            y = float(t_row["prevDateValue"]) * 100.0
            wf.add(Fact("traffic_vs_prev_year", f"{y:.0f}%", y, "% vs same week last year",
                        src.format(metric="traffic"), "prevDateValue (WK)"))

        # --- ATFM delay: network average, then the en-route / airport split ----
        d_now = _val(_record(cur["delay"], network_type="avg", date_range="WK",
                             has_share=False))
        d_prev = _val(_record(old["delay"], network_type="avg", date_range="WK",
                              has_share=False))
        if d_now is not None:
            wf.add(Fact("delay_per_flight", f"{d_now:.1f}", d_now, "min/flight",
                        src.format(metric="delay"), "networkType=avg, dateRange=WK"))
        if d_now is not None and d_prev is not None:
            d = _pct_change(d_now, d_prev)
            wf.add(Fact("delay_wow", f"{d:.0f}%", d, "% vs prev week",
                        f"derived: WK avg {week.sync_date} vs {prev.sync_date}",
                        "computed", ("delay_per_flight",)))

        # The split is *reported*, not inferred: among the WK rows carrying a
        # `share`, the per-flight pair (values in min/flight, so << the raw
        # minute totals) gives en-route under networkType=total and airport
        # under networkType=avg. Verified against W10/11/17/20/21/22.
        #
        # Do NOT reconstruct these by multiplying the network average by the
        # share and assuming en-route is the larger: en-route was 81% of delay
        # in W22 but only 40% in W11, so that heuristic silently swaps the two
        # in winter weeks.
        er = _val(_split_row(cur["delay"], "total"))
        ap = _val(_split_row(cur["delay"], "avg"))
        if er is not None:
            wf.add(Fact("delay_enroute", f"{er:.1f}", er, "min/flight",
                        src.format(metric="delay"),
                        "networkType=total, dateRange=WK, share row"))
        if ap is not None:
            wf.add(Fact("delay_airport", f"{ap:.1f}", ap, "min/flight",
                        src.format(metric="delay"),
                        "networkType=avg, dateRange=WK, share row"))

        # --- punctuality (percentage points, not percent) ----------------------
        p_now = _val(_record(cur["punctuality"], network_type="total", date_range="WK"))
        p_prev = _val(_record(old["punctuality"], network_type="total", date_range="WK"))
        if p_now is not None:
            wf.add(Fact("punctuality", _pct_text(p_now), p_now, "% arrivals on time",
                        src.format(metric="punctuality"),
                        "networkType=total, dateRange=WK"))
        if p_now is not None and p_prev is not None:
            pp = p_now - p_prev
            wf.add(Fact("punctuality_wow", f"{pp:.0f}pp", pp, "pp vs prev week",
                        f"derived: WK total {week.sync_date} vs {prev.sync_date}",
                        "computed", ("punctuality",)))

        # --- who to write bullets about ---------------------------------------
        for cat, sink in (("area_control_center", wf.acc_ranking),
                          ("airports", wf.airport_ranking)):
            rk = dataapp.fetch_ranking("delay", cat, date=week.sync_date,
                                       date_range=SYNC_DATE_RANGE, limit=10,
                                       session=session)
            sink.extend(rk.rows)

        return wf
    finally:
        if own:
            session.close()
