"""Network Situation Report Drafter.

Click a button on Monday, get last week's report: prose in house style, every
number traceable to the query that produced it, every bullet traceable to the NOP
lines it was written from.

The page shows the draft and its evidence side by side on purpose. The value here
is not that a model wrote something plausible -- it is that an analyst can check
it in seconds and publish, or see immediately where it is thin.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from aiu_chat.nsr import draft as nsr_draft
from aiu_chat.nsr import facts as nsr_facts
from aiu_chat.sources import dataapp

_HISTORY_WEEKS = 13


def _trend(metric: str, week: nsr_facts.Week, label: str, unit: str) -> go.Figure | None:
    """The last `_HISTORY_WEEKS` weeks of `metric`, so a figure has context.

    A number in a report is an assertion; a trend is an explanation. This is what
    turns "delay was 62% higher" into something an analyst can defend.
    """
    session = requests.Session()
    xs, ys = [], []
    try:
        w = week
        for _ in range(_HISTORY_WEEKS):
            try:
                res = dataapp.fetch_network(metric, date=w.sync_date, session=session)
            except Exception:  # noqa: BLE001 - a missing week must not kill the page
                break
            row = next((r for r in res.records
                        if r.get("dateRange") == "WK"
                        and r.get("share") is None
                        and r.get("networkType") == ("avg" if metric == "delay" else "total")),
                       None)
            val = nsr_facts._val(row) if row else None
            if val is not None:
                xs.append(w.label)
                ys.append(val)
            w = w.previous()
    finally:
        session.close()

    if len(xs) < 2:
        return None

    xs, ys = xs[::-1], ys[::-1]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", name=label))
    # Mark the reported week so the eye lands on it.
    fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode="markers",
                             marker=dict(size=12), name=week.label,
                             showlegend=False))
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=30, b=0),
                      title=f"{label} ({unit}), last {len(xs)} weeks",
                      showlegend=False)
    return fig


def _render_provenance(d: nsr_draft.Draft) -> None:
    """Where every number came from, and what every bullet was written from."""
    st.subheader("Provenance")

    st.caption("Every figure in the draft, and the query behind it.")
    rows = [
        {
            "Figure": f.text,
            "Means": f.key.replace("_", " "),
            "Unit": f.unit,
            "Exact value": f"{f.value:,.4f}",
            "Source": f.source,
            "Field": f.field,
        }
        for f in d.facts.facts.values()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.caption(
        f"Data App sync `{d.facts.sync_id}` dated **{d.week.sync_date}** "
        f"(week {d.week.span_text()}). Week-on-week changes are differenced from "
        f"the {d.week.previous().label} sync — the API has no week-on-week field."
    )

    st.divider()
    st.caption("Every bullet, and the NOP tactical updates it was written from.")
    for b in d.bullets:
        with st.expander(f"{b.name} — {len(b.evidence)} NOP excerpt(s)"):
            if b.delay_per_flight is not None:
                st.markdown(
                    f"**Rank {b.rank}** by ATFM delay · "
                    f"**{b.delay_per_flight:.2f}** min/flight"
                )
            for e in b.evidence:
                st.markdown(f"- *{e.date:%a %d %b}* — {e.excerpt}")
                st.caption(f"NOP message `{e.message_id}`")


def _render_draft(d: nsr_draft.Draft) -> None:
    st.subheader(f"Draft — Network situation (Week {d.week.iso_week}: {d.week.span_text()})")

    if d.clean:
        st.success(
            "Verified: every number in this draft traces back to an executed query."
        )
    else:
        st.error(
            f"{len(d.findings)} number(s) in this draft do **not** trace back to the "
            "data. Do not publish without checking them."
        )
        for f in d.findings:
            st.markdown(f"- `{f.literal}` — …{f.context}…")

    st.markdown(d.headline)

    for label, bullets in (("**For en-route ATFM delay:**", d.enroute),
                           ("**For airport ATFM delay:**", d.airport)):
        st.markdown(label)
        for b in bullets:
            st.markdown(f"- **{b.name}** {b.text}")
            if not b.reads_as_continuation:
                st.caption(
                    f"✏️ {b.name}: this bullet does not continue the name "
                    "grammatically — reword its opening before publishing."
                )
            if b.suggestion:
                st.warning(
                    f"Unverified suggestion for {b.name} — not supported by NOP, "
                    f"accept or reject: {b.suggestion}",
                    icon="⚠️",
                )

    st.divider()
    st.caption("Editable text — adjust, then copy out to publish.")
    st.text_area("Report HTML", d.to_html(), height=200, key="nsr_html")


def render() -> None:
    st.title("📝 Network Situation Report Drafter")
    st.caption(
        "Drafts the weekly Network Situation Report from the Data App API "
        "(the numbers) and the archived NOP tactical updates (the causes). "
        "The model writes prose only — it never computes a figure, and every "
        "number is checked back against the query that produced it before the "
        "draft is shown."
    )

    default_week = nsr_facts.last_complete_week()
    col1, col2 = st.columns([2, 1])
    with col1:
        day = st.date_input(
            "Report on the week containing",
            value=default_week.monday,
            help="Defaults to the week that just ended.",
        )
    week = nsr_facts.week_of(day if isinstance(day, dt.date) else default_week.monday)
    with col2:
        st.metric("Reporting week", week.label, week.span_text())

    if not st.button("Draft this week's report", type="primary"):
        st.info(
            "Numbers come from the Data App weekly (`WK`) aggregates on the "
            f"**{week.sync_date}** sync; causes come from the NOP tactical updates "
            "published during the week. Nothing is generated until you click."
        )
        return

    with st.spinner("Collecting figures, retrieving NOP evidence, drafting…"):
        try:
            d = nsr_draft.build(week)
        except Exception as exc:  # noqa: BLE001 - surface it, never paper over it
            st.error(f"Could not draft this week: {exc}")
            return

    left, right = st.columns([3, 2])
    with left:
        _render_draft(d)
    with right:
        _render_provenance(d)

    st.divider()
    st.subheader("Why these numbers")
    st.caption(
        "The reported week in context. A single week's figure is an assertion; "
        "the trend is what lets you defend it."
    )
    t1, t2, t3 = st.tabs(["Traffic", "ATFM delay", "Punctuality"])
    for tab, metric, label, unit in (
        (t1, "traffic", "Daily flights", "flights/day"),
        (t2, "delay", "ATFM delay per flight", "min/flight"),
        (t3, "punctuality", "Arrival punctuality", "%"),
    ):
        with tab:
            fig = _trend(metric, week, label, unit)
            if fig is None:
                st.info("Not enough history for this metric.")
            else:
                st.plotly_chart(fig, width="stretch")

    st.divider()
    st.caption(
        "Data © EUROCONTROL Aviation Intelligence Unit (ansperformance.eu) and the "
        "EUROCONTROL Network Operations Portal. Unofficial tool — not affiliated "
        "with or endorsed by EUROCONTROL. A draft, not a publication: review before "
        "use."
    )
