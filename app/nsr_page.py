"""Network Situation Report Drafter.

Click a button on Monday, get last week's report: prose in house style, every
number traceable to the query that produced it, every bullet traceable to the NOP
lines it was written from.

The page shows the draft and its evidence side by side on purpose. The value here
is not that a model wrote something plausible -- it is that an analyst can check
it in seconds and publish, or see immediately where it is thin.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_quill import st_quill

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
    st.caption(
        "Every bullet, the NOP updates it was written from, and a place to add "
        "what NOP does not know."
    )
    for b in d.bullets:
        marks = " 📝" if b.has_notes else ""
        with st.expander(f"{b.name} — {len(b.evidence)} NOP excerpt(s){marks}"):
            if b.delay_per_flight is not None:
                st.markdown(
                    f"**Rank {b.rank}** by ATFM delay · "
                    f"**{b.delay_per_flight:.2f}** min/flight"
                )
            for e in b.evidence:
                st.markdown(f"- *{e.date:%a %d %b}* — {e.excerpt}")
                st.caption(f"NOP message `{e.message_id}`")

            _bullet_controls(d, b)


def _bullet_controls(d: nsr_draft.Draft, b: nsr_draft.Bullet) -> None:
    """Analyst notes, and a rewrite of just this bullet.

    Rewriting one bullet rather than the whole report is the difference between a
    review loop you use and one you avoid: a single weak sentence should not cost
    a full redraft.
    """
    key = f"note_{d.week.label}_{b.kind}_{b.name}"
    notes = st.text_area(
        "Analyst notes — what NOP doesn't know",
        value=b.notes,
        key=key,
        height=80,
        placeholder=(
            "e.g. 23 diversions resulted from the Saturday drone sighting; "
            "the on-going TTMS trial contributed to higher delays."
        ),
        help=(
            "Treated as authoritative evidence and preferred over NOP where they "
            "conflict — a person verified it. Figures you give here are allowed "
            "into the prose; figures you don't are still flagged."
        ),
    )

    if st.button("↻ Rewrite this bullet", key=f"rw_{key}", width="stretch"):
        with st.spinner(f"Rewriting {b.name}…"):
            try:
                ec = _evidence_for(d.week, b.name)
                # Warmer than the initial draft: a retry that reproduces the same
                # sentence is not a retry.
                new = nsr_draft.write_bullet(ec, d.week, notes=notes,
                                             temperature=0.6)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not rewrite: {exc}")
                return
        sink = d.enroute if b.kind == "acc" else d.airport
        for i, existing in enumerate(sink):
            if existing.name == b.name:
                sink[i] = new
                break
        d.reverify()
        st.rerun()


@st.cache_data(show_spinner=False)
def _week_evidence(sunday: dt.date) -> dict:
    """Every candidate entity's NOP evidence for the week ending `sunday`.

    Cached: rewriting one bullet must not re-fetch and re-scan the whole week's
    NOP messages, or the per-bullet retry costs as much as a full redraft and
    stops being worth doing.
    """
    return nsr_draft.entity_evidence(nsr_facts.week_of(sunday))


def _evidence_for(week: nsr_facts.Week, name: str):
    return _week_evidence(week.sunday)[name]


def _verdict(findings, *, edited: bool) -> None:
    """The gate's verdict, on whatever text is currently in the editor."""
    scope = "after your edits" if edited else "as drafted"
    if not findings:
        st.success(
            f"Verified {scope}: every number traces back to an executed query.",
            icon="✅",
        )
        return
    st.error(
        f"{len(findings)} number(s) {scope} do **not** trace back to the data. "
        "Do not publish without checking them.",
        icon="🚩",
    )
    for f in findings:
        st.markdown(f"- `{f.literal}` — …{f.context}…")


def _render_draft(d: nsr_draft.Draft) -> None:
    st.subheader(f"Draft — {d.title}")

    for b in d.bullets:
        if not b.reads_as_continuation:
            st.caption(
                f"✏️ **{b.name}**: this bullet does not continue the name "
                "grammatically — reword its opening."
            )
        if b.suggestion:
            st.warning(
                f"**{b.name}** — unverified suggestion, not supported by NOP. "
                f"Accept or reject: {b.suggestion}",
                icon="⚠️",
            )

    st.caption(
        "Edit below. The report is checked again against the data every time you "
        "change it, so the verdict always describes the text you are about to "
        "publish — not the one the model first wrote."
    )

    # Quill is a real WYSIWYG editor and it emits HTML, which is the format the
    # publishing system stores -- so what is reviewed is exactly what is exported.
    #
    # The key is a hash of the draft, not just the week: Quill holds its own state
    # against its key, so a stable key would leave the old sentence on screen after
    # a bullet is rewritten -- the analyst would see their rewrite ignored.
    source = d.to_html()
    digest = hashlib.sha1(source.encode()).hexdigest()[:12]
    edited_html = st_quill(
        value=source,
        html=True,
        toolbar=[["bold", "italic"], ["link"],
                 [{"list": "ordered"}, {"list": "bullet"}], ["clean"]],
        key=f"nsr_editor_{d.week.label}_{digest}",
    ) or source

    plain = nsr_draft.html_to_text(edited_html)
    changed = _normalise(edited_html) != _normalise(source)
    # Re-check the *edited* text, through the draft so analyst notes stay allowed.
    findings = d.reverify(plain)
    _verdict(findings, edited=changed)

    st.divider()
    st.caption("Copy or download the report exactly as it stands above.")
    stem = f"nsr_W{d.week.iso_week}_{d.week.sunday:%Y%m%d}"

    c1, c2 = st.columns(2)
    with c1:
        _copy_button(edited_html, "📋 Copy HTML")
    with c2:
        _copy_button(plain, "📋 Copy text")

    c3, c4, c5 = st.columns(3)
    with c3:
        st.download_button(
            "⬇️ HTML", data=edited_html, file_name=f"{stem}.html",
            mime="text/html", width="stretch",
            help="The shape the publishing system stores.",
        )
    with c4:
        st.download_button(
            "⬇️ Plain text", data=plain,
            file_name=f"{stem}.txt", mime="text/plain", width="stretch",
            help="For pasting into an email.",
        )
    with c5:
        st.download_button(
            "⬇️ Evidence (CSV)", data=_evidence_csv(d),
            file_name=f"{stem}_evidence.csv", mime="text/csv", width="stretch",
            help="Every figure and every NOP citation behind the draft.",
        )


def _normalise(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def _copy_button(text: str, label: str) -> None:
    """Copy `text` to the clipboard.

    st.code's built-in copy affordance does the work and needs no JS of our own,
    which keeps this offline-safe (no CDN) and avoids a component dependency for
    what is a one-click nicety.
    """
    with st.popover(label, width="stretch"):
        st.caption("Click the copy icon in the top-right of the box.")
        st.code(text, language=None, wrap_lines=True)


def _evidence_csv(d: nsr_draft.Draft) -> str:
    """Every figure and every citation, flat -- the audit trail, exportable."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["kind", "subject", "value", "unit", "source", "field", "detail"])
    for f in d.facts.facts.values():
        w.writerow(["figure", f.key, f.text, f.unit, f.source, f.field,
                    f"exact={f.value}"])
    for b in d.bullets:
        for e in b.evidence:
            w.writerow(["citation", b.name, "", "", f"NOP {e.message_id}",
                        e.date.isoformat(), e.excerpt])
    return buf.getvalue()


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

    if st.button("Draft this week's report", type="primary"):
        with st.spinner("Collecting figures, retrieving NOP evidence, drafting…"):
            try:
                st.session_state["nsr_draft"] = nsr_draft.build(week)
            except Exception as exc:  # noqa: BLE001 - surface it, never paper over it
                st.error(f"Could not draft this week: {exc}")
                return

    d = st.session_state.get("nsr_draft")
    # A draft is kept in session state so a rewritten bullet survives the rerun --
    # otherwise every interaction would silently redraft the whole report.
    if d is None or d.week.label != week.label:
        st.info(
            "Numbers come from the Data App weekly (`WK`) aggregates on the "
            f"**{week.sync_date}** sync; causes come from the NOP tactical updates "
            "published during the week. Nothing is generated until you click."
        )
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
