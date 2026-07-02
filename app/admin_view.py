"""Hidden admin viewer for browsing logged conversations.

Access is gated two ways (defence by obscurity + auth):
  1. Only reachable when the URL carries the secret slug
     (?view=<AIU_ADMIN_VIEW_SLUG>). It is never listed in the app navigation.
  2. A separate password (AIU_ADMIN_VIEW_PASSWORD) — distinct from the main app
     password — must be entered.

Reads use the PocketBase superuser token (the log collections are create-only for
the app), so this page needs PB_ADMIN_USER_* to be configured.
"""
from __future__ import annotations

import hmac

import pandas as pd
import streamlit as st

from aiu_chat import config
from aiu_chat.agent import analytics
from aiu_chat.agent.chart import make_chart
from aiu_chat.sources import chatlog


def _authed() -> bool:
    """Gate the viewer on its own password (separate from the main app password)."""
    if not config.admin_viewer_configured():
        st.error(
            "The admin viewer is not configured. Set AIU_ADMIN_VIEW_PASSWORD "
            "(and PB_ADMIN_USER_* for reads)."
        )
        return False

    if st.session_state.get("admin_view_authed"):
        return True

    st.title("🔒 Conversation viewer")
    pw = st.text_input("Viewer password", type="password")
    if st.button("Unlock"):
        if hmac.compare_digest(pw, config.ADMIN_VIEW_PASSWORD):
            st.session_state["admin_view_authed"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _fmt(dt: str) -> str:
    return (dt or "").replace("T", " ").replace("Z", "")[:19]


def _render_turn(t: dict, i: int) -> None:
    """Render one logged turn (question + full trace)."""
    q = t.get("question") or "(no question)"
    st.markdown(f"#### {i}. {q}")

    meta = []
    if t.get("route"):
        meta.append(f"route: `{t['route']}`")
    if t.get("model_tier"):
        meta.append(f"model: `{t['model_tier']}`")
    if t.get("latency_ms") is not None:
        meta.append(f"{t['latency_ms']} ms")
    if t.get("needs_clarification"):
        meta.append("clarifying question")
    if t.get("error"):
        meta.append("⚠️ error")
    st.caption(" · ".join(meta) or "—")

    sa = t.get("standalone_question")
    if sa and sa != q:
        st.caption(f"Interpreted as: *{sa}*")

    if t.get("answer"):
        st.markdown("**Answer**")
        st.markdown(t["answer"])

    if t.get("error"):
        st.error(t["error"])

    if t.get("sql"):
        with st.expander("SQL"):
            st.code(t["sql"], language="sql")

    table = t.get("result_table")
    if table:
        try:
            df = pd.DataFrame(table)
        except Exception:
            df = None
        if df is not None and not df.empty:
            spec = t.get("chart_spec")
            if spec:
                fig = make_chart(spec, df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"av_chart_{i}")
            st.dataframe(df, use_container_width=True, hide_index=True, key=f"av_df_{i}")
            if t.get("row_count"):
                st.caption(f"{t['row_count']} row(s)"
                           + (" (truncated)" if t.get("truncated") else ""))

    sources = t.get("sources")
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                title = s.get("title") or "(untitled)"
                url = s.get("url") or ""
                st.markdown(f"- [{title}]({url})" if url else f"- {title}")

    live = t.get("live_payload")
    if live:
        with st.expander("Live source payload"):
            st.json(live)

    st.divider()


# Human-readable route names for the analytics breakdown.
_ROUTE_LABELS = {
    "data": "📊 Historical data", "concept": "📖 Concept / methodology",
    "both": "📊+📖 Data + concept", "nop": "📡 NOP messages",
    "dataapp": "📅 Latest daily (D-1)", "nm_live": "🟢 Real-time network",
    "catalog": "🗂️ Data catalogue", "none": "🚫 Out of scope",
    "unknown": "❔ Unknown",
}


@st.cache_data(ttl=120, show_spinner="Crunching analytics…")
def _load_analytics_rows():
    """Fetch slim turn rows + exact totals (cached briefly to avoid re-hitting
    PocketBase on every widget interaction)."""
    turns = chatlog.fetch_all(
        chatlog.TURNS,
        fields="id,route,error,needs_clarification,latency_ms,created,question",
    )
    total_sessions = chatlog.count(chatlog.SESSIONS)
    return turns, total_sessions


def _render_analytics() -> None:
    top_row = st.columns([1, 3])
    if top_row[0].button("↻ Refresh", key="av_refresh_analytics"):
        _load_analytics_rows.clear()

    try:
        turns, total_sessions = _load_analytics_rows()
    except chatlog.ChatLogError as exc:
        st.error(f"Could not read analytics: {exc}")
        return

    if not turns:
        st.info("No conversations logged yet. Analytics will appear once people "
                "start asking questions.")
        return

    a = analytics.compute(turns, total_sessions=total_sessions)

    # --- headline metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversations", f"{a.total_sessions:,}")
    c2.metric("Questions (turns)", f"{a.total_turns:,}")
    c3.metric("Turns / conversation", a.turns_per_session)
    c4.metric("Error rate", f"{a.error_rate:.1%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Clarifying asks", f"{a.clarification_count:,}")
    c6.metric("Median latency",
              f"{a.median_latency_ms/1000:.1f} s" if a.median_latency_ms else "—")
    c7.metric("Avg latency",
              f"{a.avg_latency_ms/1000:.1f} s" if a.avg_latency_ms else "—")
    c8.metric("Est. OpenAI requests", f"{a.estimated_openai_requests:,}")

    st.caption(
        "OpenAI-request estimate uses the per-route call fan-out (data ≈ 4–5 "
        "requests/question, concept ≈ 3). Use it to translate your OpenAI "
        "dashboard's request count into questions."
    )

    st.divider()

    # --- route distribution + activity over time ---
    left, right = st.columns(2)
    with left:
        st.markdown("**Questions by route**")
        if a.route_counts:
            rdf = pd.DataFrame(
                [(_ROUTE_LABELS.get(k, k), v) for k, v in a.route_counts.items()],
                columns=["Route", "Questions"],
            ).set_index("Route")
            st.bar_chart(rdf, horizontal=True)
    with right:
        st.markdown("**Activity over time**")
        if a.per_day:
            ddf = pd.DataFrame(a.per_day, columns=["Date", "Questions"])
            rdf2 = pd.DataFrame(a.requests_per_day, columns=["Date", "Est. requests"])
            merged = ddf.merge(rdf2, on="Date", how="left").set_index("Date")
            st.line_chart(merged)

    st.divider()

    # --- top questions ---
    st.markdown("**Most-asked questions**")
    if a.top_questions:
        qdf = pd.DataFrame(a.top_questions, columns=["Question", "Times asked"])
        st.dataframe(qdf, use_container_width=True, hide_index=True)
    else:
        st.caption("No repeated questions yet.")


def render() -> None:
    if not _authed():
        return

    st.title("🗂️ Conversation viewer")
    st.caption("All logged conversations. Hidden, password-protected, read-only.")

    tab_analytics, tab_sessions, tab_turns = st.tabs(
        ["📊 Analytics", "By conversation", "All turns (search)"]
    )

    # --- analytics summary ---
    with tab_analytics:
        _render_analytics()

    # --- browse by conversation ---
    with tab_sessions:
        col1, col2 = st.columns([3, 1])
        search = col1.text_input("Search sessions (id / user-agent / fingerprint / ip)",
                                 key="av_sess_search")
        limit = col2.number_input("Max", 10, 500, 100, step=10, key="av_sess_limit")
        try:
            sessions = chatlog.list_sessions(limit=int(limit), search=search.strip())
        except chatlog.ChatLogError as exc:
            st.error(f"Could not read sessions: {exc}")
            sessions = []

        if not sessions:
            st.info("No sessions found.")
        for s in sessions:
            sid = s.get("session_id", "")
            label = (f"{_fmt(s.get('created',''))} · {sid[:12]} · "
                     f"{s.get('app_mode','?')} · {s.get('model_tier','?')}")
            with st.expander(label):
                cols = st.columns(2)
                cols[0].markdown(
                    f"**Session** `{sid}`  \n"
                    f"**Browser id** `{s.get('browser_id','') or '—'}`  \n"
                    f"**Fingerprint** `{s.get('fingerprint_hash','') or '—'}`"
                )
                cols[1].markdown(
                    f"**Timezone** {s.get('timezone','') or '—'}  \n"
                    f"**Languages** {s.get('languages','') or '—'}  \n"
                    f"**Screen** {s.get('screen','') or '—'}"
                )
                if s.get("user_agent"):
                    st.caption(f"UA: {s['user_agent']}")
                if st.button("Load conversation", key=f"av_load_{s.get('id')}"):
                    st.session_state["av_open_session"] = sid

                if st.session_state.get("av_open_session") == sid:
                    try:
                        turns = chatlog.list_turns(sid)
                    except chatlog.ChatLogError as exc:
                        st.error(f"Could not read turns: {exc}")
                        turns = []
                    st.markdown(f"**{len(turns)} turn(s)**")
                    for i, t in enumerate(turns, 1):
                        _render_turn(t, i)

    # --- search across all turns ---
    with tab_turns:
        col1, col2 = st.columns([3, 1])
        q = col1.text_input("Search turns (question / answer / SQL)", key="av_turn_search")
        limit2 = col2.number_input("Max", 10, 500, 100, step=10, key="av_turn_limit")
        try:
            turns = chatlog.recent_turns(limit=int(limit2), search=q.strip())
        except chatlog.ChatLogError as exc:
            st.error(f"Could not read turns: {exc}")
            turns = []
        st.caption(f"{len(turns)} turn(s)")
        for i, t in enumerate(turns, 1):
            with st.container(border=True):
                st.caption(f"{_fmt(t.get('created',''))} · session {t.get('session_id','')[:12]}")
                _render_turn(t, i)
