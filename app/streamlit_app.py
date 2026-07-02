"""Streamlit chat UI for AIU Chat.

Thin presentation layer: it calls the agent's data path and renders prose, the
result table, and (when chart-worthy) a Plotly chart. It does not import DuckDB
or Ollama directly — everything goes through aiu_chat.agent.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
import hmac

from aiu_chat import config
from aiu_chat.agent.catalog import get_catalog
from aiu_chat.agent.chart import make_chart
from aiu_chat.agent.llm import OllamaError, OpenAIError, build_client
from aiu_chat.agent.orchestrator import answer
from aiu_chat.agent.turn_log import build_turn_record
from aiu_chat.sources import chatlog

def check_password():
    if st.session_state.get("authenticated", False):
        return True

    password = st.text_input("Password", type="password")

    if st.button("Log in"):
        if hmac.compare_digest(
            password,
            st.secrets["APP_PASSWORD"],
        ):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


if not check_password():
    st.stop()

def _about():
    # Imported lazily; `streamlit run app/streamlit_app.py` puts app/ on sys.path.
    import about_page

    about_page.render()

st.set_page_config(page_title="Aviation Intelligence + Chat", page_icon="✈️", layout="centered")

# EUROCONTROL logo at the very top of the sidebar, above the page menu.
_LOGO_PATH = str(Path(__file__).parent / "assets" / "eurocontrol-logo.svg")
if Path(_LOGO_PATH).exists():
    st.logo(_LOGO_PATH, size="large")

# Enlarge the logo (st.logo's "large" still renders small for this SVG). Injected
# at the top level so it applies on every page. Targets the several test-ids
# Streamlit has used for the sidebar logo across versions.
st.markdown(
    """
    <style>
    /* Verified Streamlit logo test-ids (img elements). */
    img[data-testid="stLogo"],
    img[data-testid="stSidebarLogo"],
    img[data-testid="stHeaderLogo"] {
        height: 3rem !important; width: auto !important; max-width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DISCLAIMER = (
    "Data © EUROCONTROL Aviation Intelligence Unit ([ansperformance.eu](https://ansperformance.eu)). "
    "Always verify: the assistant may pick the wrong data source or misinterpret it."
)

# Shown when logging is on so users know conversations are recorded (quiet, brief).
LOGGING_NOTICE = (
    "Conversations are recorded for quality and analysis."
)

# A small, scoped style refresh (kept minimal so it survives Streamlit updates).
_CSS = """
<style>
/* Tighten the top padding so the header sits higher. */
.block-container { padding-top: 2.2rem; }

/* Gradient header banner. */
.aiu-header {
    background: linear-gradient(135deg, #1565C0 0%, #1E88E5 60%, #42A5F5 100%);
    color: #fff; padding: 1.1rem 1.3rem; border-radius: 0.8rem;
    margin-bottom: 1.1rem; box-shadow: 0 2px 10px rgba(21,101,192,.18);
}
.aiu-header h1 { color:#fff; margin:0; font-size:1.6rem; font-weight:700; letter-spacing:.2px; }
.aiu-header p  { color:#E3F2FD; margin:.25rem 0 0; font-size:.95rem; }

/* Pill-style chips for route + source provenance. */
.aiu-chip {
    display:inline-block; padding:.12rem .6rem; margin:.15rem .3rem .15rem 0;
    border-radius:999px; font-size:.78rem; font-weight:600; line-height:1.5;
    background:#EEF3FB; color:#1565C0; border:1px solid #D9E2F0;
}
.aiu-chip.src { background:#F4F7FB; color:#46566B; font-weight:500; }

/* Suggestion grid: single-line column titles so rows stay aligned. */
.aiu-sugg-title {
    font-weight:700; font-size:.9rem; color:#1A2433;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
    height:1.6rem; line-height:1.6rem; margin-bottom:.4rem;
}

/* Suggestion buttons: equal-sized boxes (3-line height), centred text. */
div[data-testid="stExpander"] .stButton > button {
    display:flex; align-items:center; justify-content:center; text-align:center;
    height:5.6rem; width:100%; white-space:normal; line-height:1.3;
    padding:.6rem .7rem; margin-bottom:.45rem; overflow:hidden;
    border:1px solid #D9E2F0; border-radius:.6rem; background:#FBFCFE;
    font-weight:400; font-size:.84rem; transition:all .12s ease;
}
div[data-testid="stExpander"] .stButton > button:hover {
    border-color:#1565C0; background:#EEF3FB; transform:translateY(-1px);
    box-shadow:0 2px 6px rgba(21,101,192,.12);
}
div[data-testid="stExpander"] .stButton > button p { margin:0; }
</style>
"""


def _chip(text: str, kind: str = "", url: str | None = None) -> str:
    cls = f"aiu-chip {kind}".strip()
    if url:
        return f'<a class="{cls}" href="{url}" target="_blank" style="text-decoration:none">{text}</a>'
    return f'<span class="{cls}">{text}</span>'


# Suggested topics shown when the chat is empty. Each item is (question, label),
# where the label is the same text with the key terms in **bold** for the button.
SUGGESTIONS = [
    (
        "🟢 Live now",
        [
            ("How many aircraft are airborne right now?",
             "How many **aircraft are airborne** right now?"),
            ("Which areas have the most delay right now?",
             "Which areas have the **most delay** right now?"),
            ("Are there active weather regulations now?",
             "Are there active **weather regulations** now?"),
        ],
    ),
    (
        "📅 Latest daily",
        [
            ("How many flights did France have on the latest day?",
             "How many **flights** did **France** have on the latest day?"),
            ("What is the latest daily ATFM delay for DSNA?",
             "Latest daily **ATFM delay** for **DSNA**?"),
            ("Latest punctuality at Heathrow?",
             "Latest **punctuality** at **Heathrow**?"),
        ],
    ),
    (
        "📊 Historical",
        [
            ("Which 5 states had the most CO2 emissions in 2024?",
             "Which 5 states had the most **CO2 emissions** in **2024**?"),
            ("Show EGLL airport traffic per year as a bar chart",
             "**EGLL airport traffic** per year as a **bar chart**"),
            ("Which ANSP had the most en-route ATFM delay in 2025?",
             "Which **ANSP** had the most **en-route ATFM delay** in 2025?"),
        ],
    ),
    (
        "📡 Network ops",
        [
            ("What's the current tactical situation on the network?",
             "Current **tactical situation** on the **network**?"),
            ("Any airport regulations or airspace issues right now?",
             "Any **airport regulations** or **airspace** issues now?"),
            ("How is additional ASMA time calculated?",
             "How is **additional ASMA time** calculated?"),
        ],
    ),
]


# Human-readable explanation of each route, shown so the user sees how (and why)
# the assistant answered.
ROUTE_INFO = {
    "data": ("📊 Historical data (SQL)",
             "Recognised a question about past figures → wrote SQL and ran it on the local datasets."),
    "concept": ("📖 Concept / methodology",
                "Recognised a definition/methodology question → searched the reference docs & PDFs."),
    "both": ("📊+📖 Data and concept",
             "Needed both a figure and an explanation → combined the data and concept paths."),
    "nop": ("📡 NOP messages",
            "Recognised an operational/NOP question → fetched and interpreted recent NOP messages."),
    "dataapp": ("📅 Latest daily (D-1)",
                "Recognised a request for recent daily figures → queried the EUROCONTROL Data App API (D-1)."),
    "nm_live": ("🟢 Real-time network",
                "Recognised a 'right now' question → fetched the live Network Manager snapshot."),
    "catalog": ("🗂️ Data catalogue",
                "Recognised a 'what data do you have' question → answered from the dataset catalogue."),
    "none": ("🚫 Out of scope",
             "Judged the question to be outside European air navigation performance → declined."),
}


@st.cache_resource(show_spinner=False)
def _catalog():
    return get_catalog()


@st.cache_resource(show_spinner=False)
def _client(tier: str):
    """Cached per mode so changing the selector rebuilds the client."""
    return build_client(tier)


def _render_turn(turn, idx):
    """Render a Turn: combined prose, optional chart + table + SQL, and sources.

    `idx` makes element keys unique across replayed turns — Streamlit raises
    StreamlitDuplicateElementId if two charts/dataframes share an auto-ID.
    """
    # A clarifying question: show it plainly, no route chrome.
    if turn.needs_clarification:
        st.markdown(f"❓ {turn.answer}")
        st.caption("Please reply with the detail and I'll continue.")
        return

    # Show how the question was routed (transparency into the agent's choice).
    # Multi-source turns show a chip per source; single-source shows one.
    routes = getattr(turn, "routes", None) or [turn.route]
    chips = []
    whys = []
    for r in routes:
        label, why = ROUTE_INFO.get(r, (r, ""))
        chips.append(_chip(label))
        if why:
            whys.append(f"**{label}** — {why}")
    if len(routes) > 1:
        chips.insert(0, _chip("🔀 Multi-source"))
    st.markdown(" ".join(chips), unsafe_allow_html=True)
    detail = "\n\n".join(whys)
    if turn.standalone_question and turn.standalone_question != turn.question:
        detail += f"\n\nInterpreted your question as: *{turn.standalone_question}*"
    if detail:
        with st.expander("How I answered this"):
            st.markdown(detail)

    st.markdown(turn.answer)

    data = turn.data
    if data is not None and data.result is not None and not data.result.dataframe.empty:
        df = data.result.dataframe
        # Chart first (if the spec is valid + chart-worthy), then the table.
        fig = make_chart(data.chart_spec, df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
        st.dataframe(df, use_container_width=True, hide_index=True, key=f"df_{idx}")
        if data.result.truncated:
            st.caption(f"Showing first {data.result.row_count} rows.")
        if data.sql:
            with st.expander("Show SQL"):
                st.code(data.sql, language="sql")

    # Live NOP messages: show the underlying messages in an expander.
    if turn.nop is not None and turn.nop.messages:
        with st.expander(f"NOP messages ({len(turn.nop.messages)})"):
            for m in turn.nop.messages:
                st.markdown(f"**{m.type}** · {m.published}")
                st.text(m.text[:1500])
        st.markdown(_chip("📡 NOP · live", "src"), unsafe_allow_html=True)

    # Data App figures: D-1 (latest daily), not real-time — one chip per entity
    # looked up (fan-out shows several).
    if turn.dataapp is not None and turn.dataapp.results:
        chips = " ".join(
            _chip(f"📅 Data App · {r.entity.name} · {r.sync_date} (D-1)", "src")
            for r in turn.dataapp.results
        )
        st.markdown(chips, unsafe_allow_html=True)

    # NM live snapshot: genuinely real-time.
    if turn.nm_live is not None and turn.nm_live.snapshot is not None:
        st.markdown(_chip("🟢 Network Manager · live", "src"), unsafe_allow_html=True)

    # Cross-frame aggregate (#4): show the computed figure + its SQL (auditable).
    agg = getattr(turn, "aggregate", None)
    if agg is not None and agg.dataframe is not None and not agg.dataframe.empty:
        st.markdown(_chip("🧮 Combined figure (computed)"), unsafe_allow_html=True)
        st.dataframe(agg.dataframe, use_container_width=True, hide_index=True,
                     key=f"agg_{idx}")
        with st.expander("Show aggregation SQL"):
            st.code(agg.sql, language="sql")

    if turn.sources:
        seen = []
        for s in turn.sources:
            if s.source_url not in [u for _, u in seen]:
                seen.append((s.source_title, s.source_url))
        chips = " ".join(_chip(f"📄 {title}", "src", url=url) for title, url in seen[:6])
        st.markdown(chips, unsafe_allow_html=True)


def _render_suggestions():
    """Topic cards with example questions arranged in an aligned grid; clicking
    one asks it. Returns the chosen question, or None."""
    chosen = None
    cols = st.columns(len(SUGGESTIONS), gap="small")
    for col, (topic, questions) in zip(cols, SUGGESTIONS):
        with col:
            # Fixed single-line title so columns stay row-aligned regardless of
            # title length.
            st.markdown(f'<div class="aiu-sugg-title">{topic}</div>', unsafe_allow_html=True)
            for j, (question, label) in enumerate(questions):
                if st.button(label, key=f"sugg_{topic}_{j}", use_container_width=True):
                    chosen = question
    return chosen


def _sidebar_controls():
    """Render the mode selector and return the chosen tier key."""
    with st.sidebar:
        st.subheader("Mode")
        tier_keys = list(config.MODEL_TIERS)
        default_idx = tier_keys.index(config.DEFAULT_TIER) if config.DEFAULT_TIER in tier_keys else 0
        tier = st.radio(
            "Choose a mode",
            tier_keys,
            index=default_idx,
            format_func=lambda k: config.MODEL_TIERS[k]["label"],
            label_visibility="collapsed",
        )
        st.caption(config.MODEL_TIERS[tier]["blurb"])

        # Contact details + data disclaimer at the bottom.
        st.divider()
        st.caption(
            "**Issues or feedback?**  \n"
            "[quinten.goens@eurocontrol.int](mailto:quinten.goens@eurocontrol.int)"
        )
        st.divider()
        notice = DISCLAIMER
        if config.chat_logging_configured():
            notice += f"  \n{LOGGING_NOTICE}"
        st.caption(f"**Data & disclaimer**  \n{notice}")
    return tier


def _ensure_log_session(tier: str):
    """Capture the browser fingerprint and create the logging session once.

    Returns (fingerprint_dict, session_record_id | None). Entirely best-effort:
    any failure returns a minimal fingerprint so chat continues unaffected.
    """
    if not config.chat_logging_configured():
        return {}, None
    try:
        from fingerprint import get_fingerprint

        fp = get_fingerprint()
    except Exception:
        fp = {"session_id": st.session_state.get("session_id", "")}

    # Create the session record once per browser conversation.
    if "log_session_created" not in st.session_state:
        try:
            now = _iso_now()
            rid = chatlog.ensure_session(
                fp.get("session_id", ""),
                {
                    "started_at": now,
                    "last_seen_at": now,
                    "app_mode": "local" if config.LOCAL else "cloud",
                    "model_tier": tier,
                    "user_agent": fp.get("user_agent", ""),
                    "languages": fp.get("languages", ""),
                    "timezone": fp.get("timezone", ""),
                    "screen": fp.get("screen", ""),
                    "platform": fp.get("platform", ""),
                    "browser_id": fp.get("browser_id", ""),
                    "fingerprint_hash": fp.get("fingerprint_hash", ""),
                    "extra": {k: fp.get(k) for k in
                              ("hardware_concurrency", "device_memory") if fp.get(k) is not None},
                },
            )
            st.session_state["log_session_created"] = True
            st.session_state["log_session_record_id"] = rid
        except Exception:
            # Never let logging setup break the app.
            st.session_state["log_session_created"] = True
            st.session_state["log_session_record_id"] = None

    return fp, st.session_state.get("log_session_record_id")


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")


def _log_turn_safe(turn, *, fp, session_record_id, tier, latency_ms, error=None):
    """Serialize + append one turn. Best-effort; never raises into the UI."""
    if not config.chat_logging_configured():
        return
    try:
        record = build_turn_record(
            turn,
            turn_index=len(st.session_state.get("history", [])),
            model_tier=tier,
            latency_ms=latency_ms,
            error=error,
        ) if turn is not None else {
            "turn_index": len(st.session_state.get("history", [])),
            "created_at": _iso_now(),
            "model_tier": tier or "",
            "error": (error or "")[:20000],
            "latency_ms": int(latency_ms) if latency_ms is not None else None,
        }
        chatlog.log_turn(
            session_id=fp.get("session_id", ""),
            session_record_id=session_record_id,
            turn=record,
        )
    except Exception:
        pass  # logging must never surface to the user


def main():
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="aiu-header">'
        "<h1>✈️ Aviation Intelligence + Chat</h1>"
        "<p>Ask about European air navigation performance — historical data, "
        "the latest daily figures, the live network, and operational updates.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        catalog = _catalog()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        st.stop()

    tier = _sidebar_controls()

    # Capture the browser fingerprint + open a logging session (best-effort).
    fp, session_record_id = _ensure_log_session(tier)

    # Chat history (Turn objects) lives in session state and feeds follow-ups.
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "turn"?}
    if "history" not in st.session_state:
        st.session_state.history = []  # list of Turn

    # Suggestion cards stay at the top, always — open before the first question,
    # then collapsed (but still available) once a conversation starts.
    started = bool(st.session_state.messages)
    with st.expander("💡 Example questions", expanded=not started):
        suggested = _render_suggestions()

    # Replay history. The message index gives every element a stable unique key.
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("turn") is not None:
                _render_turn(msg["turn"], idx=i)
            else:
                st.markdown(msg["content"])

    typed = st.chat_input("e.g. Which 5 states had the most CO2 emissions in 2024?")
    prompt = typed or suggested
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_box = st.status("Thinking…", expanded=True)
            # Accumulate every stage in the one dropdown (none overwritten).
            _seen_labels: list[str] = []

            def _on_status(label, detail=None):
                if label not in _seen_labels:
                    _seen_labels.append(label)
                    status_box.write(f"**{label}**")
                if detail:
                    status_box.write(detail)
                status_box.update(label=label)

            _t0 = time.monotonic()
            _err = None
            try:
                turn = answer(
                    prompt,
                    history=st.session_state.history,
                    client=_client(tier),
                    catalog=catalog,
                    on_status=_on_status,
                )
                status_box.update(label="✓ Done", state="complete", expanded=False)
            except (OllamaError, OpenAIError) as exc:
                status_box.update(label="Failed", state="error")
                st.error(f"Could not reach the model: {exc}")
                turn = None
                _err = str(exc)
            _latency_ms = int((time.monotonic() - _t0) * 1000)

            # Log the turn (best-effort) before mutating history, so turn_index
            # reflects this turn's position.
            _log_turn_safe(
                turn, fp=fp, session_record_id=session_record_id,
                tier=tier, latency_ms=_latency_ms, error=_err,
            )

            if turn is not None:
                _render_turn(turn, idx=len(st.session_state.messages))
                st.session_state.history.append(turn)
                st.session_state.messages.append(
                    {"role": "assistant", "content": turn.answer, "turn": turn}
                )


def _admin():
    import admin_view

    admin_view.render()


# Explicit multi-page navigation so the sidebar labels are clean
# ("Aviation Intelligence + Chat" / "About"), not derived from filenames.
_pages = [
    st.Page(main, title="Aviation Intelligence + Chat", icon="✈️", default=True),
]

# Admin Panel: sits between Chat and About, shown whenever a viewer password is
# configured. The page itself is password-gated (a separate password from the
# main app login).
if config.admin_viewer_configured():
    _pages.append(st.Page(_admin, title="Admin Panel", icon="🗂️"))

_pages.append(st.Page(_about, title="About", icon="ℹ️"))

_nav = st.navigation(_pages)
_nav.run()
