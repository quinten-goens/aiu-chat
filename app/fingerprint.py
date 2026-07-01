"""Browser fingerprint capture for chat logging.

On Streamlit Cloud the Python server sits behind a proxy and does not reliably
see the real client IP or User-Agent, and cannot recognise a returning browser.
This module runs a tiny JS snippet in the user's browser that reads the
identifying attributes (User-Agent, languages, timezone, screen, platform) plus
a stable per-browser id kept in localStorage, and returns them to Python via a
Streamlit component value.

`streamlit-javascript` (a maintained micro-component) is used if installed; if it
isn't, we fall back to a session-only random id with no browser attributes, so
the app still works everywhere.
"""
from __future__ import annotations

import hashlib
import json
import uuid

import streamlit as st

# The JS run in the browser. Returns a JSON object of browser attributes and a
# stable localStorage id (created once per browser, persisted across sessions).
_JS = """
(function () {
  try {
    var KEY = 'aiu_browser_id';
    var bid = localStorage.getItem(KEY);
    if (!bid) {
      bid = (crypto && crypto.randomUUID) ? crypto.randomUUID()
            : String(Date.now()) + '-' + Math.random().toString(36).slice(2);
      localStorage.setItem(KEY, bid);
    }
    var n = window.navigator || {};
    return JSON.stringify({
      browser_id: bid,
      user_agent: n.userAgent || '',
      languages: (n.languages || [n.language || '']).join(','),
      platform: n.platform || '',
      timezone: (Intl.DateTimeFormat().resolvedOptions().timeZone) || '',
      screen: (window.screen ? (window.screen.width + 'x' + window.screen.height +
               '@' + (window.devicePixelRatio || 1)) : ''),
      hardware_concurrency: n.hardwareConcurrency || null,
      device_memory: n.deviceMemory || null
    });
  } catch (e) {
    return JSON.stringify({error: String(e)});
  }
})()
"""


def _fingerprint_hash(attrs: dict) -> str:
    """Stable hash identifying a browser across sessions.

    Uses the persistent browser_id plus stable attributes; excludes volatile ones.
    """
    basis = "|".join(str(attrs.get(k, "")) for k in
                     ("browser_id", "user_agent", "platform", "timezone", "screen"))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def get_fingerprint() -> dict:
    """Return browser attributes for the current session, cached in session_state.

    Always includes `session_id` (per-conversation) and `fingerprint_hash`.
    Runs the JS component at most once per session.
    """
    if "fingerprint" in st.session_state:
        return st.session_state["fingerprint"]

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex

    attrs: dict = {"session_id": st.session_state["session_id"]}

    try:
        from streamlit_javascript import st_javascript

        raw = st_javascript(_JS, key="aiu_fp")
        # First run returns 0/None before the browser round-trips the value.
        if raw and isinstance(raw, str) and raw.strip().startswith("{"):
            attrs.update(json.loads(raw))
    except Exception:
        # Component missing or blocked: degrade to session-only id, no attributes.
        attrs.setdefault("browser_id", st.session_state["session_id"])

    attrs["fingerprint_hash"] = _fingerprint_hash(attrs)

    # Only cache once we actually have browser attributes, so a pending first
    # render (before the JS resolves) is retried on the next rerun.
    if attrs.get("user_agent") or "streamlit_javascript" not in _sys_modules():
        st.session_state["fingerprint"] = attrs
    return attrs


def _sys_modules():
    import sys
    return sys.modules
