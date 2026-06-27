"""NOP content source: live query of EUROCONTROL Network Operations Portal
message updates stored in a PocketBase collection (`nop_content`).

Live-queried per question (no local index) so answers are always current. We
fetch the most relevant recent messages, strip the HTML to text, and hand them
to the model to interpret and answer from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from aiu_chat import config

USER_AGENT = "aiu-chat/0.1"
TIMEOUT = 30
COLLECTION = "nop_content"


class NopError(RuntimeError):
    """PocketBase unreachable or auth failed."""


@dataclass
class NopMessage:
    id: str
    type: str
    published: str
    text: str  # HTML stripped to plain text


def _strip_html(html: str) -> str:
    """Cheap HTML → text for NOP message bodies."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|h\d|li)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)  # drop remaining tags
    text = re.sub(r"&nbsp;", " ", text)
    # Normalise whitespace: trim each line, drop blank lines to single breaks.
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _auth_token(session: requests.Session) -> str:
    if not config.nop_configured():
        raise NopError(
            "NOP credentials are not set (PB_NOP_USER_EMAIL / PB_NOP_USER_PASSWORD)."
        )
    url = f"{config.PB_NOP_URL}/api/collections/users/auth-with-password"
    try:
        r = session.post(
            url,
            json={"identity": config.PB_NOP_USER_EMAIL, "password": config.PB_NOP_USER_PASSWORD},
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.RequestException as exc:
        raise NopError(f"Could not reach PocketBase at {config.PB_NOP_URL}: {exc}") from exc
    if r.status_code != 200:
        raise NopError(f"NOP auth failed (HTTP {r.status_code}).")
    return r.json()["token"]


def _escape_filter(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def fetch_messages(
    *,
    query: str | None = None,
    message_type: str | None = None,
    limit: int = 5,
    session: requests.Session | None = None,
) -> list[NopMessage]:
    """Fetch recent NOP messages, optionally filtered by keyword and/or type.

    Keyword filtering uses PocketBase's `~` (contains) on the message content;
    results are ordered newest-first.
    """
    own = session is None
    session = session or requests.Session()
    try:
        token = _auth_token(session)
        filters = []
        if query:
            filters.append(f'nop_message_content ~ "{_escape_filter(query)}"')
        if message_type:
            filters.append(f'nop_message_type = "{_escape_filter(message_type)}"')
        params = {
            "perPage": limit,
            "sort": "-nop_publish_datetime",
            "filter": " && ".join(filters) if filters else "",
        }
        url = f"{config.PB_NOP_URL}/api/collections/{COLLECTION}/records"
        try:
            r = session.get(
                url, params=params, timeout=TIMEOUT,
                headers={"User-Agent": USER_AGENT, "Authorization": token},
            )
        except requests.RequestException as exc:
            raise NopError(f"NOP query failed: {exc}") from exc
        if r.status_code != 200:
            raise NopError(f"NOP query returned HTTP {r.status_code}.")
        items = r.json().get("items", [])
    finally:
        if own:
            session.close()

    return [
        NopMessage(
            id=it.get("id", ""),
            type=it.get("nop_message_type", ""),
            published=it.get("nop_publish_datetime", ""),
            text=_strip_html(it.get("nop_message_content", "")),
        )
        for it in items
    ]
