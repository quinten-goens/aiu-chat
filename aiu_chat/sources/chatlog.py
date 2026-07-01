"""Chat logging + retrieval against the PocketBase `chat_sessions` / `chat_turns`
collections.

Two roles share this module:

* **The app appends** with the dedicated, create-only log user
  (`PB_CHAT_USER_*`). Writes are best-effort and fail-open: if PocketBase is
  slow or unreachable the chat still answers — a logging failure is recorded in
  the app log (per the "never silently swallow a failure" rule) but never shown
  to the end user.
* **The hidden viewer reads** with the superuser token (`PB_ADMIN_USER_*`),
  since the collections are create-only for the app and expose no public read.

PocketHost sits behind Cloudflare, which 403s bare urllib/requests User-Agents,
so every call sends a browser-like UA (same lesson as the NOP source).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from aiu_chat import config

logger = logging.getLogger("aiu_chat.chatlog")

# Cloudflare in front of PocketHost blocks non-browser User-Agents.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36 aiu-chat-log"
)
TIMEOUT = 15  # short: logging must never stall the UI for long

SESSIONS = "chat_sessions"
TURNS = "chat_turns"


class ChatLogError(RuntimeError):
    """PocketBase unreachable, auth failed, or a write/read was rejected."""


# --- token cache (per role) -------------------------------------------------
# Cached module-side with a soft TTL so we don't re-auth on every turn.
_TOKEN_TTL = 20 * 60  # PocketBase tokens last much longer; refresh conservatively
_cache: dict[str, tuple[str, float]] = {}


def _auth(session: requests.Session, *, collection: str, identity: str,
          password: str, cache_key: str) -> str:
    """Return a (cached) auth token for `collection`/`identity`."""
    hit = _cache.get(cache_key)
    if hit and (time.time() - hit[1]) < _TOKEN_TTL:
        return hit[0]
    url = f"{config.PB_CHAT_URL}/api/collections/{collection}/auth-with-password"
    try:
        r = session.post(
            url,
            json={"identity": identity, "password": password},
            timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.RequestException as exc:
        raise ChatLogError(f"Could not reach PocketBase at {config.PB_CHAT_URL}: {exc}") from exc
    if r.status_code != 200:
        raise ChatLogError(f"Chat-log auth failed for {collection} (HTTP {r.status_code}).")
    token = r.json().get("token")
    if not token:
        raise ChatLogError("Chat-log auth returned no token.")
    _cache[cache_key] = (token, time.time())
    return token


def _log_token(session: requests.Session) -> str:
    """Token for the create-only log user (used by the app to append)."""
    return _auth(
        session, collection="users",
        identity=config.PB_CHAT_USER_EMAIL, password=config.PB_CHAT_USER_PASSWORD,
        cache_key="log",
    )


def _admin_token(session: requests.Session) -> str:
    """Superuser token (used only by the hidden viewer to read)."""
    if not (config.PB_ADMIN_USER_EMAIL and config.PB_ADMIN_USER_PASSWORD):
        raise ChatLogError("Admin credentials (PB_ADMIN_USER_*) are not set.")
    return _auth(
        session, collection="_superusers",
        identity=config.PB_ADMIN_USER_EMAIL, password=config.PB_ADMIN_USER_PASSWORD,
        cache_key="admin",
    )


# --- writes (app, best-effort) ---------------------------------------------
def _create(session: requests.Session, token: str, collection: str, body: dict) -> dict:
    url = f"{config.PB_CHAT_URL}/api/collections/{collection}/records"
    try:
        r = session.post(
            url, json=body, timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Authorization": token},
        )
    except requests.RequestException as exc:
        raise ChatLogError(f"Chat-log write failed: {exc}") from exc
    if r.status_code not in (200, 201):
        raise ChatLogError(f"Chat-log write to {collection} returned HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def ensure_session(session_id: str, fields: dict) -> str | None:
    """Create the session record if it doesn't exist yet; return its record id.

    Idempotent per browser session: a UNIQUE index on `session_id` means a repeat
    create returns 400, which we treat as "already there" and look up the id.
    Best-effort — returns None on any failure (logging must never break chat).
    """
    if not config.chat_logging_configured():
        return None
    s = requests.Session()
    try:
        token = _log_token(s)
        body = {"session_id": session_id, **fields}
        try:
            rec = _create(s, token, SESSIONS, body)
            return rec.get("id")
        except ChatLogError as exc:
            # Likely the UNIQUE(session_id) collision on a returning session —
            # the record already exists; we don't have read access as the log
            # user, so just return None (the turn's session_id links it anyway).
            logger.debug("ensure_session create returned: %s", exc)
            return None
    except ChatLogError as exc:
        logger.warning("chat-log ensure_session failed (non-fatal): %s", exc)
        return None
    finally:
        s.close()


def log_turn(*, session_id: str, session_record_id: str | None, turn: dict) -> None:
    """Append one turn record. Best-effort; swallows errors into the app log."""
    if not config.chat_logging_configured():
        return
    s = requests.Session()
    try:
        token = _log_token(s)
        body = {"session_id": session_id, **turn}
        if session_record_id:
            body["session"] = session_record_id
        _create(s, token, TURNS, body)
    except ChatLogError as exc:
        logger.warning("chat-log log_turn failed (non-fatal): %s", exc)
    except Exception as exc:  # never let logging crash a request
        logger.warning("chat-log log_turn unexpected error (non-fatal): %s", exc)
    finally:
        s.close()


# --- reads (viewer, superuser) ---------------------------------------------
@dataclass
class SessionRow:
    id: str
    data: dict


def _list(session: requests.Session, token: str, collection: str, *,
          per_page: int = 200, page: int = 1, filt: str = "", sort: str = "",
          expand: str = "", fields: str = "") -> dict:
    params = {"perPage": per_page, "page": page}
    if filt:
        params["filter"] = filt
    if sort:
        params["sort"] = sort
    if expand:
        params["expand"] = expand
    if fields:
        params["fields"] = fields
    url = f"{config.PB_CHAT_URL}/api/collections/{collection}/records"
    try:
        r = session.get(
            url, params=params, timeout=TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Authorization": token},
        )
    except requests.RequestException as exc:
        raise ChatLogError(f"Chat-log read failed: {exc}") from exc
    if r.status_code != 200:
        raise ChatLogError(f"Chat-log read from {collection} returned HTTP {r.status_code}.")
    return r.json()


def list_sessions(*, limit: int = 200, search: str = "") -> list[dict]:
    """Return recent sessions (newest first). Viewer-only (superuser token)."""
    s = requests.Session()
    try:
        token = _admin_token(s)
        filt = ""
        if search:
            esc = search.replace("\\", "\\\\").replace('"', '\\"')
            filt = (f'session_id ~ "{esc}" || user_agent ~ "{esc}" '
                    f'|| fingerprint_hash ~ "{esc}" || client_ip ~ "{esc}"')
        data = _list(s, token, SESSIONS, per_page=limit, filt=filt, sort="-created")
        return data.get("items", [])
    finally:
        s.close()


def list_turns(session_id: str, *, limit: int = 500) -> list[dict]:
    """Return all turns for a session_id, oldest first. Viewer-only."""
    s = requests.Session()
    try:
        token = _admin_token(s)
        esc = session_id.replace("\\", "\\\\").replace('"', '\\"')
        data = _list(s, token, TURNS, per_page=limit,
                     filt=f'session_id = "{esc}"', sort="turn_index,created")
        return data.get("items", [])
    finally:
        s.close()


def recent_turns(*, limit: int = 200, search: str = "") -> list[dict]:
    """Return recent turns across all sessions (newest first). Viewer-only."""
    s = requests.Session()
    try:
        token = _admin_token(s)
        filt = ""
        if search:
            esc = search.replace("\\", "\\\\").replace('"', '\\"')
            filt = f'question ~ "{esc}" || answer ~ "{esc}" || sql ~ "{esc}"'
        data = _list(s, token, TURNS, per_page=limit, filt=filt, sort="-created")
        return data.get("items", [])
    finally:
        s.close()


# --- analytics (viewer, superuser) -----------------------------------------
def count(collection: str, *, filt: str = "") -> int:
    """Exact record count via PocketBase's totalItems (perPage=1). Viewer-only."""
    s = requests.Session()
    try:
        token = _admin_token(s)
        data = _list(s, token, collection, per_page=1, filt=filt)
        return int(data.get("totalItems", 0))
    finally:
        s.close()


def fetch_all(collection: str, *, fields: str = "", filt: str = "",
              sort: str = "-created", page_size: int = 500,
              max_records: int = 20000) -> list[dict]:
    """Fetch all records of a collection (paged), returning a slim projection.

    `fields` limits the columns pulled (e.g. "route,created,latency_ms") so
    analytics never drags down big result_table/answer blobs. Capped by
    `max_records` as a safety valve. Viewer-only (superuser token).
    """
    s = requests.Session()
    out: list[dict] = []
    try:
        token = _admin_token(s)
        page = 1
        while len(out) < max_records:
            data = _list(s, token, collection, per_page=page_size, page=page,
                         filt=filt, sort=sort, fields=fields)
            items = data.get("items", [])
            out.extend(items)
            if page >= int(data.get("totalPages", 1)) or not items:
                break
            page += 1
        return out[:max_records]
    finally:
        s.close()
