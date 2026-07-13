"""The published NSR archive (Data App `/situation_reports`).

Two uses: teaching the drafter house style by example, and giving the backtest
something to diff against. 77 reports, weekly, back to Dec 2024.
"""
from __future__ import annotations

import datetime as dt
import re

import requests

from aiu_chat import config
from aiu_chat.nsr.facts import Week
from aiu_chat.sources import dataapp

TIMEOUT = 30


def fetch_all(session: requests.Session | None = None) -> list[dict]:
    """Every published report, newest first."""
    own = session is None
    session = session or requests.Session()
    try:
        r = session.get(
            f"{config.DATAAPP_BASE}/situation_reports",
            params={"itemsPerPage": 100, "order[date]": "desc"},
            timeout=TIMEOUT,
            headers={"User-Agent": dataapp.USER_AGENT},
        )
        r.raise_for_status()
        return r.json()["data"]
    finally:
        if own:
            session.close()


def strip_html(html: str) -> str:
    text = re.sub(r"</li>|</p>", "\n", html)
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("\xa0", " ")


def headline_text(report: dict) -> str:
    """Just the three headline paragraphs -- the part whose style we imitate.

    The bullets are dropped: they are the previous week's causes and would only
    tempt the model into carrying them over.
    """
    text = strip_html(report["content"])
    keep, section = [], None
    for line in (ln.strip() for ln in text.split("\n")):
        if not line:
            continue
        if line in ("Traffic", "ATFM Delay", "Punctuality"):
            section = line
            keep.append(line)
            continue
        if line.startswith("For "):        # start of the bullet lists
            section = None
            continue
        if section:
            keep.append(line)
    return "\n".join(keep)


def recent_headlines(*, before: Week, limit: int = 3,
                     session: requests.Session | None = None) -> list[str]:
    """The `limit` most recent published headlines from before `before`."""
    cutoff = before.monday
    out = []
    for rep in fetch_all(session):
        pub = dt.date.fromisoformat(rep["date"][:10])
        if pub >= cutoff:
            continue
        out.append(headline_text(rep))
        if len(out) >= limit:
            break
    return out
