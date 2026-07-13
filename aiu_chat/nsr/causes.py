"""Causal evidence for the NSR bullets, from the archived NOP tactical updates.

The Data App tells us *how much* delay an ACC or airport had; it never says why.
The NOP tactical updates do -- they are the operational commentary published
through the week, and the published NSR bullets are visibly written from them.
For W22/2026 the NOP said:

    EDDM (Munich)
    Arrivals regulated at a reduced rate due to drones in the vicinity.

and the report said "Munich ... A drone sighting also caused delays ... on
Saturday." That is the relationship this module exposes: for each entity the
delay ranking surfaced, the NOP lines from that week that mention it, with dates,
so the drafter can write a grounded bullet and cite it -- and so the analyst can
see exactly what the sentence was built from.

**This module retrieves; it does not classify.** An earlier cut tried to reduce
each excerpt to a reason code (ATC CAPACITY / WEATHER / STAFFING). That is the
wrong shape for this data, for two reasons:

  * NOP is prose, not an enum. One week alone phrases weather as "weather (cb)",
    "weather (cbs)", "weather (ts)", "weather (lvp)", "cb activity" and "daytime
    heating", writes the Middle East on-load five different ways, and contains
    "atc capcity". Any pattern set over that is a lossy enum to maintain forever.

  * The interesting causes have no bucket. "Arrivals regulated at a reduced rate
    due to drones in the vicinity" is exactly the line the published report was
    built from, and a classifier would have thrown it away and kept "weather".

So the split is: deterministic code decides what is *true* (facts.py) and what
evidence is *admissible* (this module -- the right week, the right entity, the
verbatim line). The LLM then reads those excerpts and writes the bullet, which is
a summarisation task over supplied text rather than an act of recall. Grounding
comes from constraining the model to these excerpts and citing their message ids,
not from pre-digesting them into labels.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import requests

from aiu_chat import config
from aiu_chat.nsr.facts import Week
from aiu_chat.sources import nop

# NOP writes ICAO codes and long names; the ranking writes short names. Bridge
# the two so "Tel Aviv" finds "LLBG", and "Karlsruhe UAC" finds "Karlsruhe".
_AIRPORT_ICAO = {
    "amsterdam": "EHAM", "athens": "LGAV", "barcelona": "LEBL",
    "brussels": "EBBR", "dublin": "EIDW", "frankfurt": "EDDF",
    "geneva": "LSGG", "helsinki": "EFHK", "istanbul": "LTFM",
    "lisbon": "LPPT", "london gatwick": "EGKK", "london heathrow": "EGLL",
    "madrid": "LEMD", "manchester": "EGCC", "munich": "EDDM",
    "nice": "LFMN", "palma de mallorca": "LEPA", "paris charles de gaulle": "LFPG",
    "rome fiumicino": "LIRF", "tel aviv": "LLBG", "vienna": "LOWW",
    "warsaw": "EPWA", "zurich": "LSZH",
}

@dataclass
class Evidence:
    """One NOP line that mentions an entity, kept verbatim for citation."""

    date: dt.date
    excerpt: str
    message_id: str

    @property
    def weekday(self) -> str:
        return self.date.strftime("%A")


@dataclass
class EntityCauses:
    """Everything we can say about one ACC or airport for one week."""

    name: str
    kind: str                      # "acc" | "airport"
    rank: int
    delay_per_flight: float | None
    total_delay_min: float | None
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """Can a bullet about this entity cite anything at all?"""
        return bool(self.evidence)

    @property
    def days_mentioned(self) -> list[str]:
        return sorted({e.weekday for e in self.evidence},
                      key=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday",
                                     "Friday", "Saturday", "Sunday"].index(d))


def _aliases(name: str, kind: str) -> list[str]:
    """The strings a NOP message might use for this entity."""
    out = [name]
    low = name.lower()
    if kind == "acc":
        # "Karlsruhe UAC" -> also match a bare "Karlsruhe".
        out.append(re.sub(r"\s+(ACC|UAC)$", "", name, flags=re.I))
    else:
        icao = _AIRPORT_ICAO.get(low)
        if icao:
            out.append(icao)
        # NOP writes "Tel-Aviv" where the ranking writes "Tel Aviv".
        if " " in name:
            out.append(name.replace(" ", "-"))
    return [a for a in dict.fromkeys(out) if len(a) > 3]


def _fetch_week(week: Week, session: requests.Session) -> list[nop.NopMessage]:
    """Every NOP message published during `week`."""
    token = nop._auth_token(session)
    url = f"{config.PB_NOP_URL}/api/collections/{nop.COLLECTION}/records"
    end = week.sunday + dt.timedelta(days=1)
    messages: list[nop.NopMessage] = []
    page = 1
    while True:
        r = session.get(
            url,
            params={
                "perPage": 200,
                "page": page,
                "sort": "nop_publish_datetime",
                "filter": (f'nop_publish_datetime >= "{week.monday.isoformat()}" && '
                           f'nop_publish_datetime < "{end.isoformat()}"'),
            },
            timeout=nop.TIMEOUT,
            headers={"User-Agent": nop.USER_AGENT, "Authorization": token},
        )
        if r.status_code != 200:
            raise nop.NopError(f"NOP week query returned HTTP {r.status_code}.")
        body = r.json()
        for it in body.get("items", []):
            messages.append(nop.NopMessage(
                id=it.get("id", ""),
                type=it.get("nop_message_type", ""),
                published=it.get("nop_publish_datetime", ""),
                text=nop._strip_html(it.get("nop_message_content", "")),
            ))
        if page >= body.get("totalPages", 1):
            break
        page += 1
    return messages


def _excerpt(text: str, alias: str) -> str | None:
    """The paragraph of a NOP message that talks about `alias`.

    NOP messages are sectioned ("Weather:", "Aerodromes:", then a block per
    airport). We want the block, not the whole bulletin -- a bullet cited to
    2,000 words of network-wide weather is not a citation.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for i, line in enumerate(lines):
        if alias.lower() not in line.lower():
            continue
        # The entity is usually a heading with its detail on the lines beneath.
        block = [line]
        for nxt in lines[i + 1:]:
            if _is_heading(nxt):
                break
            block.append(nxt)
            if len(" ".join(block)) > 320:
                break
        return " ".join(block)[:400]
    return None


def _is_heading(line: str) -> bool:
    """Does this line start a new entity's block?

    Headings are ICAO-coded: "EDDM (Munich)", but also "LGGG/LGMD (Athens/
    Makedonia)" and "LFEE (Reims)". A naive `^[A-Z]{4}\\s*\\(` misses the slashed
    form, which then bleeds the next airport's text into this one's citation --
    and a bullet citing another airport's regulation is worse than no bullet.
    """
    return bool(
        re.match(r"^[A-Z]{4}(?:/[A-Z]{4})*\s*\(", line)
        or line.endswith(":")
    )


def collect(week: Week, entities: list[tuple[str, str, int, float | None, float | None]],
            *, session: requests.Session | None = None) -> list[EntityCauses]:
    """Gather NOP evidence for each ranked entity.

    `entities` is (name, kind, rank, delay_per_flight, total_delay_min).
    """
    own = session is None
    session = session or requests.Session()
    try:
        messages = _fetch_week(week, session)
        out: list[EntityCauses] = []

        for name, kind, rank, per_flight, total in entities:
            ec = EntityCauses(name=name, kind=kind, rank=rank,
                              delay_per_flight=per_flight, total_delay_min=total)
            aliases = _aliases(name, kind)
            seen: set[str] = set()

            for msg in messages:
                for alias in aliases:
                    if alias.lower() not in msg.text.lower():
                        continue
                    snippet = _excerpt(msg.text, alias)
                    if not snippet or snippet in seen:
                        continue
                    seen.add(snippet)
                    try:
                        day = dt.date.fromisoformat(msg.published[:10])
                    except ValueError:
                        continue
                    ec.evidence.append(
                        Evidence(date=day, excerpt=snippet, message_id=msg.id)
                    )
                    break

            ec.evidence.sort(key=lambda e: e.date)
            out.append(ec)
        return out
    finally:
        if own:
            session.close()
