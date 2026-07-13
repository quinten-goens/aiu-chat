"""Assemble a full Network Situation Report draft for one week.

The pipeline, and the division of labour it enforces:

    facts.collect()   -> every number, from the Data App API   (no model)
    causes.collect()  -> the NOP lines behind each entity      (no model)
    LLM               -> prose, from those numbers and lines only
    verify.check()    -> every numeral in the prose traced back to a fact

The model never computes, never recalls, and never sources. It phrases. Anything
it asserts that we cannot trace is surfaced to the analyst rather than published.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import requests

from aiu_chat.agent import prompts
from aiu_chat.agent.llm import Message, build_client
from aiu_chat.nsr import causes, facts, verify
from aiu_chat.nsr.facts import Week, WeekFacts

# How many ranked entities we offer the model per section. The published reports
# run to five bullets each; we retrieve a few more so an ungrounded entity can be
# dropped without leaving the section short.
CANDIDATES = 7
BULLETS = 5


@dataclass
class Bullet:
    """One entity's bullet, and what it was written from."""

    name: str
    kind: str
    rank: int
    delay_per_flight: float | None
    text: str
    evidence: list[causes.Evidence] = field(default_factory=list)
    grounded: bool = True
    suggestion: str | None = None      # unverified; analyst accepts or rejects

    @property
    def source_ids(self) -> list[str]:
        return [e.message_id for e in self.evidence]

    @property
    def reads_as_continuation(self) -> bool:
        """Does the text follow the printed entity name grammatically?

        House style prints the name in bold and continues it ("Barcelona ACC
        *recorded* ATC capacity delays..."), so a bullet must open with a verb. The
        model sometimes opens with a noun phrase, which renders as "Belgrade ACC
        ATC capacity regulations dominated the week".

        Flagged rather than auto-repaired: prepending a verb to text we do not
        parse turns a merely-awkward bullet into a wrong one ("EHAM recovered from
        IT issues" would become "saw recovered from IT issues"). The analyst can
        fix a flagged sentence in seconds; they cannot un-see a corrupted one.
        """
        if not self.text:
            return True
        first = self.text.split(" ", 1)[0].rstrip(",").lower()
        return first.endswith(("ed", "ke")) or first in _CONTINUATION_OPENERS


# Openers used across the published archive that are not past-tense "-ed" verbs.
_CONTINUATION_OPENERS = frozenset({
    "saw", "was", "were", "had", "also", "continued", "underwent", "met",
})


@dataclass
class Draft:
    """A complete draft, with everything needed to audit it."""

    week: Week
    facts: WeekFacts
    headline: str
    enroute: list[Bullet] = field(default_factory=list)
    airport: list[Bullet] = field(default_factory=list)
    findings: list[verify.Finding] = field(default_factory=list)
    generated_at: dt.datetime = field(default_factory=dt.datetime.now)

    @property
    def clean(self) -> bool:
        """Did every number in the prose trace back to an executed query?"""
        return not self.findings

    @property
    def bullets(self) -> list[Bullet]:
        return self.enroute + self.airport

    def to_html(self) -> str:
        """The report in the shape the publishing system stores."""
        out = [self.headline_html()]
        for label, bullets in (("For en-route ATFM delay:", self.enroute),
                               ("For airport ATFM delay:", self.airport)):
            out.append(f"<p><strong>{label}</strong></p><ul>")
            for b in bullets:
                out.append(f"<li><strong>{b.name}</strong> {b.text}</li>")
            out.append("</ul>")
        return "".join(out)

    def headline_html(self) -> str:
        html = []
        for para in self.headline.split("\n\n"):
            lines = [ln for ln in para.split("\n") if ln.strip()]
            if not lines:
                continue
            head, body = lines[0], " ".join(lines[1:])
            html.append(f"<p><strong>{head}</strong></p><p>{body}</p>")
        return "".join(html)


def _facts_block(wf: WeekFacts) -> str:
    return "\n".join(f"  {f.key} = {f.text}" for f in wf.facts.values())


def _examples(week: Week, session: requests.Session, n: int = 3) -> str:
    """Recent published headline paragraphs, as style exemplars.

    Style is taught by example rather than described: the sentence shape has been
    stable across all 77 published reports, and showing it is more reliable than
    trying to specify it.
    """
    from aiu_chat.nsr.archive import recent_headlines
    return "\n\n".join(recent_headlines(before=week, limit=n, session=session))


def _write_headline(wf: WeekFacts, session: requests.Session, client) -> str:
    msgs = [
        Message("system", prompts.NSR_HEADLINE_SYSTEM),
        Message("user", prompts.NSR_HEADLINE_USER.format(
            week=wf.week.label,
            span=wf.week.span_text(),
            prev_week=wf.week.previous().label,
            prev_year=wf.week.sunday.year - 1,
            facts=_facts_block(wf),
            examples=_examples(wf.week, session),
        )),
    ]
    return client.chat(msgs, temperature=0.0).strip()


def _write_bullet(ec: causes.EntityCauses, week: Week, client) -> Bullet:
    """One bullet, written only from that entity's NOP excerpts."""
    excerpts = "\n".join(
        f"  [{e.date:%a %d %b}] {e.excerpt}" for e in ec.evidence
    ) or "  (none)"

    msgs = [
        Message("system", prompts.NSR_BULLET_SYSTEM),
        Message("user", prompts.NSR_BULLET_USER.format(
            name=ec.name,
            kind="ACC / en-route" if ec.kind == "acc" else "airport",
            week=week.label,
            span=week.span_text(),
            excerpts=excerpts,
        )),
    ]
    raw = client.chat(msgs, temperature=0.2).strip()

    # A SUGGESTION is only meaningful when the model had nothing to go on, but it
    # will sometimes append one to a perfectly good bullet. Split it off wherever
    # it lands rather than letting unverified text into the published prose.
    suggestion = None
    m = re.search(r"\bSUGGESTION:\s*(.+)", raw, re.S)
    if m:
        suggestion = m.group(1).strip()
        raw = raw[:m.start()].strip()

    grounded = not raw.upper().startswith("INSUFFICIENT")
    text = "" if not grounded else _clean_bullet(raw, ec.name)

    return Bullet(name=ec.name, kind=ec.kind, rank=ec.rank,
                  delay_per_flight=ec.delay_per_flight, text=text,
                  evidence=ec.evidence, grounded=grounded, suggestion=suggestion)


def _clean_bullet(text: str, name: str) -> str:
    """Strip anything the model echoed back from the prompt's example format.

    The bullet is a continuation of the entity name, which the UI prints; few-shot
    examples show that by pairing a name with its text, and the model sometimes
    copies the pairing ("Karlsruhe UAC | experienced...") or restates the name.
    Neither belongs in the published sentence, so remove them here rather than
    trusting the instruction to hold every time.
    """
    text = text.strip().strip('"')

    # The model re-announces its subject in several shapes, all of which have to
    # come off the front, and it will chain them ("Maastricht UAC EDYY Maastricht
    # experienced..."). Strip repeatedly until the text starts with its verb.
    #   - the name itself, bracketed or followed by a separator
    #   - the name with a generic suffix ("Frankfurt airport")
    #   - a hyphen/space variant ("Tel-Aviv" for "Tel Aviv")
    #   - the ICAO code the excerpts use ("EDYY", "EHAM")
    # Strip the ACC/UAC suffix from the stem so a chained restatement ("Maastricht
    # UAC EDYY Maastricht ...") is fully unwound: the trailing bare "Maastricht"
    # no longer matches the full ranking name.
    bare = re.sub(r"\s+(?:ACC|UAC)$", "", name, flags=re.IGNORECASE)
    stem = re.escape(bare).replace(r"\ ", r"[\s\-]")
    # The ICAO alternative stays case-SENSITIVE on purpose: matched case-insensitively,
    # `[A-Z]{4}` swallows the opening verb of every bullet ("saw ...", "faced ...").
    pattern = re.compile(
        rf"^\s*(?:(?i:\[?\s*{stem}(?:\s+(?:airport|ACC|UAC))?\s*\]?\s*(?:\||:|-)?)"
        rf"|[A-Z]{{4}}(?:/[A-Z]{{4}})*)\s*"
    )
    for _ in range(4):
        stripped = pattern.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return text.strip()


def build(week: Week | None = None, *, session: requests.Session | None = None,
          client=None) -> Draft:
    """Draft the report for `week` (default: the week that just ended)."""
    week = week or facts.last_complete_week()
    own = session is None
    session = session or requests.Session()
    client = client or build_client()
    try:
        wf = facts.collect(week, session=session)

        entities = [
            (r["name"], "acc", r["rankNumber"], r.get("avgValue"), r.get("value"))
            for r in wf.acc_ranking[:CANDIDATES]
        ] + [
            (r["name"], "airport", r["rankNumber"], r.get("avgValue"), r.get("value"))
            for r in wf.airport_ranking[:CANDIDATES]
        ]
        evidence = causes.collect(week, entities, session=session)

        headline = _write_headline(wf, session, client)

        enroute, airport = [], []
        for ec in evidence:
            sink = enroute if ec.kind == "acc" else airport
            if len(sink) >= BULLETS:
                continue
            # An entity with no NOP evidence at all cannot be written about
            # honestly; skip it rather than let the model improvise.
            if not ec.grounded:
                continue
            sink.append(_write_bullet(ec, week, client))

        draft = Draft(week=week, facts=wf, headline=headline,
                      enroute=enroute, airport=airport)

        # The gate: every numeral in the prose must trace back to a fact. The
        # bullets are meant to carry no figures at all, so they are checked too.
        prose = "\n".join([draft.headline] + [b.text for b in draft.bullets])
        draft.findings = verify.check(prose, wf)
        return draft
    finally:
        if own:
            session.close()
