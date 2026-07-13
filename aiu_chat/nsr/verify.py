"""The gate between a generated draft and something anyone would publish.

The drafter hands the model its numbers as literals and tells it not to invent
others. This module checks that it obeyed, because a prompt is a request and not
a guarantee: every numeral in the generated prose is re-extracted and matched
against the fact table, and anything that does not trace back to an executed
query is flagged.

This is the NSR's version of the project rule that numeric answers must come from
executed queries and never from the model. Nothing here trusts the model; the
whole point is to catch it being wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from aiu_chat.nsr.facts import WeekFacts

# Numerals as they appear in the prose: 33,890 / 2.6 / -4 / 62 / 1.9
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Numbers that are part of the report's own vocabulary rather than claims about
# the network: week labels (W22), years (2025), and the ordinary small integers
# that show up in prose like "the first" -- these are not figures to verify.
_WEEK_REF = re.compile(r"\bW\d{1,2}\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# A figure spelled out in words is still a figure, and it walks straight past a
# numeral scanner -- the model really does write "twenty-three diversions" when
# asked for a count. Catching the shape is enough: the gate flags it, the analyst
# rewrites it in digits, and it is then checked like any other number.
_SPELLED = re.compile(
    r"\b(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ]"
    r"(?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen"
    r"|nineteen|hundred|thousand)\b",
    re.IGNORECASE,
)


@dataclass
class Finding:
    """One numeral in the draft that does not trace back to a fact."""

    literal: str
    context: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.literal!r} in \"...{self.context}...\""


def _allowed(facts: WeekFacts) -> set[str]:
    """Every literal the prose may legitimately contain.

    A fact's `text` is the canonical rendering, but the model may reasonably drop
    a sign the sentence already carries ("-4pp worse" vs "4pp worse"), so accept
    the unsigned form of each too.
    """
    out: set[str] = set()
    for fact in facts.facts.values():
        for form in (fact.text, fact.text.lstrip("+-")):
            out.add(_canon(form))
    return out


def _canon(literal: str) -> str:
    """Compare on the figure alone.

    A fact renders as "1.9%" / "-4pp" / "33,890" but the prose scanner extracts
    bare numerals ("1.9", "-4", "33890"), so units and separators must come off
    both sides or every percentage reads as untraceable -- a gate that cries wolf
    on real facts is a gate nobody looks at.
    """
    return re.sub(r"[,%+]|pp\b", "", literal).strip()


def check(prose: str, facts: WeekFacts, *, also_allowed: str = "") -> list[Finding]:
    """Every numeral in `prose` that is not one of `facts`.

    Week labels and years are skipped -- they are references, not measurements.

    `also_allowed` is free text whose numerals are additionally permitted: the
    analyst's own notes. The published reports carry figures no feed has ("and 23
    diversions"), and an analyst who typed that number verified it -- so the gate
    must accept it, or it would flag the one thing a human explicitly vouched for.
    It still only accepts numbers that appear *somewhere* in the notes; the model
    cannot mint a new one and claim a human said it.
    """
    allowed = _allowed(facts)
    allowed |= {_canon(m.group(0)) for m in _NUMBER.finditer(also_allowed)}
    masked = _YEAR.sub(" ", _WEEK_REF.sub(" ", prose))

    findings: list[Finding] = []
    for m in _NUMBER.finditer(masked):
        literal = m.group(0)
        canon = _canon(literal)
        if canon in allowed or canon.lstrip("-") in allowed:
            continue
        start, end = max(0, m.start() - 30), min(len(masked), m.end() + 30)
        findings.append(Finding(literal=literal,
                                context=masked[start:end].replace("\n", " ").strip()))

    for m in _SPELLED.finditer(masked):
        start, end = max(0, m.start() - 30), min(len(masked), m.end() + 30)
        findings.append(Finding(
            literal=m.group(0),
            context=(masked[start:end].replace("\n", " ").strip()
                     + "  [figure written in words — rewrite it in digits so it "
                       "can be checked]"),
        ))
    return findings


def cited_ids(bullet_sources: dict[str, list[str]]) -> set[str]:
    """The NOP message ids a draft's bullets were built from."""
    return {mid for ids in bullet_sources.values() for mid in ids}
