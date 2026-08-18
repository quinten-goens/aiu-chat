"""ACC / ATSU names → the FIR or ANSP entities that actually carry their data.

The AIU delay datasets are keyed by FIR (`enroute_delay_fir`: 62 entities such
as "Greece") and by ANSP (`enroute_delay_ansp`: 50 entities such as "HCAA").
Users naturally ask about *area control centres* — "Athens ACC", "Karlsruhe UAC"
— which are not entities in either table. Before this map, such questions were
refused outright even though the data was present: three logged conversations
asked for Athens ACC delay causes and got "I can't answer that", while Greece
FIR held the answer (Jan-Jun 2026: 409,297 min, staffing-dominated).

We deliberately do NOT auto-substitute: an ACC and its FIR are not the same
thing, so the caller asks the user to confirm.

Every candidate name below was cross-checked against the two parquet files;
`tests/test_aliases.py::test_every_candidate_is_a_real_entity` keeps it honest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ACC/UAC/airport-city token -> real entity names, most likely first.
# Keys are lowercase and suffix-free; see `_normalise`.
# FIR name first (always queryable), then the ANSP where one exists.
_NEAR_MISS: dict[str, list[str]] = {
    # Greece — the case that prompted this module.
    "athens": ["Greece", "HCAA"],
    "thessaloniki": ["Greece", "HCAA"],
    "macedonia": ["Greece", "HCAA"],
    # Germany
    "karlsruhe": ["Germany", "DFS"],
    "langen": ["Germany", "DFS"],
    "munich": ["Germany", "DFS"],
    "bremen": ["Germany", "DFS"],
    # France
    "reims": ["France", "DSNA"],
    "brest": ["France", "DSNA"],
    "bordeaux": ["France", "DSNA"],
    "marseille": ["France", "DSNA"],
    "aix": ["France", "DSNA"],
    "paris": ["France", "DSNA"],
    # United Kingdom
    "london": ["United Kingdom", "NATS (Continental)"],
    "swanwick": ["United Kingdom", "NATS (Continental)"],
    "prestwick": ["United Kingdom", "NATS (Continental)"],
    "scottish": ["United Kingdom", "NATS (Continental)"],
    # Ireland
    "shannon": ["Ireland", "AirNav Ireland"],
    "dublin": ["Ireland", "AirNav Ireland"],
    # Iberia
    "madrid": ["Spain", "ENAIRE"],
    "barcelona": ["Spain", "ENAIRE"],
    "seville": ["Spain", "ENAIRE"],
    "canarias": ["Spain Canarias", "ENAIRE"],
    "palma": ["Spain", "ENAIRE"],
    "lisbon": ["Portugal", "NAV Portugal"],
    "santa maria": ["Portugal Santa Maria", "NAV Portugal"],
    # Italy
    "rome": ["Italy", "ENAV"],
    "milan": ["Italy", "ENAV"],
    "padua": ["Italy", "ENAV"],
    "brindisi": ["Italy", "ENAV"],
    # Alpine / Benelux
    "vienna": ["Austria", "Austro Control"],
    "zurich": ["Switzerland", "Skyguide"],
    "geneva": ["Switzerland", "Skyguide"],
    "amsterdam": ["Netherlands", "LVNL"],
    "brussels": ["Belgium", "skeyes"],
    "luxembourg": ["ANA LUX"],
    # Central & Eastern Europe
    "warsaw": ["Poland", "PANSA"],
    "budapest": ["Hungary", "HungaroControl (EC)"],
    "prague": ["Czech Republic", "ANS CR"],
    "bucharest": ["Romania", "ROMATSA"],
    "sofia": ["Bulgaria", "BULATSA"],
    "zagreb": ["Croatia", "Croatia Control"],
    "ljubljana": ["Slovenia", "Slovenia Control"],
    "bratislava": ["Slovakia", "LPS"],
    "belgrade": ["Serbia and Montenegro", "SMATSA"],
    "skopje": ["North Macedonia", "M-NAV"],
    "tirana": ["Albania", "Albcontrol"],
    "sarajevo": ["Bosnia and Herzegovina"],
    "chisinau": ["Moldova", "MOLDATSA"],
    "kyiv": ["Ukraine", "UkSATSE"],
    # Nordics & Baltics
    "copenhagen": ["Denmark", "NAVIAIR"],
    "stockholm": ["Sweden", "LFV"],
    "malmo": ["Sweden", "LFV"],
    "oslo": ["Norway", "Avinor"],
    "bodo": ["Norway", "Avinor"],
    "helsinki": ["Finland", "ANS Finland"],
    "tampere": ["Finland", "ANS Finland"],
    "reykjavik": ["Iceland", "Isavia"],
    "riga": ["Latvia", "LGS"],
    "tallinn": ["Estonia", "EANS"],
    "vilnius": ["Lithuania", "Oro Navigacija"],
    # Mediterranean & Caucasus
    "ankara": ["Turkey", "DHMI"],
    "istanbul": ["Turkey", "DHMI"],
    "nicosia": ["Cyprus", "DCAC Cyprus"],
    "valletta": ["Malta", "MATS"],
    "yerevan": ["Armenia", "ARMATS"],
    "tbilisi": ["Georgia", "Sakaeronavigatsia"],
}

# Names that ARE real entities and must never be treated as a near-miss, even
# though some look like control-centre names ("Maastricht UAC" is a real ANSP).
_REAL_ENTITIES: set[str] = {
    "maastricht uac", "muac", "greece", "germany", "france", "united kingdom",
    "spain", "italy", "ireland", "portugal", "austria", "switzerland",
    "netherlands", "belgium", "poland", "hungary", "czech republic", "romania",
    "bulgaria", "croatia", "slovenia", "slovakia", "denmark", "sweden",
    "norway", "finland", "iceland", "latvia", "estonia", "lithuania", "turkey",
    "turkiye", "cyprus", "malta", "serbia", "north macedonia", "albania",
    "bosnia and herzegovina", "moldova", "ukraine", "armenia", "georgia",
}

# Suffixes that mark a control centre rather than an entity.
_SUFFIXES = (
    " area control centre", " area control center", " upper area control centre",
    " acc", " uac", " fir", " ctr", " tma", " airport", " ansp",
)


@dataclass
class NearMiss:
    """A name that is not an entity but maps to one or more that are."""

    query: str
    candidates: list[str] = field(default_factory=list)
    reason: str = ""

    def question(self) -> str:
        """The clarifying question to put to the user."""
        if len(self.candidates) == 1:
            opts = f"**{self.candidates[0]}**"
        else:
            opts = " or ".join(f"**{c}**" for c in self.candidates)
        return (
            f"“{self.query}” isn’t a separate entity in the EUROCONTROL "
            f"performance datasets — {self.reason} Did you mean {opts}?"
        )


def _normalise(name: str) -> str:
    out = " ".join((name or "").strip().lower().split())
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if out.endswith(suf):
                out = out[: -len(suf)].strip()
                changed = True
    return out


def resolve_near_miss(name: str) -> NearMiss | None:
    """Return a NearMiss if `name` looks like an ACC/airport that maps onto a
    real FIR/ANSP entity, else None (unknown text and real entities both)."""
    raw = " ".join((name or "").strip().lower().split())
    if not raw or raw in _REAL_ENTITIES:
        return None
    key = _normalise(name)
    if not key or key in _REAL_ENTITIES:
        return None
    cands = _NEAR_MISS.get(key)
    if not cands:
        return None
    return NearMiss(
        query=name.strip(),
        candidates=list(cands),
        reason=(
            "en-route delay data is published per FIR and per ANSP, "
            "not per area control centre."
        ),
    )
