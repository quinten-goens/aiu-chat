"""Deterministic entity resolver over the entity/knowledge layer.

Loads `data/entities.json` (built by ingest.build_entities) and offers:

  * `resolve(text, kind=None)`  -> best Entity for a surface form, or None.
  * `candidates(text, kind=None)` -> all matching entities (ambiguity handling).
  * `find_in_question(q)`        -> entities mentioned anywhere in a question.
  * `filter_value(entity, table)`-> the exact (column-less) literal to filter on
                                    in that table, reconciling name mismatches.

The resolver is **advisory**: it informs the SQL prompt but never hard-gates, so
an unresolved entity simply falls back to today's model-guess behaviour and can
never regress a currently-working question into a failure.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field

from aiu_chat import config

# Aliases this short are ignored entirely when scanning free-text questions:
# 2-letter ISO/ICAO codes (is, eg, lo, ...) collide with common English words.
# They remain usable via explicit resolve()/candidates() lookups.
_MIN_SCAN_ALIAS_LEN = 3


@dataclass(frozen=True)
class Entity:
    entity_id: str
    kind: str
    canonical_name: str
    icao: str = ""
    iata: str = ""
    state_id: str = ""
    lat: float | None = None
    lon: float | None = None
    filter_values: dict = field(default_factory=dict)

    def filter_value(self, table: str) -> str | None:
        """The raw literal to filter on for this entity in `table` (or None)."""
        return self.filter_values.get(table)


@dataclass(frozen=True)
class _Index:
    by_id: dict
    alias_to_ids: dict  # normalised alias -> tuple(entity_id, ...)


def _norm(text: str) -> str:
    """Same normalisation the builder uses (lowercase, punctuation->space)."""
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _load(path=None) -> _Index:
    path = path or (config.DATA_DIR / "entities.json")
    if not path.exists():
        return _Index(by_id={}, alias_to_ids={})
    raw = json.loads(path.read_text())
    by_id = {
        e["entity_id"]: Entity(
            entity_id=e["entity_id"], kind=e["kind"],
            canonical_name=e["canonical_name"], icao=e.get("icao", ""),
            iata=e.get("iata", ""), state_id=e.get("state_id", ""),
            lat=e.get("lat"), lon=e.get("lon"),
            filter_values=e.get("filter_values", {}),
        )
        for e in raw.get("entities", [])
    }
    alias_to_ids: dict[str, list[str]] = {}
    for a in raw.get("aliases", []):
        alias_to_ids.setdefault(a["alias"], []).append(a["entity_id"])
    # Freeze the id lists to tuples for hashability/immutability.
    return _Index(by_id=by_id, alias_to_ids={k: tuple(v) for k, v in alias_to_ids.items()})


@functools.lru_cache(maxsize=1)
def _index() -> _Index:
    return _load()


def reload() -> None:
    """Clear the cached index (call after a rebuild)."""
    _index.cache_clear()


def get(entity_id: str) -> Entity | None:
    return _index().by_id.get(entity_id)


def candidates(text: str, *, kind: str | None = None) -> list[Entity]:
    """All entities whose alias exactly equals the normalised `text`."""
    idx = _index()
    ids = idx.alias_to_ids.get(_norm(text), ())
    ents = [idx.by_id[i] for i in ids if i in idx.by_id]
    if kind:
        ents = [e for e in ents if e.kind == kind]
    return ents


def resolve(text: str, *, kind: str | None = None) -> Entity | None:
    """Best single entity for `text`, or None. If ambiguous, prefers an airport
    over a state (questions naming an airport are more specific), then the entity
    with the most aliases (a proxy for prominence). Use `candidates()` when you
    need to detect ambiguity."""
    cands = candidates(text, kind=kind)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # Deterministic tie-break: airports first, then by alias count desc, then id.
    idx = _index()
    alias_count = {e.entity_id: 0 for e in cands}
    for ids in idx.alias_to_ids.values():
        for i in ids:
            if i in alias_count:
                alias_count[i] += 1
    return sorted(
        cands,
        key=lambda e: (0 if e.kind == "airport" else 1, -alias_count[e.entity_id], e.entity_id),
    )[0]


def find_in_question(question: str, *, max_hits: int = 8) -> list[Entity]:
    """Entities mentioned in a question, by matching known aliases on WORD
    BOUNDARIES (never as substrings — so 'sion' won't fire inside 'emissions',
    nor 'per' inside 'per year').

    Longer aliases are tried first so a specific phrase ('london heathrow') wins
    over a generic token ('london'), and once an alias's span is claimed it isn't
    re-matched by a shorter overlapping alias. 2-letter codes are skipped
    entirely (see _MIN_SCAN_ALIAS_LEN). Returns de-duplicated entities in
    first-appearance order.
    """
    idx = _index()
    nq = _norm(question)
    if not nq:
        return []

    hits: list[tuple[int, str]] = []  # (position, entity_id)
    seen_ids: set[str] = set()
    claimed: list[tuple[int, int]] = []  # spans already matched (avoid overlaps)

    def _overlaps(a: int, b: int) -> bool:
        return any(a < end and b > start for start, end in claimed)

    for alias in sorted(idx.alias_to_ids, key=len, reverse=True):
        if len(alias) < _MIN_SCAN_ALIAS_LEN:
            continue
        m = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", nq)
        if not m or _overlaps(m.start(), m.end()):
            continue
        claimed.append((m.start(), m.end()))
        for eid in idx.alias_to_ids[alias]:
            if eid not in seen_ids:
                seen_ids.add(eid)
                hits.append((m.start(), eid))

    hits.sort(key=lambda h: h[0])
    out: list[Entity] = []
    for _, eid in hits:
        ent = idx.by_id.get(eid)
        if ent is not None:
            out.append(ent)
        if len(out) >= max_hits:
            break
    return out


def filter_value(entity_id: str, table: str) -> str | None:
    """The raw literal to filter on for `entity_id` in `table`, or None. The
    caller pairs it with the table's known entity column (e.g. STATE_NAME)."""
    ent = get(entity_id)
    return ent.filter_value(table) if ent is not None else None


# Which column each table filters an entity on (mirrors the builder's bindings).
# Airports filter on ICAO everywhere; states on their name column per table.
_STATE_FILTER_COL = {
    "co2_emissions_by_state": "STATE_NAME",
    "airport_traffic": "STATE_NAME",
    "atc_pre_departure_delays": "STATE_NAME",
    "all_pre_departure_delays": "STATE_NAME",
    "atfm_slot_adherence": "STATE_NAME",
    "airport_arrival_atfm_delay": "STATE_NAME",
    "vertical_flight_efficiency": "STATE_NAME",
    "taxi_in_additional_time": "STATE_NAME",
    "taxi_out_additional_time": "STATE_NAME",
    "asma_additional_time": "STATE_NAME",
}
_AIRPORT_FILTER_COL = "APT_ICAO"


def sql_prompt_hint(question: str, *, max_entities: int = 6) -> str:
    """Render a compact 'resolved entities' block for the SQL-generation prompt.

    For each entity found in the question, states the canonical name and the
    exact column+literal to filter on in each relevant table (this is what fixes
    the STATE_NAME case/spelling mismatch across tables). Returns "" when the
    entity layer finds nothing, so the prompt is unchanged in that case.
    """
    ents = find_in_question(question, max_hits=max_entities)
    if not ents:
        return ""
    lines = ["Resolved entities (use these exact filter values):"]
    for e in ents:
        if e.kind == "airport":
            lines.append(
                f"- {e.canonical_name} → airport {e.icao}"
                + (f" (IATA {e.iata})" if e.iata else "")
                + f": filter {_AIRPORT_FILTER_COL} = '{e.icao}' on airport tables."
            )
        elif e.kind == "state":
            # Show the per-table literal only where it differs, to keep it short.
            vals = {t: v for t, v in e.filter_values.items()}
            if vals:
                distinct = sorted(set(vals.values()))
                if len(distinct) == 1:
                    lines.append(
                        f"- {e.canonical_name} → state: filter STATE_NAME = "
                        f"'{distinct[0]}'."
                    )
                else:
                    parts = ", ".join(
                        f"{t}: '{v}'" for t, v in sorted(vals.items())
                    )
                    lines.append(
                        f"- {e.canonical_name} → state: STATE_NAME differs by "
                        f"table — {parts}."
                    )
            else:
                lines.append(f"- {e.canonical_name} → state.")
    return "\n".join(lines)
