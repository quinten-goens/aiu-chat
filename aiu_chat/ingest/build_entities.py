"""Build the entity / knowledge layer: canonical entities + aliases.

Run: python -m aiu_chat.ingest.build_entities

Derives the domain's entities (states + airports for v1) from the local parquet
datasets, enriches them from the OurAirports snapshot (IATA codes, municipality,
keyword aliases, ISO country map), and writes two tables into the DuckDB file
plus a `data/entities.json` mirror for prompting and unit tests.

Design: docs/design/entity-layer.md. Key ideas:
  * Airports are keyed by ICAO (`apt:<icao>`), states by ISO country code
    (`state:<iso>`) so the mismatched STATE_NAME variants across tables all
    resolve to one canonical id.
  * `entity_aliases` records every surface form (normalised) -> entity_id.
  * `filter_values` records, per (entity, table), the exact raw value to filter
    on (e.g. state:GB -> co2 table 'UNITED KINGDOM', airport table 'United
    Kingdom'), so generated SQL filters on the right literal.
  * Enrichment never invents queryable data; it only populates entity rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from aiu_chat import config
from aiu_chat.agent.catalog import get_catalog
from aiu_chat.ingest.download_ourairports import ourairports_dir

# Which dataset column holds which entity. Bound here (not in catalog.json) so
# the builder is self-contained; kinds beyond state/airport are added in a later
# slice.
ENTITY_BINDINGS = {
    "co2_emissions_by_state":   {"kind": "state",   "name_col": "STATE_NAME", "code_col": "STATE_CODE"},
    "airport_traffic":          {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "atc_pre_departure_delays": {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "all_pre_departure_delays": {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "atfm_slot_adherence":      {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "airport_arrival_atfm_delay": {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "vertical_flight_efficiency": {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "taxi_in_additional_time":  {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "taxi_out_additional_time": {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
    "asma_additional_time":     {"kind": "airport", "icao_col": "APT_ICAO", "name_col": "APT_NAME", "state_col": "STATE_NAME"},
}

# STATE_NAME variants OurAirports countries.csv doesn't match by name -> ISO code.
# Small, curated, only for the handful that don't reconcile automatically.
STATE_NAME_ISO_OVERRIDES = {
    "CZECHIA": "CZ",
    "TURKIYE": "TR", "TÜRKIYE": "TR",
    "MOLDOVA, REPUBLIC OF": "MD",
    "REPUBLIC OF NORTH MACEDONIA": "MK",
    # Canary Islands is a Spanish region EUROCONTROL reports separately; keep it a
    # distinct entity keyed by a name slug (handled by the fallback), not ES.
}


# Single-token aliases equal to one of these common words are dropped (they turn
# up as noise in OurAirports keywords and would false-match ordinary questions).
_ALIAS_STOPWORDS = {
    "per", "the", "and", "for", "new", "old", "north", "south", "east", "west",
    "city", "town", "port", "field", "international", "airport", "air", "base",
    "central", "national", "regional", "island", "bay", "point", "park", "lake",
    "hill", "mount", "river", "valley", "sion",  # 'sion' -> Sion is rarely asked
}


def _norm(text: str) -> str:
    """Normalise a surface form for alias matching: lowercase, collapse spacing
    and punctuation to single spaces, strip. Keeps letters/digits and spaces."""
    t = (text or "").lower().strip()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)  # punctuation -> space
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


@dataclass
class Entity:
    entity_id: str
    kind: str
    canonical_name: str
    icao: str = ""
    iata: str = ""
    state_id: str = ""
    lat: float | None = None
    lon: float | None = None
    aliases: set[str] = field(default_factory=set)
    # (table -> raw value to filter on in that table)
    filter_values: dict[str, str] = field(default_factory=dict)

    def add_alias(self, text: str) -> None:
        n = _norm(text)
        # Drop single-token aliases that are common English words (they come from
        # noisy OurAirports keywords, e.g. 'per' for LHPR) — they cause false
        # matches inside ordinary questions. Multi-word aliases are always kept.
        if n and not (" " not in n and n in _ALIAS_STOPWORDS):
            self.aliases.add(n)


# --- OurAirports snapshot loading ------------------------------------------
def _load_countries(oa_dir: Path) -> dict[str, str]:
    """Map UPPER(country name) -> ISO alpha-2 code, plus a code->name index."""
    name_to_iso: dict[str, str] = {}
    path = oa_dir / "countries.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name_to_iso[r["name"].strip().upper()] = r["code"].strip()
    return name_to_iso


def _iso_name(oa_dir: Path) -> dict[str, str]:
    """ISO code -> canonical country name (from OurAirports)."""
    out: dict[str, str] = {}
    with (oa_dir / "countries.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["code"].strip()] = r["name"].strip()
    return out


def _load_airports(oa_dir: Path) -> dict[str, dict]:
    """Map ICAO -> OurAirports airport row (only rows with an ICAO code)."""
    out: dict[str, dict] = {}
    with (oa_dir / "airports.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            icao = (r.get("icao_code") or r.get("ident") or "").strip().upper()
            if icao:
                out[icao] = r
    return out


def _state_iso_for(name: str, name_to_iso: dict[str, str]) -> str | None:
    up = (name or "").strip().upper()
    if up in STATE_NAME_ISO_OVERRIDES:
        return STATE_NAME_ISO_OVERRIDES[up]
    return name_to_iso.get(up)


# --- build ------------------------------------------------------------------
def build_entities(*, duckdb_path=None, catalog=None, oa_dir=None,
                   out_dir=None, write_json: bool = True,
                   write_db: bool = True) -> dict:
    """Build entities + aliases; write DuckDB tables + entities.json. Returns the
    JSON-serialisable structure (also useful for tests).

    `out_dir` overrides where entities.json is read/written (defaults to
    config.DATA_DIR); tests pass a temp dir so they never touch the real file.
    """
    catalog = catalog or get_catalog()
    oa_dir = oa_dir or ourairports_dir()
    duckdb_path = duckdb_path or config.DUCKDB_PATH
    out_dir = Path(out_dir) if out_dir else config.DATA_DIR

    if not (oa_dir / "countries.csv").exists():
        raise FileNotFoundError(
            f"OurAirports snapshot not found in {oa_dir}. Run "
            f"`python -m aiu_chat.ingest.download_ourairports` first."
        )

    name_to_iso = _load_countries(oa_dir)
    iso_name = _iso_name(oa_dir)
    oa_airports = _load_airports(oa_dir)
    paths = {d.table: d.parquet_path for d in catalog.datasets}

    con = duckdb.connect()  # in-memory reader over parquet
    try:
        states: dict[str, Entity] = {}      # entity_id -> Entity
        airports: dict[str, Entity] = {}

        # States first so airports can link + backfill each state's per-table
        # filter value.
        ordered = sorted(
            ENTITY_BINDINGS.items(),
            key=lambda kv: 0 if kv[1]["kind"] == "state" else 1,
        )
        for table, binding in ordered:
            ppath = paths.get(table)
            if not ppath or not Path(ppath).exists():
                continue
            if binding["kind"] == "state":
                _ingest_states(con, ppath, table, binding, name_to_iso, iso_name, states)
            elif binding["kind"] == "airport":
                _ingest_airports(con, ppath, table, binding, name_to_iso, iso_name,
                                 oa_airports, states, airports)
    finally:
        con.close()

    all_entities = list(states.values()) + list(airports.values())
    _merge_alias_seed({e.entity_id: e for e in all_entities})
    result = _to_json(all_entities)

    if write_json:
        _drift_report(result, out_dir)  # loudly log entity additions/removals
        _write_json(result, out_dir)    # JSON mirror (prompting + tests)
    if write_db:
        _write_db(all_entities, duckdb_path)

    print(f"Built {len(states)} states + {len(airports)} airports "
          f"({sum(len(e.aliases) for e in all_entities)} aliases).")
    return result


def _ingest_states(con, ppath, table, binding, name_to_iso, iso_name, states):
    name_col = binding["name_col"]
    code_col = binding.get("code_col")
    cols = f"{name_col}" + (f", {code_col}" if code_col else "")
    rows = con.execute(
        f"SELECT DISTINCT {cols} FROM read_parquet(?) WHERE {name_col} IS NOT NULL",
        [ppath],
    ).fetchall()
    for row in rows:
        raw_name = row[0]
        icao_prefix = row[1] if code_col and len(row) > 1 else None
        iso = _state_iso_for(raw_name, name_to_iso)
        if iso:
            eid = f"state:{iso}"
            canon = iso_name.get(iso, raw_name.title())
        else:
            # Non-ISO EUROCONTROL entity (e.g. Canary Islands) -> name-slug key.
            eid = f"state:{_slug(raw_name)}"
            canon = raw_name.title()
        ent = states.get(eid)
        if ent is None:
            ent = Entity(entity_id=eid, kind="state", canonical_name=canon,
                         icao=(icao_prefix or ""))
            if iso:
                ent.iata = ""  # states have no IATA
            ent.add_alias(canon)
            states[eid] = ent
        ent.add_alias(raw_name)
        if iso:
            ent.add_alias(iso)                 # "GB"
        if icao_prefix:
            ent.add_alias(icao_prefix)         # ICAO 2-letter, e.g. "EB"
        # Exact raw value to filter on in *this* table.
        ent.filter_values[table] = raw_name


def _ingest_airports(con, ppath, table, binding, name_to_iso, iso_name,
                     oa_airports, states, airports):
    icao_col = binding["icao_col"]
    name_col = binding.get("name_col")
    state_col = binding.get("state_col")
    sel = ", ".join(c for c in (icao_col, name_col, state_col) if c)
    rows = con.execute(
        f"SELECT DISTINCT {sel} FROM read_parquet(?) WHERE {icao_col} IS NOT NULL",
        [ppath],
    ).fetchall()
    for row in rows:
        icao = (row[0] or "").strip().upper()
        if not icao:
            continue
        raw_name = row[1] if name_col and len(row) > 1 else ""
        raw_state = row[2] if state_col and len(row) > 2 else ""
        eid = f"apt:{icao}"
        ent = airports.get(eid)
        if ent is None:
            oa = oa_airports.get(icao, {})
            canon = (oa.get("name") or raw_name or icao).strip()
            iso = (oa.get("iso_country") or "").strip()
            state_id = ""
            if iso:
                state_id = f"state:{iso}"
            elif raw_state:
                # Fall back to reconciling the dataset's own state name.
                riso = _state_iso_for(raw_state, name_to_iso)
                state_id = f"state:{riso}" if riso else f"state:{_slug(raw_state)}"
            ent = Entity(
                entity_id=eid, kind="airport", canonical_name=canon, icao=icao,
                iata=(oa.get("iata_code") or "").strip(),
                state_id=state_id,
                lat=_to_float(oa.get("latitude_deg")),
                lon=_to_float(oa.get("longitude_deg")),
            )
            ent.add_alias(icao)
            ent.add_alias(canon)
            if ent.iata:
                ent.add_alias(ent.iata)                      # "LHR"
            if oa.get("municipality"):
                ent.add_alias(oa["municipality"])            # "London"
            for kw in (oa.get("keywords") or "").split(","):
                ent.add_alias(kw)                            # "Londres", "LON"
            airports[eid] = ent
        # Every dataset name variant becomes an alias + the per-table filter value.
        if raw_name:
            ent.add_alias(raw_name)
        ent.filter_values[table] = icao  # airports filter on ICAO everywhere

        # Also record the STATE's raw filter value in this (airport) table, so a
        # cross-source state join knows to filter e.g. STATE_NAME='United Kingdom'
        # here vs 'UNITED KINGDOM' in the state table.
        if raw_state and ent.state_id and ent.state_id in states:
            states[ent.state_id].filter_values.setdefault(table, raw_state)


def _to_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# --- serialisation ----------------------------------------------------------
def _to_json(entities: list[Entity]) -> dict:
    ents = []
    aliases = []
    for e in entities:
        ents.append({
            "entity_id": e.entity_id, "kind": e.kind,
            "canonical_name": e.canonical_name, "icao": e.icao, "iata": e.iata,
            "state_id": e.state_id, "lat": e.lat, "lon": e.lon,
            "filter_values": e.filter_values,
        })
        for a in sorted(e.aliases):
            aliases.append({"alias": a, "entity_id": e.entity_id})
    return {"entities": ents, "aliases": aliases}


def _merge_alias_seed(by_id: dict[str, Entity]) -> None:
    """Merge curated colloquial aliases (data/entity_aliases_seed.json) that the
    OurAirports data doesn't carry. Only ADDS aliases to existing entities; an
    unknown entity_id in the seed is skipped (logged), never fatal."""
    seed_path = config.DATA_DIR / "entity_aliases_seed.json"
    if not seed_path.exists():
        return
    try:
        seed = json.loads(seed_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ! alias seed unreadable ({exc}); skipping")
        return
    added = skipped = 0
    for eid, aliases in seed.items():
        if eid.startswith("_"):
            continue
        ent = by_id.get(eid)
        if ent is None:
            skipped += 1
            continue
        for a in aliases:
            ent.add_alias(a)
            added += 1
    print(f"  merged alias seed: +{added} aliases"
          + (f" ({skipped} unknown entity ids skipped)" if skipped else ""))


def _drift_report(result: dict, out_dir: Path) -> None:
    """Compare against the previous entities.json and log additions/removals
    loudly (a rename/format change upstream should never be silent)."""
    out = out_dir / "entities.json"
    if not out.exists():
        print("  (no previous entities.json — first build)")
        return
    try:
        prev = json.loads(out.read_text())
    except (json.JSONDecodeError, OSError):
        print("  ! previous entities.json unreadable; skipping drift check")
        return
    old = {e["entity_id"] for e in prev.get("entities", [])}
    new = {e["entity_id"] for e in result.get("entities", [])}
    added, removed = new - old, old - new
    if added:
        print(f"  drift: +{len(added)} new entities (e.g. {sorted(added)[:5]})")
    if removed:
        print(f"  ! drift: -{len(removed)} entities REMOVED "
              f"(e.g. {sorted(removed)[:5]}) — verify the data wasn't broken")
    if not added and not removed:
        print("  drift: no entity additions/removals")


def _write_json(result: dict, out_dir: Path) -> None:
    out = out_dir / "entities.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  -> wrote {out}")


def _write_db(entities: list[Entity], duckdb_path) -> None:
    """Write entities + entity_aliases tables into the DuckDB file (idempotent)."""
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute("DROP TABLE IF EXISTS entity_aliases")
        con.execute("DROP TABLE IF EXISTS entities")
        con.execute("""
            CREATE TABLE entities (
                entity_id TEXT PRIMARY KEY, kind TEXT, canonical_name TEXT,
                icao TEXT, iata TEXT, state_id TEXT, lat DOUBLE, lon DOUBLE,
                filter_values JSON
            )
        """)
        con.execute("CREATE TABLE entity_aliases (alias TEXT, entity_id TEXT)")
        for e in entities:
            con.execute(
                "INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?)",
                [e.entity_id, e.kind, e.canonical_name, e.icao, e.iata,
                 e.state_id, e.lat, e.lon, json.dumps(e.filter_values)],
            )
            for a in sorted(e.aliases):
                con.execute("INSERT INTO entity_aliases VALUES (?,?)", [a, e.entity_id])
        con.execute("CREATE INDEX idx_alias ON entity_aliases (alias)")
        print(f"  -> wrote entities/entity_aliases into {duckdb_path}")
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the entity/knowledge layer.")
    parser.add_argument("--no-db", action="store_true", help="Write JSON only, skip DuckDB tables.")
    args = parser.parse_args(argv)
    build_entities(write_db=not args.no_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
