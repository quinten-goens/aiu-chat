"""Parse the acronyms glossary into a structured lookup table.

Acronym lookup ("what does ATFM stand for?") is a structured-data problem, not a
semantic-retrieval one. Vector search over a large mixed corpus dilutes short
glossary entries, so we keep a dedicated CODE -> definition table for exact and
prefix matching, used alongside vector retrieval in the concept path.
"""
from __future__ import annotations

import re

import duckdb

from aiu_chat import config
from aiu_chat.ingest.docs import DOC_SOURCES
from aiu_chat.ingest.scrape_docs import fetch_text

ACRONYM_TABLE = "acronyms"

# Lines look like "ATFM - Air Traffic Flow Management" (dash or en/em dash).
_ENTRY_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9/+.\-]{0,15})\s*[-–—]\s*(.+?)\s*$")


def parse_acronyms(text: str) -> list[tuple[str, str]]:
    """Extract (code, definition) pairs from the glossary page text."""
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _ENTRY_RE.match(line)
        if not m:
            continue
        code, definition = m.group(1).strip(), m.group(2).strip()
        # Filter obvious non-entries (e.g. a stray word, or definition == code).
        if len(code) >= 2 and definition and definition.upper() != code.upper():
            pairs.append((code, definition))
    return pairs


def build_acronym_table(con: duckdb.DuckDBPyConnection) -> int:
    """Fetch the acronyms page and (re)build the structured table. Returns count."""
    source = next((s for s in DOC_SOURCES if s.glossary), None)
    if source is None:
        return 0
    text = fetch_text(source)
    if not text:
        print("  ! acronyms page unavailable; skipping acronym table")
        return 0

    pairs = parse_acronyms(text)
    con.execute(f"DROP TABLE IF EXISTS {ACRONYM_TABLE}")
    con.execute(
        f"CREATE TABLE {ACRONYM_TABLE} (code VARCHAR, definition VARCHAR, source_url VARCHAR)"
    )
    if pairs:
        con.executemany(
            f"INSERT INTO {ACRONYM_TABLE} VALUES (?, ?, ?)",
            [(c, d, source.url) for c, d in pairs],
        )
    print(f"  + acronym table: {len(pairs)} entries")
    return len(pairs)


# Match a query that is essentially asking for an acronym, e.g.
# "what does ATFM stand for", "what is ASMA", "ATFM meaning", or just "ATFM".
_ACRONYM_QUERY_RE = re.compile(
    r"\b([A-Z]{2,7}[0-9]*)\b",
)


def lookup_acronyms(query: str, con: duckdb.DuckDBPyConnection | None = None) -> list[dict]:
    """Return acronym definitions for uppercase tokens in the query (exact match)."""
    codes = _ACRONYM_QUERY_RE.findall(query)
    if not codes:
        return []

    own_con = con is None
    con = con or duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if ACRONYM_TABLE not in tables:
            return []
        placeholders = ", ".join("?" for _ in codes)
        rows = con.execute(
            f"SELECT code, definition, source_url FROM {ACRONYM_TABLE} "
            f"WHERE UPPER(code) IN ({placeholders})",
            [c.upper() for c in codes],
        ).fetchall()
    finally:
        if own_con:
            con.close()
    return [{"code": c, "definition": d, "source_url": u} for (c, d, u) in rows]
