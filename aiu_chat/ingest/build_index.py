"""Embed scraped doc chunks and store them in a DuckDB VSS table.

Run: python -m aiu_chat.ingest.build_index

Creates/replaces a `doc_chunks` table in the DuckDB file with an HNSW index over
the embedding column, so the concept path can do similarity search in the same
engine as the structured data.
"""
from __future__ import annotations

import duckdb

from aiu_chat import config
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.ingest.scrape_docs import Chunk, scrape_all

TABLE = "doc_chunks"


def _connect() -> duckdb.DuckDBPyConnection:
    config.ensure_dirs()
    con = duckdb.connect(str(config.DUCKDB_PATH))
    con.execute("INSTALL vss")
    con.execute("LOAD vss")
    # Allow the persisted HNSW index to be written to the database file.
    con.execute("SET hnsw_enable_experimental_persistence = true")
    return con


def build_index(client: OllamaClient | None = None) -> int:
    """Scrape, embed, and store doc chunks. Returns the number stored."""
    client = client or OllamaClient()
    dim = config.EMBEDDING_DIM

    print("Scraping reference pages...")
    chunks: list[Chunk] = scrape_all()
    if not chunks:
        raise RuntimeError("No document chunks scraped; nothing to index.")

    print(f"Embedding {len(chunks)} chunks with '{client.embedding_model}'...")
    rows = []
    for i, ch in enumerate(chunks, 1):
        vec = client.embed(ch.text)
        if len(vec) != dim:
            raise RuntimeError(
                f"Embedding dim {len(vec)} != configured {dim}. "
                f"Set AIU_EMBEDDING_DIM to match '{client.embedding_model}'."
            )
        rows.append((i, ch.text, ch.source_url, ch.source_title, ch.ordinal, vec))
        if i % 20 == 0:
            print(f"  ...{i}/{len(chunks)}")

    con = _connect()
    try:
        con.execute(f"DROP TABLE IF EXISTS {TABLE}")
        con.execute(
            f"""
            CREATE TABLE {TABLE} (
                id INTEGER,
                text VARCHAR,
                source_url VARCHAR,
                source_title VARCHAR,
                ordinal INTEGER,
                embedding FLOAT[{dim}]
            )
            """
        )
        con.executemany(
            f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?, ?, ?)", rows
        )
        con.execute(
            f"CREATE INDEX doc_chunks_hnsw ON {TABLE} USING HNSW(embedding) "
            f"WITH (metric = 'cosine')"
        )
        (count,) = con.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()
    finally:
        con.close()

    print(f"Indexed {count} chunks -> {config.DUCKDB_PATH} ({TABLE})")
    return count


def main(argv: list[str] | None = None) -> int:
    build_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
