"""Embed scraped doc chunks and store them in a DuckDB VSS table.

Run: python -m aiu_chat.ingest.build_index

Creates/replaces a `doc_chunks` table in the DuckDB file with an HNSW index over
the embedding column, so the concept path can do similarity search in the same
engine as the structured data.
"""
from __future__ import annotations

import duckdb

from aiu_chat import config
from aiu_chat.agent.llm import embed_text
from aiu_chat.ingest.acronyms import build_acronym_table
from aiu_chat.ingest.discover_pdfs import discover_pdf_links
from aiu_chat.ingest.pdf import scrape_pdfs
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


def build_index(include_pdfs: bool = True) -> int:
    """Scrape pages + PDFs, embed, and store doc chunks. Returns the number stored.

    Embeddings use the deployment's configured provider (Ollama nomic locally,
    OpenAI text-embedding-3-small in the cloud), so the index dimension matches.
    """
    dim = config.EMBEDDING_DIM

    print("Scraping reference pages...")
    chunks: list[Chunk] = scrape_all()

    if include_pdfs:
        print("Discovering and ingesting PDFs...")
        try:
            pdf_links = discover_pdf_links()
            chunks.extend(scrape_pdfs(pdf_links))
        except Exception as exc:
            # PDF ingestion is additive; don't let a repo/clone hiccup wipe the
            # page-based index.
            print(f"  ! PDF ingestion skipped due to error: {exc}")

    if not chunks:
        raise RuntimeError("No document chunks scraped; nothing to index.")

    print(f"Embedding {len(chunks)} chunks with '{config.EMBEDDING_MODEL}' "
          f"({config.EMBEDDING_PROVIDER}, dim={dim})...")
    rows = []
    for i, ch in enumerate(chunks, 1):
        vec = embed_text(ch.text)
        if len(vec) != dim:
            raise RuntimeError(
                f"Embedding dim {len(vec)} != configured {dim} for "
                f"'{config.EMBEDDING_MODEL}'."
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
        # Structured acronym table for exact lookups (complements vector search).
        build_acronym_table(con)
    finally:
        con.close()

    print(f"Indexed {count} chunks -> {config.DUCKDB_PATH} ({TABLE})")
    return count


def main(argv: list[str] | None = None) -> int:
    build_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
