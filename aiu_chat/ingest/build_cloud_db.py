"""Build the self-contained cloud DuckDB for a Streamlit Cloud deployment.

Run (with OPENAI_KEY set):
    LOCAL=false python -m aiu_chat.ingest.build_cloud_db

Produces ONE file (config.DUCKDB_PATH, e.g. data/aiu_cloud.duckdb) containing:
  * every dataset as a table (copied from the local Parquet files), so the app
    needs no Parquet directory in the cloud;
  * the document vector index (doc_chunks) embedded with OpenAI
    text-embedding-3-small (1536-dim) + its HNSW index;
  * the acronyms lookup table.

This file is committed and shipped with the app — no Ollama, no Parquet, no
separate database server needed on Streamlit Cloud.
"""
from __future__ import annotations

import duckdb

from aiu_chat import config
from aiu_chat.agent.catalog import load_catalog
from aiu_chat.ingest import build_index as bi


def _copy_datasets(con: duckdb.DuckDBPyConnection) -> int:
    """Copy each dataset's Parquet into a real table inside the cloud DuckDB."""
    catalog = load_catalog()
    n = 0
    for d in catalog.datasets:
        safe = d.parquet_path.replace("'", "''")
        con.execute(f"DROP TABLE IF EXISTS {d.table}")
        con.execute(f"CREATE TABLE {d.table} AS SELECT * FROM read_parquet('{safe}')")
        n += 1
        print(f"  + table {d.table}")
    return n


def main(argv: list[str] | None = None) -> int:
    if config.LOCAL:
        raise SystemExit(
            "Run with LOCAL=false so embeddings use OpenAI (cloud) — "
            "otherwise the index dimension won't match the cloud deployment."
        )
    if not config.OPENAI_KEY:
        raise SystemExit("OPENAI_KEY must be set to build the cloud index.")

    config.ensure_dirs()
    print(f"Building cloud DuckDB at {config.DUCKDB_PATH}")
    print(f"Embeddings: {config.EMBEDDING_MODEL} (dim {config.EMBEDDING_DIM})\n")

    # 1) Datasets -> tables (so no Parquet dir is needed in the cloud).
    print("Copying datasets into the DuckDB...")
    con = duckdb.connect(str(config.DUCKDB_PATH))
    try:
        con.execute("INSTALL vss")
        con.execute("LOAD vss")
        con.execute("SET hnsw_enable_experimental_persistence = true")
        ndatasets = _copy_datasets(con)
    finally:
        con.close()
    print(f"  copied {ndatasets} dataset tables\n")

    # 2) Docs + PDFs -> vector index (OpenAI embeddings), into the SAME file.
    print("Building the document vector index with OpenAI embeddings...")
    nchunks = bi.build_index()

    print(f"\nDone. {ndatasets} dataset tables + {nchunks} doc chunks -> {config.DUCKDB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
