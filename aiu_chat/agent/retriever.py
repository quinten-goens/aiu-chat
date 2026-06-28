"""Concept path: embed a query and retrieve relevant doc chunks via DuckDB VSS."""
from __future__ import annotations

from dataclasses import dataclass

import duckdb

from aiu_chat import config
from aiu_chat.agent.llm import embed_text
from aiu_chat.ingest.build_index import TABLE


@dataclass
class RetrievedChunk:
    text: str
    source_url: str
    source_title: str
    similarity: float  # cosine similarity in [0, 1]


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    con.execute("INSTALL vss")
    con.execute("LOAD vss")
    return con


def index_exists() -> bool:
    if not config.DUCKDB_PATH.exists():
        return False
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        return TABLE in tables
    finally:
        con.close()


def retrieve(
    query: str,
    *,
    client=None,  # accepted for call-site compatibility; embedding uses embed_text
    top_k: int | None = None,
    min_similarity: float | None = None,
) -> list[RetrievedChunk]:
    """Return the most similar doc chunks to the query, filtered by similarity.

    Embeds the query with the deployment's configured provider (Ollama or OpenAI),
    matching the index it queries.
    """
    top_k = top_k or config.TOP_K
    min_similarity = config.MIN_SIMILARITY if min_similarity is None else min_similarity

    qvec = embed_text(query)
    dim = config.EMBEDDING_DIM

    con = _connect()
    try:
        # array_cosine_similarity returns [-1, 1]; HNSW index accelerates the
        # ordering. We fetch top_k then filter by the similarity floor.
        rows = con.execute(
            f"""
            SELECT text, source_url, source_title,
                   array_cosine_similarity(embedding, ?::FLOAT[{dim}]) AS sim
            FROM {TABLE}
            ORDER BY sim DESC
            LIMIT ?
            """,
            [qvec, top_k],
        ).fetchall()
    finally:
        con.close()

    return [
        RetrievedChunk(text=t, source_url=u, source_title=title, similarity=sim)
        for (t, u, title, sim) in rows
        if sim >= min_similarity
    ]
