"""Deterministic cross-frame aggregation executor (feature #4).

Registers the frames a turn already produced (the SQL result, the merged fan-out
records) as named DuckDB views, then validates + executes ONE model-emitted
aggregation SELECT over them. This is how "combined total across France, Germany
and Spain" is computed WITHOUT the model doing arithmetic: the model writes the
aggregation SQL, deterministic code runs it.

Separate from sql_tool's main path so it can allow the frame view names
(`data`, `dataapp`) instead of the catalog tables, without weakening either the
catalog restriction or the shared single-SELECT / no-DDL / no-file-function
safety checks (those are reused via validate_sql(allowed_tables=...)).
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from aiu_chat import config
from aiu_chat.agent.sql_tool import UnsafeSQLError, validate_sql

# Fixed view names the aggregation SQL may reference; nothing else is allowed.
FRAME_VIEWS = ("data", "dataapp")


@dataclass
class AggResult:
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    truncated: bool


def collect_frames(turn) -> dict[str, pd.DataFrame]:
    """Gather a turn's tabular frames into named DataFrames (fixed view names).

    Only frames with rows are included. Non-tabular sources (nm_live/nop) are
    excluded in v1.
    """
    frames: dict[str, pd.DataFrame] = {}

    data = getattr(turn, "data", None)
    result = getattr(data, "result", None) if data is not None else None
    df = getattr(result, "dataframe", None) if result is not None else None
    if df is not None and not df.empty:
        frames["data"] = df

    dataapp = getattr(turn, "dataapp", None)
    results = getattr(dataapp, "results", None) if dataapp is not None else None
    if results:
        rows = []
        for r in results:
            for rec in r.records:
                rows.append({"entity": r.entity.name, **rec})
        if rows:
            frames["dataapp"] = pd.DataFrame(rows)

    return frames


def run_aggregation(sql: str, frames: dict[str, pd.DataFrame]) -> AggResult:
    """Validate + execute an aggregation SELECT over the given frames.

    Allowed tables are exactly the frame names present. Read-only, in-memory,
    row-capped. Raises UnsafeSQLError on a safety violation."""
    if not frames:
        raise UnsafeSQLError("No frames to aggregate.")

    allowed = set(frames)
    validate_sql(sql, allowed_tables=allowed)  # shared safety checks + view allow-set

    con = duckdb.connect()  # in-memory; no disk, no catalog data
    try:
        for name, df in frames.items():
            con.register(name, df)
        cap = config.MAX_RESULT_ROWS
        out = con.execute(f"SELECT * FROM ({sql}) LIMIT {cap + 1}").fetch_df()
    finally:
        con.close()

    truncated = len(out) > cap
    if truncated:
        out = out.head(cap)
    return AggResult(sql=sql, dataframe=out, row_count=len(out), truncated=truncated)
