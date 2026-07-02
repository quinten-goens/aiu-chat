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


def run_transform(
    sql: str,
    frames: dict[str, pd.DataFrame],
    *,
    catalog=None,
) -> AggResult:
    """Validate + execute a model-emitted TRANSFORM SELECT over `frames` (named
    in-memory daily-series tables) — and, if `catalog` is given, also the bundled
    historical dataset tables (read-only views) in the SAME query.

    This is the "manipulator": resample weekly/monthly, derive per-flight ratios,
    rolling averages, etc. The model writes ONE SELECT; deterministic code runs
    it, so figures always come from an executed query, never model arithmetic.

    Same safety envelope as run_aggregation (single read-only SELECT, no DDL /
    PRAGMA / file functions), with the allow-set WIDENED to the frame names plus
    the catalog table names. Raises UnsafeSQLError on any violation."""
    if not frames:
        raise UnsafeSQLError("No frames to transform.")

    allowed = set(frames)
    if catalog is not None:
        allowed |= set(catalog.table_names)
    validate_sql(sql, allowed_tables=allowed)

    con = duckdb.connect()  # in-memory
    try:
        for name, df in frames.items():
            con.register(name, df)
        if catalog is not None:
            _register_catalog_views(con, catalog)
        cap = config.MAX_RESULT_ROWS
        out = con.execute(f"SELECT * FROM ({sql}) LIMIT {cap + 1}").fetch_df()
    finally:
        con.close()

    truncated = len(out) > cap
    if truncated:
        out = out.head(cap)
    return AggResult(sql=sql, dataframe=out, row_count=len(out), truncated=truncated)


def _register_catalog_views(con, catalog) -> None:
    """Expose each bundled dataset as a read-only view on `con` (Parquet locally,
    or the attached bundled DuckDB in the cloud). Mirrors sql_tool's connection so
    the manipulator can join live period frames against historical baselines. The
    view SQL is ours (trusted); the model's query only ever sees the views."""
    from pathlib import Path

    attached = False
    for d in catalog.datasets:
        if Path(d.parquet_path).exists():
            safe_path = d.parquet_path.replace("'", "''")
            con.execute(f"CREATE VIEW {d.table} AS SELECT * FROM read_parquet('{safe_path}')")
        else:
            if not attached:
                safe_db = str(config.DUCKDB_PATH).replace("'", "''")
                con.execute(f"ATTACH '{safe_db}' AS bundled (READ_ONLY)")
                attached = True
            con.execute(f"CREATE VIEW {d.table} AS SELECT * FROM bundled.{d.table}")
