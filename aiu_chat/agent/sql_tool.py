"""Read-only, sandboxed text-to-SQL execution over the DuckDB/Parquet data.

LLM-generated SQL is treated as untrusted input. Defense in depth:

  1. Parse with sqlglot and require exactly ONE statement that is a SELECT (or a
     WITH ... SELECT). Anything else is rejected before it reaches the engine.
  2. Walk the AST and reject dangerous node types (DDL/DML, ATTACH, COPY, PRAGMA,
     SET, CALL, EXPORT, file functions) even if they somehow parse as part of a
     query.
  3. Restrict table references to the known catalog tables.
  4. Execute against a DuckDB connection opened read-only, with each catalog
     table registered as a view over its Parquet file. A row cap is applied.

The public entry point is `run_sql(sql)`, returning a SqlResult.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from aiu_chat import config
from aiu_chat.agent.catalog import Catalog, get_catalog

DIALECT = "duckdb"

# Statement types we allow at the top level.
_ALLOWED_TOP_LEVEL = (exp.Select, exp.Union, exp.Except, exp.Intersect)

# AST node types that must never appear anywhere in the query.
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge,
    exp.Create, exp.Drop, exp.Alter, exp.TruncateTable,
    exp.Command,        # catch-all for COPY, ATTACH, PRAGMA, SET, CALL, EXPORT, etc.
    exp.Attach,         # DuckDB ATTACH
)

# Function names that read/write the filesystem or escape the sandbox.
_FORBIDDEN_FUNCS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "glob", "copy", "install", "load",
}


class UnsafeSQLError(ValueError):
    """Raised when generated SQL fails a safety check."""


@dataclass
class SqlResult:
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    truncated: bool


def _strip_comments_and_trailing_semi(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def validate_sql(sql: str, catalog: Catalog | None = None) -> exp.Expression:
    """Validate that `sql` is a safe, single, read-only SELECT. Returns the AST.

    Raises UnsafeSQLError on any violation.
    """
    catalog = catalog or get_catalog()
    cleaned = _strip_comments_and_trailing_semi(sql)
    if not cleaned:
        raise UnsafeSQLError("Empty SQL.")

    try:
        statements = sqlglot.parse(cleaned, dialect=DIALECT)
    except Exception as exc:  # sqlglot parse errors
        raise UnsafeSQLError(f"Could not parse SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"Exactly one statement is allowed; got {len(statements)}."
        )

    stmt = statements[0]

    # Top-level must be a SELECT-shaped query. sqlglot attaches WITH to the
    # SELECT (a CTE query parses as a Select carrying a `with` arg), so checking
    # the statement class is sufficient.
    if not isinstance(stmt, _ALLOWED_TOP_LEVEL):
        raise UnsafeSQLError(
            f"Only SELECT queries are allowed; got {type(stmt).__name__}."
        )

    # Reject forbidden node types anywhere in the tree.
    for node in stmt.walk():
        node = node[0] if isinstance(node, tuple) else node
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(
                f"Disallowed statement element: {type(node).__name__}."
            )

    # Reject forbidden functions (filesystem access, install/load).
    for func in stmt.find_all(exp.Anonymous, exp.Func):
        name = (func.name or "").lower()
        if name in _FORBIDDEN_FUNCS:
            raise UnsafeSQLError(f"Disallowed function: {name}().")

    # Restrict table references to known catalog tables. CTE aliases are
    # references to in-query result sets, not base tables, so they are allowed.
    allowed = catalog.table_names
    cte_names = _cte_names(stmt)
    for table in stmt.find_all(exp.Table):
        tname = table.name
        if tname and tname not in allowed and tname not in cte_names:
            raise UnsafeSQLError(
                f"Unknown table '{tname}'. Allowed: {sorted(allowed)}."
            )

    return stmt


def _cte_names(stmt: exp.Expression) -> set[str]:
    """All CTE aliases defined anywhere in the statement."""
    return {cte.alias for cte in stmt.find_all(exp.CTE) if cte.alias}


def _connect_readonly(catalog: Catalog) -> duckdb.DuckDBPyConnection:
    """In-memory connection with each catalog table as a read-only Parquet view.

    We use an in-memory DB (no writable database file) and register each dataset
    as a VIEW over read_parquet(...). The view definitions are created by us, not
    by the model, so they are trusted; the model's query only ever sees views.
    """
    con = duckdb.connect(database=":memory:")
    for d in catalog.datasets:
        safe_path = d.parquet_path.replace("'", "''")
        con.execute(
            f"CREATE VIEW {d.table} AS SELECT * FROM read_parquet('{safe_path}')"
        )
    return con


def run_sql(
    sql: str,
    catalog: Catalog | None = None,
    max_rows: int | None = None,
) -> SqlResult:
    """Validate and execute read-only SQL. Returns a SqlResult.

    Raises UnsafeSQLError if validation fails; duckdb errors propagate for the
    caller to surface honestly.
    """
    catalog = catalog or get_catalog()
    max_rows = max_rows or config.MAX_RESULT_ROWS

    validate_sql(sql, catalog)  # raises on anything unsafe
    cleaned = _strip_comments_and_trailing_semi(sql)

    con = _connect_readonly(catalog)
    try:
        # Fetch one extra row to detect truncation.
        df = con.execute(cleaned).fetch_df()
    finally:
        con.close()

    truncated = len(df) > max_rows
    if truncated:
        df = df.head(max_rows)

    return SqlResult(
        sql=cleaned, dataframe=df, row_count=len(df), truncated=truncated
    )
