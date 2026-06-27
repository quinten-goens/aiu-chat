"""The data path: question -> SQL -> execute -> grounded answer.

This is the text-to-SQL pipeline without routing or charts yet (those arrive in
later slices). It is the heart of the "numbers always come from executed SQL"
guarantee: the model writes SQL and narrates rows, but never computes numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from aiu_chat.agent.catalog import Catalog, get_catalog
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.agent import prompts
from aiu_chat.agent.sql_tool import SqlResult, UnsafeSQLError, run_sql

CANNOT_ANSWER = "CANNOT_ANSWER"


@dataclass
class DataAnswer:
    question: str
    sql: str | None
    result: SqlResult | None
    answer: str
    ok: bool  # True if a grounded answer was produced


def _clean_sql(raw: str) -> str:
    """Strip markdown fences / stray prose the model may add around the SQL."""
    text = raw.strip()
    # Remove ```sql ... ``` fences if present.
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return text.strip().rstrip(";").strip()


def _as_of_for(result: SqlResult, catalog: Catalog) -> str | None:
    """Best-effort as-of date from any catalog table referenced (first match)."""
    for d in catalog.datasets:
        if d.table in (result.sql or ""):
            return d.as_of
    return catalog.datasets[0].as_of if catalog.datasets else None


def answer_data_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    catalog: Catalog | None = None,
) -> DataAnswer:
    client = client or OllamaClient()
    catalog = catalog or get_catalog()

    # 1. Generate SQL.
    sql_messages = prompts.build_sql_messages(catalog.prompt_text(), question)
    raw_sql = client.chat(sql_messages, temperature=0.0)
    sql = _clean_sql(raw_sql)

    if CANNOT_ANSWER in sql.upper():
        return DataAnswer(
            question=question,
            sql=None,
            result=None,
            answer="I can't answer that from the available datasets.",
            ok=False,
        )

    # 2. Validate + execute (read-only, sandboxed).
    try:
        result = run_sql(sql, catalog=catalog)
    except UnsafeSQLError as exc:
        return DataAnswer(
            question=question,
            sql=sql,
            result=None,
            answer=f"The generated query was rejected by the safety check: {exc}",
            ok=False,
        )
    except Exception as exc:  # duckdb execution error — surface honestly
        return DataAnswer(
            question=question,
            sql=sql,
            result=None,
            answer=f"The query failed to execute: {exc}",
            ok=False,
        )

    # 3. Narrate grounded in the executed rows.
    rows_json = _rows_to_json(result.dataframe)
    answer_messages = prompts.build_answer_messages(
        question, result.sql, rows_json, _as_of_for(result, catalog)
    )
    answer = client.chat(answer_messages, temperature=0.0)

    return DataAnswer(
        question=question, sql=result.sql, result=result, answer=answer.strip(), ok=True
    )


def _rows_to_json(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Compact JSON of the result for the narration prompt (capped for context)."""
    return df.head(max_rows).to_json(orient="records", date_format="iso")
