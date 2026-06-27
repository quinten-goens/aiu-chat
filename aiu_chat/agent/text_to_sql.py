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

# Words signalling the user explicitly wants a visualisation.
_CHART_INTENT_RE = re.compile(
    r"\b(chart|graph|plot|bar ?chart|line ?chart|visuali[sz]e|"
    r"bars?|histogram|pie|trend line)\b",
    re.IGNORECASE,
)


def wants_chart(question: str) -> bool:
    """True if the question explicitly asks for a chart/visualisation."""
    return bool(_CHART_INTENT_RE.search(question))


@dataclass
class DataAnswer:
    question: str
    sql: str | None
    result: SqlResult | None
    answer: str
    ok: bool  # True if a grounded answer was produced
    chart_spec: dict | None = None  # LLM-emitted chart spec (validated at render time)


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
    with_chart: bool = True,
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

    # 4. Ask for a chart spec when the result is plausibly chart-worthy (>=2
    # rows), or whenever the user explicitly asked for a chart. Failures here
    # never break the answer — charts are best-effort.
    chart_spec = None
    force = wants_chart(question)
    if with_chart and (len(result.dataframe) >= 2 or force):
        chart_spec = _maybe_chart_spec(client, question, result.dataframe, force=force)

    return DataAnswer(
        question=question,
        sql=result.sql,
        result=result,
        answer=answer.strip(),
        ok=True,
        chart_spec=chart_spec,
    )


def _maybe_chart_spec(client, question, df, force: bool = False):
    """Best-effort chart spec from the model; None on any problem."""
    try:
        messages = prompts.build_chart_messages(
            question, list(df.columns), _rows_to_json(df, max_rows=20), force=force
        )
        return client.chat_json(messages, temperature=0.0)
    except Exception:
        return None  # charts are optional; never fail the answer over one


def _rows_to_json(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Compact JSON of the result for a prompt (capped for context).

    For a result that fits, send it whole. For a larger result, send the head
    AND tail so the model never wrongly concludes data is 'missing' beyond the
    window (e.g. claiming no data after the 100th row of a time series).
    """
    if len(df) <= max_rows:
        return df.to_json(orient="records", date_format="iso")
    half = max_rows // 2
    head = df.head(half).to_json(orient="records", date_format="iso")
    tail = df.tail(half).to_json(orient="records", date_format="iso")
    return f'{{"_note": "showing first {half} and last {half} of {len(df)} rows", "head": {head}, "tail": {tail}}}'
