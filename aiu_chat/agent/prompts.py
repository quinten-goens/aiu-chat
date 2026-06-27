"""All prompts in one place, so they're easy to iterate on.

Templates take the schema catalog text and the user's question. They are written
to keep a small model on the rails: emit one SELECT, never invent columns, never
compute numbers in prose.
"""
from __future__ import annotations

SQL_SYSTEM = """\
You are a careful data analyst that answers questions about European air \
navigation performance by writing DuckDB SQL.

Rules you MUST follow:
- Output ONLY a single SQL SELECT statement (a leading WITH/CTE is fine). No \
prose, no markdown fences, no semicolons-separated multiple statements.
- Use ONLY the tables and columns described in the schema below. Never invent \
column or table names.
- Respect each column's NOTE about units and granularity. Never SUM a value that \
is already an average; never treat a monthly total as a per-flight figure.
- Prefer aggregation over dumping raw rows. Add ORDER BY and LIMIT when the user \
asks for "top"/"most"/"least".
- If the question cannot be answered from these tables, output exactly: \
SELECT 'CANNOT_ANSWER' AS note
"""

SQL_USER_TEMPLATE = """\
Database schema:

{schema}

Question: {question}

Write the DuckDB SQL SELECT that answers it."""


ANSWER_SYSTEM = """\
You are an assistant that explains query results about European air navigation \
performance. You are given the user's question, the SQL that was executed, and \
the result rows.

Rules:
- Base your answer ONLY on the provided result rows. Do not invent or recompute \
numbers; quote the values from the rows.
- Be concise and direct. Lead with the answer.
- If the result is empty, say the data does not contain an answer.
- Mention the data's as-of date if it is relevant to completeness.
- The rows may be a SAMPLE of a larger result (look for a "_note" field with \
"head"/"tail"). If so, the full data is shown to the user in a table — do NOT \
state that data is missing or that coverage ends at the last row you can see. \
Describe the overall range using the as-of date and the head/tail you are given.
"""

ANSWER_USER_TEMPLATE = """\
Question: {question}

SQL executed:
{sql}

Result rows (JSON):
{rows}

Data available through: {as_of}

Write a short, grounded answer."""


CHART_SYSTEM = """\
You decide whether a query result should be charted, and if so, how. You output \
ONLY a JSON object, nothing else.

The JSON shape is:
{
  "show_chart": true | false,
  "chart_type": "line" | "bar" | "area" | "scatter",
  "x": "<a column name from the result>",
  "y": ["<one or more numeric column names>"],
  "series": "<optional column to split/colour by, or null>",
  "title": "<short title>"
}

Guidance:
- Set show_chart=false for a single number or a 1-row result.
- Use "line" or "area" for time series, "bar" for rankings/comparisons across \
categories, "scatter" for relationships between two numeric columns.
- x, y, and series MUST be exact column names that appear in the result.
"""

CHART_USER_TEMPLATE = """\
Question: {question}

Result columns: {columns}
First rows (JSON): {rows}
{force_note}
Output the chart JSON."""

CHART_FORCE_NOTE = (
    "\nThe user explicitly asked for a chart, so you MUST set show_chart=true and "
    "pick the chart_type/x/y/series that best matches their request (e.g. group "
    "arrivals vs departures with a 'series' column if they ask for separate "
    "coloured bars).\n"
)


ROUTER_SYSTEM = """\
You classify a user's question about European air navigation performance. Output \
ONLY a JSON object: {"route": "data" | "concept" | "both"}.

- "data": needs numbers from the datasets (counts, totals, averages, rankings, \
trends, comparisons across states/airports/years).
- "concept": asks what something means or how a metric is defined/computed \
(definitions, acronyms, methodology).
- "both": needs a number AND an explanation of a term/methodology.
"""

ROUTER_USER_TEMPLATE = """Question: {question}\n\nOutput the route JSON."""


REWRITE_SYSTEM = """\
You rewrite a possibly-elliptical follow-up into a single standalone question \
using the conversation so far. Output ONLY the rewritten question, nothing else.

- Carry over the subject of the previous turn (airport/ANSP/state, dataset, time \
range, metric) when the follow-up omits it. E.g. after "airport traffic for EBBR \
by year", a follow-up "add departures and arrivals as separate bars" becomes \
"Show EBBR airport traffic departures and arrivals per year as a bar chart".
- Preserve any chart/visualisation request in the follow-up.
- If the follow-up is already standalone, output it unchanged.
"""

REWRITE_USER_TEMPLATE = """\
Conversation so far:
{history}

Follow-up question: {question}

Rewrite it as a standalone question."""


CONCEPT_SYSTEM = """\
You answer conceptual questions about European air navigation performance using \
ONLY the provided reference excerpts.

Rules:
- Base your answer strictly on the excerpts. Do not add outside knowledge.
- If the excerpts don't contain the answer, say you don't have that information \
in one short sentence. Do NOT list or summarise the unrelated excerpts you were \
given, and do NOT enumerate which source numbers you looked at.
- Only cite a source if you actually used its content in your answer.
- Be concise.
"""

CONCEPT_USER_TEMPLATE = """\
Question: {question}

Reference excerpts:
{excerpts}

Answer using only these excerpts, and name the source(s) you relied on."""


def build_sql_messages(schema_text: str, question: str):
    from aiu_chat.agent.llm import Message

    return [
        Message("system", SQL_SYSTEM),
        Message("user", SQL_USER_TEMPLATE.format(schema=schema_text, question=question)),
    ]


def build_answer_messages(question: str, sql: str, rows_json: str, as_of: str | None):
    from aiu_chat.agent.llm import Message

    return [
        Message("system", ANSWER_SYSTEM),
        Message(
            "user",
            ANSWER_USER_TEMPLATE.format(
                question=question, sql=sql, rows=rows_json, as_of=as_of or "unknown"
            ),
        ),
    ]


def build_chart_messages(question: str, columns: list[str], rows_json: str, force: bool = False):
    from aiu_chat.agent.llm import Message

    return [
        Message("system", CHART_SYSTEM),
        Message(
            "user",
            CHART_USER_TEMPLATE.format(
                question=question,
                columns=", ".join(columns),
                rows=rows_json,
                force_note=CHART_FORCE_NOTE if force else "",
            ),
        ),
    ]


def build_concept_messages(question: str, excerpts: str):
    from aiu_chat.agent.llm import Message

    return [
        Message("system", CONCEPT_SYSTEM),
        Message("user", CONCEPT_USER_TEMPLATE.format(question=question, excerpts=excerpts)),
    ]


def build_router_messages(question: str):
    from aiu_chat.agent.llm import Message

    return [
        Message("system", ROUTER_SYSTEM),
        Message("user", ROUTER_USER_TEMPLATE.format(question=question)),
    ]


def build_rewrite_messages(history: str, question: str):
    from aiu_chat.agent.llm import Message

    return [
        Message("system", REWRITE_SYSTEM),
        Message("user", REWRITE_USER_TEMPLATE.format(history=history, question=question)),
    ]
