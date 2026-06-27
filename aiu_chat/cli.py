"""Minimal CLI for the text-to-SQL data path. Run: aiu-chat-cli  (or python -m aiu_chat.cli)

A bare REPL to exercise the pipeline end to end before the Streamlit UI exists.
"""
from __future__ import annotations

import sys

from aiu_chat.agent.catalog import get_catalog
from aiu_chat.agent.llm import OllamaClient, OllamaError
from aiu_chat.agent.text_to_sql import answer_data_question

BANNER = """\
AIU Chat (CLI) — ask a question about European ANS performance data.
Type 'exit' or Ctrl-D to quit. Type 'sql' to toggle showing the generated SQL.
"""


def main(argv: list[str] | None = None) -> int:
    try:
        catalog = get_catalog()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    client = OllamaClient()
    tables = ", ".join(sorted(catalog.table_names))
    print(BANNER)
    print(f"Available tables: {tables}\n")

    show_sql = True
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0
        if question.lower() == "sql":
            show_sql = not show_sql
            print(f"(show SQL: {show_sql})")
            continue

        try:
            ans = answer_data_question(question, client=client, catalog=catalog)
        except OllamaError as exc:
            print(f"error: {exc}\n", file=sys.stderr)
            continue

        if show_sql and ans.sql:
            print(f"\n  SQL: {ans.sql}")
        if ans.result is not None and not ans.result.dataframe.empty:
            print()
            print(ans.result.dataframe.head(20).to_string(index=False))
            if ans.result.truncated:
                print(f"  ... (truncated to {ans.result.row_count} rows)")
        print(f"\nbot> {ans.answer}\n")


if __name__ == "__main__":
    raise SystemExit(main())
