"""Minimal CLI for the text-to-SQL data path. Run: aiu-chat-cli  (or python -m aiu_chat.cli)

A bare REPL to exercise the pipeline end to end before the Streamlit UI exists.
"""
from __future__ import annotations

import sys

from aiu_chat.agent.catalog import get_catalog
from aiu_chat.agent.llm import OllamaClient, OllamaError
from aiu_chat.agent.orchestrator import answer

BANNER = """\
AIU Chat (CLI) — ask about European ANS performance (data + concepts).
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
    history = []
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
            turn = answer(question, history=history, client=client, catalog=catalog)
        except OllamaError as exc:
            print(f"error: {exc}\n", file=sys.stderr)
            continue

        print(f"  (route: {turn.route})")
        if show_sql and turn.data is not None and turn.data.sql:
            print(f"  SQL: {turn.data.sql}")
        if turn.data is not None and turn.data.result is not None \
                and not turn.data.result.dataframe.empty:
            print()
            print(turn.data.result.dataframe.head(20).to_string(index=False))
        print(f"\nbot> {turn.answer}\n")
        history.append(turn)


if __name__ == "__main__":
    raise SystemExit(main())
