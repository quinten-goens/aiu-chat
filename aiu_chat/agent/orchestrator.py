"""Top-level agent: rewrite follow-ups, route, dispatch, and combine.

This ties the data path and the concept path together into one entry point,
`answer()`, used by both the CLI and the Streamlit UI. Every turn is logged
(rewritten question, route, SQL, sources) for debuggability — a core requirement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aiu_chat.agent import prompts
from aiu_chat.agent.catalog import Catalog, get_catalog
from aiu_chat.agent.concept import ConceptAnswer, answer_concept_question
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.agent.text_to_sql import DataAnswer, answer_data_question

logger = logging.getLogger("aiu_chat.agent")

VALID_ROUTES = {"data", "concept", "both"}


@dataclass
class Turn:
    """A user turn and the agent's response, carrying everything the UI renders."""
    question: str               # the original user text
    standalone_question: str    # after follow-up rewriting
    route: str
    data: DataAnswer | None = None
    concept: ConceptAnswer | None = None
    answer: str = ""
    sources: list = field(default_factory=list)


def _history_text(history: list[Turn], max_turns: int = 4) -> str:
    """Render recent turns for the rewrite prompt."""
    recent = history[-max_turns:]
    lines = []
    for t in recent:
        lines.append(f"User: {t.question}")
        if t.answer:
            lines.append(f"Assistant: {t.answer[:300]}")
    return "\n".join(lines)


def rewrite_followup(question: str, history: list[Turn], client: OllamaClient) -> str:
    """Rewrite an elliptical follow-up into a standalone question."""
    if not history:
        return question
    try:
        messages = prompts.build_rewrite_messages(_history_text(history), question)
        rewritten = client.chat(messages, temperature=0.0).strip()
        return rewritten or question
    except Exception:
        return question  # rewriting is best-effort; fall back to the raw question


def route_question(question: str, client: OllamaClient) -> str:
    """Classify a standalone question as data | concept | both."""
    try:
        result = client.chat_json(prompts.build_router_messages(question))
        route = str(result.get("route", "")).lower()
        return route if route in VALID_ROUTES else "data"
    except Exception:
        return "data"  # default to the data path if routing fails


def answer(
    question: str,
    *,
    history: list[Turn] | None = None,
    client: OllamaClient | None = None,
    catalog: Catalog | None = None,
) -> Turn:
    client = client or OllamaClient()
    catalog = catalog or get_catalog()
    history = history or []

    standalone = rewrite_followup(question, history, client)
    route = route_question(standalone, client)
    logger.info("turn route=%s | q=%r | standalone=%r", route, question, standalone)

    turn = Turn(question=question, standalone_question=standalone, route=route)

    if route in ("data", "both"):
        turn.data = answer_data_question(standalone, client=client, catalog=catalog)
        logger.info("  data ok=%s sql=%r", turn.data.ok, turn.data.sql)
    if route in ("concept", "both"):
        turn.concept = answer_concept_question(standalone, client=client)
        logger.info(
            "  concept ok=%s sources=%s",
            turn.concept.ok,
            [s.source_title for s in turn.concept.sources],
        )

    turn.answer, turn.sources = _combine(turn)
    return turn


def _combine(turn: Turn) -> tuple[str, list]:
    """Merge path outputs into one answer + source list."""
    parts: list[str] = []
    sources: list = []

    if turn.concept is not None and turn.concept.ok:
        parts.append(turn.concept.answer)
        sources.extend(turn.concept.sources)
    if turn.data is not None:
        parts.append(turn.data.answer)

    if not parts:
        # Nothing useful from either path.
        if turn.data is not None:
            return turn.data.answer, sources
        if turn.concept is not None:
            return turn.concept.answer, sources
        return "I couldn't find an answer to that.", sources

    return "\n\n".join(parts), sources
