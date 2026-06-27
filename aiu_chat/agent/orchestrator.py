"""Top-level agent: rewrite follow-ups, route, dispatch, and combine.

This ties the data path and the concept path together into one entry point,
`answer()`, used by both the CLI and the Streamlit UI. Every turn is logged
(rewritten question, route, SQL, sources) for debuggability — a core requirement.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from aiu_chat.agent import prompts
from aiu_chat.agent.catalog import Catalog, get_catalog
from aiu_chat.agent.concept import ConceptAnswer, answer_concept_question
from aiu_chat.agent.dataapp_answer import DataAppAnswer, answer_dataapp_question
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.agent.nop_answer import NopAnswer, answer_nop_question
from aiu_chat.agent.text_to_sql import DataAnswer, answer_data_question

logger = logging.getLogger("aiu_chat.agent")

VALID_ROUTES = {"data", "concept", "both", "nop", "dataapp"}

# Questions about what data/reports the system holds — answer from the catalog.
_AVAILABILITY_RE = re.compile(
    r"\b(what|which)\b.*\b(data|datasets?|tables?|reports?)\b.*"
    r"\b(have|available|access|hold|got|contain|offer)\b"
    r"|\b(data|datasets?)\b.*\b(available|do you have)\b"
    r"|\bwhat can you (tell me about|answer)\b",
    re.IGNORECASE,
)


def _is_availability_question(question: str) -> bool:
    return bool(_AVAILABILITY_RE.search(question))


@dataclass
class Turn:
    """A user turn and the agent's response, carrying everything the UI renders."""
    question: str               # the original user text
    standalone_question: str    # after follow-up rewriting
    route: str
    data: DataAnswer | None = None
    concept: ConceptAnswer | None = None
    nop: NopAnswer | None = None
    dataapp: DataAppAnswer | None = None
    answer: str = ""
    sources: list = field(default_factory=list)


def _history_text(history: list[Turn], max_turns: int = 4) -> str:
    """Render recent turns for the rewrite prompt.

    Includes the prior SQL so the rewriter knows the concrete subject (which
    airport/ANSP/dataset/columns) when a follow-up omits it.
    """
    recent = history[-max_turns:]
    lines = []
    for t in recent:
        lines.append(f"User: {t.question}")
        if t.data is not None and t.data.sql:
            lines.append(f"(SQL used: {t.data.sql})")
        if t.answer:
            lines.append(f"Assistant: {t.answer[:200]}")
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

    # Data-availability questions ("what data/datasets do you have?", "what data
    # is available for X") are answered from the catalog, not vector search —
    # the docs corpus dilutes them and gives wrong "no info" answers.
    if _is_availability_question(standalone):
        logger.info("turn route=catalog | q=%r", question)
        turn = Turn(question=question, standalone_question=standalone, route="catalog")
        turn.answer = catalog.describe()
        return turn

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
    if route == "nop":
        turn.nop = answer_nop_question(standalone, client=client)
        logger.info("  nop ok=%s messages=%d", turn.nop.ok, len(turn.nop.messages))
    if route == "dataapp":
        turn.dataapp = answer_dataapp_question(standalone, client=client)
        logger.info(
            "  dataapp ok=%s entity=%s",
            turn.dataapp.ok,
            turn.dataapp.result.entity.name if turn.dataapp.result else None,
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
    if turn.nop is not None:
        parts.append(turn.nop.answer)
    if turn.dataapp is not None:
        parts.append(turn.dataapp.answer)

    if not parts:
        # Nothing useful from any path.
        for sub in (turn.data, turn.concept, turn.nop, turn.dataapp):
            if sub is not None:
                return sub.answer, sources
        return "I couldn't find an answer to that.", sources

    return "\n\n".join(parts), sources
