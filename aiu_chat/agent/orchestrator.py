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
from aiu_chat.agent.nm_answer import NmLiveAnswer, answer_nm_question
from aiu_chat.agent.nop_answer import NopAnswer, answer_nop_question
from aiu_chat.agent.text_to_sql import DataAnswer, answer_data_question

logger = logging.getLogger("aiu_chat.agent")

VALID_ROUTES = {"data", "concept", "both", "nop", "dataapp", "nm_live", "none"}

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
    nm_live: NmLiveAnswer | None = None
    answer: str = ""
    sources: list = field(default_factory=list)
    needs_clarification: bool = False  # True when the turn is a clarifying question


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
            # Flag a clarifying question so the rewriter treats the user's next
            # message as the answer to it and merges them into one question.
            prefix = "Assistant (clarifying question)" if t.needs_clarification else "Assistant"
            lines.append(f"{prefix}: {t.answer[:200]}")
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


# Routes where a missing subject (which airport/ANSP/country?) blocks an answer.
_CLARIFIABLE_ROUTES = {"data", "both", "dataapp", "nm_live"}


def needs_clarification(question: str, route: str, client: OllamaClient) -> str | None:
    """Return a single clarifying question if an essential detail is missing,
    else None. Conservative: only fires when the agent genuinely can't proceed.
    Failures default to None (proceed) so a hiccup never blocks an answer."""
    if route not in _CLARIFIABLE_ROUTES:
        return None
    try:
        result = client.chat_json(prompts.build_clarify_messages(question, route))
        if result.get("needs_clarification") and result.get("question"):
            return str(result["question"]).strip()
    except Exception:
        return None
    return None


# What each route is doing while it works — shown live in the UI status.
_ROUTE_STATUS = {
    "data": "Querying the datasets…",
    "both": "Querying the data and reference docs…",
    "concept": "Searching the reference docs & PDFs…",
    "nop": "Fetching NOP messages…",
    "dataapp": "Looking up the latest daily figures…",
    "nm_live": "Fetching the live network snapshot…",
}

# Short reason for each routing choice — shown as the step detail.
_ROUTE_WHY = {
    "data": "historical figures from the local datasets",
    "both": "needs a figure and an explanation",
    "concept": "a definition / methodology question",
    "nop": "the operational situation in NOP messages",
    "dataapp": "latest daily (D-1) figures for an entity",
    "nm_live": "the real-time network state",
    "none": "outside air navigation performance",
}


def answer(
    question: str,
    *,
    history: list[Turn] | None = None,
    client: OllamaClient | None = None,
    catalog: Catalog | None = None,
    on_status=None,
) -> Turn:
    """Answer a question.

    `on_status(label, detail=None)`, if given, is called at each real stage:
    `label` is the short rotating status title; `detail` is an optional line
    describing what that step actually produced (rewritten question, chosen
    route, SQL, sources) — surfaced live in the UI. Thinking mode is off, so this
    shows the real per-step artifacts, not a chain-of-thought.
    """
    client = client or OllamaClient()
    catalog = catalog or get_catalog()
    history = history or []

    def status(label: str, detail: str | None = None) -> None:
        if on_status is not None:
            on_status(label, detail)

    if history:
        status("Reading the conversation…")
    standalone = rewrite_followup(question, history, client)
    if standalone != question:
        status("Reading the conversation…", f"Interpreted as: *{standalone}*")

    # Data-availability questions ("what data/datasets do you have?", "what data
    # is available for X") are answered from the catalog, not vector search —
    # the docs corpus dilutes them and gives wrong "no info" answers.
    if _is_availability_question(standalone):
        logger.info("turn route=catalog | q=%r", question)
        status("Choosing the best source…", "Route: **catalogue** (listing available data)")
        turn = Turn(question=question, standalone_question=standalone, route="catalog")
        turn.answer = catalog.describe()
        return turn

    status("Choosing the best source…")
    route = route_question(standalone, client)
    logger.info("turn route=%s | q=%r | standalone=%r", route, question, standalone)
    status("Choosing the best source…", f"Route: **{route}** — {_ROUTE_WHY.get(route, '')}")

    turn = Turn(question=question, standalone_question=standalone, route=route)

    # Out-of-scope questions decline cleanly rather than forcing a path.
    if route == "none":
        turn.answer = (
            "That's outside what I can help with — I answer questions about "
            "European air navigation performance (traffic, delays, efficiency, "
            "emissions, the live network, and EUROCONTROL methodology)."
        )
        return turn

    # Ask one clarifying question if an essential detail is missing (conservative
    # — only when we genuinely can't proceed). The user's reply flows back as a
    # follow-up and is merged with this question by rewrite_followup next turn.
    status("Checking the question…")
    clarification = needs_clarification(standalone, route, client)
    if clarification:
        logger.info("  needs clarification: %r", clarification)
        turn.needs_clarification = True
        turn.answer = clarification
        return turn

    status(_ROUTE_STATUS.get(route, "Gathering data…"))
    if route in ("data", "both"):
        turn.data = answer_data_question(standalone, client=client, catalog=catalog)
        logger.info("  data ok=%s sql=%r", turn.data.ok, turn.data.sql)
        if turn.data.sql:
            status(_ROUTE_STATUS.get(route, "Gathering data…"),
                   f"SQL:\n```sql\n{turn.data.sql}\n```")
    if route in ("concept", "both"):
        turn.concept = answer_concept_question(standalone, client=client)
        logger.info(
            "  concept ok=%s sources=%s",
            turn.concept.ok,
            [s.source_title for s in turn.concept.sources],
        )
        srcs = ", ".join(sorted({s.source_title for s in turn.concept.sources}))
        if srcs:
            status("Searching the reference docs & PDFs…", f"Retrieved from: {srcs}")
    if route == "nop":
        turn.nop = answer_nop_question(standalone, client=client)
        logger.info("  nop ok=%s messages=%d", turn.nop.ok, len(turn.nop.messages))
        status("Fetching NOP messages…", f"Found {len(turn.nop.messages)} relevant message(s)")
    if route == "dataapp":
        turn.dataapp = answer_dataapp_question(standalone, client=client)
        ent = turn.dataapp.result.entity.name if turn.dataapp.result else None
        logger.info("  dataapp ok=%s entity=%s", turn.dataapp.ok, ent)
        if ent:
            status("Looking up the latest daily figures…",
                   f"Resolved entity: **{ent}** (Data App, D-1)")
    if route == "nm_live":
        turn.nm_live = answer_nm_question(standalone, client=client)
        logger.info("  nm_live ok=%s", turn.nm_live.ok)
        status("Fetching the live network snapshot…", "Live Network Manager data")

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
    if turn.nm_live is not None:
        parts.append(turn.nm_live.answer)

    if not parts:
        # Nothing useful from any path.
        for sub in (turn.data, turn.concept, turn.nop, turn.dataapp, turn.nm_live):
            if sub is not None:
                return sub.answer, sources
        return "I couldn't find an answer to that.", sources

    return "\n\n".join(parts), sources
