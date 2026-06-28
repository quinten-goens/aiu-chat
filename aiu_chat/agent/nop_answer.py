"""NOP path: fetch relevant NOP messages live, then answer from them."""
from __future__ import annotations

from dataclasses import dataclass, field

from aiu_chat.agent import prompts
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.sources.nop import NopError, NopMessage, fetch_messages

import re

# Words signalling a HISTORICAL/topical search (vs. the current situation).
_HISTORICAL_RE = re.compile(
    r"\b(was|were|last week|last month|yesterday|previous|history|"
    r"ever|happened|past|earlier)\b",
    re.IGNORECASE,
)

# Stopwords stripped when deriving a keyword for a historical search.
_STOP = {
    "what", "is", "the", "a", "an", "are", "was", "were", "do", "does", "did",
    "show", "me", "tell", "about", "any", "there", "in", "on", "for", "of", "to",
    "nop", "message", "messages", "latest", "recent", "current", "today",
    "give", "can", "you", "please", "and", "or", "with", "this", "that",
    "situation", "happening", "network", "right", "now",
}


def _keyword(question: str) -> str | None:
    """Pick a keyword for the PocketBase filter, ONLY for historical/topical
    searches. For 'current situation' questions we fetch the latest messages
    unfiltered, since each tactical update already covers weather, aerodromes
    and airspace comprehensively (and uses ICAO codes, not plain names)."""
    if not _HISTORICAL_RE.search(question):
        return None
    words = [w.strip(".,?!:;\"'()").lower() for w in question.split()]
    candidates = [w for w in words if len(w) > 3 and w not in _STOP]
    return max(candidates, key=len) if candidates else None


@dataclass
class NopAnswer:
    question: str
    answer: str
    messages: list[NopMessage] = field(default_factory=list)
    ok: bool = True


def _format_messages(messages: list[NopMessage]) -> str:
    parts = []
    for i, m in enumerate(messages, 1):
        parts.append(f"[{i}] type={m.type} published={m.published}\n{m.text}")
    return "\n\n".join(parts)


def answer_nop_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    fetch=fetch_messages,
) -> NopAnswer:
    client = client or OllamaClient()

    try:
        messages = fetch(query=_keyword(question), limit=5)
        # If a keyword search found nothing, fall back to the latest messages.
        if not messages:
            messages = fetch(query=None, limit=5)
    except NopError as exc:
        return NopAnswer(question=question, answer=f"NOP source unavailable: {exc}", ok=False)

    if not messages:
        return NopAnswer(
            question=question,
            answer="No NOP messages are available right now.",
            messages=[],
            ok=False,
        )

    chat_messages = prompts.build_nop_messages(question, _format_messages(messages))
    answer = client.chat(chat_messages, temperature=0.0)
    return NopAnswer(question=question, answer=answer.strip(), messages=messages, ok=True)
