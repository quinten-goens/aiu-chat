"""The concept path: question -> retrieve doc chunks -> grounded answer."""
from __future__ import annotations

from dataclasses import dataclass, field

from aiu_chat.agent import prompts
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.agent.retriever import RetrievedChunk, retrieve


@dataclass
class ConceptAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    ok: bool = True


def _format_excerpts(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] Source: {c.source_title} ({c.source_url})\n{c.text}")
    return "\n\n".join(parts)


def answer_concept_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    retriever=retrieve,
) -> ConceptAnswer:
    client = client or OllamaClient()
    chunks = retriever(question, client=client)

    if not chunks:
        return ConceptAnswer(
            question=question,
            answer="I don't have reference material that covers that.",
            sources=[],
            ok=False,
        )

    messages = prompts.build_concept_messages(question, _format_excerpts(chunks))
    answer = client.chat(messages, temperature=0.0)
    return ConceptAnswer(
        question=question, answer=answer.strip(), sources=chunks, ok=True
    )
