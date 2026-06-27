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


def _acronym_chunks(question: str) -> list[RetrievedChunk]:
    """Exact acronym matches as high-confidence chunks (structured lookup).

    Complements vector search: short glossary entries get diluted in a large
    mixed corpus, so an exact CODE match is surfaced directly.
    """
    try:
        from aiu_chat.ingest.acronyms import lookup_acronyms

        hits = lookup_acronyms(question)
    except Exception:
        return []
    return [
        RetrievedChunk(
            text=f"{h['code']} stands for {h['definition']}.",
            source_url=h["source_url"],
            source_title="Acronyms",
            similarity=1.0,  # exact match
        )
        for h in hits
    ]


def answer_concept_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    retriever=retrieve,
) -> ConceptAnswer:
    client = client or OllamaClient()

    # Exact acronym matches first, then vector retrieval. Dedup by text.
    acronyms = _acronym_chunks(question)
    retrieved = retriever(question, client=client)
    seen = {a.text for a in acronyms}
    chunks = acronyms + [c for c in retrieved if c.text not in seen]

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
