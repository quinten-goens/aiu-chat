"""Data App path: LLM picks metric+entity, deterministic resolver fetches, narrate."""
from __future__ import annotations

import json
from dataclasses import dataclass

from aiu_chat.agent import prompts
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.sources.dataapp import (
    DataAppError,
    DataAppResult,
    METRIC_ENDPOINTS,
    fetch_metric,
)

VALID_KINDS = {"country", "airport", "ansp", "aircraft_operator"}


@dataclass
class DataAppAnswer:
    question: str
    answer: str
    result: DataAppResult | None = None
    ok: bool = True


def answer_dataapp_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    fetch=fetch_metric,
) -> DataAppAnswer:
    client = client or OllamaClient()

    # 1. LLM extracts metric + entity (it never builds the API calls itself).
    try:
        spec = client.chat_json(prompts.build_dataapp_extract_messages(question))
    except Exception as exc:
        return DataAppAnswer(question=question, answer=f"Could not parse the request: {exc}", ok=False)

    metric = (spec.get("metric") or "").lower()
    kind = (spec.get("entity_kind") or "").lower()
    entity = (spec.get("entity") or "").strip()
    if metric not in METRIC_ENDPOINTS or kind not in VALID_KINDS or not entity:
        return DataAppAnswer(
            question=question,
            answer="I can't map that to the live Data App API (supported metrics: "
                   "traffic, delay, CO2, punctuality — for a country, airport, ANSP, or airline).",
            ok=False,
        )

    # 2. Deterministic 3-hop resolve + fetch.
    try:
        result = fetch(metric, kind, entity)
    except DataAppError as exc:
        return DataAppAnswer(question=question, answer=f"Data App lookup failed: {exc}", ok=False)

    if not result.records:
        return DataAppAnswer(
            question=question,
            answer=f"No current {metric} data found for {result.entity.name}.",
            result=result,
            ok=False,
        )

    # 3. Narrate grounded in the fetched records.
    messages = prompts.build_dataapp_answer_messages(
        question, metric, result.entity.name, result.sync_date,
        json.dumps(result.records),
    )
    answer = client.chat(messages, temperature=0.0)
    return DataAppAnswer(question=question, answer=answer.strip(), result=result, ok=True)
