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
    result: DataAppResult | None = None      # first/only entity (back-compat)
    results: list = None                      # all fetched entities (fan-out)
    ok: bool = True

    def __post_init__(self):
        if self.results is None:
            self.results = [self.result] if self.result is not None else []


def _extract_entities(spec: dict) -> list[tuple[str, str]]:
    """Normalise the extract JSON into a list of (kind, entity) pairs.

    Accepts the new `entities: [...]` list and the old single `entity`/
    `entity_kind` (back-compat). Invalid/empty entries are dropped."""
    pairs: list[tuple[str, str]] = []
    raw = spec.get("entities")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                k = (item.get("entity_kind") or "").lower()
                e = (item.get("entity") or "").strip()
                if k in VALID_KINDS and e:
                    pairs.append((k, e))
    else:
        k = (spec.get("entity_kind") or "").lower()
        e = (spec.get("entity") or "").strip()
        if k in VALID_KINDS and e:
            pairs.append((k, e))
    # De-duplicate, preserve order.
    seen = []
    for p in pairs:
        if p not in seen:
            seen.append(p)
    return seen


def answer_dataapp_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    fetch=fetch_metric,
) -> DataAppAnswer:
    from aiu_chat import config

    client = client or OllamaClient()

    # 1. LLM extracts metric + the list of entities (it never builds the API calls).
    try:
        spec = client.chat_json(prompts.build_dataapp_extract_messages(question))
    except Exception as exc:
        return DataAppAnswer(question=question, answer=f"Could not parse the request: {exc}", ok=False)

    metric = (spec.get("metric") or "").lower()
    entities = _extract_entities(spec)
    if metric not in METRIC_ENDPOINTS or not entities:
        return DataAppAnswer(
            question=question,
            answer="I can't map that to the live Data App API (supported metrics: "
                   "traffic, delay, CO2, punctuality — for a country, airport, ANSP, or airline).",
            ok=False,
        )

    # Fan-out is capped; when disabled, only the first entity is used.
    if not config.FANOUT:
        entities = entities[:1]
    else:
        entities = entities[: config.MAX_FANOUT]

    # 2. Deterministic per-entity fetch loop. One entity's failure never sinks the
    # others — failures are collected and reported alongside the successes.
    results: list[DataAppResult] = []
    errors: list[str] = []
    for kind, entity in entities:
        try:
            result = fetch(metric, kind, entity)
        except DataAppError as exc:
            errors.append(f"{entity}: {exc}")
            continue
        if not result.records:
            errors.append(f"{result.entity.name}: no current {metric} data")
            continue
        results.append(result)

    if not results:
        detail = "; ".join(errors) if errors else "no data"
        return DataAppAnswer(
            question=question,
            answer=f"No current {metric} data found ({detail}).",
            ok=False,
        )

    # 3. Narrate grounded in the merged records (each tagged with its entity).
    merged = []
    for r in results:
        for rec in r.records:
            merged.append({"entity": r.entity.name, **rec})
    entity_label = ", ".join(r.entity.name for r in results)
    sync_date = results[0].sync_date
    messages = prompts.build_dataapp_answer_messages(
        question, metric, entity_label, sync_date, json.dumps(merged),
    )
    answer = client.chat(messages, temperature=0.0).strip()
    if errors:
        answer += "\n\n_(No data for: " + "; ".join(errors) + ".)_"

    return DataAppAnswer(
        question=question, answer=answer,
        result=results[0], results=results, ok=True,
    )
