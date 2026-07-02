"""Data App path: LLM picks the query shape (entity / network / ranking) and its
inputs; deterministic resolvers fetch; the model narrates the fetched rows.

Three shapes, all off the same sync model (see sources/dataapp.py):
  * entity   — a figure for one or more named entities (with per-source fan-out).
  * network  — a whole-network figure, optionally on a specific date.
  * ranking  — a top/bottom-N ("which airport/country/pair/operator is highest").
The LLM never builds API calls; it only fills a small JSON spec.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from aiu_chat.agent import prompts
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.sources.dataapp import (
    DataAppError,
    DataAppResult,
    RankingResult,
    METRIC_ENDPOINTS,
    RANKING_CATEGORIES,
    fetch_metric,
    fetch_network,
    fetch_ranking,
)

VALID_KINDS = {"country", "airport", "ansp", "aircraft_operator"}
VALID_PERIODS = {"DY", "WK", "MM", "Y2D"}


@dataclass
class DataAppAnswer:
    question: str
    answer: str
    result: DataAppResult | None = None      # first/only entity (back-compat)
    results: list = None                      # all fetched entities (fan-out)
    ranking: RankingResult | None = None      # populated for ranking queries
    network: DataAppResult | None = None      # populated for whole-network queries
    ok: bool = True

    def __post_init__(self):
        if self.results is None:
            self.results = [self.result] if self.result is not None else []


def _extract_entities(spec: dict) -> list[tuple[str, str]]:
    """Normalise the extract JSON into a list of (kind, entity) pairs.

    Accepts the `entities: [...]` list and the old single `entity`/`entity_kind`
    (back-compat). Invalid/empty entries are dropped, order preserved, deduped."""
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
    seen = []
    for p in pairs:
        if p not in seen:
            seen.append(p)
    return seen


def _period(spec: dict) -> str:
    p = str(spec.get("period") or "DY").upper()
    return p if p in VALID_PERIODS else "DY"


def _date(spec: dict) -> str | None:
    d = spec.get("date")
    if isinstance(d, str) and len(d) == 10 and d[4] == "-" and d[7] == "-":
        return d
    return None


def answer_dataapp_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    fetch=fetch_metric,
    fetch_net=fetch_network,
    fetch_rank=fetch_ranking,
) -> DataAppAnswer:
    client = client or OllamaClient()

    # 1. LLM fills the query spec (it never builds the API calls).
    try:
        spec = client.chat_json(prompts.build_dataapp_extract_messages(question))
    except Exception as exc:
        return DataAppAnswer(question=question, answer=f"Could not parse the request: {exc}", ok=False)

    metric = (spec.get("metric") or "").lower()
    if metric not in METRIC_ENDPOINTS:
        return DataAppAnswer(
            question=question,
            answer="I can't map that to the live Data App API (supported metrics: "
                   "traffic, delay, CO2, punctuality).",
            ok=False,
        )

    kind = (spec.get("query_kind") or "entity").lower()
    if kind == "ranking":
        return _answer_ranking(question, metric, spec, client, fetch_rank)
    if kind == "network":
        return _answer_network(question, metric, spec, client, fetch_net)
    return _answer_entities(question, metric, spec, client, fetch)


def _answer_network(question, metric, spec, client, fetch_net) -> DataAppAnswer:
    date = _date(spec)
    try:
        result = fetch_net(metric, date=date)
    except DataAppError as exc:
        return DataAppAnswer(question=question, answer=f"No network {metric} data: {exc}", ok=False)
    if not result.records:
        return DataAppAnswer(
            question=question,
            answer=f"No network {metric} data found for that period.", ok=False)
    messages = prompts.build_dataapp_network_messages(
        question, metric, result.sync_date, json.dumps(result.records))
    answer = client.chat(messages, temperature=0.0).strip()
    return DataAppAnswer(question=question, answer=answer, network=result, ok=True)


def _answer_ranking(question, metric, spec, client, fetch_rank) -> DataAppAnswer:
    from aiu_chat.sources.dataapp import METRIC_ENDPOINTS as _ME
    category = (spec.get("ranking_category") or "").lower()
    if category not in RANKING_CATEGORIES:
        return DataAppAnswer(
            question=question,
            answer="I couldn't tell what to rank (airports, countries, airport "
                   "pairs, or airlines).", ok=False)
    if _ME[metric][1] is None:
        return DataAppAnswer(
            question=question,
            answer=f"The Data App API doesn't expose {metric} rankings.", ok=False)

    scope_kind = (spec.get("scope_kind") or "").lower() or None
    scope = (spec.get("scope") or "").strip() or None
    if scope_kind not in VALID_KINDS:
        scope_kind = None
    order = str(spec.get("order") or "highest").lower()
    both = order == "both"
    ascending = order in ("lowest", "asc", "ascending", "least")
    date = _date(spec)
    period = _period(spec)

    try:
        # A "both extremes" question needs the WHOLE ranking so the tail is the
        # true minimum, not the bottom of a top-N slice — fetch a large window.
        ranking = fetch_rank(
            metric, category,
            scope_kind=scope_kind, scope_query=scope,
            date=date, date_range=period, ascending=ascending,
            limit=200 if both else 15,
        )
    except DataAppError as exc:
        return DataAppAnswer(question=question, answer=f"Ranking lookup failed: {exc}", ok=False)
    if not ranking.rows:
        return DataAppAnswer(
            question=question,
            answer=f"No {metric} ranking data found for that {category} query.", ok=False)

    if both:
        # Ranking is sorted highest-first: head = highest, tail = lowest. Give the
        # narration only those two extremes (labelled) so it can't misread a slice.
        direction = "both"
        rows_json = json.dumps({
            "highest": ranking.rows[0],
            "lowest": ranking.rows[-1],
        })
    else:
        direction = "lowest" if ascending else "highest"
        rows_json = json.dumps(ranking.rows)

    messages = prompts.build_dataapp_ranking_messages(
        question, metric, category, ranking.scope, ranking.sync_date,
        period, direction, rows_json,
    )
    answer = client.chat(messages, temperature=0.0).strip()
    return DataAppAnswer(question=question, answer=answer, ranking=ranking, ok=True)


def _answer_entities(question, metric, spec, client, fetch) -> DataAppAnswer:
    from aiu_chat import config

    entities = _extract_entities(spec)
    if not entities:
        return DataAppAnswer(
            question=question,
            answer="I can't map that to the live Data App API (name a country, "
                   "airport, ANSP, or airline — or ask for a network figure or ranking).",
            ok=False,
        )

    if not config.FANOUT:
        entities = entities[:1]
    else:
        entities = entities[: config.MAX_FANOUT]

    date = _date(spec)
    results: list[DataAppResult] = []
    errors: list[str] = []
    for ekind, entity in entities:
        try:
            result = fetch(metric, ekind, entity, date=date)
        except DataAppError as exc:
            errors.append(f"{entity}: {exc}")
            continue
        if not result.records:
            errors.append(f"{result.entity.name}: no {metric} data")
            continue
        results.append(result)

    if not results:
        detail = "; ".join(errors) if errors else "no data"
        return DataAppAnswer(
            question=question,
            answer=f"No {metric} data found ({detail}).", ok=False)

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
