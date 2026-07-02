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
    TimeseriesResult,
    METRIC_ENDPOINTS,
    RANKING_CATEGORIES,
    fetch_metric,
    fetch_network,
    fetch_ranking,
    fetch_timeseries,
)

VALID_KINDS = {"country", "airport", "ansp", "aircraft_operator"}
VALID_PERIODS = {"DY", "WK", "MM", "Y2D"}
VALID_METRICS = set(METRIC_ENDPOINTS)


@dataclass
class TimeseriesAnswer:
    """The period-series result carried on a DataAppAnswer: the fetched daily
    series (one per metric), the optional manipulated/derived frame, the chart
    spec, and the final DataFrame the UI renders."""
    metric_line: str                 # human label, e.g. "traffic" or "delay, traffic"
    entity: str
    start: str
    end: str
    series: list                     # list[TimeseriesResult] (one per metric)
    dataframe: object = None         # pandas DataFrame the UI shows (raw or transformed)
    transform_sql: str | None = None  # the manipulator SQL, if one ran
    chart_spec: dict | None = None
    truncated: bool = False


@dataclass
class DataAppAnswer:
    question: str
    answer: str
    result: DataAppResult | None = None      # first/only entity (back-compat)
    results: list = None                      # all fetched entities (fan-out)
    ranking: RankingResult | None = None      # populated for ranking queries
    network: DataAppResult | None = None      # populated for whole-network queries
    timeseries: TimeseriesAnswer | None = None  # populated for period queries
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
    fetch_ts=fetch_timeseries,
) -> DataAppAnswer:
    from aiu_chat import config

    client = client or OllamaClient()

    # 1. LLM fills the query spec (it never builds the API calls).
    try:
        spec = client.chat_json(prompts.build_dataapp_extract_messages(question))
    except Exception as exc:
        return DataAppAnswer(question=question, answer=f"Could not parse the request: {exc}", ok=False)

    kind = (spec.get("query_kind") or "entity").lower()

    # A period series is dispatched first: it may carry SEVERAL metrics (in
    # `metrics`) rather than the single top-level `metric` the others use.
    if kind == "timeseries" and config.DATAAPP_TIMESERIES:
        return _answer_timeseries(question, spec, client, fetch_ts)

    metric = (spec.get("metric") or "").lower()
    if metric not in METRIC_ENDPOINTS:
        return DataAppAnswer(
            question=question,
            answer="I can't map that to the live Data App API (supported metrics: "
                   "traffic, delay, CO2, punctuality).",
            ok=False,
        )

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


# --- period time series (feature #6) ---------------------------------------

def _ts_metrics(spec: dict) -> list[str]:
    """The metrics a period query needs: the `metrics` list, else the single
    `metric`. Deduped, order-preserved, validated."""
    raw = spec.get("metrics")
    if not isinstance(raw, list) or not raw:
        raw = [spec.get("metric")]
    out: list[str] = []
    for m in raw:
        m = (m or "").lower()
        if m in VALID_METRICS and m not in out:
            out.append(m)
    return out


def _frame_name(metric: str) -> str:
    """The in-memory table name a metric's daily series is registered under."""
    return f"{metric}_ts"


def _answer_timeseries(question, spec, client, fetch_ts) -> DataAppAnswer:
    from aiu_chat import config

    import pandas as pd

    metrics = _ts_metrics(spec)
    if not metrics:
        return DataAppAnswer(
            question=question,
            answer="I couldn't tell which metric to chart over that period "
                   "(traffic, delay, CO2, or punctuality).", ok=False)

    start, end = spec.get("start"), spec.get("end")
    if not (_date({"date": start}) and _date({"date": end})):
        return DataAppAnswer(
            question=question,
            answer="I couldn't read the date range — give a start and end date "
                   "(e.g. from 1 January 2026 to 1 May 2026).", ok=False)
    start, end = _date({"date": start}), _date({"date": end})

    # One series per named entity (compare France vs Germany daily traffic on one
    # chart), or the whole network when none is named. Each (entity, metric) pair
    # is fetched; per-metric rows for ALL entities go into one long frame tagged
    # with an `entity` column, so the chart can split by it and the manipulator
    # can GROUP BY / PARTITION BY it. Cap the fan-out to be a polite scraper.
    targets = _extract_entities(spec) or [(None, None)]
    if not config.FANOUT:
        targets = targets[:1]
    else:
        targets = targets[: config.MAX_FANOUT]
    multi_entity = len(targets) > 1

    # per_metric[metric] -> list of TimeseriesResult (one per entity that had data)
    per_metric: dict[str, list[TimeseriesResult]] = {}
    series: list[TimeseriesResult] = []
    truncated = False
    errors: list[str] = []
    entity_names: list[str] = []
    for kind, query in targets:
        for metric in metrics:
            try:
                ts = fetch_ts(metric, start=start, end=end, kind=kind, query=query)
            except DataAppError as exc:
                who = query or "the network"
                errors.append(f"{who} {metric}: {exc}")
                continue
            if not ts.rows:
                who = query or "the network"
                errors.append(f"{who} {metric}: no data in period")
                continue
            series.append(ts)
            per_metric.setdefault(metric, []).append(ts)
            truncated = truncated or ts.truncated
            if ts.entity.name not in entity_names:
                entity_names.append(ts.entity.name)

    if not series:
        detail = "; ".join(errors) if errors else "no data"
        return DataAppAnswer(
            question=question,
            answer=f"No Data App series found for that period ({detail}).", ok=False)

    # Build one long frame per metric: [date, (entity,) value, avgValue]. The
    # `entity` column is only added when more than one entity actually returned
    # data, so single-entity/network frames stay exactly as before.
    include_entity = multi_entity and len(entity_names) > 1
    frames: dict = {}
    for metric, results in per_metric.items():
        parts = []
        for ts in results:
            df = pd.DataFrame(ts.rows)
            if include_entity:
                df.insert(1, "entity", ts.entity.name)
            parts.append(df)
        combined = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        combined = combined.sort_values("date").reset_index(drop=True)
        frames[_frame_name(metric)] = combined

    entity_name = ", ".join(entity_names) if include_entity else entity_names[0]
    metric_line = ", ".join(sorted({s.metric for s in series}, key=metrics.index))

    # Build the frame the user sees: raw single series, or (optionally) the
    # manipulated/derived result. The manipulator runs deterministic SQL — the
    # model never does the arithmetic.
    result_df, transform_sql = _maybe_manipulate(question, spec, frames, series, client)

    # Visualiser: a validated chart spec over the (possibly transformed) frame.
    chart_spec = _timeseries_chart_spec(question, result_df, client)

    # Narrate the series/result (grounded in the rows; a sample if long).
    rows_json = result_df.head(60).to_json(orient="records")
    messages = prompts.build_dataapp_timeseries_messages(
        question, metric_line, entity_name, start, end, rows_json, capped=truncated)
    answer = client.chat(messages, temperature=0.0).strip()
    if errors:
        answer += "\n\n_(No data for: " + "; ".join(errors) + ".)_"

    ts_answer = TimeseriesAnswer(
        metric_line=metric_line, entity=entity_name, start=start, end=end,
        series=series, dataframe=result_df, transform_sql=transform_sql,
        chart_spec=chart_spec, truncated=truncated,
    )
    return DataAppAnswer(question=question, answer=answer, timeseries=ts_answer, ok=True)


def _maybe_manipulate(question, spec, frames, series, client):
    """Run the manipulator over the daily frames when the question needs an
    aggregation/derivation (weekly/monthly, per-flight ratio, rolling avg, ...).

    Returns (result_dataframe, transform_sql_or_None). Best-effort: on any
    failure or NO_TRANSFORM, falls back to the raw daily frame. A transform is
    ALWAYS attempted when 2+ metrics were fetched (they must be combined into one
    figure), or when the spec's `transform` describes a manipulation."""
    import json as _json

    import pandas as pd

    from aiu_chat import config
    from aiu_chat.agent import agg_tool
    from aiu_chat.agent.catalog import get_catalog

    # The default frame the user sees if no transform runs: the single raw series
    # (or, for multi-metric, a date-joined wide frame so the table is coherent).
    # A frame may carry an `entity` column (multi-entity comparison); join on both
    # date AND entity then so metrics line up per (day, entity), never cross-join.
    if len(frames) == 1:
        raw_df = next(iter(frames.values()))
    else:
        raw_df = None
        for name, df in frames.items():
            metric = name[:-3]  # strip "_ts"
            renamed = df.rename(columns={"value": metric, "avgValue": f"{metric}_avg"})
            join_on = ["date"] + (["entity"] if "entity" in renamed.columns else [])
            raw_df = renamed if raw_df is None else raw_df.merge(
                renamed, on=join_on, how="outer")
        sort_cols = ["date"] + (["entity"] if "entity" in raw_df.columns else [])
        raw_df = raw_df.sort_values(sort_cols).reset_index(drop=True)

    transform = (spec.get("transform") or "").strip()
    need_transform = len(frames) > 1 or bool(transform)
    if not need_transform:
        return raw_df, None

    try:
        tables_desc = "\n".join(
            f"- {name}: {', '.join(map(str, df.columns))}" for name, df in frames.items())
        samples = {name: df.head(5).to_dict("records") for name, df in frames.items()}
        catalog = get_catalog()
        catalog_note = ", ".join(sorted(catalog.table_names))
        messages = prompts.build_manipulate_sql_messages(
            question, tables_desc, _json.dumps(samples, default=str), catalog_note)
        sql = client.chat(messages, temperature=0.0).strip()
        m = _strip_fences(sql)
        if not m or "NO_TRANSFORM" in m.upper():
            return raw_df, None
        agg = agg_tool.run_transform(m, frames, catalog=catalog)
        if agg.dataframe is None or agg.dataframe.empty:
            return raw_df, None
        return agg.dataframe, m
    except Exception:
        return raw_df, None  # manipulation is best-effort; raw series still shown


def _timeseries_chart_spec(question, df, client) -> dict | None:
    """A validated chart spec for the series frame (reuses the data path's chart
    step). Returns the raw spec dict (validated against the frame) or None."""
    try:
        import pandas as pd  # noqa: F401

        from aiu_chat.agent.chart import parse_spec

        if df is None or len(df) < 2:
            return None
        cols = list(df.columns)
        rows_json = df.head(20).to_json(orient="records")
        messages = prompts.build_chart_messages(question, cols, rows_json, force=True)
        spec = client.chat_json(messages)
        # Validate against the real frame; keep only a chart-worthy, valid spec.
        return spec if parse_spec(spec, df) is not None else None
    except Exception:
        return None


def _strip_fences(sql: str) -> str:
    import re
    m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1)
    return sql.strip().rstrip(";").strip()
