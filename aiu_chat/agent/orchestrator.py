"""Top-level agent: rewrite follow-ups, route, dispatch, and combine.

This ties the data path and the concept path together into one entry point,
`answer()`, used by both the CLI and the Streamlit UI. Every turn is logged
(rewritten question, route, SQL, sources) for debuggability — a core requirement.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from aiu_chat import config
from aiu_chat.agent import aliases, prompts
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
    route: str                  # primary route (first of `routes`), for UI chrome
    routes: list = field(default_factory=list)  # full route set (multi-source)
    data: DataAnswer | None = None
    concept: ConceptAnswer | None = None
    nop: NopAnswer | None = None
    dataapp: DataAppAnswer | None = None
    nm_live: NmLiveAnswer | None = None
    answer: str = ""
    sources: list = field(default_factory=list)
    needs_clarification: bool = False  # True when the turn is a clarifying question
    aggregate: object = None  # optional cross-frame AggResult (feature #4)
    aggregate_answer: str = ""  # narrated prose of the aggregate result
    sub_turns: list = field(default_factory=list)  # per-part turns (decomposed #5)


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


def decompose_question(question: str, client: OllamaClient) -> list[str]:
    """Split a compound question into independent standalone sub-questions.

    Returns a list of 1..MAX_SUBQUESTIONS questions in the order asked. Most
    questions are single -> a one-element list (the question itself). Best-effort
    and conservative: any failure, an empty/malformed result, or a single part
    falls back to `[question]`, so decomposition never breaks a plain question."""
    try:
        result = client.chat_json(prompts.build_decompose_messages(question))
    except Exception:
        return [question]
    raw = result.get("questions")
    if not isinstance(raw, list):
        return [question]
    parts = [str(q).strip() for q in raw if isinstance(q, str) and str(q).strip()]
    if len(parts) <= 1:
        return [question]  # single part -> answer the original question verbatim
    return parts[: config.MAX_SUBQUESTIONS]


def route_question(question: str, client: OllamaClient) -> str:
    """Classify a standalone question to a single primary route (back-compat).

    Prefer `plan_routes()` for multi-source. This returns the first planned route
    so existing callers/tests keep working with either router output shape."""
    try:
        result = client.chat_json(prompts.build_router_messages(question))
        return _normalise_routes(result, max_routes=1)[0]
    except Exception:
        return "data"  # default to the data path if routing fails


# Routes that actually gather/answer (a set may contain these); `both` expands to
# data+concept, `none`/`catalog` are terminal and never combined.
_COMBINABLE_ROUTES = {"data", "concept", "dataapp", "nm_live", "nop"}


def _normalise_routes(result: dict, max_routes: int) -> list[str]:
    """Turn a router JSON result into a validated, de-duplicated route list.

    Accepts either `routes: [...]` (new) or `route: "..."` (old); expands `both`
    to [data, concept]; keeps order; caps length. Returns ["data"] as a safe
    default when nothing valid is present."""
    raw: list[str] = []
    if isinstance(result.get("routes"), list):
        raw = [str(r).lower() for r in result["routes"]]
    elif result.get("route") is not None:
        raw = [str(result["route"]).lower()]

    expanded: list[str] = []
    for r in raw:
        if r == "both":
            expanded += ["data", "concept"]
        else:
            expanded.append(r)

    # A terminal route wins alone and is never combined.
    if "none" in expanded:
        return ["none"]
    if "catalog" in expanded:
        return ["catalog"]

    seen: list[str] = []
    for r in expanded:
        if r in _COMBINABLE_ROUTES and r not in seen:
            seen.append(r)
    seen = seen[:max_routes]
    return seen or ["data"]


def plan_routes(question: str, client: OllamaClient, *, max_routes: int) -> list[str]:
    """Return the ordered set of routes to answer `question` from (1..max_routes).

    Multi-source planning: the router may name several sources. Defensive — any
    failure or malformed output falls back to the single best route, so a set is
    never worse than today's one-route behaviour."""
    try:
        result = client.chat_json(prompts.build_router_messages(question))
        return _normalise_routes(result, max_routes)
    except Exception:
        return ["data"]


# Routes where a missing subject (which airport/ANSP/country?) blocks an answer.
_CLARIFIABLE_ROUTES = {"data", "both", "dataapp", "nm_live"}


def _candidate_entity_names(question: str) -> list[str]:
    """Multi-word proper-noun-ish spans from the question, longest first.

    Cheap and deliberately over-generous: `aliases.resolve_near_miss` is the
    authority on what is actually a near-miss, so a false candidate costs
    nothing but a dict lookup."""
    spans = re.findall(
        r"\b([A-Z][\w’'-]*(?:\s+(?:[A-Z][\w’'-]*|ACC|UAC|FIR))*)", question)
    return sorted({s.strip() for s in spans if s.strip()}, key=len, reverse=True)


def needs_clarification(question: str, route: str, client: OllamaClient) -> str | None:
    """Return a single clarifying question if an essential detail is missing,
    else None. Conservative: only fires when the agent genuinely can't proceed.
    Failures default to None (proceed) so a hiccup never blocks an answer."""
    # A named ACC/UAC is a near-miss for a real FIR/ANSP entity. Three logged
    # conversations about "Athens ACC" were refused outright even though Greece
    # FIR holds the answer — ask instead of failing closed. Checked BEFORE the
    # route guard: the refusals came back on routes that never reach the LLM
    # clarification below.
    for token in _candidate_entity_names(question):
        nm = aliases.resolve_near_miss(token)
        if nm is not None:
            return nm.question()

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
    "dataapp": "Querying the EUROCONTROL Data App…",
    "nm_live": "Fetching the live network snapshot…",
}

# Short reason for each routing choice — shown as the step detail.
_ROUTE_WHY = {
    "data": "historical figures from the local datasets",
    "both": "needs a figure and an explanation",
    "concept": "a definition / methodology question",
    "nop": "the operational situation in NOP messages",
    "dataapp": "daily Data App figures — a specific day, the network, or a ranking",
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
    # Default client is provider-aware: honours LOCAL/AIU_MODEL_TIER so the same
    # entry point runs on local Ollama or cloud OpenAI without code changes.
    if client is None:
        from aiu_chat.agent.llm import build_client
        client = build_client()
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

    # Decompose a compound question ("flights on the network on 10 Mar 2026 AND
    # punctuality at LEBL on 10 Mar 2025", "traffic in France and delays in
    # Germany") into independent standalone sub-questions. Each sub-question is
    # answered through the FULL route pipeline, then the grounded sub-answers are
    # synthesised into one. A single-part question yields one sub-question and
    # behaves exactly as before.
    subqs = decompose_question(standalone, client) if config.DECOMPOSE else [standalone]
    if len(subqs) > 1:
        status("Planning the answer…",
               "Split into "
               + str(len(subqs))
               + " parts:\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(subqs, 1)))
        return _answer_compound(question, standalone, subqs,
                                client=client, catalog=catalog, status=status)

    return _answer_single(question, standalone, client=client, catalog=catalog, status=status)


def _answer_single(
    question: str,
    standalone: str,
    *,
    client: OllamaClient,
    catalog: Catalog,
    status,
) -> Turn:
    """Route, dispatch, and combine a SINGLE standalone question into one Turn.

    This is the per-question pipeline: it is called once for a plain question and
    once per sub-question for a decomposed compound one."""
    status("Choosing the best source…")
    max_routes = config.MAX_ROUTES if config.MULTI_SOURCE else 1
    routes = plan_routes(standalone, client, max_routes=max_routes)
    # The Turn's `route` keeps a single primary label (first route) for the UI's
    # existing route chrome; the full set drives dispatch.
    primary = routes[0]
    logger.info("turn routes=%s | q=%r | standalone=%r", routes, question, standalone)
    if len(routes) > 1:
        status("Choosing the best source…",
               f"Routes: **{' + '.join(routes)}** (multi-source)")
    else:
        status("Choosing the best source…", f"Route: **{primary}** — {_ROUTE_WHY.get(primary, '')}")

    turn = Turn(question=question, standalone_question=standalone, route=primary)
    turn.routes = routes

    # Out-of-scope questions decline cleanly rather than forcing a path.
    if primary == "none":
        turn.answer = (
            "That's outside what I can help with — I answer questions about "
            "European air navigation performance (traffic, delays, efficiency, "
            "emissions, the live network, and EUROCONTROL methodology)."
        )
        return turn

    # Ask one clarifying question if an essential detail is missing (conservative
    # — only when we genuinely can't proceed). Checked once, on the primary route,
    # to avoid nagging. The user's reply flows back as a follow-up next turn.
    status("Checking the question…")
    clarification = needs_clarification(standalone, primary, client)
    if clarification:
        logger.info("  needs clarification: %r", clarification)
        turn.needs_clarification = True
        turn.answer = clarification
        return turn

    # Dispatch every route in the set; each fills its own slot on the Turn.
    status(_ROUTE_STATUS.get(primary, "Gathering data…"))
    if "data" in routes:
        turn.data = answer_data_question(standalone, client=client, catalog=catalog)
        logger.info("  data ok=%s sql=%r", turn.data.ok, turn.data.sql)
        if turn.data.sql:
            status(_ROUTE_STATUS.get("data", "Gathering data…"),
                   f"SQL:\n```sql\n{turn.data.sql}\n```")
    if "concept" in routes:
        turn.concept = answer_concept_question(standalone, client=client)
        logger.info(
            "  concept ok=%s sources=%s",
            turn.concept.ok,
            [s.source_title for s in turn.concept.sources],
        )
        srcs = ", ".join(sorted({s.source_title for s in turn.concept.sources}))
        if srcs:
            status("Searching the reference docs & PDFs…", f"Retrieved from: {srcs}")
    if "nop" in routes:
        turn.nop = answer_nop_question(standalone, client=client)
        logger.info("  nop ok=%s messages=%d", turn.nop.ok, len(turn.nop.messages))
        status("Fetching NOP messages…", f"Found {len(turn.nop.messages)} relevant message(s)")
    if "dataapp" in routes:
        turn.dataapp = answer_dataapp_question(standalone, client=client)
        da = turn.dataapp
        if da.timeseries is not None:
            ts = da.timeseries
            detail = (f"Series: **{ts.metric_line}** for {ts.entity}, "
                      f"{ts.start}→{ts.end}"
                      + (" · transformed" if ts.transform_sql else ""))
        elif da.ranking is not None:
            detail = (f"Ranking: **{da.ranking.category}** by {da.ranking.metric} "
                      f"(scope: {da.ranking.scope}, {da.ranking.sync_date})")
        elif da.network is not None:
            detail = f"Network **{da.network.metric}** as of {da.network.sync_date}"
        elif da.result is not None:
            detail = f"Resolved entity: **{da.result.entity.name}** (Data App, {da.result.sync_date})"
        else:
            detail = None
        logger.info("  dataapp ok=%s | %s", da.ok, detail)
        if detail:
            status("Looking up the Data App figures…", detail)
    if "nm_live" in routes:
        turn.nm_live = answer_nm_question(standalone, client=client)
        logger.info("  nm_live ok=%s", turn.nm_live.ok)
        status("Fetching the live network snapshot…", "Live Network Manager data")

    # Optional cross-frame aggregation (#4): compute a combined total / diff /
    # ranking across the frames this turn produced — via a deterministic SQL
    # query, never model arithmetic. Off by default; guarded.
    if config.AGGREGATION:
        _maybe_aggregate(turn, standalone, client, status)

    turn.answer, turn.sources = _combine(turn, client=client, status=status)
    return turn


def _answer_compound(
    question: str,
    standalone: str,
    subqs: list[str],
    *,
    client: OllamaClient,
    catalog: Catalog,
    status,
) -> Turn:
    """Answer each independent sub-question through the full pipeline, then merge
    the grounded sub-answers into one synthesised Turn.

    The merged Turn keeps every sub-turn (for the UI's per-part charts/tables and
    for logging) and unions their sources. If a sub-question itself needs
    clarification, we ask that one question and stop — the user's reply re-runs
    the whole compound question next turn."""
    sub_turns: list[Turn] = []
    for i, sub in enumerate(subqs, 1):
        status("Answering part " + str(i) + f" of {len(subqs)}…", f"*{sub}*")
        # Each sub-question is already standalone -> its own question and
        # standalone_question are the sub text (no further rewrite).
        st = _answer_single(sub, sub, client=client, catalog=catalog, status=status)
        # A clarifying sub-question can't be silently merged — surface it and stop.
        if st.needs_clarification:
            merged = Turn(question=question, standalone_question=standalone, route=st.route)
            merged.needs_clarification = True
            merged.answer = st.answer
            return merged
        sub_turns.append(st)

    # Merge: union routes/sources, collect (label, grounded answer) parts.
    routes: list[str] = []
    for st in sub_turns:
        for r in st.routes or [st.route]:
            if r not in routes:
                routes.append(r)

    primary = sub_turns[0].route if sub_turns else "data"
    turn = Turn(question=question, standalone_question=standalone, route=primary)
    turn.routes = routes
    turn.sub_turns = sub_turns

    sources: list = []
    seen_urls: set = set()
    for st in sub_turns:
        for s in st.sources:
            url = getattr(s, "source_url", None)
            if url not in seen_urls:
                seen_urls.add(url)
                sources.append(s)
    turn.sources = sources

    # Synthesise the sub-answers into one, each labelled by its sub-question so the
    # synthesis prompt keeps every figure with its own context. Never recomputes.
    labelled = [(f"Part {i}: {st.standalone_question}", st.answer)
                for i, st in enumerate(sub_turns, 1)]
    if len(labelled) == 1:
        turn.answer = labelled[0][1]
    else:
        synth = _synthesize(standalone, labelled, client, status) if config.MULTI_SOURCE else None
        turn.answer = synth or "\n\n".join(
            f"**{st.standalone_question}**\n\n{st.answer}" for st in sub_turns)
    logger.info("compound turn parts=%d routes=%s", len(sub_turns), routes)
    return turn


def _maybe_aggregate(turn, question: str, client, status) -> None:
    """Run a deterministic cross-frame aggregation when the question asks for one.

    Best-effort: any failure leaves the per-frame answers untouched (never raises
    into the turn)."""
    try:
        from aiu_chat.agent import agg_tool
        from aiu_chat.agent.sql_tool import UnsafeSQLError
        import json as _json

        frames = agg_tool.collect_frames(turn)
        if not frames:
            return
        # Only worth aggregating across 2+ rows total.
        if sum(len(df) for df in frames.values()) < 2:
            return

        views_desc = "\n".join(f"- {n}: {', '.join(map(str, df.columns))}"
                               for n, df in frames.items())
        samples = {n: df.head(10).to_dict("records") for n, df in frames.items()}
        status("Combining the figures…", "Computing a cross-source total/comparison")
        messages = prompts.build_agg_sql_messages(
            question, views_desc, _json.dumps(samples, default=str))
        sql = client.chat(messages, temperature=0.0).strip()
        # Strip fences if any.
        m = re.search(r"```(?:sql)?\s*(.*?)```", sql, re.DOTALL | re.IGNORECASE)
        if m:
            sql = m.group(1).strip()
        sql = sql.rstrip(";").strip()
        if not sql or "NO_AGG" in sql.upper():
            return
        agg = agg_tool.run_aggregation(sql, frames)
        turn.aggregate = agg
        logger.info("  aggregate rows=%s sql=%r", agg.row_count, agg.sql)

        # Narrate the executed aggregation result (grounded; no recomputation).
        rows_json = agg.dataframe.head(50).to_json(orient="records")
        ans_messages = prompts.build_answer_messages(question, agg.sql, rows_json, None)
        turn.aggregate_answer = client.chat(ans_messages, temperature=0.0).strip()
    except Exception as exc:  # UnsafeSQLError, execution error, anything
        logger.info("  aggregation skipped: %s", exc)


def _combine(turn: Turn, *, client: OllamaClient | None = None, status=None) -> tuple[str, list]:
    """Merge path outputs into one answer + source list.

    For a single source this is just that source's answer. For 2+ grounded
    sources it optionally runs a synthesis pass (LLM) that stitches the already-
    grounded sub-answers into one cohesive answer WITHOUT recomputing any figure;
    on any problem it falls back to blank-line concatenation.
    """
    # (label, text) parts in a stable order, so the synthesis prompt and the
    # fallback read consistently.
    labelled: list[tuple[str, str]] = []
    sources: list = []

    # A cross-frame aggregate (#4) is the direct answer — lead with it.
    if getattr(turn, "aggregate_answer", ""):
        labelled.append(("Combined figure", turn.aggregate_answer))
    if turn.concept is not None and turn.concept.ok:
        labelled.append(("Definition / methodology", turn.concept.answer))
        sources.extend(turn.concept.sources)
    if turn.data is not None:
        labelled.append(("Historical data", turn.data.answer))
    if turn.dataapp is not None:
        labelled.append(("EUROCONTROL Data App (daily)", turn.dataapp.answer))
    if turn.nm_live is not None:
        labelled.append(("Live network", turn.nm_live.answer))
    if turn.nop is not None:
        labelled.append(("Network operations (NOP)", turn.nop.answer))

    if not labelled:
        # Nothing useful from any path.
        for sub in (turn.data, turn.concept, turn.nop, turn.dataapp, turn.nm_live):
            if sub is not None:
                return sub.answer, sources
        return "I couldn't find an answer to that.", sources

    parts = [text for _, text in labelled]
    if len(labelled) == 1:
        return parts[0], sources

    # 2+ sources: synthesize (guarded) when enabled and a client is available.
    if config.MULTI_SOURCE and client is not None:
        synth = _synthesize(turn.standalone_question, labelled, client, status)
        if synth:
            return synth, sources

    return "\n\n".join(parts), sources


def _synthesize(question: str, labelled: list[tuple[str, str]], client, status) -> str | None:
    """LLM pass that stitches grounded sub-answers into one answer, preserving
    every figure verbatim. Returns None on any failure (caller falls back)."""
    try:
        if status is not None:
            status("Combining the sources…", "Synthesising a single answer")
        messages = prompts.build_synthesis_messages(question, labelled)
        out = client.chat(messages, temperature=0.0).strip()
        return out or None
    except Exception:
        return None
