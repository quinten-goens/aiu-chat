"""Serialize a completed `Turn` into a PocketBase `chat_turns` record.

Kept separate from the orchestrator so the logging concern doesn't clutter the
answer flow, and separate from the PocketBase client so the wire layer stays
dumb. Everything the UI can render is captured: the raw + rewritten question,
route, final prose, generated SQL and its result table, the chart spec, cited
sources, and the live-source payloads (NOP / Data App / NM live).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


def _pb_now() -> str:
    """PocketBase date format (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%fZ")


def _json_safe(value: Any) -> Any:
    """Make a value JSON-serialisable and free of NaN/Inf (PocketBase rejects them)."""
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _dataframe_records(df, *, max_rows: int = 500) -> list[dict]:
    """Convert a result DataFrame to a capped list of JSON-safe row dicts."""
    if df is None or df.empty:
        return []
    head = df.head(max_rows)
    # to_dict on a stringified copy keeps dates/decimals JSON-friendly.
    records = head.astype(object).where(head.notna(), None).to_dict(orient="records")
    return _json_safe(records)


def _sources_payload(sources: list) -> list[dict]:
    out = []
    for s in sources or []:
        out.append({
            "title": getattr(s, "source_title", None),
            "url": getattr(s, "source_url", None),
            "text": (getattr(s, "text", "") or "")[:1000],
            "score": getattr(s, "score", None),
        })
    return _json_safe(out)


def _live_payload(turn) -> dict | None:
    """Capture whichever live source produced this turn (NOP / Data App / NM)."""
    payload: dict = {}

    nop = getattr(turn, "nop", None)
    if nop is not None and getattr(nop, "messages", None):
        payload["nop"] = [
            {"id": m.id, "type": m.type, "published": m.published, "text": m.text[:4000]}
            for m in nop.messages
        ]

    dataapp = getattr(turn, "dataapp", None)
    if dataapp is not None:
        results = getattr(dataapp, "results", None) or (
            [dataapp.result] if getattr(dataapp, "result", None) is not None else [])
        entries = []
        for r in results:
            ent = getattr(r, "entity", None)
            entries.append({
                "entity": getattr(ent, "name", None),
                "entity_kind": getattr(ent, "kind", None),
                "sync_date": getattr(r, "sync_date", None),
                "metric": getattr(r, "metric", None),
                "records": _json_safe(getattr(r, "records", None)),
            })
        net = getattr(dataapp, "network", None)
        if net is not None:
            entries.append({
                "entity": "Network", "sync_date": getattr(net, "sync_date", None),
                "metric": getattr(net, "metric", None),
                "records": _json_safe(getattr(net, "records", None)),
            })
        rank = getattr(dataapp, "ranking", None)
        if rank is not None:
            entries.append({
                "ranking_category": getattr(rank, "category", None),
                "scope": getattr(rank, "scope", None),
                "sync_date": getattr(rank, "sync_date", None),
                "metric": getattr(rank, "metric", None),
                "rows": _json_safe(getattr(rank, "rows", None)),
            })
        if entries:
            payload["dataapp"] = entries

    nm = getattr(turn, "nm_live", None)
    if nm is not None and getattr(nm, "snapshot", None) is not None:
        s = nm.snapshot
        payload["nm_live"] = {
            "airborne": getattr(s, "airborne", None),
            "total_delay_min": getattr(s, "total_delay_min", None),
            "top_delays": _json_safe(getattr(s, "top_delays", None)),
            "active_regulations": len(getattr(s, "regulations", []) or []),
        }

    return _json_safe(payload) or None


def _evidence(turn) -> dict:
    """The grounded artifacts of ONE turn: SQL, chart spec, result table, and any
    live-source payload. Reused for a plain turn and each sub-turn of a compound
    (decomposed) answer, so no per-part figure is lost from the log."""
    out: dict = {}
    data = getattr(turn, "data", None)
    result = getattr(data, "result", None) if data is not None else None
    df = getattr(result, "dataframe", None) if result is not None else None
    if data is not None:
        out["sql"] = getattr(data, "sql", None) or ""
        out["chart_spec"] = _json_safe(getattr(data, "chart_spec", None))
    if result is not None:
        out["row_count"] = int(getattr(result, "row_count", 0) or 0)
        out["truncated"] = bool(getattr(result, "truncated", False))
        out["result_table"] = _dataframe_records(df)
    live = _live_payload(turn)
    if live:
        out["live_payload"] = live
    return out


def build_turn_record(turn, *, turn_index: int, model_tier: str | None,
                      latency_ms: int | None, error: str | None = None) -> dict:
    """Build the `chat_turns` record body for a completed Turn."""
    routes = getattr(turn, "routes", None) or [getattr(turn, "route", "") or ""]
    record: dict = {
        "turn_index": turn_index,
        "created_at": _pb_now(),
        "question": getattr(turn, "question", "") or "",
        "standalone_question": getattr(turn, "standalone_question", "") or "",
        # For a compound turn the primary route alone understates it — record the
        # union so the log reflects every path used.
        "route": "+".join(r for r in routes if r) or (getattr(turn, "route", "") or ""),
        "needs_clarification": bool(getattr(turn, "needs_clarification", False)),
        "answer": getattr(turn, "answer", "") or "",
        "model_tier": model_tier or "",
    }
    if latency_ms is not None:
        record["latency_ms"] = int(latency_ms)
    if error:
        record["error"] = str(error)[:20000]

    sub_turns = getattr(turn, "sub_turns", None)
    if sub_turns:
        # Compound answer: capture each part's question + route + evidence so the
        # viewer can show every sub-answer's SQL/table/live payload.
        record["sub_turns"] = _json_safe([
            {
                "question": getattr(st, "standalone_question", "") or "",
                "route": "+".join(getattr(st, "routes", None)
                                  or [getattr(st, "route", "") or ""]),
                "answer": getattr(st, "answer", "") or "",
                **_evidence(st),
            }
            for st in sub_turns
        ])
    else:
        record.update(_evidence(turn))

    record["sources"] = _sources_payload(getattr(turn, "sources", []))
    return record
