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
    if dataapp is not None and getattr(dataapp, "result", None) is not None:
        r = dataapp.result
        ent = getattr(r, "entity", None)
        payload["dataapp"] = {
            "entity": getattr(ent, "name", None),
            "entity_kind": getattr(ent, "kind", None),
            "sync_date": getattr(r, "sync_date", None),
            "metric": getattr(r, "metric", None),
            "values": _json_safe(getattr(r, "values", None)),
        }

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


def build_turn_record(turn, *, turn_index: int, model_tier: str | None,
                      latency_ms: int | None, error: str | None = None) -> dict:
    """Build the `chat_turns` record body for a completed Turn."""
    data = getattr(turn, "data", None)
    result = getattr(data, "result", None) if data is not None else None
    df = getattr(result, "dataframe", None) if result is not None else None

    record: dict = {
        "turn_index": turn_index,
        "created_at": _pb_now(),
        "question": getattr(turn, "question", "") or "",
        "standalone_question": getattr(turn, "standalone_question", "") or "",
        "route": getattr(turn, "route", "") or "",
        "needs_clarification": bool(getattr(turn, "needs_clarification", False)),
        "answer": getattr(turn, "answer", "") or "",
        "model_tier": model_tier or "",
    }
    if latency_ms is not None:
        record["latency_ms"] = int(latency_ms)
    if error:
        record["error"] = str(error)[:20000]

    if data is not None:
        record["sql"] = getattr(data, "sql", None) or ""
        record["chart_spec"] = _json_safe(getattr(data, "chart_spec", None))
    if result is not None:
        record["row_count"] = int(getattr(result, "row_count", 0) or 0)
        record["truncated"] = bool(getattr(result, "truncated", False))
        record["result_table"] = _dataframe_records(df)

    record["sources"] = _sources_payload(getattr(turn, "sources", []))
    live = _live_payload(turn)
    if live:
        record["live_payload"] = live

    return record
