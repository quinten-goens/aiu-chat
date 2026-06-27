"""NM live path: fetch the real-time network snapshot, then answer from it."""
from __future__ import annotations

import json
from dataclasses import dataclass

from aiu_chat.agent import prompts
from aiu_chat.agent.llm import OllamaClient
from aiu_chat.sources.nm_live import NmLiveError, NmLiveSnapshot, fetch_snapshot


@dataclass
class NmLiveAnswer:
    question: str
    answer: str
    snapshot: NmLiveSnapshot | None = None
    ok: bool = True


def _snapshot_json(s: NmLiveSnapshot, max_regs: int = 8) -> str:
    return json.dumps(
        {
            "airborne_now": s.airborne,
            "landed_today": s.landed,
            "planned_today": s.planned,
            "total_today": s.total,
            "total_network_delay_min": s.total_delay_min,
            "top_delayed_accs": s.top_delays[:8],
            "active_regulations": [
                {
                    "location": r.location, "reason": r.reason,
                    "delay_min": r.delay_min, "impacted_flights": r.impacted_flights,
                }
                for r in s.regulations[:max_regs]
            ],
            "active_regulations_total": len(s.regulations),
        }
    )


def answer_nm_question(
    question: str,
    *,
    client: OllamaClient | None = None,
    fetch=fetch_snapshot,
) -> NmLiveAnswer:
    client = client or OllamaClient()
    try:
        snapshot = fetch()
    except NmLiveError as exc:
        return NmLiveAnswer(question=question, answer=f"Live network data unavailable: {exc}", ok=False)

    messages = prompts.build_nm_live_messages(question, _snapshot_json(snapshot))
    answer = client.chat(messages, temperature=0.0)
    return NmLiveAnswer(question=question, answer=answer.strip(), snapshot=snapshot, ok=True)
