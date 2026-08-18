# Conversation-Audit Fixes, Model Refresh & Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three real failure modes found by auditing all 23 logged conversations, refresh the OpenAI model tiers to the current generation, and cut multi-route latency — without regressing the 263-passing test baseline.

**Architecture:** Three independent fix areas, each isolated to one or two modules. (1) The Data App client gains date-cursor pagination and batched per-day metric reads, replacing the silent 100-row truncation. (2) The entity layer gains an ACC/airport→FIR/ANSP alias map that drives a *clarifying question* instead of a dead-end refusal. (3) The orchestrator parallelises independent route fan-out. Model tiers move to the `gpt-5.6-*` family behind existing env vars.

**Tech Stack:** Python 3.11+, DuckDB, pandas, requests, Streamlit, pytest, OpenAI Chat Completions API, EUROCONTROL Data App API.

**Spec:** This document is self-contained — the evidence section below *is* the spec. It was derived by querying the `chat_turns`/`chat_sessions` PocketBase collections (23 turns, 33 sessions, 2026-07-03 → 2026-08-18) and reproducing each failure against the live API and local Parquet.

## Global Constraints

- **No test degradation.** Baseline is **263 passed, 0 failed, ~13s** (`.venv/bin/python -m pytest -q`). Every task ends by re-running the full suite; the pass count must be ≥263 with 0 failures. A genuinely-wrong test may be corrected, but never deleted or `skip`ped to make a number go green.
- **Numeric answers come from executed SQL / executed API results only** — never model arithmetic (CLAUDE.md hard constraint).
- **Polite scraper.** Pagination is uncapped in window length (user decision) but MUST be bounded by a call-count ceiling and an inter-call delay. Never silently trim a user's window — if a ceiling is hit, say so in the answer.
- **Read-only, single-statement SQL** validation stays intact; no new file-reading SQL functions.
- **No LLM-generated plotting code** — charts stay chart-spec driven.
- **All new knobs configurable via `aiu_chat/config.py`** reading `AIU_*` env vars, with the current behaviour as the default where a default is ambiguous.
- Python: use `from __future__ import annotations`, match surrounding style (dataclasses, type hints, docstrings explaining *why*).
- **Never commit secrets.** `.env` stays gitignored.

---

## Evidence: what the 23 logged conversations show

Pulled via superuser auth against `PB_CHAT_URL` (see the `chat-logging-pocketbase` memory).

**Route distribution:** dataapp 11, nm_live 3, data 3, nop 2, dataapp+concept 1, catalog 1, none 1, empty 1.

**Verdict per turn:** 15 clearly succeeded. **7 failed or degraded.** 1 was correct behaviour (out-of-scope refusal, turn 6).

### Failure A — silent time-series truncation (4 turns: 18, 19, 21, 22)

Every one asked for a multi-month daily series; every one has `row_count=100` and an answer that quietly narrates a *shorter* window than requested:

- Turn 18: asked 1 Jan → 1 Jun 2026 (Madrid). Answered "covers 1 January to 1 March".
- Turns 19/21/22: asked 1 Jan → 15 Aug 2026 (SAS Group). All answered "covers 1 January to 1 March".

**Root cause (reproduced live):** the Data App API hard-caps `itemsPerPage` at 100 and **ignores it silently** — requesting 375 returns `meta: {itemsPerPage: 100, totalItems: 226}`. `find_syncs_in_range` (`aiu_chat/sources/dataapp.py:278`) sets `itemsPerPage = max(MAX_PERIOD_DAYS, 30) + 5` and makes exactly one call, so it can never see more than 100 days.

**Also reproduced:** the `page=` parameter is *also* ignored — paging 3 times returned 300 rows that were the same 100 days repeated (still ending 2026-04-10). So offset paging does not work; **date-cursor** walking does: advancing `syncDate[after]` past the last-seen date retrieved all **226 days in 4 calls**.

The model behaved correctly given bad input — it narrated exactly the rows it was handed. This is a data-layer bug, not a prompt bug.

### Failure B — answerable question refused (3 turns: 3, 12, 14)

- Turn 12: "main causes of ATFM delay in Athens ACC on Week 27 this year" → "I can't answer that from the available datasets."
- Turn 14: same for "first six months of 2026" → same refusal, after **91.4 s** (the slowest turn logged).
- Turn 3: monthly traffic + YoY variation for 2026 + causes → same refusal.

**Root cause (reproduced locally):** the data exists. `enroute_delay_fir.parquet` carries per-cause en-route ATFM delay columns and has an entity literally named `Greece`; `enroute_delay_ansp.parquet` has 50 ANSP entities. Neither contains the string "Athens ACC" — ANSP has *no* Greek match at all, FIR has `Greece`. Executed proof for turn 14's exact question:

```
Greece FIR, Jan–Jun 2026, en-route ATFM delay minutes:
  capacity  1,372 | staffing 105,609 | weather 4,015 | TOTAL 409,297
```

Staffing-dominated — a genuinely valuable finding that the tool refused to surface three times. The resolver simply has no ACC→FIR/ANSP bridge, so it fails closed.

**User decision:** on a near-miss, **ask a clarifying question** ("Athens ACC isn't a separate entity — did you mean Greece (FIR)?") rather than silently substituting.

### Failure C — one hard API error (turn 15)

`OpenAI API returned HTTP 500` (request id `req_f5fc73df…`), empty question and empty answer, route `''`. A transient upstream 500 with **no retry** — the turn was simply lost.

### Latency (perf, not correctness)

Mean latency across successful turns is ~32 s; the tail is bad: **91.4 s**, 65.4 s, 58.1 s, 55.3 s. Two structural causes, both confirmed by reading the code:

1. `_answer_single` is called **sequentially per route** in `orchestrator._answer_compound` — a 3-route fan-out costs the sum, not the max.
2. `fetch_timeseries` (`dataapp.py:324`) loops **one HTTP call per day** to read each sync's metric. A 226-day series is ~230 sequential round-trips. This gets *worse* once pagination unlocks longer windows, so it must be fixed in the same task.

Neither entity resolution nor sync lookup is cached (`lru_cache` appears only at `entities.py:83`).

### Model inventory (probed live against the account)

Config currently pins `gpt-5.5` (max), `gpt-5.4-mini`, `gpt-5.4-nano`. Newer models are available: **`gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`**, plus `gpt-5.5-pro`.

Smoke-tested — all four respond correctly. On a representative delay-cause SQL task (all produced valid SQL):

| model | latency | completion tok | reasoning tok |
|---|---|---|---|
| `gpt-5.5` (current max) | 8.7 s | 458 | 245 |
| `gpt-5.6-luna` | 5.7 s | 593 | 396 |
| `gpt-5.6-sol` | **4.6 s** | **201** | 114 |
| `gpt-5.6-terra` | 8.3 s | 668 | 456 |

The `5.6` names don't follow the nano/mini/max convention, so Task 5 benchmarks them on *this project's actual* prompts before assigning tiers rather than guessing from names.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `aiu_chat/sources/dataapp.py` | Data App HTTP client | Modify — add `_get_all_pages` date-cursor pagination; batch per-day metric reads; add throttle |
| `aiu_chat/config.py` | Central env config | Modify — add pagination/throttle/parallelism knobs; retarget model tiers |
| `aiu_chat/agent/entities.py` | Entity index + resolution | Modify — add ACC/airport→FIR/ANSP alias lookup returning candidates |
| `aiu_chat/agent/aliases.py` | ACC/ATSU → FIR/ANSP alias map | **Create** — pure data + lookup, no I/O |
| `aiu_chat/agent/orchestrator.py` | Route planning + fan-out | Modify — parallel fan-out; wire near-miss clarification |
| `aiu_chat/agent/llm.py` | LLM client | Modify — retry with backoff on 5xx/429 |
| `tests/test_dataapp_pagination.py` | Pagination unit tests | **Create** |
| `tests/test_aliases.py` | Alias-map unit tests | **Create** |
| `tests/test_llm_retry.py` | Retry unit tests | **Create** |

Each task is independently testable and committed separately.

---

## Task 0: Rotate the leaked OpenAI key (do this first, manually)

**This is a human action, not a code change.** During the audit, `OPENAI_KEY` was read out of `.env` and printed in plaintext into a chat transcript. Treat it as compromised.

- [ ] **Step 1: Revoke and reissue**

Go to https://platform.openai.com/api-keys, revoke the key beginning `sk-proj-Fo9qZog…`, create a replacement, and put the new value in `.env` (which is already gitignored).

- [ ] **Step 2: Confirm the key never entered git history**

```bash
git log -p --all -S 'sk-proj-' -- .env | head
git check-ignore -v .env
```

Expected: no output from the first command (the secret was never committed); the second confirms `.env` is ignored. If the first prints anything, stop and purge history before continuing.

- [ ] **Step 3: Verify the new key works**

```bash
.venv/bin/python -c "
from aiu_chat import config; print('key loaded:', bool(config.OPENAI_KEY), 'len:', len(config.OPENAI_KEY))"
```

Expected: `key loaded: True` and a plausible length. **The value itself must never be echoed.**

---

## Task 1: Date-cursor pagination for sync lookup

Fixes Failure A. This is the highest-value change in the plan — it converts four wrong answers into right ones.

**Files:**
- Modify: `aiu_chat/sources/dataapp.py` (`find_syncs_in_range`, ~lines 254-283)
- Modify: `aiu_chat/config.py` (new knobs near the Data App section, ~line 244)
- Test: `tests/test_dataapp_pagination.py` (create)

**Interfaces:**
- Consumes: existing `_get(session, path, params) -> dict`, `ENTITY_ENDPOINTS`, `NETWORK_DATATYPE`, `Entity`.
- Produces: `find_syncs_in_range(session, *, start, end, entity=None) -> list[tuple[int, str]]` — unchanged signature, now complete over the window. New module constant `API_PAGE_SIZE = 100`. New config values `DATAAPP_MAX_PAGES`, `DATAAPP_THROTTLE_S`.

- [ ] **Step 1: Add the config knobs**

In `aiu_chat/config.py`, immediately after the `DATAAPP_BASE` line:

```python
# The Data App API silently caps itemsPerPage at 100 and ignores `page`, so long
# windows must be walked with a syncDate cursor. The window itself is never
# trimmed (a user asking for 3 years gets 3 years), but we bound total calls so a
# pathological request cannot hammer a public API — polite-scraper constraint.
DATAAPP_MAX_PAGES = int(os.getenv("AIU_DATAAPP_MAX_PAGES", "40"))
# Seconds to sleep between consecutive paged calls.
DATAAPP_THROTTLE_S = float(os.getenv("AIU_DATAAPP_THROTTLE_S", "0.15"))
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_dataapp_pagination.py`. These use a fake `_get` — no network.

```python
"""Pagination tests for the Data App sync lookup.

The live API caps itemsPerPage at 100 and ignores `page`; the only way to walk a
longer window is a syncDate cursor. These tests pin that behaviour with a fake
API that reproduces the cap exactly.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from aiu_chat.sources import dataapp


class FakeAPI:
    """Mimics the real API: returns at most 100 rows, honours syncDate[after],
    and IGNORES `page` (exactly like the live endpoint)."""

    def __init__(self, first_day: str, n_days: int):
        self.days = [
            (date.fromisoformat(first_day) + timedelta(days=i)).isoformat()
            for i in range(n_days)
        ]
        self.calls = 0

    def __call__(self, session, path, params):
        self.calls += 1
        after = params.get("syncDate[after]")
        before = params.get("syncDate[before]")
        sel = [d for d in self.days if (not after or d >= after) and (not before or d <= before)]
        page = sel[:100]  # hard cap, `page` param deliberately ignored
        return {
            "meta": {"itemsPerPage": 100, "totalItems": len(sel)},
            "data": [
                {"id": 1000 + self.days.index(d), "syncDate": f"{d}T00:00:00+00:00"}
                for d in page
            ],
        }


def test_short_window_single_call(monkeypatch):
    fake = FakeAPI("2026-01-01", 30)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2026-01-01", end="2026-01-30")
    assert len(out) == 30
    assert fake.calls == 1


def test_long_window_is_complete(monkeypatch):
    """226 days must all come back — the bug was stopping silently at 100."""
    fake = FakeAPI("2026-01-01", 226)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2026-01-01", end="2026-08-14")
    assert len(out) == 226
    assert out[0][1] == "2026-01-01"
    assert out[-1][1] == "2026-08-14"
    assert fake.calls > 1


def test_results_sorted_and_deduped(monkeypatch):
    fake = FakeAPI("2026-01-01", 150)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2026-01-01", end="2026-05-30")
    dates = [d for _, d in out]
    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))


def test_page_ceiling_is_respected(monkeypatch):
    """A pathological window must not fire unbounded calls."""
    monkeypatch.setattr(dataapp.config, "DATAAPP_MAX_PAGES", 3)
    fake = FakeAPI("2020-01-01", 2000)
    monkeypatch.setattr(dataapp, "_get", fake)
    out = dataapp.find_syncs_in_range(None, start="2020-01-01", end="2025-06-23")
    assert fake.calls <= 3
    assert len(out) <= 300
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_dataapp_pagination.py -v
```

Expected: `test_long_window_is_complete` FAILS with `assert 100 == 226`. That failure *is* the reproduction of the logged bug.

- [ ] **Step 4: Implement the cursor walk**

In `aiu_chat/sources/dataapp.py`, add near the other module constants (after `TIMEOUT = 30`):

```python
# The API silently caps page size at 100 regardless of what we ask for.
API_PAGE_SIZE = 100
```

Replace the body of `find_syncs_in_range` after the `params` dict is built (i.e. replace the `params["itemsPerPage"] = ...` line and the single `_get` + dedupe block) with:

```python
    params["syncDate[after]"] = start
    params["syncDate[before]"] = end
    params["order[syncDate]"] = "asc"
    params["itemsPerPage"] = API_PAGE_SIZE

    # Walk the window with a syncDate cursor. Offset paging does NOT work here:
    # the API ignores `page` and re-serves the same first 100 rows, so we instead
    # advance `syncDate[after]` past the last day we saw on each pass.
    by_day: dict[str, int] = {}
    cursor = start
    for _ in range(max(1, config.DATAAPP_MAX_PAGES)):
        page_params = dict(params, **{"syncDate[after]": cursor})
        data = _get(session, "/syncs", page_params).get("data", [])
        if not data:
            break
        last_seen = cursor
        for row in data:
            d = (row.get("syncDate") or "")[:10]
            if d and d not in by_day:
                by_day[d] = row["id"]
            if d > last_seen:
                last_seen = d
        # No forward progress (all rows on/before the cursor) -> window exhausted.
        if last_seen <= cursor or len(data) < API_PAGE_SIZE:
            break
        cursor = last_seen
        if config.DATAAPP_THROTTLE_S:
            time.sleep(config.DATAAPP_THROTTLE_S)

    return [(sid, d) for d, sid in sorted(by_day.items())]
```

Add `import time` to the module imports if not already present.

Note the boundary subtlety: because `syncDate[after]` is inclusive in this API, the last-seen day is re-fetched on the next pass; `by_day` dedupes it, so this is correct but costs one duplicate row per page — acceptable.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_dataapp_pagination.py -v
```

Expected: all 4 PASS.

- [ ] **Step 6: Verify against the live API**

```bash
.venv/bin/python -c "
import requests
from aiu_chat.sources import dataapp
s = requests.Session()
out = dataapp.find_syncs_in_range(s, start='2026-01-01', end='2026-08-15')
print('days:', len(out), 'first:', out[0][1], 'last:', out[-1][1])"
```

Expected: `days: 226 first: 2026-01-01 last: 2026-08-14` (matches the reproduction).

- [ ] **Step 7: Run the full suite — no degradation**

```bash
.venv/bin/python -m pytest -q
```

Expected: **≥263 passed, 0 failed** (263 baseline + 4 new = 267).

- [ ] **Step 8: Commit**

```bash
git add aiu_chat/sources/dataapp.py aiu_chat/config.py tests/test_dataapp_pagination.py
git commit -m "Data App: paginate sync lookup with a syncDate cursor

The API silently caps itemsPerPage at 100 and ignores \`page\`, so any window
longer than 100 days was truncated without warning and the model narrated a
shorter period than the user asked for (4 logged conversations). Walk the
window with a date cursor instead, bounded by a page ceiling and throttle."
```

---

## Task 2: Batch the per-day metric reads

`fetch_timeseries` issues one HTTP call per day. Task 1 makes long windows reachable, which would make this ~230 sequential calls. Fix it in the same breath.

**Files:**
- Modify: `aiu_chat/sources/dataapp.py` (`fetch_timeseries`, ~lines 286-345)
- Modify: `aiu_chat/config.py`
- Test: `tests/test_dataapp_pagination.py` (extend)

**Interfaces:**
- Consumes: `find_syncs_in_range` from Task 1; existing `METRIC_ENDPOINTS`, `_pick_day_record`, `TimeseriesResult`.
- Produces: `fetch_timeseries(...) -> TimeseriesResult` — unchanged signature and row shape (`{"date","value","avgValue"}`), now concurrent. New config `DATAAPP_CONCURRENCY`.

- [ ] **Step 1: Add the concurrency knob**

In `aiu_chat/config.py`, after `DATAAPP_THROTTLE_S`:

```python
# Parallel per-day metric reads in a time series. Modest by design: enough to
# make a 200-day series usable, low enough to stay a polite client.
DATAAPP_CONCURRENCY = int(os.getenv("AIU_DATAAPP_CONCURRENCY", "8"))
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_dataapp_pagination.py`:

```python
def test_timeseries_reads_every_day(monkeypatch):
    """Every sync day must appear as a row, in date order, with no gaps."""
    days = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(150)]
    syncs = [(1000 + i, d) for i, d in enumerate(days)]
    monkeypatch.setattr(dataapp, "find_syncs_in_range", lambda *a, **k: syncs)

    def fake_get(session, path, params):
        sid = params.get("traffic.sync.id")
        return {"data": [{
            "dateRange": "DY", "networkType": "total",
            "value": float(sid), "avgValue": float(sid),
        }]}

    monkeypatch.setattr(dataapp, "_get", fake_get)
    res = dataapp.fetch_timeseries("traffic", start="2026-01-01", end="2026-05-30")
    assert len(res.rows) == 150
    assert [r["date"] for r in res.rows] == days
    assert res.rows[0]["value"] == 1000.0
    assert res.rows[-1]["value"] == 1149.0
```

- [ ] **Step 3: Run it to verify it passes-or-fails meaningfully**

```bash
.venv/bin/python -m pytest tests/test_dataapp_pagination.py::test_timeseries_reads_every_day -v
```

Expected: PASS even before the change (correctness is already right — it is *slow*, not wrong). This test is the safety net that the concurrency rewrite must not break. If it fails, stop and fix that first.

- [ ] **Step 4: Parallelise the read loop**

In `fetch_timeseries`, replace the sequential `for sync_id, sync_date in syncs:` block with:

```python
        endpoint, _, prefix = METRIC_ENDPOINTS[metric]

        def _read_day(item: tuple[int, str]) -> dict | None:
            sync_id, sync_date = item
            data = _get(
                session, endpoint,
                {f"{prefix}.sync.id": sync_id, "itemsPerPage": 30},
            ).get("data", [])
            recs = [{k: v for k, v in r.items() if k != prefix} for r in data]
            dy = _pick_day_record(recs)
            if dy is None:
                return None
            return {"date": sync_date, "value": dy.get("value"),
                    "avgValue": dy.get("avgValue")}

        # One call per day is unavoidable (the API has no bulk endpoint), but they
        # are independent, so run a bounded pool instead of ~200 serial requests.
        rows: list[dict] = []
        workers = max(1, min(config.DATAAPP_CONCURRENCY, len(syncs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_read_day, syncs):
                if row is not None:
                    rows.append(row)
```

Add to the module imports:

```python
from concurrent.futures import ThreadPoolExecutor
```

`requests.Session` is not formally thread-safe for mutation, but concurrent plain GETs on one session are the standard, widely-used pattern and each worker only reads. Keep the existing `rows.sort(key=lambda r: r["date"])` after the block — `pool.map` preserves input order, but the explicit sort is cheap insurance.

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_dataapp_pagination.py -v
```

Expected: all PASS.

- [ ] **Step 6: Measure the real improvement**

```bash
.venv/bin/python -c "
import time, requests
from aiu_chat.sources import dataapp
s = requests.Session()
t = time.time()
r = dataapp.fetch_timeseries('traffic', start='2026-01-01', end='2026-06-01',
                             kind='country', query='Spain', session=s)
print(f'rows={len(r.rows)} first={r.rows[0][\"date\"]} last={r.rows[-1][\"date\"]} in {time.time()-t:.1f}s')"
```

Expected: ~152 rows spanning the **full** requested window (this is the Madrid/turn-18 scenario, previously capped at 100 rows ending 1 March), in well under the previous serial time.

- [ ] **Step 7: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ≥267 passed, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add aiu_chat/sources/dataapp.py aiu_chat/config.py tests/test_dataapp_pagination.py
git commit -m "Data App: read time-series days concurrently

A 200-day series was ~200 serial HTTP round-trips, which dominated the worst
logged latencies. Run the independent per-day reads in a bounded thread pool."
```

---

## Task 3: ACC/airport near-miss → clarifying question

Fixes Failure B. Per the user's decision, a near-miss **asks** rather than substitutes.

**Files:**
- Create: `aiu_chat/agent/aliases.py`
- Modify: `aiu_chat/agent/orchestrator.py` (`needs_clarification`, ~line 180)
- Test: `tests/test_aliases.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks (independent).
- Produces: `aiu_chat.agent.aliases.resolve_near_miss(name: str) -> NearMiss | None`, where `NearMiss` is a dataclass with fields `query: str`, `candidates: list[str]`, `reason: str`. Also `NearMiss.question() -> str` returning the user-facing clarifying sentence.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_aliases.py`:

```python
"""ACC/ATSU names are not entities in the AIU datasets.

Three logged conversations asked about "Athens ACC" and were refused, even
though Greece FIR carries the answer. A near-miss must produce a clarifying
question naming real candidates, never a silent substitution.
"""
from __future__ import annotations

from aiu_chat.agent import aliases


def test_athens_acc_offers_greece():
    nm = aliases.resolve_near_miss("Athens ACC")
    assert nm is not None
    assert "Greece" in nm.candidates


def test_case_and_suffix_insensitive():
    for probe in ("athens acc", "ATHENS ACC", "Athens", "Athens Area Control Centre"):
        nm = aliases.resolve_near_miss(probe)
        assert nm is not None, probe
        assert "Greece" in nm.candidates, probe


def test_question_names_the_candidate_and_the_problem():
    q = aliases.resolve_near_miss("Athens ACC").question()
    assert "Athens" in q
    assert "Greece" in q
    assert "?" in q


def test_known_entity_is_not_a_near_miss():
    """A name that IS a real entity must pass straight through."""
    assert aliases.resolve_near_miss("Greece") is None
    assert aliases.resolve_near_miss("Maastricht UAC") is None


def test_unrelated_text_is_not_a_near_miss():
    assert aliases.resolve_near_miss("fuel prices") is None
    assert aliases.resolve_near_miss("") is None
```

- [ ] **Step 2: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_aliases.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'aiu_chat.agent.aliases'`.

- [ ] **Step 3: Implement the alias map**

Create `aiu_chat/agent/aliases.py`:

```python
"""ACC / ATSU names → the FIR or ANSP entities that actually carry their data.

The AIU delay datasets are keyed by FIR (`enroute_delay_fir`: 62 entities such
as "Greece") and by ANSP (`enroute_delay_ansp`: 50 entities such as "BULATSA").
Users naturally ask about *area control centres* — "Athens ACC", "Karlsruhe UAC"
— which are not entities in either table. Before this map, such questions were
refused outright even though the data was present.

We deliberately do NOT auto-substitute: an ACC and its FIR are not the same
thing, so the caller asks the user to confirm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ACC/UAC/airport-city token -> real entity names, most likely first.
# Keys are lowercase and suffix-free; see `_normalise`.
_NEAR_MISS: dict[str, list[str]] = {
    "athens": ["Greece"],
    "thessaloniki": ["Greece"],
    "macedonia": ["Greece"],
    "karlsruhe": ["Germany", "DFS"],
    "langen": ["Germany", "DFS"],
    "munich": ["Germany", "DFS"],
    "bremen": ["Germany", "DFS"],
    "reims": ["France", "DSNA"],
    "brest": ["France", "DSNA"],
    "bordeaux": ["France", "DSNA"],
    "marseille": ["France", "DSNA"],
    "paris": ["France", "DSNA"],
    "london": ["United Kingdom", "NATS (Continental)"],
    "swanwick": ["United Kingdom", "NATS (Continental)"],
    "prestwick": ["United Kingdom", "NATS (Continental)"],
    "scottish": ["United Kingdom", "NATS (Continental)"],
    "shannon": ["Ireland", "AirNav Ireland"],
    "madrid": ["Spain", "ENAIRE"],
    "barcelona": ["Spain", "ENAIRE"],
    "seville": ["Spain", "ENAIRE"],
    "canarias": ["Spain", "ENAIRE"],
    "lisbon": ["Portugal", "NAV Portugal (Continental)"],
    "rome": ["Italy", "ENAV"],
    "milan": ["Italy", "ENAV"],
    "padua": ["Italy", "ENAV"],
    "brindisi": ["Italy", "ENAV"],
    "vienna": ["Austria", "Austro Control"],
    "zurich": ["Switzerland", "Skyguide"],
    "geneva": ["Switzerland", "Skyguide"],
    "amsterdam": ["Netherlands", "LVNL"],
    "brussels": ["Belgium", "skeyes"],
    "warsaw": ["Poland", "PANSA"],
    "budapest": ["Hungary", "HungaroControl"],
    "prague": ["Czech Republic", "ANS CR"],
    "bucharest": ["Romania", "ROMATSA"],
    "sofia": ["Bulgaria", "BULATSA"],
    "zagreb": ["Croatia", "Croatia Control"],
    "ljubljana": ["Slovenia", "Slovenia Control"],
    "bratislava": ["Slovakia", "LPS"],
    "copenhagen": ["Denmark", "Naviair"],
    "stockholm": ["Sweden", "LFV"],
    "malmo": ["Sweden", "LFV"],
    "oslo": ["Norway", "Avinor"],
    "helsinki": ["Finland", "ANS Finland"],
    "riga": ["Latvia", "LGS"],
    "tallinn": ["Estonia", "EANS"],
    "vilnius": ["Lithuania", "Oro Navigacija"],
    "ankara": ["Turkey", "DHMI"],
    "istanbul": ["Turkey", "DHMI"],
    "nicosia": ["Cyprus", "DCAC Cyprus"],
    "valletta": ["Malta", "MATS"],
    "belgrade": ["Serbia", "SMATSA"],
    "skopje": ["North Macedonia"],
    "tirana": ["Albania", "Albcontrol"],
    "sarajevo": ["Bosnia and Herzegovina", "BHANSA"],
    "chisinau": ["Moldova", "MOLDATSA"],
    "kyiv": ["Ukraine", "UkSATSE"],
    "yerevan": ["Armenia", "ARMATS"],
    "tbilisi": ["Georgia", "Sakaeronavigatsia"],
}

# Names that ARE real entities (or real ANSPs) and must never be treated as a
# near-miss, even though they look like control-centre names.
_REAL_ENTITIES: set[str] = {
    "maastricht uac", "greece", "germany", "france", "united kingdom", "spain",
    "italy", "ireland", "portugal", "austria", "switzerland", "netherlands",
    "belgium", "poland", "hungary", "czech republic", "romania", "bulgaria",
    "croatia", "slovenia", "slovakia", "denmark", "sweden", "norway", "finland",
    "latvia", "estonia", "lithuania", "turkey", "cyprus", "malta", "serbia",
    "north macedonia", "albania", "bosnia and herzegovina", "moldova", "ukraine",
    "armenia", "georgia",
}

# Suffixes that mark a control centre rather than an entity.
_SUFFIXES = (
    " area control centre", " area control center", " upper area control centre",
    " acc", " uac", " fir", " ctr", " tma", " airport", " ansp",
)


@dataclass
class NearMiss:
    """A name that is not an entity but maps to one or more that are."""

    query: str
    candidates: list[str] = field(default_factory=list)
    reason: str = ""

    def question(self) -> str:
        """The clarifying question to put to the user."""
        if len(self.candidates) == 1:
            opts = f"**{self.candidates[0]}**"
        else:
            opts = " or ".join(f"**{c}**" for c in self.candidates)
        return (
            f"“{self.query}” isn’t a separate entity in the EUROCONTROL "
            f"performance datasets — {self.reason} Did you mean {opts}?"
        )


def _normalise(name: str) -> str:
    out = " ".join((name or "").strip().lower().split())
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if out.endswith(suf):
                out = out[: -len(suf)].strip()
                changed = True
    return out


def resolve_near_miss(name: str) -> NearMiss | None:
    """Return a NearMiss if `name` looks like an ACC/airport that maps onto a
    real FIR/ANSP entity, else None (unknown text and real entities both)."""
    raw = " ".join((name or "").strip().lower().split())
    if not raw or raw in _REAL_ENTITIES:
        return None
    key = _normalise(name)
    if not key or key in _REAL_ENTITIES:
        return None
    cands = _NEAR_MISS.get(key)
    if not cands:
        return None
    return NearMiss(
        query=name.strip(),
        candidates=list(cands),
        reason=(
            "en-route delay data is published per FIR and per ANSP, "
            "not per area control centre."
        ),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_aliases.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Wire it into clarification**

Read `aiu_chat/agent/orchestrator.py:180-216` (`needs_clarification`) first to match its existing return contract — it returns `str | None` where the string is the question shown to the user.

Add at the top of the function body, before the existing LLM-based logic, so a known near-miss short-circuits without an LLM round-trip:

```python
    # A named ACC/UAC is a near-miss for a real FIR/ANSP entity. Three logged
    # conversations about "Athens ACC" were refused outright even though Greece
    # FIR holds the answer — ask instead of failing closed.
    from aiu_chat.agent import aliases

    for token in _candidate_entity_names(question):
        nm = aliases.resolve_near_miss(token)
        if nm is not None:
            return nm.question()
```

If no `_candidate_entity_names` helper exists in the module, add this alongside it:

```python
def _candidate_entity_names(question: str) -> list[str]:
    """Multi-word proper-noun-ish spans from the question, longest first.

    Cheap and deliberately over-generous: `aliases.resolve_near_miss` is the
    authority on what is actually a near-miss, so a false candidate costs
    nothing but a dict lookup."""
    import re

    spans = re.findall(
        r"\b([A-Z][\w’'-]*(?:\s+(?:[A-Z][\w’'-]*|ACC|UAC|FIR))*)", question)
    return sorted({s.strip() for s in spans if s.strip()}, key=len, reverse=True)
```

- [ ] **Step 6: Verify the logged failure now clarifies**

```bash
.venv/bin/python -c "
from aiu_chat.agent import aliases
for q in ['Athens ACC', 'Karlsruhe UAC', 'Greece', 'fuel prices']:
    nm = aliases.resolve_near_miss(q)
    print(repr(q), '->', nm.question() if nm else None)"
```

Expected: the first two produce clarifying questions naming Greece / Germany; `Greece` and `fuel prices` produce `None`.

- [ ] **Step 7: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ≥272 passed, 0 failed. **If any existing orchestrator test now fails**, the near-miss check is firing too eagerly — narrow `_candidate_entity_names` rather than weakening the test.

- [ ] **Step 8: Commit**

```bash
git add aiu_chat/agent/aliases.py aiu_chat/agent/orchestrator.py tests/test_aliases.py
git commit -m "Ask about ACC near-misses instead of refusing

\"Athens ACC delay causes\" was refused three times although Greece FIR carries
the data (Jan-Jun 2026: 409k min, staffing-dominated). Map ACC/UAC names to the
FIR/ANSP entities that hold their data and ask the user to confirm."
```

---

## Task 4: Retry transient LLM failures

Fixes Failure C — one turn lost to an unretried HTTP 500.

**Files:**
- Modify: `aiu_chat/agent/llm.py`
- Modify: `aiu_chat/config.py`
- Test: `tests/test_llm_retry.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: retry behaviour internal to the LLM client; new config `LLM_MAX_RETRIES`, `LLM_RETRY_BASE_S`. No public signature changes.

- [ ] **Step 1: Add config**

In `aiu_chat/config.py`, near the other LLM settings:

```python
# Transient upstream failures (HTTP 5xx / 429) cost a whole logged turn once.
# Retry a couple of times with exponential backoff before surfacing the error.
LLM_MAX_RETRIES = int(os.getenv("AIU_LLM_MAX_RETRIES", "2"))
LLM_RETRY_BASE_S = float(os.getenv("AIU_LLM_RETRY_BASE_S", "1.0"))
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_llm_retry.py`:

```python
"""A transient OpenAI 500 cost a whole logged turn (empty question and answer).
Retryable statuses must be retried; client errors must not be."""
from __future__ import annotations

import pytest

from aiu_chat.agent import llm


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] < 3:
            raise llm.LLMError("OpenAI API returned HTTP 500: server error")
        return "fine"

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    assert llm._with_retries(flaky) == "fine"
    assert calls["n"] == 3


def test_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    def always_500(*a, **k):
        calls["n"] += 1
        raise llm.LLMError("OpenAI API returned HTTP 503: unavailable")

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    with pytest.raises(llm.LLMError):
        llm._with_retries(always_500)
    assert calls["n"] == 3


def test_does_not_retry_client_error(monkeypatch):
    calls = {"n": 0}

    def bad_request(*a, **k):
        calls["n"] += 1
        raise llm.LLMError("OpenAI API returned HTTP 400: bad request")

    monkeypatch.setattr(llm.config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(llm.config, "LLM_RETRY_BASE_S", 0)
    with pytest.raises(llm.LLMError):
        llm._with_retries(bad_request)
    assert calls["n"] == 1
```

- [ ] **Step 3: Run to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_llm_retry.py -v
```

Expected: FAIL — `_with_retries` does not exist.

- [ ] **Step 4: Implement**

Read `aiu_chat/agent/llm.py` to confirm the exception class name (the tests assume `LLMError`; if it differs, use the real one consistently in both). Add:

```python
_RETRYABLE = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")


def _is_retryable(exc: Exception) -> bool:
    return any(code in str(exc) for code in _RETRYABLE)


def _with_retries(fn, *args, **kwargs):
    """Call `fn`, retrying transient upstream failures with exponential backoff.

    A single unretried HTTP 500 lost a whole conversation turn in the logs."""
    attempts = max(0, config.LLM_MAX_RETRIES) + 1
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below
            if not _is_retryable(exc) or i == attempts - 1:
                raise
            last = exc
            if config.LLM_RETRY_BASE_S:
                time.sleep(config.LLM_RETRY_BASE_S * (2 ** i))
    assert last is not None
    raise last
```

Add `import time` if absent. Then wrap the actual HTTP call in the chat/completion method — find the method that performs the request and route its body through `_with_retries`, e.g. change a direct `return self._post(...)` into `return _with_retries(self._post, ...)`. Do **not** wrap at a level that would re-run prompt construction or double-log a turn.

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_llm_retry.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ≥275 passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add aiu_chat/agent/llm.py aiu_chat/config.py tests/test_llm_retry.py
git commit -m "Retry transient LLM 5xx/429 with backoff

A single unretried OpenAI HTTP 500 lost an entire logged turn."
```

---

## Task 5: Refresh the OpenAI model tiers

**Files:**
- Modify: `aiu_chat/config.py` (`_OPENAI_TIERS`, ~lines 96-113)
- Modify: `.env.example`
- Test: benchmark script (scratchpad, not committed)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: updated `_OPENAI_TIERS` defaults. Env var names (`AIU_OPENAI_NANO` / `_MINI` / `_MAX`) are unchanged, so any existing override keeps working.

Available now (probed live): `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.5-pro`, plus the current `gpt-5.5` / `gpt-5.4-mini` / `gpt-5.4-nano`.

- [ ] **Step 1: Benchmark on this project's real prompts**

The `5.6` names are opaque, so measure before assigning. Write a scratchpad script that runs each candidate over ~6 prompts drawn from `aiu_chat/agent/prompts.py` (one router classification, one text-to-SQL, one chart spec, one synthesis, one follow-up rewrite, one clarification) and records latency, token use, and whether the output parses (valid JSON where JSON is expected; a single `SELECT` where SQL is expected).

```bash
.venv/bin/python /private/tmp/claude-501/-Users-quintengoens-Repos-aiu-chat/1cc0d791-25fe-4812-a028-12f8509ba36a/scratchpad/bench_models.py
```

Record the table in the commit message. Prior smoke test on one SQL task: `sol` 4.6 s/201 tok, `luna` 5.7 s/593 tok, `terra` 8.3 s/668 tok, `gpt-5.5` 8.7 s/458 tok — all correct.

- [ ] **Step 2: Assign tiers from the measurements**

Update `_OPENAI_TIERS` in `aiu_chat/config.py`. Assign the fastest reliable model to `gpt_nano`, the best latency/quality balance to `gpt_mini`, and the highest-quality to `gpt_max`. Based on the smoke test the expected shape is:

```python
_OPENAI_TIERS = {
    "gpt_nano": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_NANO", "gpt-5.4-nano"),
        "label": "⚡ Fast · GPT nano",
        "blurb": "OpenAI's smallest GPT-5 model. Fast and inexpensive.",
    },
    "gpt_mini": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_MINI", "gpt-5.6-sol"),
        "label": "🧠 Balanced · GPT 5.6 sol",
        "blurb": "Balanced OpenAI model — a good default.",
    },
    "gpt_max": {
        "provider": "openai",
        "model": os.getenv("AIU_OPENAI_MAX", "gpt-5.6-terra"),
        "label": "🚀 Max · GPT 5.6 terra (most capable)",
        "blurb": "OpenAI's most capable general model. Best quality, higher cost.",
    },
}
```

**Only commit the assignment the benchmark actually supports** — if `sol` beats `terra` on quality too, promote it and say so. Do not ship a slower model to a faster tier.

- [ ] **Step 3: Document the new vars**

Ensure `.env.example` lists `AIU_OPENAI_NANO`, `AIU_OPENAI_MINI`, `AIU_OPENAI_MAX` with the new defaults as comments, plus the Task 1/2/4 knobs (`AIU_DATAAPP_MAX_PAGES`, `AIU_DATAAPP_THROTTLE_S`, `AIU_DATAAPP_CONCURRENCY`, `AIU_LLM_MAX_RETRIES`, `AIU_LLM_RETRY_BASE_S`).

- [ ] **Step 4: Smoke-test each tier end to end**

```bash
.venv/bin/python -c "
from aiu_chat import config
for k, v in config._OPENAI_TIERS.items():
    print(k, '->', v['model'])"
```

Then run one real question through the app path per tier and confirm a grounded answer comes back.

- [ ] **Step 5: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ≥275 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add aiu_chat/config.py .env.example
git commit -m "Move OpenAI tiers to the gpt-5.6 family

Benchmarked luna/sol/terra against gpt-5.5 on this project's own router, SQL,
chart-spec and synthesis prompts; table in the plan. Env overrides unchanged."
```

---

## Task 6: Parallel route fan-out

The largest remaining latency win: a 3-route question currently costs the sum of its routes.

**Files:**
- Modify: `aiu_chat/agent/orchestrator.py` (`_answer_compound`, ~line 384)
- Modify: `aiu_chat/config.py`
- Test: `tests/test_orchestrator.py` (extend)

**Interfaces:**
- Consumes: existing `_answer_single(...)`, `Turn`, the `status(label, detail)` callback from `answer`.
- Produces: unchanged `_answer_compound` signature and output ordering. New config `ROUTE_CONCURRENCY`.

- [ ] **Step 1: Add config**

```python
# Independent routes in a multi-source answer are fanned out in parallel; the
# sequential version made a 3-route question cost the sum of its parts.
ROUTE_CONCURRENCY = int(os.getenv("AIU_ROUTE_CONCURRENCY", "3"))
```

- [ ] **Step 2: Write the failing test**

Read the existing fan-out tests in `tests/test_orchestrator.py` first and match their fixtures. Add:

```python
def test_compound_routes_run_in_parallel(monkeypatch):
    """Three 0.3s routes must finish in well under the 0.9s serial cost."""
    import time
    from aiu_chat.agent import orchestrator

    def slow_single(question, route, *a, **k):
        time.sleep(0.3)
        return {"route": route, "answer": f"ans-{route}"}

    monkeypatch.setattr(orchestrator, "_answer_single", slow_single)
    monkeypatch.setattr(orchestrator.config, "ROUTE_CONCURRENCY", 3)

    t = time.time()
    out = orchestrator._answer_compound(...)  # match the real signature
    elapsed = time.time() - t

    assert elapsed < 0.7, f"routes ran serially ({elapsed:.2f}s)"
    assert [o["route"] for o in out] == ["data", "concept", "dataapp"]  # order preserved
```

Adapt the call to the real `_answer_compound` signature and the real return type — read the function before writing the assertion.

- [ ] **Step 3: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_orchestrator.py::test_compound_routes_run_in_parallel -v
```

Expected: FAIL on the elapsed-time assertion (~0.9 s).

- [ ] **Step 4: Implement**

In `_answer_compound`, replace the sequential per-route loop with a bounded pool, **preserving result order** so the synthesis prompt and citations stay deterministic:

```python
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, min(config.ROUTE_CONCURRENCY, len(routes)))
    if workers == 1 or len(routes) == 1:
        results = [_answer_single(question, r, ...) for r in routes]
    else:
        # Routes are independent (separate backends, no shared mutable state), so
        # fan out and keep input order — synthesis and citations must be stable.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda r: _answer_single(question, r, ...), routes))
```

Two constraints to respect while editing:
- The `status()` progress callback writes to Streamlit. Calling it from worker threads can misbehave, so either emit a single "querying N sources…" status before the pool and a summary after, or collect per-route status strings and emit them from the main thread once the pool joins.
- Preserve existing error semantics: if one route raises, keep the current behaviour (whatever `_answer_compound` does today for a failing route) rather than letting one failure kill the whole answer.

- [ ] **Step 5: Run the test**

```bash
.venv/bin/python -m pytest tests/test_orchestrator.py -v
```

Expected: the new test PASSES and all pre-existing orchestrator tests still pass.

- [ ] **Step 6: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ≥276 passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add aiu_chat/agent/orchestrator.py aiu_chat/config.py tests/test_orchestrator.py
git commit -m "Run independent routes in parallel

A multi-source answer cost the sum of its routes; the worst logged turn took
91s. Fan out to a bounded pool, preserving route order for stable synthesis."
```

---

## Task 7: Cache entity and sync lookups

**Files:**
- Modify: `aiu_chat/sources/dataapp.py` (`resolve_entity`, `latest_sync`)
- Modify: `aiu_chat/config.py`
- Test: `tests/test_dataapp_pagination.py` (extend)

**Interfaces:**
- Consumes: `resolve_entity(kind, query, session) -> Entity` from Task 1's module.
- Produces: same signature, now memoised. New config `DATAAPP_CACHE_TTL_S`; new function `clear_caches()` for tests.

- [ ] **Step 1: Add config**

```python
# Entity lookups ("France" -> id) are stable; cache them for a session to avoid
# re-resolving the same name on every turn. Sync ids change daily, so the TTL is
# short enough to pick up a new day's data.
DATAAPP_CACHE_TTL_S = int(os.getenv("AIU_DATAAPP_CACHE_TTL_S", "900"))
```

- [ ] **Step 2: Write the failing test**

```python
def test_entity_resolution_is_cached(monkeypatch):
    calls = {"n": 0}

    def counting_get(session, path, params):
        calls["n"] += 1
        return {"data": [{"id": 7, "name": "France", "code": "FR"}]}

    dataapp.clear_caches()
    monkeypatch.setattr(dataapp, "_get", counting_get)
    a = dataapp.resolve_entity("country", "France", None)
    b = dataapp.resolve_entity("country", "France", None)
    assert a.id == b.id == 7
    assert calls["n"] == 1, "second lookup should hit the cache"
```

- [ ] **Step 3: Run to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_dataapp_pagination.py::test_entity_resolution_is_cached -v
```

Expected: FAIL — either `clear_caches` is missing or `calls["n"] == 2`.

- [ ] **Step 4: Implement a small TTL cache**

Add to `aiu_chat/sources/dataapp.py`:

```python
_entity_cache: dict[tuple[str, str], tuple[float, Entity]] = {}


def clear_caches() -> None:
    """Drop memoised lookups (tests, and after a data refresh)."""
    _entity_cache.clear()
```

Wrap the existing `resolve_entity` body: check `_entity_cache` for a fresh
`(kind, query.lower())` entry before hitting the network, and store the result
on the way out. Keep the original resolution logic intact — rename it to
`_resolve_entity_uncached` and have `resolve_entity` be the caching wrapper, so
the network path stays exactly as tested today.

Do **not** cache `latest_sync` beyond the TTL — sync ids roll daily and a stale
id would silently serve yesterday's figure as today's.

- [ ] **Step 5: Run the tests**

```bash
.venv/bin/python -m pytest tests/test_dataapp_pagination.py -v
```

Expected: all PASS.

- [ ] **Step 6: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ≥277 passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add aiu_chat/sources/dataapp.py aiu_chat/config.py tests/test_dataapp_pagination.py
git commit -m "Cache Data App entity lookups with a short TTL"
```

---

## Task 8: Regression-test the four originally-failing questions

Turns the audit into a permanent guard so these specific failures cannot come back.

**Files:**
- Create: `tests/test_logged_regressions.py`

**Interfaces:**
- Consumes: `dataapp.fetch_timeseries` (Tasks 1-2), `aliases.resolve_near_miss` (Task 3).
- Produces: no production code.

- [ ] **Step 1: Write the tests**

```python
"""Regressions for failures found by auditing the logged conversations.

Each test names the logged turn it came from. Network-touching tests are marked
`live` so the default suite stays fast and offline:
    pytest -m live      # run these
    pytest -m "not live"
"""
from __future__ import annotations

import pytest

from aiu_chat.agent import aliases


def test_athens_acc_no_longer_dead_ends():
    """Turns 12 and 14: refused outright; must now clarify toward Greece."""
    nm = aliases.resolve_near_miss("Athens ACC")
    assert nm is not None and "Greece" in nm.candidates


@pytest.mark.live
def test_long_series_covers_the_whole_window():
    """Turns 18/19/21/22: answers silently stopped at 1 March (100-row cap)."""
    import requests
    from aiu_chat.sources import dataapp

    s = requests.Session()
    res = dataapp.fetch_timeseries(
        "traffic", start="2026-01-01", end="2026-06-01",
        kind="country", query="Spain", session=s)
    assert len(res.rows) > 100, "still truncated at the API page cap"
    assert res.rows[-1]["date"] >= "2026-05-25", res.rows[-1]["date"]
```

- [ ] **Step 2: Register the marker**

Add to `pyproject.toml` under `[tool.pytest.ini_options]` (create the table if absent):

```toml
markers = ["live: hits a live external API; excluded from the default run"]
```

- [ ] **Step 3: Run both ways**

```bash
.venv/bin/python -m pytest tests/test_logged_regressions.py -v -m "not live"
.venv/bin/python -m pytest tests/test_logged_regressions.py -v -m live
```

Expected: the offline test passes in the first run; the live test passes in the second.

- [ ] **Step 4: Full suite**

```bash
.venv/bin/python -m pytest -q
```

Expected: ≥278 passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_logged_regressions.py pyproject.toml
git commit -m "Pin regressions for the four audited conversation failures"
```

---

## Task 9: Re-audit and report

- [ ] **Step 1: Re-run the audit queries**

Re-run the PocketBase pull and confirm the failure signatures are gone from *new* turns: no `row_count == 100` on a multi-month request, no "I can't answer that" on a delay-cause question with a resolvable entity.

- [ ] **Step 2: Replay the four failing questions through the app**

```
Could you fetch the number of daily flights in SAS group airline from 1 january 2026 till 15 august 2026; visualise it
Could you analyse the number of daily flights in madrid airport from 1 january 2026 till 1 june 2026; visualise it
What are the delay causes for Athens ACC for the first six months of 2026?
Can you tell me the main causes of ATFM delay in Athens ACC on Week 27 this year?
```

Expected: the first two now narrate the **full** requested window (or say plainly why not); the last two ask the Greece clarifying question and then answer with real per-cause figures.

- [ ] **Step 3: Confirm the baseline held**

```bash
.venv/bin/python -m pytest -q
```

Expected: **0 failed**, pass count ≥263.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "Audit follow-up: verify logged failures are resolved"
```

---

## Self-Review

**Spec coverage.** Failure A → Tasks 1, 2, 8. Failure B → Tasks 3, 8. Failure C → Task 4. New OpenAI models → Task 5. Optimization (user chose "correctness + perf + improvements you deem needed") → Tasks 2, 6, 7. Leaked key → Task 0. Verification → Task 9. Every audited failure maps to a task.

**Deliberately out of scope.** The user chose *not* to restructure the oversized modules (`prompts.py` 1053 lines, `streamlit_app.py` 679, `orchestrator.py` 551), so no task splits them. Turn 6's out-of-scope refusal was correct behaviour and needs no fix. Turn 20 (`route=catalog`, 1 ms) worked as designed.

**Placeholder scan.** No TBDs. Two tasks intentionally require reading real code before finalising an edit — Task 6's `_answer_compound` signature and Task 4's exception class name — and both say so explicitly with the file and line to read, rather than inventing a signature the plan cannot verify.

**Type consistency.** `find_syncs_in_range -> list[tuple[int, str]]` is used consistently in Tasks 1, 2, 7. `NearMiss` fields (`query`, `candidates`, `reason`) and `.question()` are consistent across Tasks 3 and 8. `resolve_near_miss -> NearMiss | None` is used identically in both. Config names are unique and each is introduced exactly once.

**Risk note.** Task 3's alias map contains ~55 hand-written ANSP/FIR names. The FIR names were verified against `enroute_delay_fir.parquet` (62 entities, includes "Greece", "North Macedonia"); the **ANSP names were not individually verified** against the 50-entity ANSP list. Since the map only ever produces a *suggestion the user confirms*, a wrong ANSP string degrades to a slightly-off clarifying question rather than a wrong number — but Task 3 should cross-check the ANSP column against the parquet and drop any name that does not appear.
