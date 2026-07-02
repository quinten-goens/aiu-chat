# Design: Per-source fan-out (feature #3)

Status: **proposed** · Target: `aiu-chat` · Depends on: entity layer (#1, done)

Let one question trigger **several requests to the same source**, each for a
different entity, merged into one result before narration. The motivating case:
the **Data App API is per-entity** — "compare traffic for France, Germany and
Spain" needs three separate fetches today's path can't do (it resolves one
entity).

---

## 1. Why

`answer_dataapp_question()` extracts exactly one `entity` and fetches it once. So:

- *"Compare today's traffic for France, Germany and Spain"* → only one country
  answered (or the model arbitrarily picks one).
- *"Latest delay for DSNA and DFS"* → same.

The SQL path doesn't need this (one query with `WHERE state IN (...)` handles
multi-entity), but the **live per-entity APIs** (Data App, and later NM/NOP
per-airport) genuinely require N calls. This is fan-out: extract a **list** of
entities, fetch each, merge.

## 2. Scope

- **v1: the Data App path only** (the clear per-entity case). NM/NOP fan-out can
  follow the same shape later if needed.
- Not SQL (multi-entity is one query). Not cross-source (#2) nor numeric
  cross-entity aggregation (#4 — though a merged multi-entity table is a natural
  input to it).
- Numbers-from-source rule unchanged: each fetch is grounded; narration quotes
  the merged records, never recomputed.

## 3. Design

### 3.1 Extract a list of entities

Extend the Data App extract prompt to return `entities: [...]` (each an
`{entity, entity_kind}`), while still accepting the single `entity`/`entity_kind`
(back-compat). One `metric` applies to all (the common case: same metric across
several entities). Cap at **`AIU_MAX_FANOUT`** (default 5) to bound cost.

Optionally resolve/normalise each surface form through the **entity resolver
(#1)** so "UK"/"United Kingdom"/"GB" collapse before hitting the API.

### 3.2 Fan-out fetch loop

Deterministic loop (no LLM): for each (metric, kind, entity) call the existing
`fetch_metric()`. Per-entity failures are **isolated** — one entity that fails to
resolve/fetch doesn't sink the others; it's recorded and reported. Results are
collected into a list of `DataAppResult`.

### 3.3 Merge + narrate

Tag each result's records with its entity name and concatenate into one record
set (a small table: entity | metric | period | value | as-of). Narrate once over
the **merged** records so the answer compares them directly. `DataAppAnswer`
grows an optional `results: list[DataAppResult]` (the single `result` stays for
back-compat / the one-entity case).

## 4. Config / flags

- `AIU_FANOUT` (default on) — enable multi-entity fan-out; off → today's single
  entity.
- `AIU_MAX_FANOUT` (default 5) — cap on entities per question.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| N API calls per question (latency/cost) | Capped at 5; only fires when the model actually extracts multiple entities; single-entity stays the common path. |
| One bad entity fails the whole answer | Per-entity try/except; failures collected and surfaced ("no data for X"), others still answered. |
| Model over-extracts entities | Prompt: only list entities explicitly named; cap enforced server-side; extras dropped. |
| Regression on single-entity questions | `entities` falls back to the single `entity`; `DataAppAnswer.result` preserved; flag-gated; existing dataapp eval cases must stay green. |
| Merged narration recomputes/compares wrongly | Narration prompt forbids recomputation; each value is quoted per entity from its own fetch. |

## 6. Build order (slices)

1. Extract: prompt returns `entities` list (back-compat single entity);
   `answer_dataapp_question` reads a normalised entity list. Unit-tested.
2. Fan-out loop + merge: fetch each, isolate failures, tag records by entity,
   build merged record set. Unit-tested with a fake `fetch`.
3. Merged-narration prompt + `DataAppAnswer.results`; UI note that N entities
   were looked up. Flag-gated. Gold eval: add a multi-entity Data App case.

Each slice is independently revertable; the flag off = exactly today's behaviour.
```
