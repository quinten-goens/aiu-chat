# Design: Aggregation agent (feature #4)

Status: **proposed** · Target: `aiu-chat` · Depends on: #2 (multi-source), #3 (fan-out)

A second pass that reconciles the frames produced by a multi-source (#2) or
fan-out (#3) turn and computes **cross-frame** figures — totals, differences,
"which grew most", shares — via a **deterministic DuckDB query over the collected
frames**, then narrates. The model decides *what* to compute; it never does the
arithmetic itself (the core trustworthiness rule).

---

## 1. Why

After #2/#3 a turn can hold several result frames:

- a SQL `DataAnswer` dataframe (historical),
- N fan-out `DataAppResult` record sets (per entity),

but each is narrated on its own. Questions like *"combined traffic across France,
Germany and Spain"* or *"which of these three grew the most vs last year"* need a
figure computed **across** frames. Today the only options are (a) the model does
the mental math (forbidden — unreliable) or (b) no answer. #4 adds a safe path:
compute it with SQL over the frames.

## 2. Scope & non-goals

- **In:** a single extra deterministic aggregation query over the frames a turn
  already produced, when the question asks for a cross-frame figure; then a short
  narration of that executed result.
- **Out:** open-ended multi-step agent loops; re-fetching new data; any model
  arithmetic. If no cross-frame computation is requested, #4 does nothing.
- **Hard rule preserved:** numbers come from an executed query. The aggregation
  SQL is validated and run by deterministic code; the model only picks the
  aggregation intent and narrates the result.

## 3. Design

### 3.1 Collect frames

After dispatch, gather the turn's frames into named pandas DataFrames:

- `data` — the SQL result dataframe (if any).
- `dataapp` — the merged, entity-tagged fan-out records (if any), as one frame.
- (extensible: nm_live/nop are not tabular, so excluded for v1.)

If fewer than one tabular frame with >1 row exists, skip #4 entirely.

### 3.2 Decide whether to aggregate

A cheap LLM step (or a rule) decides if the question wants a cross-frame figure
(keywords: total/combined/sum, difference/vs, share/percentage, which
most/least, average across). If not → skip; the per-frame answers stand.

### 3.3 Deterministic aggregation query

The model emits **one** aggregation SQL `SELECT` over the frames, referenced by
fixed view names (`data`, `dataapp`). It is executed by a **separate, tightly
scoped executor** (`agg_tool.py`) that:

- registers each frame as a DuckDB view under its fixed name (in-memory,
  read-only connection);
- reuses the existing `sqlglot` validation (single SELECT, no DDL/DML, no file
  functions) **but** with the allowed-table set = the registered view names (not
  the catalog). This keeps the main SQL path's catalog restriction intact while
  letting aggregation touch only the in-memory frames.
- caps rows; never touches disk or the catalog data.

This is why it's a *separate* executor: it must allow different table names
(`data`/`dataapp`) than the catalog-restricted main path, without weakening
either.

### 3.4 Narrate

Narrate the executed aggregation result with the existing grounded-answer prompt
(quote values, no recomputation). The aggregation answer is appended to / merged
with the turn's answer (via the same synthesis path as #2).

## 4. Config / flags

- `AIU_AGGREGATION` (default **off** initially — it's the most complex and least
  common; opt-in until proven). Master switch.
- Reuses `AIU_MAX_RESULT_ROWS` for the row cap.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Model does arithmetic in prose anyway | Narration prompt forbids it; the figure comes from the executed aggregation frame; eval traps a wrong total. |
| Aggregation SQL escapes the sandbox | Same sqlglot validation as the main path (single SELECT, no DDL/DML/file funcs); executor is in-memory, read-only, view-only. |
| Frame schemas vary (column names) | The model is shown each frame's columns + a few rows before writing the aggregation SQL (like the main SQL prompt); bad column → validation/execution error → skip, per-frame answers stand. |
| Over-triggering adds latency to simple turns | Default off; the aggregate-intent check is conservative; only fires on 2+ frames. |
| Complexity/regression | Fully additive, flag-gated (off), separate module; existing paths unchanged; skip on any failure. |

## 6. Build order (slices)

1. `agent/agg_tool.py`: register frames as views + validate (reuse sqlglot core
   with a view-name allow-set) + execute read-only. Unit-tested with hand-written
   aggregation SQL over sample frames (no model).
2. Aggregate-intent detector + aggregation-SQL prompt; wire into the orchestrator
   after dispatch, behind `AIU_AGGREGATION`. Unit-tested with a fake client.
3. Narration + merge into the turn answer; gold eval case (combined total across
   fan-out entities). Full eval run before enabling by default.

Each slice is independently revertable; the flag defaults OFF, so the system
behaves exactly as before #4 until explicitly enabled.
```
