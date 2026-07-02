# Design: Multi-source planner (feature #2)

Status: **proposed** · Target: `aiu-chat` · Depends on: entity layer (#1, done)

Let one question draw on **several** sources and synthesize a single grounded
answer — extending today's single-route dispatch (and the special-case `both`)
into a general N-route capability.

---

## 1. Why

Today `router.py` picks exactly **one** route, except `both` which is a
hard-coded data+concept pair. Real questions often span sources:

- *"How does Heathrow's delay this year compare to its 5-year average, and why is
  ASMA measured that way?"* → **dataapp** (this year) + **data** (historical avg)
  + **concept** (methodology).
- *"What's the live network situation and how does today's traffic compare to
  the yearly trend?"* → **nm_live** + **data**.

The pieces already exist — every path function (`answer_data_question`,
`answer_concept_question`, `answer_nop_question`, `answer_dataapp_question`,
`answer_nm_question`) is implemented, and `_combine()` already merges multiple
sub-answers. What's missing is (a) a router that can pick **more than one** route
and (b) an orchestrator loop that runs them and a synthesis step that reconciles
them into one answer.

## 2. Non-goals

- Not an open-ended agent loop (that's a later, heavier option). The planner
  picks a **set** of routes up front, deterministically dispatched.
- Not per-source fan-out (#3 — multiple calls to the *same* source) nor
  cross-source numeric aggregation (#4). Those layer on top.
- The **numbers-from-execution** rule is unchanged: each path still grounds its
  own figures; synthesis only stitches prose, never recomputes.

## 3. Design

### 3.1 Router returns a route *set*

Extend the router to optionally emit `routes: [...]` (1–3 routes) alongside the
existing `route: "..."`. Backward-compatible:

- Old single `route` still accepted → treated as a one-element set.
- `both` still accepted → expands to `["data", "concept"]` (keeps existing
  behaviour/eval green).
- New: `routes` may list any combination of `data | concept | dataapp | nm_live |
  nop` (not `none`; `none` is terminal and never combined).
- Cap at **3** routes to bound latency/cost; drop duplicates; ignore `none`
  inside a set.

The router prompt gains a short instruction + one example showing a
multi-source question mapping to a `routes` list. Parsing is defensive: any
malformed/oversized list falls back to the single best route (never worse than
today).

### 3.2 Orchestrator dispatches the set

`answer()` already runs per-route blocks. Refactor so the route-handling blocks
run for **each route in the set** (data, concept, dataapp, nm_live, nop), each
populating its slot on the `Turn` (`turn.data`, `turn.concept`, …) exactly as
now. `_combine()` already merges whatever slots are filled — so multi-route
"just works" once several slots are populated.

Clarification: only ask if a route in the set *needs* a subject and it's missing
(reuse `needs_clarification`, checked once on the primary route to avoid nagging).

### 3.3 Synthesis

`_combine()` today concatenates sub-answers with blank lines. For 2+ sources add
an optional **synthesis pass** (LLM) that takes the already-grounded sub-answers
+ their sources and writes one cohesive answer that:

- preserves every figure verbatim (no recomputation — a rule in the prompt),
- states each figure's source/as-of,
- reads as one answer, not stitched fragments.

Guarded: if synthesis fails or is disabled, fall back to today's concatenation.
Single-route answers **skip** synthesis (no behaviour change, no extra cost).

## 4. Config / flags

- `AIU_MULTI_SOURCE` (default on) — master switch for multi-route planning.
  Off → router collapses to one route (today's behaviour) and no synthesis pass.
- Route-set cap (`AIU_MAX_ROUTES`, default 3).

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Over-eager multi-routing balloons latency/cost (every question hits 3 sources) | Router prompt is conservative ("only add a route if the question genuinely needs it"); cap at 3; single-route stays the common case. Eval watches route counts. |
| Synthesis recomputes/garbles a figure | Prompt forbids recomputation; each sub-answer already carries the grounded number; synthesis is prose-only and guarded with a concat fallback. |
| Regression on existing single-route + `both` questions | `both` still maps to data+concept; single `route` unchanged; flag-gated; gold eval must stay green (esp. route-assertion cases). |
| A weak router picks the wrong *set* | Falls back to the single best route on any parse issue; no set is ever worse than one route. |

## 6. Build order (slices)

1. Router: accept + validate a `routes` set (with `route`/`both` back-comp);
   `route_question()` returns `list[str]`. Unit-tested on JSON shapes.
2. Orchestrator: dispatch the set (loop the existing per-route blocks); populate
   multiple `Turn` slots. Unit-tested with a fake client.
3. Synthesis pass in `_combine()` for 2+ sources, behind the flag, with concat
   fallback. Unit-tested (figures preserved).
4. Router prompt example + gold eval cases for genuine multi-source questions;
   full eval run.

Each slice is independently revertable; the flag defaults on but flips off to
exactly today's behaviour.
```
