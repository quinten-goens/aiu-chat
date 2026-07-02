# Design: Entity / Knowledge Layer

Status: **proposed** · Author: design pass · Target: `aiu-chat`

A lightweight, deterministic entity layer that makes the domain's entities
(states, airports, ANSPs, FIRs) and their relationships explicit — the
foundation for multi-source questions (#2), per-source fan-out (#3), and
cross-source aggregation (#4). **This is not a graph database** and does not add
a third backend; it stays within the "just Ollama + DuckDB" constraint.

---

## 1. Why (grounded in the real data)

An inspection of the 13 datasets found three problems that a small entity layer
solves and that nothing in the current pipeline addresses:

| Problem | Evidence from the data | Consequence today |
|---|---|---|
| **State names don't match across tables** | `co2_emissions_by_state.STATE_NAME` = `ALBANIA`, `BELGIUM` (UPPERCASE); `airport_traffic.STATE_NAME` = `United Kingdom`, `France` (Title Case). A join on `STATE_NAME` between the two returns **0 rows**. | Any cross-source question about a country silently returns nothing or wrong data. A single JOIN the model writes will fail quietly. |
| **Entity names inconsistent within a table** | `LFPG` appears as both `Paris-Charles-de-Gaulle` and `Paris - Charles-de-Gaulle`. | Grouping/filtering by name double-counts or misses rows; only `APT_ICAO` is reliable. |
| **Three separate entity vocabularies** | `STATE_NAME`/`STATE_CODE` (ICAO 2-letter), `APT_ICAO`/`APT_NAME`, `ENTITY_NAME`/`ENTITY_TYPE` (ANSP names like `DSNA`; FIR names mixing `Austria` COUNTRY(FIR) and `Baltic FAB` FAB(FIR)). | The model must reconcile these in its head per query; no deterministic mapping exists. |

Cardinality is small and curatable: **44 states, 336 airports, 50 ANSPs**, ~30
FIRs/FABs. This is enumerable, not "big data".

### What the entity layer enables (concretely)

1. **Deterministic entity resolution.** "Heathrow" / "LHR" / "EGLL" / "London
   Heathrow" → canonical `EGLL`. "UK" / "United Kingdom" / "GB" → canonical
   state. Removes a whole class of wrong-filter bugs.
2. **Cross-source joins that actually work.** A canonical `state_id` bridges
   `ALBANIA` (CO2 table) and `Albania` (FIR table) and `United Kingdom`
   (airport table) — the prerequisite for feature #2.
3. **Multi-hop / relationship questions.** `airport → state → ANSP`,
   `airport → FIR`. "Which ANSP covers Heathrow?" becomes a lookup, not a guess.
4. **Better SQL prompting.** Inject a compact "here are the canonical entities
   and their aliases" block so the model filters on the right column/value.
5. **Fan-out targeting (#3).** "Compare FR, DE, ES" resolves to three canonical
   entities the fetch loop can iterate deterministically.

### Why NOT a graph database

- Violates the two-backend constraint (Ollama + DuckDB only).
- The relationships here are shallow (1–2 hops) and nearly static; a graph
  engine's traversal power is unused.
- DuckDB already does the joins once a shared key exists. The value is the
  **canonical keys + alias table**, not a traversal engine.

---

## 2. Data model

Two artefacts, both built at ingest, both living **inside the existing DuckDB**
(and mirrored to JSON for prompting/tests). No new service.

### 2.1 `entities` (canonical nodes)

| column | type | notes |
|---|---|---|
| `entity_id` | text (PK) | stable canonical id, e.g. `apt:EGLL`, `state:GB`, `ansp:DSNA`, `fir:LFEE` |
| `kind` | text | `state` \| `airport` \| `ansp` \| `fir` |
| `canonical_name` | text | display name, e.g. `London — Heathrow`, `United Kingdom` |
| `icao` | text | ICAO code where applicable (`EGLL`, or 2-letter state prefix) |
| `iata` | text | airports only, where known (`LHR`) — best-effort |
| `state_id` | text | FK → the state entity this belongs to (airport/ANSP/FIR → state) |

### 2.2 `entity_aliases` (all the surface forms → canonical)

| column | type | notes |
|---|---|---|
| `alias` | text | a normalised surface form (`heathrow`, `united kingdom`, `uk`, `gb`, `london heathrow`, `paris charles de gaulle`) |
| `entity_id` | text | FK → `entities.entity_id` |
| `source` | text | where the alias came from: `data` (a value seen in a table), `iata`, `manual` |

Aliases are **normalised** (lowercased, punctuation/spacing collapsed) so
`Paris-Charles-de-Gaulle` and `Paris - Charles-de-Gaulle` both map to `apt:LFPG`.

### 2.3 Relationships

Relationships are **columns on `entities`** (`state_id`), not a separate edge
table — the graph is shallow enough that FK columns suffice. If deeper edges are
ever needed (airport→FIR, ANSP→FIRs-managed), add a thin `entity_edges(src,
rel, dst)` table later. Not needed for v1.

### 2.4 Column map (which table column holds which entity)

A small static map (in `catalog.json` or a sibling file) records, per dataset,
which column is the entity key and of what kind:

```json
"entity_binding": {
  "co2_emissions_by_state":  {"kind": "state",   "name_col": "STATE_NAME", "code_col": "STATE_CODE"},
  "airport_traffic":         {"kind": "airport", "code_col": "APT_ICAO",   "name_col": "APT_NAME", "state_col": "STATE_NAME"},
  "enroute_delay_ansp":      {"kind": "ansp",    "name_col": "ENTITY_NAME"},
  "enroute_delay_fir":       {"kind": "fir",     "name_col": "ENTITY_NAME"}
}
```

This is what lets resolution know *"a state filter on `airport_traffic` uses
`STATE_NAME='United Kingdom'`, but on `co2_emissions_by_state` it's
`STATE_NAME='UNITED KINGDOM'` (uppercase)"* — i.e. the canonical entity carries
**per-table filter values**.

---

## 3. How it's built (ingest)

New module `ingest/build_entities.py`, run as part of `scripts/refresh.sh`:

1. For each dataset with an `entity_binding`, `SELECT DISTINCT` the entity
   column(s) via DuckDB (fast, already how the catalog is built).
2. **Canonicalise:**
   - States keyed by **ISO country code** (from OurAirports `countries.csv`) →
     `state:<iso>` (e.g. `state:GB`). Map every `STATE_NAME` variant (`ALBANIA`,
     `Albania`, `United Kingdom`) and the ICAO 2-letter `STATE_CODE` to it as
     aliases, and record the per-table exact value for filtering.
   - Airports keyed by `APT_ICAO` → `apt:<icao>`. First-seen `APT_NAME` becomes
     `canonical_name`; all name variants become aliases; link `state_id` via the
     row's `STATE_NAME` → resolved state.
   - ANSPs keyed by a slug of `ENTITY_NAME` → `ansp:<slug>`.
   - FIRs keyed by `ENTITY_NAME`, `kind=fir` (skip FAB aggregates or mark
     `kind=fab`).
3. **Enrich from OurAirports (§3.1).** Join the external reference data (fetched
   at ingest, cached, committed as a snapshot) to the entities derived above,
   keyed by ICAO / ISO country code. This replaces hand-curated seeds with
   automatic, full-coverage enrichment for all 336 airports and 44 states.
4. Write `entities` + `entity_aliases` tables into DuckDB **and** dump
   `data/entities.json` for prompting and unit tests.
5. **Schema-drift guard** (per the freshness requirement): if a dataset's
   entity column disappears or a previously-known entity vanishes, log loudly.

Cost: trivial — a handful of `SELECT DISTINCT`s over already-local parquet, plus
two small CSV joins.

### 3.1 External enrichment: OurAirports (Role: reference, not queryable)

Two public-domain CSVs from [OurAirports](https://ourairports.com/data/) enrich
the entity layer. They are **never SQL-queryable performance data** — they only
populate the canonical entity/alias rows. Fetched at ingest, cached, and a
snapshot committed so ingestion stays reproducible offline. **License: public
domain** ("released to the Public Domain … credit appreciated but not required")
— compatible with the project's terms; no attribution constraint.

**`airports.csv`** — join to your 336 airports on `icao_code = APT_ICAO` (take
only the intersection; OA has ~85k airports, you want your 336). Fields used:

| OA field | Feeds |
|---|---|
| `icao_code` | join key → `apt:<icao>` |
| `iata_code` (e.g. `LHR`, `CDG`) | **new** — you have zero IATA today; becomes an alias + `entities.iata` |
| `name` (`London Heathrow Airport`) | canonical name candidate + alias |
| `municipality` (`London`, `Paris`) | alias (colloquial "London", "Paris") |
| `keywords` (`LON, Londres` / `PAR, Roissy Airport`) | **pre-made aliases**, incl. foreign-language forms — replaces manual curation |
| `latitude_deg`, `longitude_deg` | stored on `entities` (future geo/nearest queries) |
| `iso_country` (`GB`, `FR`) | links airport → canonical state (below) |

**`countries.csv`** — small ISO `code ↔ name` map. This is the piece that closes
the **state-name smoking gun**: your tables disagree (`ALBANIA` vs `Albania` vs
`United Kingdom`), but every one resolves to a neutral ISO code
(`AL`, `GB`) via this map, which becomes the canonical `state:<iso>` id. Each
raw `STATE_NAME` variant is then recorded as a per-table filter value against
that canonical id (§2.4). ANSPs and FIRs are **not** covered by OurAirports and
stay derived from your own data (deferred relationship enrichment — see §9 Q4).

> Note the canonical state id shifts from `state:<ICAO 2-letter>` to
> `state:<ISO country code>` now that `countries.csv` gives a clean ISO anchor;
> the ICAO 2-letter `STATE_CODE` is retained as an alias.

---

## 4. How it's used (agent)

New module `agent/entities.py` with a pure, cached resolver:

```python
resolve("heathrow")            -> Entity(id="apt:EGLL", kind="airport", state_id="state:GB", ...)
resolve("UK", kind="state")    -> Entity(id="state:GB", ...)
filter_value("apt:EGLL", table="airport_traffic")  -> ("APT_ICAO", "EGLL")
filter_value("state:GB", table="co2_emissions_by_state") -> ("STATE_NAME", "UNITED KINGDOM")
```

Three integration points, each **additive and independently shippable**:

1. **SQL prompt enrichment (first, lowest-risk).** When the question mentions an
   entity, inject a short resolved block into the SQL-gen prompt:
   *"Resolved entities: Heathrow → airport EGLL; on airport tables filter
   `APT_ICAO='EGLL'`."* This alone fixes most wrong-filter/wrong-case bugs with
   zero change to the control flow. Measure with the eval set.
2. **Data App / live fan-out targeting (feeds #3).** `resolve()` turns a list of
   mentioned entities into canonical ids the fetch loop iterates.
3. **Cross-source key (feeds #2/#4).** The canonical `state_id`/`apt` id is the
   join key when combining a SQL frame with a Data App frame.

The resolver is **advisory** to the LLM, never a hard gate — an unresolved
entity falls back to today's behaviour (model guesses), so it can't regress a
currently-working question into a failure.

---

## 5. Prompts

- Add an optional `resolved_entities` section to `build_sql_messages`.
- No new free-form model output to validate (resolution is deterministic Python),
  so **no new safety surface** — this is a big plus vs. giving the model more
  latitude.

---

## 6. Evaluation

Add gold cases to `tests/eval/` that are **known to break today**:

- "CO2 for the UK in 2024" — must filter `UNITED KINGDOM` (uppercase) correctly.
- "Compare traffic and CO2 for Germany" — cross-source, must join on a canonical
  state, not raw `STATE_NAME` (which mismatches).
- "Delays at Charles de Gaulle" — alias + name-variant resolution to `LFPG`.
- "Which state is Heathrow in?" — relationship lookup (`apt:EGLL.state_id`).
- A trap: an entity that exists in one table but not another → graceful "not in
  that dataset", not a silent empty.

Unit tests for `agent/entities.py`: alias normalisation, ambiguous names,
per-table filter-value mapping, unknown entity fallback.

**Gate:** run the full eval set before/after; the entity block must not lower the
pass rate on existing cases and should raise it on the new ones.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Ambiguous aliases** (e.g. "London" → LHR/LGW/LCY/STN) | Resolver returns *candidates*; if >1 and the question is entity-critical, surface a clarifying question (the pipeline already supports clarification) rather than guessing. |
| **Stale entities after a data refresh** | Rebuilt every ingest from `SELECT DISTINCT`; drift guard logs additions/removals. |
| **OurAirports drift** (upstream CSV changes/removes an airport) | Snapshot committed at ingest for reproducibility; join is by ICAO to *your* 336 airports, so upstream additions are ignored and a missing match degrades to name-only aliases, never a hard failure. |
| **Over-engineering into a real KG** | Explicit non-goal. FK columns only; add `entity_edges` **only** if a concrete multi-hop question needs it. |
| **Regression on working questions** | Resolver is advisory; eval-gated; feature can sit behind a config flag `AIU_ENTITY_LAYER=true` for a safe rollout. |

---

## 8. Build order (vertical slices)

0. `ingest/download_ourairports.py` → fetch + cache + commit a snapshot of
   `airports.csv` and `countries.csv` (public domain). One-time-ish; part of the
   refresh job.
1. `ingest/build_entities.py` → `entities`/`entity_aliases` tables +
   `entities.json`, for **states + airports only** (the two with the proven
   mismatch), enriched from the OurAirports snapshot. Drift guard. Unit tests on
   the builder.
2. `agent/entities.py` resolver + `filter_value()`. Unit-tested against
   `entities.json`. No agent wiring yet.
3. Wire **SQL prompt enrichment** (integration point #1) behind
   `AIU_ENTITY_LAYER`. Add the eval cases. Measure.
4. Extend to **ANSPs + FIRs**; add the relationship (`state_id`) lookups.
5. Expose `resolve()` to the Data App path as the targeting primitive for #3.

Each slice is independently useful and independently revertable.

---

## 9. Open questions for you

1. ~~**Seed curation scope**~~ — **resolved:** airport IATA/aliases and state
   ISO anchors come automatically from OurAirports (§3.1), for all 336 airports
   and 44 states. No manual seed. Grow further from the logged miss-list only if
   needed.
2. **Ambiguity policy** — for "London", prefer (a) clarify, (b) default to the
   busiest (EGLL), or (c) return all and let SQL group? I lean (a).
3. **Flag vs. always-on** — ship behind `AIU_ENTITY_LAYER` first, or straight in
   once eval passes? I lean flag-first.
4. **FIR/ANSP relationship enrichment** — **deferred** (your call): no clean open
   source, so airport→FIR / ANSP→FIR edges wait until logged questions show
   demand. ANSPs/FIRs are still resolved from your own data (names + aliases),
   just without external relationship edges. (Also TBD: treat FABs like Baltic
   FAB / BLUE MED FAB as entities, or only country-FIRs?)
