# Multi-source intelligence — design docs

This branch (`feature/multi-source-intelligence`) implements four related
capabilities that together let the assistant answer richer questions by drawing
on several data sources, iterating over a source, and aggregating across them.
They layer on top of each other:

```
#1 entity/knowledge layer  →  feeds  →  #2 multi-source planner
                                             ↓ dispatches
                                       #3 per-source fan-out (N fetches)
                                             ↓ collects frames
                                       #4 deterministic aggregation + synthesis
```

| # | Feature | Design doc | Status |
|---|---------|-----------|--------|
| 1 | **Entity / knowledge layer** — canonical entities + aliases (states, airports; ANSPs/FIRs later) built at ingest inside DuckDB, enriched from OurAirports; deterministic resolver. The backbone that makes the rest reliable. | [entity-layer.md](entity-layer.md) | design ✅ · **impl ✅** |
| 2 | **Multi-source planner** — router returns *multiple* routes; orchestrator runs each and synthesizes one grounded answer. | [multi-source-planner.md](multi-source-planner.md) | design ✅ · **impl ✅** |
| 3 | **Per-source fan-out** — extract a list of entities/metrics and loop N fetches to the same source (e.g. Data App per-country), merged into one table. | [per-source-fanout.md](per-source-fanout.md) | design ✅ · **impl ✅** |
| 4 | **Aggregation agent** — a second pass that reconciles multi-source frames via a **deterministic DuckDB query** (model never does arithmetic) and narrates. | _tbd_ | not started |

## Principles carried from CLAUDE.md

- **Numbers come from executed queries, never the model.** #4's aggregation is a
  validated DuckDB query over collected frames, not model math.
- **Two backends only** (Ollama + DuckDB). #1 is tables inside the existing
  DuckDB, not a graph database.
- **Additive & eval-gated.** Each feature ships behind a flag where sensible and
  must not lower the gold-eval pass rate on existing questions.

Build order and per-feature slices are in each design doc. Start: #1, slice 1
(`ingest/build_entities.py` for states + airports).
