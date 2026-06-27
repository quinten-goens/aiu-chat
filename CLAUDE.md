# CLAUDE.md — AIU Chat

Guidance for Claude Code when working in this repository.

## What this project is

A **fully local, offline AI chatbot** that answers questions about European Air
Navigation Services (ANS) performance using the open data and reference content
published by EUROCONTROL's Aviation Intelligence Unit (AIU) at
**https://ansperformance.eu**.

It is a **hybrid agentic RAG system**. The local LLM acts as a router/agent that,
per question, chooses between two grounded answer paths:

1. **Text-to-SQL over structured data** — for numeric/quantitative questions
   (delays, CO2 emissions, flight efficiency, traffic, taxi times, economics).
   The model writes SQL, DuckDB executes it against local Parquet files, and the
   model narrates the *executed* result. The model must **never** do arithmetic
   itself — numbers always come from query results. This is what makes a small
   (7B) model trustworthy on data. When the result is chart-worthy, the same rows
   are **also visualized** (see Visualization below) — a chart is just another
   rendering of the executed query result, not a separate data path.
2. **Vector retrieval over documents** — for conceptual/definitional questions
   (what is ASMA additional time? how is horizontal flight efficiency measured?).
   Retrieval runs over chunked-and-embedded reference/methodology/definitions
   pages stored as vectors in DuckDB (via the **VSS** extension), embedded with a
   local Ollama embedding model.

> **Architecture is deliberately just two local pieces:** **Ollama** (serves both
> the chat model *and* the embedding model) and **DuckDB** (holds both the
> structured Parquet data *and* the document vectors). No separate vector DB, no
> separate embedding stack. Streamlit is a thin HTTP/render layer on top.

The model may also answer "both" questions by combining a SQL result with
retrieved definitions.

## Hard constraints / non-negotiables

- **Fully local & offline after ingestion.** No external LLM APIs, no API keys.
  The only network access is the ingestion step that downloads data from
  ansperformance.eu / eurocontrol.int.
- **Numeric answers MUST come from executed SQL**, never from the model's own
  computation or memory. If a data question cannot be answered from the available
  tables, say so — do not estimate.
- **Generated SQL is read-only and sandboxed.** Open DuckDB in `read_only=True`.
  Reject/strip any statement that is not a single `SELECT`/`WITH` (no DDL/DML,
  no `PRAGMA`, no `ATTACH`, no `COPY ... TO`, no file functions writing to disk).
- **Treat LLM-generated SQL as untrusted input** (it gates the validation above).
- **No LLM-generated plotting/Python code.** Charts come from a small, validated
  JSON **chart spec** the model emits; deterministic code renders it. The model
  never writes matplotlib/Plotly/pandas code that gets executed.
- **Ground every answer.** Data answers cite the table(s) and the SQL used;
  document answers cite the source page/section. No source → say "I don't know".
- **State the data "as-of" date** on data answers; never imply the latest month is
  complete when it may be partial or not yet reported.
- **Always show an attribution/disclaimer footer:** data © EUROCONTROL AIU
  (ansperformance.eu); this is an **unofficial** tool, **not affiliated with or
  endorsed by EUROCONTROL**. Confirm reuse/attribution terms before publishing.
- **Log every turn** (route decision + generated SQL + chart spec + sources) for
  debugging; never silently swallow a failure.
- **Apple Silicon target.** Default model runs on Ollama, which uses **MLX** under
  the hood on Apple Silicon (Ollama ≥0.19) for the fastest Mac path. Keep memory
  footprint reasonable (default model ~6.6 GB at Q4 in unified memory).

## Tech stack (decided)

| Concern            | Choice                                                        |
|--------------------|--------------------------------------------------------------|
| UI                 | **Streamlit** chat app (`st.chat_message`), HTTP client only |
| Local LLM serving  | **Ollama** (OpenAI-compatible API at `http://localhost:11434`); uses MLX on Apple Silicon |
| Default model      | **`qwen3.5:9b`** (configurable via env var; Qwen3 7B for smaller Macs) |
| Structured engine  | **DuckDB** (read-only) over local **Parquet** files          |
| Data Q&A pattern   | **Text-to-SQL** (LLM writes SQL → DuckDB executes)           |
| Charts             | **Plotly** rendered from a validated LLM **chart spec** (JSON)|
| Doc embeddings     | **`nomic-embed-text` via Ollama** (same local service as the chat model) |
| Vector store       | **DuckDB VSS** extension (HNSW) — same engine as the data, no separate DB |
| Refresh            | **Scheduled** (launchd/cron, ~monthly)                       |
| Language           | Python 3.11+                                                  |

The UI never imports model/DB libraries directly — it talks to the agent layer,
which talks to only two backends: **Ollama** (chat + embeddings) and **DuckDB**
(data + vectors). Keep these layers separate.

## Data source: ansperformance.eu

EUROCONTROL Aviation Intelligence Portal (AIU, supporting the Performance Review
Commission). Two distinct content kinds — this split drives the whole design:

**Structured datasets (→ Parquet → DuckDB).** ~14 dataset groups, monthly
granularity, 2008–2026, dimensions like state / airport / ANSP / FIR. Includes:
ATC & all pre-departure delays, ATFM slot adherence, CO2 emissions by state,
horizontal & vertical flight efficiency, taxi-in/taxi-out additional time, ASMA
additional time, airport traffic, en-route delays by ANSP/FIR, airport arrival
ATFM delays, ACE economic/operational data. Available as CSV/Parquet (some
`.csv.bz2`); naming convention roughly `<dataset_group>_<YYYY>.csv[.bz2]`.
**Prefer Parquet** where available (typed, columnar, fast in DuckDB).

**Reference / textual content (→ chunks → embeddings → DuckDB VSS).** Definitions,
Methodology, Acronyms, Bibliography, dataset metadata, and reports (e.g. ACE
benchmarking). Use these to answer conceptual questions and to enrich the SQL
schema catalog with human-readable column/metric descriptions.

> Always verify current dataset names, columns, and download URLs against the
> live site during ingestion — do not hardcode assumptions from this doc. Be a
> polite scraper: identify a User-Agent, rate-limit, cache, and respect
> robots.txt / terms of use.

## Target repository layout

```
aiu-chat/
├── CLAUDE.md
├── README.md
├── pyproject.toml            # or requirements.txt
├── .env.example              # OLLAMA_HOST, MODEL_NAME, paths, etc.
├── config.py                 # central config from env
├── data/
│   ├── parquet/              # downloaded structured datasets (gitignored)
│   ├── aiu.duckdb            # DuckDB file: data views + doc-vector table (gitignored)
│   └── catalog.json          # schema catalog: tables, columns, descriptions
├── ingest/
│   ├── download_datasets.py  # fetch CSV/Parquet → data/parquet/
│   ├── build_catalog.py      # introspect parquet + merge metadata → catalog.json
│   ├── scrape_docs.py        # fetch reference/methodology/definitions pages
│   └── build_index.py        # chunk + embed (Ollama) → DuckDB VSS table
├── agent/
│   ├── llm.py                # thin Ollama client (chat + embeddings)
│   ├── router.py             # classify question: data | concept | both
│   ├── sql_tool.py           # schema-prompt → SQL → validate → DuckDB execute
│   ├── chart.py              # validate chart spec → build Plotly figure
│   ├── retriever.py          # embed query (Ollama) → DuckDB VSS similarity search
│   └── orchestrator.py       # ties router + tools + final synthesis
├── app/
│   └── streamlit_app.py      # chat UI; calls agent.orchestrator only
├── scripts/
│   └── refresh.sh            # run full ingestion (for cron/launchd)
└── tests/
    ├── eval/                 # gold Q→answer set + runner (run on every change)
    └── ...                   # unit tests (sql safety, chart spec, etc.)
```

(Adjust as the build proceeds, but keep ingest / agent / app cleanly separated.)

## How the agent answers a question

0. **Resolve follow-ups.** Before routing, rewrite the user's message into a
   **standalone question** using recent conversation history (e.g. "what about
   France?" after a Germany query → "Show <same metric> for France"). All later
   steps operate on this rewritten, self-contained question.
1. **Route.** `router.py` classifies the user question as `data`, `concept`, or
   `both` (LLM classification with a tight prompt, or rules + LLM fallback).
2. **Data path.** Inject the **schema catalog** (table names, columns, types,
   descriptions, a few example values, and notes on units/granularity) into the
   prompt. Model emits **one** SQL `SELECT`. `sql_tool.py` validates it
   (read-only, single statement, allowed tables), executes via DuckDB against
   `data/parquet/`, and returns rows.
3. **Concept path.** `retriever.py` embeds the query (Ollama `nomic-embed-text`)
   and pulls top-k chunks via DuckDB VSS similarity search, with their source
   URLs/sections.
4. **Visualize (data path only).** If the executed result is chart-worthy (a time
   series, a ranking/top-N, multi-row comparison), the model emits a **chart spec**
   (see Visualization) and `chart.py` renders a Plotly figure from the *same rows*.
   The prose answer is **always** produced regardless of whether a chart is shown.
5. **Synthesize.** The model writes the final answer grounded **only** in the
   executed rows and/or retrieved chunks, citing tables/SQL and/or doc sources,
   and the UI shows the chart (if any) alongside it.
   On empty/failed SQL or no relevant chunks → "I don't know / not in the data".

## Visualization (chart spec)

Charts are an extension of the data path, never a separate one — they render the
**same rows** DuckDB already returned. The model does **not** write plotting code;
it emits a small JSON **chart spec** that `chart.py` validates and turns into a
Plotly figure.

- **Trigger: auto, prose always.** Every data answer includes prose; a chart is
  *additionally* shown when the result is chart-worthy. Never replace prose with a
  chart. For tiny results (a single scalar/row) skip the chart.
- **Spec shape (keep minimal):**
  ```json
  {
    "show_chart": true,
    "chart_type": "line | bar | area | scatter",
    "x": "<column name from result>",
    "y": ["<one or more numeric column names>"],
    "series": "<optional column to split/color by>",
    "title": "<short title>"
  }
  ```
- **Validate the spec, don't trust it.** `chart.py` checks `chart_type` is in the
  allowed set and that `x`/`y`/`series` are **actual column names in the result
  DataFrame**. On any mismatch, skip the chart and just show prose + table — never
  error out the whole answer over a bad chart spec.
- **Render deterministically** with Plotly (`st.plotly_chart`). Always also offer
  the raw result table (e.g. `st.dataframe`) so the chart is auditable.

## Trustworthiness (this is what separates demo from tool)

A 9B model writing SQL over unfamiliar domain data will sometimes produce SQL that
**runs fine but is semantically wrong** (wrong aggregation, summing an already-
averaged column, mixing ATFM vs. all-causes delay). Sandboxing stops *malice*;
these measures stop *plausible-but-wrong answers*.

**1. Evaluation — gold test set (`tests/eval/`).**
- A YAML/JSON set of ~20–30 questions, each with the expected SQL *or* expected
  numeric answer (with tolerance) and/or expected source. Cover: simple lookups,
  aggregations, time series, top-N, filters, and at least a few known **trap**
  cases from the domain glossary below.
- A runner executes each question end-to-end and reports pass/fail. **Run it on
  every prompt or schema change** — it's the only objective signal that a tweak
  helped. Treat a drop in pass rate as a regression.

**2. Semantic guardrails in the schema catalog (`catalog.json`).**
For *each column*, record not just type but: **unit**, **granularity** (e.g.
"already a monthly average — do not SUM"), allowed/typical values, and gotchas.
This text is injected into the SQL-generation prompt. Pair it with a concise
**ANS domain glossary** (`docs/glossary` or part of the catalog) the model can
lean on. Key distinctions to encode:
- ATFM delay vs. all-causes delay; en-route vs. airport delay.
- "Additional time" metrics are vs. an unimpeded **baseline** — define it.
- IFR-flight counts vs. service units vs. movements.
- Per-flight averages vs. totals — never sum a per-flight/per-period average.

**3. Freshness & schema-drift checks (ingestion).**
- Record an **as-of date** per dataset at ingest; expose it so answers can state
  it and flag a partial latest month.
- On each scheduled refresh, **validate the schema** (expected columns, types)
  against the catalog and **fail loudly** on renames/format changes rather than
  silently producing broken queries. Surface the failure in logs / the refresh job.

## Conventions for Claude when building this

- **Build in vertical slices, smallest end-to-end first.** Suggested order:
  1. Ingestion of a *single* dataset → Parquet + a minimal `catalog.json`.
  2. `sql_tool.py` with read-only validation + DuckDB execution (unit-tested
     with hand-written SQL before involving the LLM).
  3. `llm.py` Ollama client; bare CLI loop that does text-to-SQL on that one
     dataset, end to end.
  4. **Stand up the gold eval set early** (even ~8 cases) against that one
     dataset, so every later change is measured. Grow it as datasets are added.
  5. Streamlit UI wrapping that loop (prose + raw result table).
  6. `chart.py`: chart-spec validation + Plotly rendering; wire auto-charting into
     the data path (unit-test spec validation against sample result frames).
  7. Doc scraping + embed (Ollama) into a DuckDB VSS table; add the concept path.
  8. Router + orchestrator to combine paths; add follow-up rewriting; expand to
     all datasets (with per-column semantic notes in the catalog).
  9. Scheduled refresh script with schema-drift validation + as-of capture.
- **Validate SQL safety with tests**, not just prompting. Adversarial cases
  (`DROP`, `ATTACH`, `COPY TO`, multiple statements, comments hiding payloads)
  must be rejected.
- **Validate chart specs with tests too.** Bad `chart_type`, columns not present
  in the result, missing `y` → render no chart (prose + table still shown), never
  crash the answer.
- **Make everything configurable via env** (`config.py`): `OLLAMA_HOST`,
  `MODEL_NAME`, `EMBEDDING_MODEL`, DuckDB/Parquet paths, `TOP_K`, etc.
- **Keep prompts in one place** (e.g. `agent/prompts.py`) so they're easy to
  iterate on.
- **Don't let the UI import DuckDB/Ollama directly** — go through `agent/`.
- **The VSS extension is loaded at runtime** (`INSTALL vss; LOAD vss;`); the
  read-only data connection and the vector store live in the same DuckDB file but
  the agent should not let LLM-generated SQL touch the vector table.
- **Fail loudly and honestly.** Surface SQL errors and empty results to the user
  rather than letting the model paper over them.
- **Log every turn** to a structured log: rewritten question, route decision,
  generated SQL, row count, chart spec, retrieved sources, latency. This is the
  primary tool for diagnosing wrong answers after the fact.
- **Run the gold eval set before considering any prompt/schema change done.**

## Future / nice-to-have (backlog — don't block the core build)

Not required for the core system to work; revisit once it's solid end-to-end.

- **Streaming responses** — stream tokens + show a "running query…" state so the
  SQL-then-prose latency of a 9B model feels responsive.
- **Caching** — memoize identical (rewritten) questions to skip the full chain.
- **Export** — let users download the result table (CSV) and chart (PNG/HTML).
  (Cheap in Streamlit; users always ask for it.)

(Observability/logging and the attribution+disclaimer footer are **not** backlog —
they're core requirements above.)

## Running (intended)

```bash
# one-time
ollama pull qwen3.5:9b                 # chat model (or qwen3:7b on smaller Macs)
ollama pull nomic-embed-text           # embedding model
pip install -e .                      # or: pip install -r requirements.txt
cp .env.example .env

# ingest data + build indexes (also the scheduled job)
bash scripts/refresh.sh

# run the app
streamlit run app/streamlit_app.py
```

## Open items to confirm during build

- Exact live dataset filenames / download URLs and whether Parquet is offered for
  each (fall back to CSV→Parquet conversion at ingest time if not).
- Which reference pages to scrape for the doc corpus (Definitions, Methodology,
  Acronyms at minimum) and how to chunk them.
- Scheduling mechanism on macOS (launchd plist vs. cron) for the monthly refresh.
