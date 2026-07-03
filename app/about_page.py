"""About page content (rendered via st.navigation from streamlit_app.py).

This page is intentionally detailed: it doubles as living documentation of how the
system is built. Diagrams are drawn with Mermaid, rendered by the
``streamlit-mermaid`` component (its own bundled frontend — no CDN and no
Graphviz/Mermaid system binary on the host), which auto-sizes so the diagrams
don't overlap the surrounding text.
"""
from __future__ import annotations

import streamlit as st
from streamlit_mermaid import st_mermaid

from aiu_chat import config


def _term(title: str, body: str) -> None:
    """A small collapsible 'jargon buster' box for one IT/LLM/RAG term."""
    with st.expander(f"📖 {title}"):
        st.markdown(body)


def _mermaid(diagram: str, *, height: int = 480) -> None:
    """Render a Mermaid diagram via the streamlit-mermaid component.

    Streamlit has no native Mermaid support (a ```mermaid fence renders as an
    inert code block), so we use the maintained ``streamlit-mermaid`` component.
    An explicit pixel height is passed (sized per diagram) so the block reserves
    the right vertical space and the figure never overlaps the text around it.
    """
    st_mermaid(diagram.strip(), height=f"{height}px")


def render():
    st.title("ℹ️ About Aviation Intelligence Chat")
    st.caption(
        "A hybrid AI assistant for European air navigation performance data — "
        "and a guided tour of exactly how it works under the hood."
    )

    llm_phrase = (
        "a local model running on-device via Ollama"
        if config.LOCAL
        else "an OpenAI GPT model"
    )

    st.markdown(
        f"""
This app answers questions about European Air Navigation Services (ANS)
performance, built on the **data and expertise of the EUROCONTROL Aviation
Intelligence Unit (AIU)**.

The substance comes from the AIU: authoritative datasets and live operational
sources, a **smart routing system** that sends each question to the right one,
and **deterministic SQL and retrieval** so figures are read straight from the
data — never made up. {llm_phrase.capitalize()} is used only as the language
layer: it turns your question into a query and the results into a readable
answer. The numbers and facts come from the AIU data, not the model.

> **The one idea to take away:** the AI never *calculates* a number and never
> *recalls* a figure from memory. It only ever (a) picks the right source, (b)
> writes a **query** or **search**, and (c) puts the **executed result** into
> words. Every figure you see was produced by running code against real data.
"""
    )

    # ---------------------------------------------------------------- glossary
    st.divider()
    st.header("🧠 Jargon buster — the concepts behind it")
    st.markdown(
        "This system combines several AI / data-engineering techniques. Expand any "
        "term below for a plain-language explanation of *what it is* and *how it's "
        "used here*."
    )

    _term(
        "RAG — Retrieval-Augmented Generation",
        """
**What it is.** Instead of trusting a language model to *remember* facts, RAG
*retrieves* the relevant facts first and hands them to the model as context, so
the answer is grounded in real source material rather than the model's memory.

**How it's used here.** This app is a **hybrid *agentic* RAG** system. "Hybrid"
because it retrieves from two very different kinds of source — **structured data**
(numbers, via SQL) and **unstructured text** (documents, via vector search).
"Agentic" because a router first *decides* which source(s) to use per question,
rather than always doing the same thing.
""",
    )
    _term(
        "Text-to-SQL",
        """
**What it is.** The model translates a plain-English question ("which 5 states
emitted the most CO2 in 2024?") into a database query (`SELECT ... ORDER BY ...
LIMIT 5`). A database engine runs that query and returns the actual rows.

**How it's used here.** This is how *quantitative* questions are answered. The
model writes **one** read-only `SELECT`; DuckDB executes it against the bundled
datasets; the model then narrates the **executed rows**. Because the numbers come
from the query result, even a small model stays trustworthy on data — it can't
fudge a total it never computed.
""",
    )
    _term(
        "Embeddings & vector search",
        """
**What it is.** An *embedding* turns a piece of text into a list of numbers (a
*vector*) that captures its meaning. Texts with similar meaning end up close
together in this numeric space. To find relevant passages, you embed the question
and look for the document chunks whose vectors are **nearest** to it (measured by
*cosine similarity*).

**How it's used here.** Reference pages (definitions, methodology, acronyms) and
PDFs are split into chunks, each embedded once at build time. At question time the
query is embedded (`{emb}`, {dim}-dimensional) and the closest chunks are
retrieved to answer conceptual questions like *"what is ASMA additional time?"*.
""".format(emb=config.EMBEDDING_MODEL, dim=config.EMBEDDING_DIM),
    )
    _term(
        "Vector store & the DuckDB VSS / HNSW index",
        """
**What it is.** A *vector store* holds all those embedding vectors and searches
them quickly. **HNSW** (Hierarchical Navigable Small World) is a graph index that
finds nearest neighbours in roughly *logarithmic* time instead of comparing
against every vector.

**How it's used here.** Rather than run a *separate* vector database, the vectors
live in **DuckDB** via its **VSS** extension (`INSTALL vss; LOAD vss;`) — the same
engine that holds the numeric datasets. One file, one dependency, both jobs:
`array_cosine_similarity(...)` ranks the chunks and returns the top **{k}** above
a similarity floor of **{sim}**.
""".format(k=config.TOP_K, sim=config.MIN_SIMILARITY),
    )
    _term(
        "Routing (the 'agent' part)",
        """
**What it is.** A classifier that reads the question and decides *which tool /
source* should answer it — and it may pick **several** at once.

**How it's used here.** A tight LLM prompt maps each question to one or more of
six routes (**data · concept · nop · dataapp · nm_live · none**). Multi-source is
on, so a question that needs *a figure and a definition* can be sent to two paths
and their grounded answers stitched together.
""",
    )
    _term(
        "Question decomposition & follow-up rewriting",
        """
**Decomposition.** A compound question ("flights on 10 Mar 2026 **and** on 10 Mar
2025?") is split into independent **standalone sub-questions**, each routed and
answered on its own, then merged — so the two halves can use different dates,
metrics or subjects.

**Follow-up rewriting.** An elliptical follow-up ("what about France?") is
rewritten into a self-contained question using the recent conversation (including
the previous SQL), so every later step sees a complete question.
""",
    )
    _term(
        "Chart spec (why the AI never writes plotting code)",
        """
**What it is.** Letting a model emit and run plotting code is a security and
correctness risk. Instead the model emits a tiny **JSON description** of the chart
— *type, x-axis, y-column(s), optional series* — and deterministic code we control
turns it into a figure.

**How it's used here.** The spec is **validated against the real result columns**
(unknown column, wrong type, or too many rows per point → no chart, but prose +
table still shown). Valid specs are rendered with **Plotly**. The model never
touches matplotlib/pandas/plotting code.
""",
    )
    _term(
        "Sandboxed, read-only SQL (treating model output as untrusted)",
        """
**What it is.** Generated SQL is treated as **untrusted input**. Defence in depth
stops both malice (`DROP`, `ATTACH`, writing files) and accidents.

**How it's used here.** Every query is (1) parsed with `sqlglot` and required to
be **exactly one `SELECT`/`WITH`**; (2) walked to reject DDL/DML/`PRAGMA`/`ATTACH`/
`COPY` and file-reading functions; (3) restricted to **known catalog tables**; (4)
executed against a **read-only** DuckDB connection with a row cap. Anything that
fails is rejected *before* it reaches the engine.
""",
    )
    _term(
        "\"D-1\", sync model & as-of date",
        """
**D-1** means "yesterday": the daily **Data App** API publishes one *sync* per day,
so the freshest figure available is the previous day's. The app states the **as-of
date** on data answers and never implies an incomplete latest period is final.

The Data App works on a **sync** model: one call lists the sync ids in a date
window, then each day's metric is read from its sync. The genuinely *real-time*
figures (aircraft airborne *now*) come from a different source — the **Network
Manager** live API — not the Data App.
""",
    )

    # ------------------------------------------------------------ data sources
    st.divider()
    st.header("📚 Data sources")
    st.markdown(
        """
**1. Historical performance datasets — bundled**
13 EUROCONTROL Aviation Intelligence Unit (AIU) datasets
([ansperformance.eu](https://ansperformance.eu)) covering CO2 emissions, airport
& en-route traffic, ATFM delays (airport, en-route by ANSP/FIR), flight
efficiency (horizontal & vertical), and additional taxi/ASMA time. Stored in a
**DuckDB** database and queried with SQL — answers are exact, not estimated.

**2. Reference documents & methodology — bundled**
Definitions, acronyms, and methodology pages scraped from ansperformance.eu, plus
**PDF documents** discovered from the public portal repository. Embedded into a
vector index and used to answer conceptual questions ("what is ASMA additional
time?").

**3. Network Operations Portal (NOP) messages — live**
Operational updates (weather/CB advisories, tactical updates) fetched live from a
PocketBase service per question and interpreted for you.

**4. EUROCONTROL Data App API — daily figures**
Daily-granularity traffic, ATFM delay, CO2, and punctuality — for **a specific
day** (including past dates, e.g. 10 March 2026), a period (this week /
month / year-to-date), **the whole network**, a named country/airport/ANSP/
airline, or a **ranking** ("which airport had the highest punctuality", "busiest
airport pair for an airline"). It can also return a **daily series over a date
range** ("daily traffic from 1 Jan to 1 May 2026"), including **several entities
compared on one chart** ("France vs Germany daily traffic"), that is then
optionally **aggregated or derived** (weekly/monthly totals, delay-per-flight,
rolling averages — computed with validated SQL, never by the model) and
**charted**.
Updated once a day — **not real-time**; the latest available day is D-1
(yesterday).

**5. EUROCONTROL Network Manager (NM) — real-time**
The genuinely live network picture behind
[eurocontrol.int/performance/live](https://www.eurocontrol.int/performance/live.html):
aircraft airborne right now, current total network delay, the most-delayed area
control centres, and active ATFM regulations.

> Data © EUROCONTROL AIU. Answers are model-generated and may contain errors.
"""
    )

    # ------------------------------------------------------- the master flow
    st.divider()
    st.header("🗺️ The journey of a question (overview)")
    st.markdown(
        "Every question travels the same pipeline before it reaches a source. "
        "The **language model** appears only at the ✎ steps; every **figure** is "
        "produced by executing code (SQL / API), shown at the ⚙ steps."
    )
    _mermaid(
        r"""
flowchart TD
    Q([YOUR QUESTION]) --> R["✎ 1 · Rewrite follow-up<br/><i>'what about France?' → a standalone question</i>"]
    R --> AV{"2 · Availability question?<br/><i>'what data do you have?' — plain regex, no AI</i>"}
    AV -- yes --> CAT[["Answer from the catalog"]]
    AV -- no --> DEC["✎ 3 · Decompose<br/><i>split 'A and B?' into standalone parts</i>"]
    DEC --> RT["✎ 4 · Route<br/><i>pick 1–3 of: data · concept · nop · dataapp · nm_live · none</i>"]
    RT --> CL{"✎ 5 · Missing an essential detail?"}
    CL -- yes --> ASK[["Ask ONE clarifying question, then stop"]]
    CL -- no --> DIS["⚙ 6 · Dispatch each chosen route<br/><i>run SQL · call the live API · vector-search docs</i>"]
    DIS --> SY["✎ 7 · Synthesise<br/><i>narrate executed results, cite sources, add as-of date, render any chart</i>"]
    SY --> ANS([GROUNDED ANSWER])

    classDef ai fill:#eef4ff,stroke:#3b6fb5,color:#12325c;
    classDef det fill:#eefaf0,stroke:#3aa25a,color:#14532d;
    classDef term fill:#f4f4f6,stroke:#8a8a99,color:#333;
    class R,DEC,RT,CL,SY ai;
    class DIS det;
    class Q,ANS,CAT,ASK,AV term;
""",
        height=640,
    )
    st.caption(
        "Blue ✎ = a language-model step (routing / wording / query-writing). "
        "Green ⚙ = a deterministic step that produces the actual numbers."
    )

    # ------------------------------------------------------- per-route flows
    st.divider()
    st.header("🔬 Each answer path, in detail")
    st.markdown(
        "Step 6 above sends the question to one or more of six paths. Here is exactly "
        "what each one does, with an example question you can try."
    )

    # -- data path ----------------------------------------------------------
    st.subheader("① `data` — historical numbers (Text-to-SQL)")
    st.markdown(
        "**Use it for:** figures from the bundled history — by year/month, rankings, "
        "totals, trends. **Example:** *“Which 5 states had the most CO2 emissions in 2024?”*"
    )
    _mermaid(
        r"""
flowchart LR
    Q([question]) --> W["✎ Write ONE SELECT<br/><i>schema catalog + resolved<br/>entity names in the prompt</i>"]
    W --> G{"🛡 Safety gate<br/><i>sqlglot: single SELECT · no DDL/DML/<br/>file fns · known tables only</i>"}
    G -- rejected --> X[["Honest error<br/>(no answer invented)"]]
    G -- ok --> D["⚙ DuckDB runs it<br/><i>read-only views over Parquet</i>"]
    D --> N["✎ Narrate the executed rows<br/><i>+ as-of date · cites table & SQL</i>"]
    D --> C["⚙ Optional chart from a validated spec<br/><i>Plotly · skipped if not chart-worthy</i>"]
    N --> A([grounded answer])
    C -.-> A

    classDef ai fill:#eef4ff,stroke:#3b6fb5,color:#12325c;
    classDef det fill:#eefaf0,stroke:#3aa25a,color:#14532d;
    classDef gate fill:#fff4e5,stroke:#d08a00,color:#7a4d00;
    classDef term fill:#f4f4f6,stroke:#8a8a99,color:#333;
    class W,N ai;
    class D,C det;
    class G gate;
    class Q,A,X term;
""",
        height=360,
    )
    st.markdown(
        "**Why it's trustworthy:** the model *writes* SQL but never *computes* the "
        "answer — the total is whatever DuckDB returned. The safety gate rejects "
        "anything that isn't a single read-only `SELECT` over known tables. A **schema "
        "catalog** (with per-column units and gotchas, e.g. *“already a monthly "
        "average — do not SUM”*) is injected so the model writes *semantically* correct "
        "SQL, and an **entity layer** reconciles name spellings across tables."
    )

    # -- concept path -------------------------------------------------------
    st.subheader("② `concept` — definitions & methodology (vector RAG)")
    st.markdown(
        "**Use it for:** what a term/metric *means* or how it's computed. "
        "**Example:** *“How is horizontal flight efficiency measured?”*"
    )
    _mermaid(
        r"""
flowchart LR
    Q([question]) --> E["✎ Embed the query<br/><i>vector, DIM dims</i>"]
    Q --> AC["⚙ Exact acronym lookup<br/><i>surfaced directly</i>"]
    E --> V["⚙ DuckDB VSS search<br/><i>HNSW + cosine similarity</i>"]
    V --> K["Top-K chunks<br/><i>+ their source URLs / titles</i>"]
    AC --> K
    K --> N["✎ Answer from the retrieved text ONLY<br/><i>cites the sources · says 'I don't know' if nothing clears the floor</i>"]
    N --> A([grounded answer])

    classDef ai fill:#eef4ff,stroke:#3b6fb5,color:#12325c;
    classDef det fill:#eefaf0,stroke:#3aa25a,color:#14532d;
    classDef term fill:#f4f4f6,stroke:#8a8a99,color:#333;
    class E,N ai;
    class V,AC det;
    class Q,A,K term;
""".replace("DIM", str(config.EMBEDDING_DIM)),
        height=320,
    )
    st.markdown(
        "The corpus is definitions, methodology, acronyms and PDFs — chunked and "
        "embedded once at build time. An **exact acronym lookup** runs alongside the "
        "vector search (short glossary entries get diluted in a big corpus, so a code "
        "match is surfaced directly). If nothing clears the similarity floor, it says "
        "so rather than guessing."
    )

    # -- dataapp path -------------------------------------------------------
    st.subheader("③ `dataapp` — daily figures (EUROCONTROL Data App)")
    st.markdown(
        "**Use it for:** daily traffic / ATFM delay / CO2 / punctuality — a specific "
        "day (incl. past dates), the whole network, a named entity, a ranking, or a "
        "**series over a date range**. **Example:** *“How many flights were on the "
        "network on 10 March 2026?”*"
    )
    _mermaid(
        r"""
flowchart TD
    Q([question]) --> S["✎ Fill a small JSON spec<br/><i>query_kind · metric · date · entities · …</i>"]
    S --> ENT["⚙ entity(-ies)<br/><i>1 fetch / entity · fan-out ≤ FAN</i>"]
    S --> NET["⚙ network<br/><i>1 fetch</i>"]
    S --> RANK["⚙ ranking<br/><i>top / bottom-N</i>"]
    S --> TS["⚙ timeseries (date range)<br/><i>1 /syncs range call + 1 read per day<br/>per-metric long frame, tagged by entity</i>"]
    TS --> M["⚙ Optional: MANIPULATE via validated SQL<br/><i>weekly/monthly totals · delay-per-flight ·<br/>rolling avg — model writes SQL, not maths</i>"]
    ENT --> N["✎ Narrate the fetched rows<br/><i>+ chart from a validated spec</i>"]
    NET --> N
    RANK --> N
    M --> N
    N --> A([grounded answer])

    classDef ai fill:#eef4ff,stroke:#3b6fb5,color:#12325c;
    classDef det fill:#eefaf0,stroke:#3aa25a,color:#14532d;
    classDef term fill:#f4f4f6,stroke:#8a8a99,color:#333;
    class S,N ai;
    class ENT,NET,RANK,TS,M det;
    class Q,A term;
""".replace("FAN", str(config.MAX_FANOUT)),
        height=560,
    )
    st.markdown(
        "The model **only fills a JSON spec** — it never builds API calls. Deterministic "
        "resolvers do the sync→metric/ranking lookups. For a date range, exactly **one** "
        "`/syncs` call gets every day in the window (capped at "
        f"**{config.MAX_PERIOD_DAYS} days** to stay a polite scraper), then one read per "
        "day. **Multi-entity comparisons** ('France vs Germany') stack each entity's "
        "series into one long frame tagged with an `entity` column, so a single chart "
        "can split by it — and any aggregation joins/groups *per entity*, never mixing "
        "them. Any derivation (weekly totals, delay ÷ flights) is a **validated SQL** "
        "step, so the arithmetic is executed, not guessed."
    )

    # -- nop path -----------------------------------------------------------
    st.subheader("④ `nop` — the operational situation (NOP messages)")
    st.markdown(
        "**Use it for:** the current tactical picture — weather/CB advisories, airspace "
        "and aerodrome issues. **Example:** *“What's the current tactical situation on "
        "the network?”*"
    )
    _mermaid(
        r"""
flowchart TD
    Q([question]) --> H{"Historical / topical?<br/><i>regex on 'was / last week / …'</i>"}
    H -- yes --> KW["⚙ Fetch by KEYWORD from PocketBase"]
    H -- "no (current situation)" --> LT["⚙ Fetch the LATEST messages<br/><i>each tactical update already covers<br/>weather / airspace / aerodromes</i>"]
    KW -- none found --> LT
    KW --> N["✎ Interpret the fetched messages for you"]
    LT --> N
    N --> A([grounded answer])

    classDef ai fill:#eef4ff,stroke:#3b6fb5,color:#12325c;
    classDef det fill:#eefaf0,stroke:#3aa25a,color:#14532d;
    classDef term fill:#f4f4f6,stroke:#8a8a99,color:#333;
    class N ai;
    class KW,LT det;
    class Q,A,H term;
""",
        height=400,
    )

    # -- nm_live path -------------------------------------------------------
    st.subheader("⑤ `nm_live` — the real-time network (Network Manager)")
    st.markdown(
        "**Use it for:** the genuinely *live* picture right now. "
        "**Example:** *“How many aircraft are airborne right now?”*"
    )
    _mermaid(
        r"""
flowchart LR
    Q([question]) --> F["⚙ Fetch the LIVE NM snapshot<br/><i>Network Manager API · real-time — NOT D-1</i>"]
    F --> S["airborne now · total network delay ·<br/>most-delayed ACCs · active regulations"]
    S --> N["✎ Answer from the snapshot fields"]
    N --> A([grounded answer])

    classDef ai fill:#eef4ff,stroke:#3b6fb5,color:#12325c;
    classDef det fill:#eefaf0,stroke:#3aa25a,color:#14532d;
    classDef term fill:#f4f4f6,stroke:#8a8a99,color:#333;
    class N ai;
    class F det;
    class Q,A,S term;
""",
        height=300,
    )
    st.markdown(
        "This is the **only** real-time path. Everything on the Data App path is D-1 "
        "(yesterday); *airborne now* / *current delay* come from here instead."
    )

    # -- compound / multi-source -------------------------------------------
    st.subheader("⑥ Compound & multi-source questions")
    st.markdown(
        "**Use it for:** a question that bundles independent parts, or that needs a "
        "*number and an explanation*. **Example:** *“How many flights on the network "
        "on 10 March 2026, and on 10 March 2025?”* or *“What was Heathrow's ASMA "
        "additional time this year, and how is ASMA additional time defined?”*"
    )
    _mermaid(
        r"""
flowchart TD
    Q([compound question]) --> DEC["✎ Decompose"]
    DEC --> P1["Part 1 → full pipeline<br/><i>route → dispatch → narrate</i>"]
    DEC --> P2["Part 2 → full pipeline<br/><i>route → dispatch → narrate</i>"]
    DEC --> PN["…"]
    P1 --> SY["✎ Synthesise the grounded parts into one answer<br/><i>keeps every figure verbatim — never recomputes</i>"]
    P2 --> SY
    PN --> SY
    SY --> A([grounded answer])

    Q2([single question, two sources]) --> BR["✎ Route = data + concept"]
    BR --> RUN["Run BOTH paths"]
    RUN --> SY2["✎ Synthesise into one grounded answer"]
    SY2 --> A2([grounded answer])

    classDef ai fill:#eef4ff,stroke:#3b6fb5,color:#12325c;
    classDef det fill:#eefaf0,stroke:#3aa25a,color:#14532d;
    classDef term fill:#f4f4f6,stroke:#8a8a99,color:#333;
    class DEC,SY,BR,SY2 ai;
    class P1,P2,RUN det;
    class Q,A,PN,Q2,A2 term;
""",
        height=560,
    )
    st.markdown(
        "Each part is answered **on its own** through the full pipeline (so the two "
        "halves may use different dates, metrics or subjects), then a synthesis pass "
        "stitches the *already-grounded* sub-answers together **without recomputing any "
        "figure**. If a part is missing an essential detail, the assistant asks that one "
        "clarifying question and stops; your reply re-runs the whole question next turn."
    )

    # -------------------------------------------------------- routing table
    st.divider()
    st.header("🧭 Routing at a glance")
    st.markdown(
        """
| Route | When | How it answers |
|---|---|---|
| **data** | historical numbers (by year/month, rankings, totals) | the model writes **SQL**, validated (read-only, sandboxed) and run against the bundled DuckDB; the model narrates the executed rows |
| **concept** | what a term/metric means or how it's computed | vector search over docs + PDFs, plus an exact acronym lookup; answered only from the retrieved text |
| **nop** | the network operational situation in NOP messages | fetches recent NOP messages and interprets them |
| **dataapp** | daily traffic/delay/CO2/punctuality — a specific day (incl. past dates), the whole network, an entity, a ranking, or a date-range series | a deterministic resolver does the Data App API's sync→metric/ranking lookup; the model only picks the query shape, metric, date & entity |
| **nm_live** | the *real-time* network state right now | fetches the live NM snapshot (airborne, delay, regulations) |
| **both** | a number *and* an explanation | combines the data and concept paths |
| **none** | outside ANS performance | declines politely |

The example panel has a **Simple / Advanced** toggle: the *Advanced* set shows off
the multi-step answers — a **daily series over a date range**, an
**aggregate/derive** step (weekly totals, or ATFM delay **per flight** = delay
minutes ÷ flights, computed with validated SQL) rendered as a **chart**, and
**multi-part** questions answered in one go.
"""
    )

    # -------------------------------------------------- model & technologies
    st.divider()
    st.header("🛠️ Model & technologies")

    if config.LOCAL:
        llm_line = (
            "- **Language layer (local):** a [Ollama](https://ollama.com) model "
            "running on-device — **⚡ Fast** (qwen3.5:4b) or **🧠 Smart** "
            f"(qwen3.5:9b), selectable in the sidebar. Embeddings use "
            f"`{config.EMBEDDING_MODEL}`. It only writes queries and phrases the "
            "results — the data does the answering."
        )
    else:
        llm_line = (
            "- **Language layer (cloud):** an [OpenAI](https://openai.com) GPT "
            "model (**Fast** / **Balanced** / **Max**, selectable in the sidebar), "
            f"with OpenAI `{config.EMBEDDING_MODEL}` embeddings. It only writes "
            "queries and phrases the results — the AIU data does the answering."
        )

    st.markdown(
        f"""
- **AIU data & routing (the core):** EUROCONTROL Aviation Intelligence Unit
  datasets and live sources, with a routing system that picks the right one per
  question and answers from the data — not from the model's memory.
{llm_line}
- **Data + vectors:** [DuckDB](https://duckdb.org) holds the datasets and, via
  its VSS extension, the document vector index (HNSW) — one engine for both.
- **Text-to-SQL safety:** generated SQL is parsed with `sqlglot` and executed
  read-only with a row cap; only known tables are allowed.
- **Charts:** [Plotly](https://plotly.com/python/) rendered from a validated
  chart spec the model emits (it never writes plotting code).
- **UI:** [Streamlit](https://streamlit.io).
- **Live sources:** PocketBase (NOP), the EUROCONTROL Data App REST API (daily,
  latest day is D-1), and the EUROCONTROL Network Manager live API.

Built and tested in vertical slices, with a gold evaluation set that scores
answers and routing against known-correct cases.
"""
    )

    st.info(
        "**How many AI calls does one question make?** A typical single-source "
        "question uses a handful of small language-model calls (rewrite → decompose "
        "→ route → clarify-check → query-write → narrate, plus a chart-spec step when "
        "a chart applies). A **date range doesn't add AI calls** — it only adds "
        "deterministic per-day API reads. A **compound** question multiplies roughly "
        "by the number of parts, plus one synthesis call. Every figure in the answer "
        "still comes from executed code, no matter how many parts there are.",
        icon="💡",
    )

    # ------------------------------------------------------------- privacy
    if config.chat_logging_configured():
        st.divider()
        st.header("🔒 Privacy")
        st.markdown(
            "Conversations are **recorded for quality and analysis** — the questions "
            "asked, the answers given, and basic technical details about your browser "
            "(so returning sessions can be recognised). Nothing you type leaves this "
            "purpose. Please don't enter personal or sensitive information."
        )

    st.divider()
    st.markdown(
        "**Developed by** Quinten Goens · ATD/AIU/OPS  \n"
        "[quinten.goens@eurocontrol.int](mailto:quinten.goens@eurocontrol.int)"
    )
