"""About page content (rendered via st.navigation from streamlit_app.py)."""
from __future__ import annotations

import streamlit as st

from aiu_chat import config


def render():
    st.title("ℹ️ About Aviation Intelligence Chat")
    st.caption(
        "A hybrid AI assistant for European air navigation performance data."
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
"""
    )

    st.divider()
    st.header("Data sources")
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

**4. EUROCONTROL Data App API — daily (D-1)**
The latest *daily* figures (yesterday / latest available day, this week,
year-to-date) for traffic, ATFM delay, CO2, and punctuality, for a specific
country, airport, ANSP, or airline. Updated once a day — **not real-time**.

**5. EUROCONTROL Network Manager (NM) — real-time**
The genuinely live network picture behind
[eurocontrol.int/performance/live](https://www.eurocontrol.int/performance/live.html):
aircraft airborne right now, current total network delay, the most-delayed area
control centres, and active ATFM regulations.

> Data © EUROCONTROL AIU. Answers are model-generated and may contain errors.
"""
    )

    st.divider()
    st.header("How it works")
    st.markdown(
        """
Every question is **routed** to one of these paths:

| Route | When | How it answers |
|---|---|---|
| **data** | historical numbers (by year/month, rankings, totals) | the model writes **SQL**, validated (read-only, sandboxed) and run against the bundled DuckDB; the model narrates the executed rows |
| **concept** | what a term/metric means or how it's computed | vector search over docs + PDFs, plus an exact acronym lookup; answered only from the retrieved text |
| **nop** | the network operational situation in NOP messages | fetches recent NOP messages and interprets them |
| **dataapp** | latest *daily* (D-1) traffic, delay, CO2, punctuality for a country/airport/ANSP/airline | a deterministic resolver does the Data App API's multi-step lookup; the model picks only the metric + entity |
| **nm_live** | the *real-time* network state right now | fetches the live NM snapshot (airborne, delay, regulations) |
| **both** | a number *and* an explanation | combines the data and concept paths |
| **none** | outside ANS performance | declines politely |

**Key principle — numbers come from data, not the model.** For quantitative
questions the model writes a query; the actual figures always come from executing
that query (or a live API), never from the model's memory. Results can be shown
as **interactive charts** when useful.

If a question is missing an essential detail (e.g. "show me the delays" — for
which airport?), the assistant **asks one clarifying question** instead of
guessing; your reply is merged with the original question. It also rewrites
follow-ups into standalone questions (so "what about France?" works) and states
the **as-of date** of the data.
"""
    )

    st.divider()
    st.header("Model & technologies")

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
  its VSS extension, the document vector index — one engine for both.
- **Text-to-SQL safety:** generated SQL is parsed with `sqlglot` and executed
  read-only with a row cap; only known tables are allowed.
- **Charts:** [Plotly](https://plotly.com/python/) rendered from a validated
  chart spec the model emits (it never writes plotting code).
- **UI:** [Streamlit](https://streamlit.io).
- **Live sources:** PocketBase (NOP), the EUROCONTROL Data App REST API (D-1),
  and the EUROCONTROL Network Manager live API.

Built and tested in vertical slices, with a gold evaluation set that scores
answers and routing against known-correct cases.
"""
    )

    if config.chat_logging_configured():
        st.divider()
        st.header("Privacy")
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