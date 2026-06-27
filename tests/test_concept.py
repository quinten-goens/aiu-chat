"""Tests for the concept path: text extraction, chunking, and answering.

Scraping and embedding need network/model, so those are exercised live during
ingestion; here we unit-test the pure logic and the answer orchestration with
fakes.
"""
from __future__ import annotations

from aiu_chat.agent.concept import answer_concept_question
from aiu_chat.agent.retriever import RetrievedChunk
from aiu_chat.ingest.scrape_docs import _extract_main_text, chunk_glossary, chunk_text


# --- extraction ------------------------------------------------------------

def test_extract_prefers_largest_region_not_just_main():
    html = """
    <html><body>
      <main><h1>Title</h1></main>
      <div class="content">
        <p>ATFM means Air Traffic Flow Management.</p>
        <p>ASMA is Arrival Sequencing and Metering Area.</p>
      </div>
    </body></html>
    """
    text = _extract_main_text(html)
    assert "Air Traffic Flow Management" in text
    assert "Arrival Sequencing" in text


def test_extract_drops_nav_and_script():
    html = """
    <html><body>
      <nav>menu junk</nav>
      <script>var x = 1;</script>
      <div><p>Real content here.</p></div>
    </body></html>
    """
    text = _extract_main_text(html)
    assert "Real content here." in text
    assert "menu junk" not in text
    assert "var x" not in text


# --- chunking --------------------------------------------------------------

def test_chunk_splits_long_text():
    text = "\n".join(f"Paragraph number {i} with some words." for i in range(200))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)  # max_chars + overlap slack


def test_chunk_short_text_single_chunk():
    assert len(chunk_text("Just one short paragraph.")) == 1


def test_glossary_chunking_splits_per_entry():
    # Mimics the acronyms page: entries separated by an em-dash line.
    text = "\n—\n".join(f"AC{i} - Term number {i}" for i in range(24))
    chunks = chunk_glossary(text, entries_per_chunk=6)
    assert len(chunks) == 4  # 24 entries / 6 per chunk
    # A specific acronym lands in a small, focused chunk.
    atfm_chunks = [c for c in chunks if "AC5 - Term number 5" in c]
    assert len(atfm_chunks) == 1
    assert atfm_chunks[0].count("\n") <= 5  # at most 6 entries


def test_glossary_falls_back_for_non_glossary_text():
    # No em-dash separators -> behaves like normal chunking (single chunk here).
    assert chunk_glossary("Just a normal paragraph, not a list.") == \
        chunk_text("Just a normal paragraph, not a list.")


# --- concept answering (faked retriever + client) --------------------------

class FakeClient:
    def __init__(self, response):
        self.response = response

    def chat(self, messages, temperature=0.0, json_mode=False):
        return self.response

    def embed(self, text):
        return [0.0]


def test_answer_uses_retrieved_chunks():
    chunks = [
        RetrievedChunk("ASMA = Arrival Sequencing and Metering Area.",
                       "http://x", "Acronyms", 0.8)
    ]
    # Use a non-acronym query so the structured acronym lookup stays out of the
    # way and we test the retrieval->answer path in isolation.
    ans = answer_concept_question(
        "how is this metric defined?",
        client=FakeClient("It is the Arrival Sequencing and Metering Area."),
        retriever=lambda q, client=None: chunks,
    )
    assert ans.ok
    assert chunks[0] in ans.sources


def test_no_chunks_returns_dont_know():
    ans = answer_concept_question(
        "what is the meaning of life?",
        client=FakeClient("unused"),
        retriever=lambda q, client=None: [],
    )
    assert not ans.ok
    assert ans.sources == []
