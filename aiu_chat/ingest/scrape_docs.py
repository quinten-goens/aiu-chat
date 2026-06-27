"""Scrape reference pages and chunk them for the concept path.

Run is via build_index (which embeds the chunks); this module only fetches and
chunks. Polite client: User-Agent, tolerates 404s on guessed URLs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from aiu_chat.ingest.docs import DOC_SOURCES, DocSource

USER_AGENT = "aiu-chat/0.1 (local research tool; +https://ansperformance.eu)"
REQUEST_TIMEOUT = 60


@dataclass
class Chunk:
    text: str
    source_url: str
    source_title: str
    ordinal: int  # position within the page


def fetch_text(source: DocSource) -> str | None:
    """Fetch a page and return cleaned main text, or None if unavailable."""
    try:
        resp = requests.get(
            source.url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.text.strip():
        return None
    return _extract_main_text(resp.text)


def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Drop non-content elements.
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
        tag.decompose()
    # Pick the candidate content region with the MOST text. On this site the
    # real content sometimes lives outside <main> (which holds only the page
    # header), so we don't blindly trust <main>/<article> — we compare.
    candidates = soup.find_all(["main", "article"]) + ([soup.body] if soup.body else [])
    root = max(candidates, key=lambda el: len(el.get_text()), default=soup)
    text = root.get_text(separator="\n")
    # Collapse whitespace; keep paragraph breaks.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def chunk_text(
    text: str, *, max_chars: int = 1000, overlap: int = 150
) -> list[str]:
    """Split text into overlapping, paragraph-aware chunks.

    Paragraphs are accumulated until max_chars; a tail overlap is carried into
    the next chunk so context isn't lost at boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 1 > max_chars:
            chunks.append(buf.strip())
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n" + para).strip()
        else:
            buf = (buf + "\n" + para).strip() if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def scrape_all(sources: list[DocSource] | None = None) -> list[Chunk]:
    """Fetch and chunk all sources. Skips pages that don't resolve."""
    sources = sources or DOC_SOURCES
    out: list[Chunk] = []
    for src in sources:
        text = fetch_text(src)
        if not text:
            print(f"  - skip {src.url} (unavailable)")
            continue
        page_chunks = chunk_text(text)
        for i, ch in enumerate(page_chunks):
            out.append(Chunk(text=ch, source_url=src.url, source_title=src.title, ordinal=i))
        print(f"  + {src.title}: {len(page_chunks)} chunks")
    return out
