"""Download PDFs and extract their full text into doc chunks.

Used by build_index to fold PDF content into the same document index as the
scraped web pages, so the concept path can answer from methodology handbooks and
technical notes. Discovery of which PDFs to fetch is done by discover_pdfs.py.
"""
from __future__ import annotations

import io

import requests

from aiu_chat.ingest.scrape_docs import Chunk, chunk_text

USER_AGENT = "aiu-chat/0.1 (local research tool; +https://ansperformance.eu)"
REQUEST_TIMEOUT = 120  # PDFs can be large


def _title_from_url(url: str) -> str:
    """Human-ish title from the file name."""
    name = url.rstrip("/").rsplit("/", 1)[-1]
    name = name[:-4] if name.lower().endswith(".pdf") else name
    return name.replace("_", " ").replace("-", " ").strip() or url


def fetch_pdf_text(url: str) -> str | None:
    """Download a PDF and extract all page text, or None if unavailable/empty."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    ctype = resp.headers.get("content-type", "")
    if resp.status_code != 200 or "pdf" not in ctype.lower():
        # Some links 404 or redirect to an HTML page; skip non-PDF responses.
        return None
    return extract_text(resp.content)


def extract_text(pdf_bytes: bytes) -> str | None:
    """Extract text from PDF bytes using pypdf. Returns None if no text."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except (PdfReadError, Exception):
        # Encrypted/corrupt/scanned PDFs may not yield text; treat as empty.
        return None
    text = "\n".join(p.strip() for p in pages if p.strip())
    return text or None


def pdf_to_chunks(url: str) -> list[Chunk]:
    """Fetch one PDF and return its chunks (empty list if unusable)."""
    text = fetch_pdf_text(url)
    if not text:
        return []
    title = _title_from_url(url)
    return [
        Chunk(text=ch, source_url=url, source_title=title, ordinal=i)
        for i, ch in enumerate(chunk_text(text))
    ]


def scrape_pdfs(urls: list[str]) -> list[Chunk]:
    """Fetch and chunk all given PDF URLs. Skips ones that don't yield text."""
    out: list[Chunk] = []
    for url in urls:
        chunks = pdf_to_chunks(url)
        if chunks:
            out.extend(chunks)
            print(f"  + PDF {_title_from_url(url)}: {len(chunks)} chunks")
        else:
            print(f"  - skip PDF {url} (no text / unavailable)")
    return out
