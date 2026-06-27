"""Tests for PDF discovery (link scanning/normalisation) and chunking.

Network/clone are exercised live during ingestion; here we unit-test the pure
logic with synthetic repo files and patched fetches.
"""
from __future__ import annotations

from unittest.mock import patch

from aiu_chat.ingest import discover_pdfs, pdf


# --- discovery -------------------------------------------------------------

def test_scan_finds_and_normalises_links(tmp_path):
    content = tmp_path / "content"
    content.mkdir()
    (content / "page.md").write_text(
        'See [handbook](/library/ace-handbook.pdf) and '
        '<a href="https://www.eurocontrol.int/foo/report.pdf">report</a>. '
        'Also a repo-relative one: ../x/skip-me.pdf'
    )
    links = discover_pdfs.scan_repo_for_pdfs(tmp_path)
    assert "https://ansperformance.eu/library/ace-handbook.pdf" in links
    assert "https://www.eurocontrol.int/foo/report.pdf" in links
    # repo-relative (../) is not resolvable -> skipped
    assert not any("skip-me" in l for l in links)


def test_scan_dedupes(tmp_path):
    content = tmp_path / "content"
    content.mkdir()
    (content / "a.md").write_text("/library/x.pdf")
    (content / "b.md").write_text("/library/x.pdf")
    links = discover_pdfs.scan_repo_for_pdfs(tmp_path)
    assert links.count("https://ansperformance.eu/library/x.pdf") == 1


def test_scan_ignores_non_text_suffixes(tmp_path):
    content = tmp_path / "content"
    content.mkdir()
    (content / "image.png").write_text("/library/should-not-scan.pdf")
    assert discover_pdfs.scan_repo_for_pdfs(tmp_path) == []


# --- extraction / chunking -------------------------------------------------

def test_title_from_url():
    assert pdf._title_from_url(
        "https://ansperformance.eu/library/AddASMA_indicator_documentation_mar23.pdf"
    ) == "AddASMA indicator documentation mar23"


def test_pdf_to_chunks_skips_when_no_text():
    with patch.object(pdf, "fetch_pdf_text", return_value=None):
        assert pdf.pdf_to_chunks("http://x/y.pdf") == []


def test_pdf_to_chunks_builds_chunks():
    long_text = "\n".join(f"Methodology paragraph {i} with content." for i in range(100))
    with patch.object(pdf, "fetch_pdf_text", return_value=long_text):
        chunks = pdf.pdf_to_chunks("http://x/methodology.pdf")
    assert len(chunks) > 1
    assert all(c.source_url == "http://x/methodology.pdf" for c in chunks)
    assert chunks[0].source_title == "methodology"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_extract_text_handles_garbage_bytes():
    # Not a real PDF -> pypdf fails -> None, never raises.
    assert pdf.extract_text(b"not a pdf at all") is None
