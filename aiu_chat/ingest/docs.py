"""Registry of reference/document pages to scrape for the concept path.

These feed the document-RAG half of the system: definitions, acronyms, and
methodology narrative that answer conceptual questions ("what is ASMA additional
time?"). Keep this list curated and small — quality over coverage.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocSource:
    url: str
    title: str
    glossary: bool = False  # split per-entry (for acronym/definition list pages)


DOC_SOURCES: list[DocSource] = [
    DocSource("https://ansperformance.eu/acronym/", "Acronyms", glossary=True),
    DocSource("https://ansperformance.eu/methodology/", "Methodology (index)"),
    DocSource(
        "https://ansperformance.eu/methodology/additional-asma-time/",
        "Methodology: Additional ASMA Time",
    ),
    DocSource(
        "https://ansperformance.eu/methodology/additional-taxi-out-time/",
        "Methodology: Additional Taxi-Out Time",
    ),
    DocSource(
        "https://ansperformance.eu/methodology/additional-taxi-in-time/",
        "Methodology: Additional Taxi-In Time",
    ),
    DocSource(
        "https://ansperformance.eu/methodology/horizontal-flight-efficiency/",
        "Methodology: Horizontal Flight Efficiency",
    ),
    DocSource(
        "https://ansperformance.eu/methodology/atfm-delay/",
        "Methodology: ATFM Delay",
    ),
]
