"""Programmatically discover PDF document links from the AIU portal source repo.

Rationale: the live site is statically generated from the public Hugo repo
`euctrl-pru/aiu-portal`, so every PDF link lives in that repo's content/layout
files. Cloning + scanning the repo is more complete and reliable than scraping
rendered pages, and — importantly — it is repeatable: re-running this picks up
any PDFs added since last time.

Run: python -m aiu_chat.ingest.discover_pdfs    # prints the discovered links
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from aiu_chat import config

PORTAL_REPO = "https://github.com/euctrl-pru/aiu-portal.git"
SITE_BASE = "https://ansperformance.eu"

# Directories within the repo worth scanning (content + templates + static config).
SCAN_DIRS = ["content", "layouts", "static", "data"]
SCAN_SUFFIXES = {".md", ".rmd", ".html", ".toml", ".yaml", ".yml", ".r", ".json"}

# Match absolute (http...) or root-relative (/...) .pdf references. The
# lookbehind on the root-relative form prevents matching the "/x/..." tail of a
# repo-relative path like "../x/y.pdf" (which isn't resolvable against the site).
_PDF_RE = re.compile(
    r"""(https?://[^\s"'()<>]+?\.pdf)"""        # absolute
    r"""|(?<![\w.])(/[^\s"'()<>]+?\.pdf)""",    # root-relative, not mid-path
    re.IGNORECASE,
)


def _repo_dir() -> Path:
    return config.DATA_DIR / "aiu-portal"


def clone_or_update_repo(repo_dir: Path | None = None) -> Path:
    """Shallow-clone the portal repo, or pull if it already exists."""
    repo_dir = repo_dir or _repo_dir()
    config.ensure_dirs()
    if (repo_dir / ".git").exists():
        print(f"Updating portal repo at {repo_dir}...")
        subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only", "--depth", "1"],
            check=True, capture_output=True, text=True,
        )
    else:
        print(f"Cloning portal repo to {repo_dir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", PORTAL_REPO, str(repo_dir)],
            check=True, capture_output=True, text=True,
        )
    return repo_dir


def _normalise(link: str) -> str | None:
    """Resolve a raw match to an absolute URL; drop anything unusable."""
    link = link.strip().rstrip(".,);")
    if link.lower().startswith("http"):
        return link
    if link.startswith("/"):
        return f"{SITE_BASE}{link}"
    return None  # repo-relative or odd paths: skip


def scan_repo_for_pdfs(repo_dir: Path | None = None) -> list[str]:
    """Scan the repo's text files for .pdf links and return sorted unique URLs."""
    repo_dir = repo_dir or _repo_dir()
    found: set[str] = set()
    for sub in SCAN_DIRS:
        base = repo_dir / sub
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for groups in _PDF_RE.findall(text):
                # Two alternatives -> findall yields a tuple; take the matched one.
                m = next((g for g in groups if g), "")
                url = _normalise(m)
                if url:
                    found.add(url)
    return sorted(found)


def discover_pdf_links(update: bool = True) -> list[str]:
    """Full discovery: (optionally) refresh the repo, then scan it for PDF links."""
    repo_dir = clone_or_update_repo() if update else _repo_dir()
    links = scan_repo_for_pdfs(repo_dir)
    print(f"Discovered {len(links)} PDF links.")
    return links


def main(argv: list[str] | None = None) -> int:
    for link in discover_pdf_links():
        print(link)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
