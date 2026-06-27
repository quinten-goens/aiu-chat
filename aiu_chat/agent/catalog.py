"""Load the schema catalog and render it for prompts / validation."""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass

from aiu_chat import config


@dataclass(frozen=True)
class DatasetCatalogEntry:
    table: str
    title: str
    description: str
    granularity: str
    parquet_path: str
    as_of: str | None
    columns: list[dict]


@dataclass(frozen=True)
class Catalog:
    datasets: list[DatasetCatalogEntry]

    @property
    def table_names(self) -> set[str]:
        return {d.table for d in self.datasets}

    def get(self, table: str) -> DatasetCatalogEntry | None:
        return next((d for d in self.datasets if d.table == table), None)

    def describe(self) -> str:
        """A user-facing summary of the available datasets (for 'what data do
        you have?' questions). Answered from the catalog, not vector search."""
        lines = ["I have these EUROCONTROL performance datasets:"]
        for d in self.datasets:
            cols = ", ".join(c["name"] for c in d.columns[:8])
            more = "…" if len(d.columns) > 8 else ""
            through = f" (through {d.as_of})" if d.as_of else ""
            lines.append(f"- **{d.title}** (`{d.table}`){through}: {d.description}")
            lines.append(f"  Columns: {cols}{more}")
        return "\n".join(lines)

    def prompt_text(self) -> str:
        """Human-readable schema + semantics for the SQL-generation prompt."""
        lines: list[str] = []
        for d in self.datasets:
            lines.append(f"## Table: {d.table}")
            lines.append(f"{d.title} — {d.description}")
            if d.granularity:
                lines.append(f"Granularity: {d.granularity}")
            if d.as_of:
                lines.append(f"Data available through: {d.as_of}")
            lines.append("Columns:")
            for col in d.columns:
                bits = [f"  - {col['name']} ({col['type']})"]
                if col.get("description"):
                    bits.append(col["description"])
                if col.get("unit"):
                    bits.append(f"[unit: {col['unit']}]")
                if col.get("note"):
                    bits.append(f"NOTE: {col['note']}")
                lines.append(" ".join(bits))
            lines.append("")
        return "\n".join(lines).strip()


def load_catalog(path=None) -> Catalog:
    path = path or config.CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found at {path}. Run the ingestion steps first "
            f"(download_datasets + build_catalog)."
        )
    raw = json.loads(path.read_text())
    datasets = [
        DatasetCatalogEntry(
            table=d["table"],
            title=d["title"],
            description=d["description"],
            granularity=d.get("granularity", ""),
            parquet_path=d["parquet_path"],
            as_of=d.get("as_of"),
            columns=d["columns"],
        )
        for d in raw["datasets"]
    ]
    return Catalog(datasets=datasets)


@functools.lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Cached catalog for the running process."""
    return load_catalog()
