"""Registry of EUROCONTROL AIU datasets we ingest.

Each entry describes how to fetch a dataset and how to expose it as a DuckDB
table, plus the semantic notes that get injected into the SQL-generation prompt
(units, granularity, gotchas) so the model doesn't produce plausible-but-wrong
SQL. See CLAUDE.md > Trustworthiness.

URLs and filename spellings are taken verbatim from the live site (note the
"co2_emmissions" double-m typo in EUROCONTROL's own filenames). Always verify
against ansperformance.eu/csv/ — do not "correct" the spelling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASE_URL = "https://www.eurocontrol.int/performance/data/download/csv"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # logical type label for the catalog (e.g. "DATE", "INTEGER", "DOUBLE", "VARCHAR", "BOOLEAN")
    description: str
    unit: str | None = None
    note: str | None = None  # gotchas, e.g. "monthly total — safe to SUM across months"


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # logical dataset key / DuckDB table name
    title: str
    description: str
    filename_pattern: str  # uses {year}
    first_year: int
    columns: list[ColumnSpec] = field(default_factory=list)
    granularity: str = ""  # e.g. "one row per state per month"

    def url_for_year(self, year: int) -> str:
        return f"{BASE_URL}/{self.filename_pattern.format(year=year)}"


# --- Dataset registry ------------------------------------------------------
# Start with one dataset end-to-end (vertical slice). Expand later.

CO2_BY_STATE = DatasetSpec(
    key="co2_emissions_by_state",
    title="CO2 Emissions by State",
    description=(
        "Monthly gate-to-gate CO2 emissions and IFR flight counts attributed to "
        "each European state's airspace, from EUROCONTROL AIU."
    ),
    filename_pattern="co2_emmissions_by_state_{year}.csv",  # sic: EUROCONTROL's spelling
    first_year=2010,
    granularity="one row per state per calendar month",
    columns=[
        ColumnSpec("FLIGHT_MONTH", "DATE", "First day of the reporting month.",
                   note="Use for time-series ordering; combine with YEAR/MONTH for filtering."),
        ColumnSpec("YEAR", "INTEGER", "Calendar year."),
        ColumnSpec("MONTH", "INTEGER", "Calendar month number (1-12)."),
        ColumnSpec("STATE_NAME", "VARCHAR", "State name in uppercase (e.g. 'FRANCE')."),
        ColumnSpec("STATE_CODE", "VARCHAR", "ICAO-style state/region code prefix (e.g. 'LF')."),
        ColumnSpec("CO2_QTY_TONNES", "DOUBLE", "CO2 emitted.", unit="tonnes",
                   note="Monthly TOTAL for the state — safe to SUM across months/states; "
                        "do NOT average it as if it were a per-flight figure."),
        ColumnSpec("TF", "INTEGER", "Total IFR flights in the month for the state.", unit="flights",
                   note="Monthly TOTAL count — safe to SUM. Per-flight CO2 = CO2_QTY_TONNES / TF."),
        ColumnSpec("NOTE", "BOOLEAN", "Data-quality flag set by EUROCONTROL for the row."),
    ],
)

DATASETS: dict[str, DatasetSpec] = {
    CO2_BY_STATE.key: CO2_BY_STATE,
}
