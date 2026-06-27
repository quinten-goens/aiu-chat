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
    filename_pattern: str  # uses {year}; include the .csv.bz2 suffix if compressed
    first_year: int
    columns: list[ColumnSpec] = field(default_factory=list)
    granularity: str = ""  # e.g. "one row per state per month"
    compressed: bool = False  # True for .csv.bz2 sources

    def url_for_year(self, year: int) -> str:
        return f"{BASE_URL}/{self.filename_pattern.format(year=year)}"


# --- Dataset registry ------------------------------------------------------
# All datasets are monthly. Column names below are taken verbatim from the live
# CSV headers (verified by probing the files). Many delay datasets use coded
# "by-cause" columns (e.g. DLY_ERT_C_1 = capacity, DLY_ERT_W_1 = weather); the
# per-column notes explain the codes so the model can use them correctly.

# Standard delay-cause codes used across ATFM delay datasets (ICAO-style):
#   A=accepted/other  C=ATC capacity  D=de-icing  E=equipment(ATC)
#   G=aerodrome capacity  I=industrial action(ATC)  M=airspace mgmt/staffing
#   N=ind. action(non-ATC)  O=other  P=special event  R=ATC routeing
#   S=ATC staffing  T=equip(non-ATC)  V=environmental/weather  W=weather  NA=not attributed
_DELAY_CAUSE_HINT = (
    "Per-cause ATFM delay minutes by code (C=ATC capacity, S=ATC staffing, "
    "W/V=weather, G=aerodrome capacity, etc.). Monthly TOTAL minutes — safe to SUM."
)

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

# --- Common column specs (shared shapes across many datasets) ---------------

def _time_cols() -> list[ColumnSpec]:
    return [
        ColumnSpec("YEAR", "INTEGER", "Calendar year."),
        ColumnSpec("MONTH_NUM", "INTEGER", "Calendar month number (1-12)."),
        ColumnSpec("MONTH_MON", "VARCHAR", "Three-letter month abbreviation (e.g. 'JAN')."),
    ]


def _airport_cols() -> list[ColumnSpec]:
    return [
        ColumnSpec("APT_ICAO", "VARCHAR", "Airport ICAO code (e.g. 'EGLL')."),
        ColumnSpec("APT_NAME", "VARCHAR", "Airport name."),
        ColumnSpec("STATE_NAME", "VARCHAR", "State the airport is in (e.g. 'France')."),
    ]


# --- Airport traffic --------------------------------------------------------

AIRPORT_TRAFFIC = DatasetSpec(
    key="airport_traffic",
    title="Airport Traffic",
    description="Daily departures, arrivals and total movements per airport.",
    filename_pattern="airport_traffic_{year}.csv",
    first_year=2016,
    granularity="one row per airport per day",
    columns=[
        *_time_cols(),
        ColumnSpec("FLT_DATE", "DATE", "Flight date (the day)."),
        *_airport_cols(),
        ColumnSpec("FLT_DEP_1", "INTEGER", "Departures (all flights).", unit="flights",
                   note="Daily count — SUM across days for monthly/yearly totals."),
        ColumnSpec("FLT_ARR_1", "INTEGER", "Arrivals (all flights).", unit="flights",
                   note="Daily count — safe to SUM."),
        ColumnSpec("FLT_TOT_1", "INTEGER", "Total movements (departures + arrivals).",
                   unit="flights", note="Daily count — safe to SUM."),
        ColumnSpec("FLT_DEP_IFR_2", "INTEGER", "IFR departures.", unit="flights"),
        ColumnSpec("FLT_ARR_IFR_2", "INTEGER", "IFR arrivals.", unit="flights"),
        ColumnSpec("FLT_TOT_IFR_2", "INTEGER", "Total IFR movements.", unit="flights"),
    ],
)

# --- Pre-departure delays ---------------------------------------------------

ATC_PRE_DEPARTURE_DELAYS = DatasetSpec(
    key="atc_pre_departure_delays",
    title="ATC Pre-Departure Delays",
    description="ATC pre-departure delay minutes by airport and day.",
    filename_pattern="atc_pre_departure_delays_{year}.csv",
    first_year=2016,
    granularity="one row per airport per day",
    columns=[
        *_time_cols(),
        ColumnSpec("FLT_DATE", "DATE", "Flight date."),
        *_airport_cols(),
        ColumnSpec("FLT_DEP_1", "INTEGER", "Departures (all flights).", unit="flights"),
        ColumnSpec("FLT_DEP_IFR_2", "INTEGER", "IFR departures (reference for the metric).",
                   unit="flights"),
        ColumnSpec("DLY_ATC_PRE_2", "DOUBLE", "ATC pre-departure delay.", unit="minutes",
                   note="Daily TOTAL minutes — SUM across days. Avg delay/flight = "
                        "SUM(DLY_ATC_PRE_2)/SUM(FLT_DEP_IFR_2)."),
        ColumnSpec("FLT_DEP_3", "INTEGER", "Departures (alt. reference period).", unit="flights"),
        ColumnSpec("DLY_ATC_PRE_3", "DOUBLE", "ATC pre-departure delay (alt. period).",
                   unit="minutes", note="Daily TOTAL minutes — safe to SUM."),
    ],
)

ALL_PRE_DEPARTURE_DELAYS = DatasetSpec(
    key="all_pre_departure_delays",
    title="All-Causes Pre-Departure Delays",
    description="All-causes pre-departure delay minutes by airport and day.",
    filename_pattern="all_pre_departure_delays_{year}.csv",
    first_year=2020,
    granularity="one row per airport per day",
    columns=[
        *_time_cols(),
        ColumnSpec("FLT_DATE", "DATE", "Flight date."),
        *_airport_cols(),
        ColumnSpec("FLT_DEP_1", "INTEGER", "Departures (all flights).", unit="flights"),
        ColumnSpec("FLT_DEP_IFR_2", "INTEGER", "IFR departures (reference).", unit="flights"),
        ColumnSpec("DLY_ALL_PRE_2", "DOUBLE", "All-causes pre-departure delay.", unit="minutes",
                   note="Daily TOTAL minutes (all causes, not just ATC) — safe to SUM."),
    ],
)

ATFM_SLOT_ADHERENCE = DatasetSpec(
    key="atfm_slot_adherence",
    title="ATFM Slot Adherence",
    description="Adherence of departures to assigned ATFM slots, by airport and day.",
    filename_pattern="atfm_slot_adherence_{year}.csv",
    first_year=2016,
    granularity="one row per airport per day",
    columns=[
        *_time_cols(),
        ColumnSpec("FLT_DATE", "DATE", "Flight date."),
        *_airport_cols(),
        ColumnSpec("FLT_DEP_1", "INTEGER", "Departures (all flights).", unit="flights"),
        ColumnSpec("FLT_DEP_REG_1", "INTEGER", "Departures that were ATFM-regulated.",
                   unit="flights", note="Daily count — safe to SUM."),
        ColumnSpec("FLT_DEP_OUT_EARLY_1", "INTEGER", "Departed before the slot window.",
                   unit="flights"),
        ColumnSpec("FLT_DEP_IN_1", "INTEGER", "Departed within the slot window (adherent).",
                   unit="flights", note="Daily count — adherence rate = SUM(FLT_DEP_IN_1)/SUM(FLT_DEP_REG_1)."),
        ColumnSpec("FLT_DEP_OUT_LATE_1", "INTEGER", "Departed after the slot window.",
                   unit="flights"),
    ],
)

# --- Airport arrival ATFM delays (compressed) -------------------------------

AIRPORT_ARRIVAL_ATFM_DELAY = DatasetSpec(
    key="airport_arrival_atfm_delay",
    title="Airport Arrival ATFM Delays",
    description="Airport arrival ATFM delay minutes, broken down by cause, per airport/day.",
    filename_pattern="apt_dly_{year}.csv.bz2",
    first_year=2014,
    compressed=True,
    granularity="one row per airport per day",
    columns=[
        *_time_cols(),
        ColumnSpec("FLT_DATE", "TIMESTAMP", "Flight date (timestamp at midnight UTC)."),
        *_airport_cols(),
        ColumnSpec("FLT_ARR_1", "INTEGER", "Arrivals (reference).", unit="flights"),
        ColumnSpec("DLY_APT_ARR_1", "DOUBLE", "Total airport arrival ATFM delay.", unit="minutes",
                   note="Daily TOTAL minutes — safe to SUM. The DLY_APT_ARR_*_1 columns split "
                        "this by cause and sum back to it."),
        ColumnSpec("DLY_APT_ARR_C_1", "DOUBLE", "Arrival delay: ATC capacity.", unit="minutes",
                   note=_DELAY_CAUSE_HINT),
        ColumnSpec("DLY_APT_ARR_W_1", "DOUBLE", "Arrival delay: weather.", unit="minutes",
                   note=_DELAY_CAUSE_HINT),
        ColumnSpec("DLY_APT_ARR_S_1", "DOUBLE", "Arrival delay: ATC staffing.", unit="minutes",
                   note=_DELAY_CAUSE_HINT),
        ColumnSpec("FLT_ARR_1_DLY", "INTEGER", "Arrivals that were delayed.", unit="flights"),
        ColumnSpec("FLT_ARR_1_DLY_15", "INTEGER", "Arrivals delayed >15 min.", unit="flights"),
    ],
)

# --- En-route ATFM delays (compressed) --------------------------------------

def _enroute_delay_cols() -> list[ColumnSpec]:
    return [
        *_time_cols(),
        ColumnSpec("FLT_DATE", "TIMESTAMP", "Flight date (timestamp at midnight UTC)."),
        ColumnSpec("ENTITY_NAME", "VARCHAR", "Name of the ANSP or FIR/country."),
        ColumnSpec("ENTITY_TYPE", "VARCHAR", "Entity kind (e.g. 'ANSP (AUA)' or 'COUNTRY (FIR)')."),
        ColumnSpec("FLT_ERT_1", "INTEGER", "En-route flights (reference).", unit="flights",
                   note="Daily count — safe to SUM."),
        ColumnSpec("DLY_ERT_1", "DOUBLE", "Total en-route ATFM delay.", unit="minutes",
                   note="Daily TOTAL minutes — safe to SUM. DLY_ERT_*_1 split this by cause."),
        ColumnSpec("DLY_ERT_C_1", "DOUBLE", "En-route delay: ATC capacity.", unit="minutes",
                   note=_DELAY_CAUSE_HINT),
        ColumnSpec("DLY_ERT_S_1", "DOUBLE", "En-route delay: ATC staffing.", unit="minutes",
                   note=_DELAY_CAUSE_HINT),
        ColumnSpec("DLY_ERT_W_1", "DOUBLE", "En-route delay: weather.", unit="minutes",
                   note=_DELAY_CAUSE_HINT),
        ColumnSpec("FLT_ERT_1_DLY", "INTEGER", "En-route flights delayed.", unit="flights"),
        ColumnSpec("FLT_ERT_1_DLY_15", "INTEGER", "En-route flights delayed >15 min.", unit="flights"),
    ]


ENROUTE_DELAY_ANSP = DatasetSpec(
    key="enroute_delay_ansp",
    title="En-route ATFM Delays by ANSP",
    description="En-route ATFM delay minutes by Air Navigation Service Provider, per day.",
    filename_pattern="ert_dly_ansp_{year}.csv.bz2",
    first_year=2008,
    compressed=True,
    granularity="one row per ANSP per day",
    columns=_enroute_delay_cols(),
)

ENROUTE_DELAY_FIR = DatasetSpec(
    key="enroute_delay_fir",
    title="En-route ATFM Delays by FIR",
    description="En-route ATFM delay minutes by Flight Information Region / country, per day.",
    filename_pattern="ert_dly_fir_{year}.csv.bz2",
    first_year=2013,
    compressed=True,
    granularity="one row per FIR/country per day",
    columns=_enroute_delay_cols(),
)

# --- Flight efficiency ------------------------------------------------------

HORIZONTAL_FLIGHT_EFFICIENCY = DatasetSpec(
    key="horizontal_flight_efficiency",
    title="Horizontal Flight Efficiency",
    description="Lateral (horizontal) en-route flight efficiency: flown vs. direct distance.",
    filename_pattern="horizontal_flight_efficiency_{year}.csv",
    first_year=2015,
    granularity="one row per entity (state/FIR) per month per trajectory model",
    columns=[
        *_time_cols(),
        ColumnSpec("ENTRY_DATE", "DATE", "First day of the reporting month."),
        ColumnSpec("ENTITY_NAME", "VARCHAR", "State / FIR name."),
        ColumnSpec("ENTITY_TYPE", "VARCHAR", "Entity kind (e.g. 'State (FIR)')."),
        ColumnSpec("TYPE_MODEL", "VARCHAR", "Trajectory model: 'CPF' (actual) or 'FTFM' (planned).",
                   note="Filter to one model (usually CPF) to avoid double counting."),
        ColumnSpec("DIST_FLOWN_KM", "DOUBLE", "Distance actually flown.", unit="km",
                   note="Monthly TOTAL — SUM. Efficiency ratio = SUM(DIST_DIRECT_KM)/SUM(DIST_FLOWN_KM)."),
        ColumnSpec("DIST_DIRECT_KM", "DOUBLE", "Great-circle direct distance.", unit="km",
                   note="Monthly TOTAL — safe to SUM."),
        ColumnSpec("DIST_ACHIEVED_KM", "DOUBLE", "Achieved distance (KEP benchmark).", unit="km",
                   note="Monthly TOTAL — safe to SUM."),
    ],
)

VERTICAL_FLIGHT_EFFICIENCY = DatasetSpec(
    key="vertical_flight_efficiency",
    title="Vertical Flight Efficiency (CCO/CDO)",
    description="Climb/descent efficiency: level-off distance/time and continuous climb/descent counts.",
    filename_pattern="vertical_flight_efficiency_{year}.csv",
    first_year=2015,
    granularity="one row per airport per month",
    columns=[
        *_time_cols(),
        *_airport_cols(),
        ColumnSpec("NBR_FLIGHTS_DESCENT", "INTEGER", "Flights analysed on descent.", unit="flights",
                   note="Monthly count — safe to SUM."),
        ColumnSpec("NBR_CDO_FLIGHTS", "INTEGER", "Continuous Descent Operation flights.",
                   unit="flights", note="CDO share = SUM(NBR_CDO_FLIGHTS)/SUM(NBR_FLIGHTS_DESCENT)."),
        ColumnSpec("TOT_TIME_LEVEL_SECONDS_DESCENT", "DOUBLE", "Total time flown level during descent.",
                   unit="seconds", note="Monthly TOTAL — safe to SUM."),
        ColumnSpec("NBR_FLIGHTS_CLIMB", "INTEGER", "Flights analysed on climb.", unit="flights"),
        ColumnSpec("NBR_CCO_FLIGHTS", "INTEGER", "Continuous Climb Operation flights.",
                   unit="flights", note="CCO share = SUM(NBR_CCO_FLIGHTS)/SUM(NBR_FLIGHTS_CLIMB)."),
        ColumnSpec("TOT_TIME_LEVEL_SECONDS_CLIMB", "DOUBLE", "Total time flown level during climb.",
                   unit="seconds", note="Monthly TOTAL — safe to SUM."),
    ],
)

# --- Additional time (taxi / ASMA) ------------------------------------------

def _additional_time_cols(metric_label: str) -> list[ColumnSpec]:
    return [
        *_time_cols(),
        *_airport_cols(),
        ColumnSpec("TF", "INTEGER", "Total flights at the airport in the month.", unit="flights",
                   note="Monthly TOTAL — safe to SUM."),
        ColumnSpec("VALID_FL", "INTEGER", "Flights with valid data for the metric.", unit="flights"),
        ColumnSpec("TOTAL_REF_NB_FL", "INTEGER", "Flights matched to a reference group.",
                   unit="flights"),
        ColumnSpec("TOTAL_REF_TIME_MIN", "DOUBLE", f"Total reference (unimpeded) {metric_label} time.",
                   unit="minutes", note="Monthly TOTAL — safe to SUM."),
        ColumnSpec("TOTAL_ADD_TIME_MIN", "DOUBLE",
                   f"Total ADDITIONAL {metric_label} time vs. the unimpeded reference.",
                   unit="minutes",
                   note="Monthly TOTAL additional minutes — SUM across months/airports. "
                        "Additional time per flight = SUM(TOTAL_ADD_TIME_MIN)/SUM(TOTAL_REF_NB_FL)."),
    ]


TAXI_IN_ADDITIONAL_TIME = DatasetSpec(
    key="taxi_in_additional_time",
    title="Additional Taxi-In Time",
    description="Additional taxi-in (arrival ground) time vs. an unimpeded reference, per airport/month.",
    filename_pattern="taxi_in_additional_time_{year}.csv",
    first_year=2018,
    granularity="one row per airport per month",
    columns=_additional_time_cols("taxi-in"),
)

TAXI_OUT_ADDITIONAL_TIME = DatasetSpec(
    key="taxi_out_additional_time",
    title="Additional Taxi-Out Time",
    description="Additional taxi-out (departure ground) time vs. an unimpeded reference, per airport/month.",
    filename_pattern="taxi_out_additional_time_{year}.csv",
    first_year=2018,
    granularity="one row per airport per month",
    columns=_additional_time_cols("taxi-out"),
)

ASMA_ADDITIONAL_TIME = DatasetSpec(
    key="asma_additional_time",
    title="Additional ASMA Time",
    description="Additional arrival sequencing & metering (ASMA) time vs. an unimpeded reference.",
    filename_pattern="asma_additional_time_{year}.csv",
    first_year=2018,
    granularity="one row per airport per month",
    columns=_additional_time_cols("ASMA arrival"),
)


DATASETS: dict[str, DatasetSpec] = {
    d.key: d
    for d in [
        CO2_BY_STATE,
        AIRPORT_TRAFFIC,
        ATC_PRE_DEPARTURE_DELAYS,
        ALL_PRE_DEPARTURE_DELAYS,
        ATFM_SLOT_ADHERENCE,
        AIRPORT_ARRIVAL_ATFM_DELAY,
        ENROUTE_DELAY_ANSP,
        ENROUTE_DELAY_FIR,
        HORIZONTAL_FLIGHT_EFFICIENCY,
        VERTICAL_FLIGHT_EFFICIENCY,
        TAXI_IN_ADDITIONAL_TIME,
        TAXI_OUT_ADDITIONAL_TIME,
        ASMA_ADDITIONAL_TIME,
    ]
}
