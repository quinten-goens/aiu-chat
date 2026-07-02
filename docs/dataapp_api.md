# EUROCONTROL Data App API — reference

Authoritative, **verified-against-live** notes for the `dataapp` query path. Base
URL: `https://api-data-app.eurocontrol.int/api` (config `AIU_DATAAPP_BASE`).

- **Public, read-only, no auth.** GET only.
- **API Platform** (Symfony). Responses are `{ "meta": {...}, "data": [...] }` for
  collections, with `meta.totalItems`, `meta.currentPage`, `meta.itemsPerPage`.
- API Platform **silently ignores unknown filter params** (it returns the full
  set instead of erroring). So a filter that "does nothing" is usually a wrong
  filter name — check against the exact names below, which come from the OpenAPI
  spec (`components.schemas`).

## The sync model — everything hangs off a *sync*

Metric rows are keyed by a **sync**: a per-stakeholder, per-**day** snapshot. You
cannot ask for "traffic in France" directly; you resolve a sync first, then read
the metric off it. Two hops (or three if resolving a named entity):

1. **Pick the sync.** `GET /syncs`, filtered by `dataType` (+ entity id, + date):
   - `dataType` values are the live API's, **not** the tidy names you'd guess:
     | conceptual kind | live `dataType` string |
     |-----------------|------------------------|
     | whole network   | `network-wide`         |
     | a country/state | `state-specific`       |
     | an airport      | `airport`              |
     | an ANSP         | `air-navigation-service-provider` |
     | an airline      | `aircraft-operator`    |
   - Pin to a **specific day**: `syncDate[after]=YYYY-MM-DD&syncDate[before]=YYYY-MM-DD`
     (equal bounds ⇒ that one day). Omit for the newest: `order[syncDate]=desc&itemsPerPage=1`.
   - Filter by entity FK: `country.id`, `airport.id`, `airNavigationServiceProvider.id`,
     `aircraftOperator.id`.
   - `syncs` is ~460k rows — **always** filter + order + `itemsPerPage`.
2. **Read the metric / ranking** filtered by the sync id (nested filter syntax below).

To resolve a **named entity → id** first: `GET /countries?name=France` (or
`?iso2=FR`), `GET /airports?code=EGLL` (or `?name=`), similarly
`/air_navigation_service_providers`, `/aircraft_operators`.

## Three query shapes (what the agent implements)

### 1. Whole-network figure (optionally on a date)
`GET /syncs?dataType=network-wide&syncDate[after]=2026-03-10&syncDate[before]=2026-03-10`
→ sync id, then
`GET /traffic_networks?traffic.sync.id=<sid>&itemsPerPage=30`.

Rows carry `networkType` (`total`/`avg`), `dateRange` (`DY` day / `WK` 7-day /
`MM` month / `Y2D` year-to-date), and `value` or `avgValue`.

> **Verified:** flights on the network on **2026-03-10** = `DY total value` =
> **24 864.0**.

Same shape for `delay_networks`, `co2_networks`, `punctualities_networks`.

### 2. Cross-entity ranking ("which one is highest/lowest")
`GET /<metric>_ranking_datas` filtered by the sync + `rankingCategory`, ordered:
```
GET /punctualities_ranking_datas
  ?punctualityRanking.punctuality.sync.id=<sid>
  &punctualityRanking.punctuality.rankingCategory=airports
  &dateRange=DY&order[value]=desc&itemsPerPage=30
```
- The filter chain is nested one level deeper than the metric filter:
  `<prefix>Ranking.<prefix>.sync.id` and `.rankingCategory`
  (prefixes: `traffic`, `delay`, `punctuality`; **co2 has no rankings**).
- `rankingCategory` selects WHAT is ranked (OpenAPI `Traffic-read` enum):
  `states`, `airports`, `airport_pairs`, `aircraft_operators`,
  `area_control_center`, `map_area_control_center`, `network`,
  `flight_breakdown`, `market_segments`, `breakdown`.
  Using the right category avoids the interleaving you get otherwise (a raw
  ranking mixes countries + airports).
- Order by `value` for a day (`DY`/`WK`/`MM`); by **`avgValue`** for `Y2D`
  (yearly), where `avgValue` is the daily average. `order[...]=asc` for "lowest".
- Rows: `name`, `value`, `avgValue`, `rankNumber`, `share`, `dateRange`.

> **Verified (network-wide sync for the day):**
> - highest arrival punctuality on **2025-03-10** (`airports`, DY, desc) →
>   **Yerevan** (value 1.0).
> - most ATFM delay on **2025-03-10** (`delay`, `states`, DY, desc) →
>   **Portugal** (5 573 min).

**Scoped rankings** (a ranking *within* one entity) use that entity's sync:
- busiest **airport pairs for an airline**: resolve the airline → its
  `aircraft-operator` sync → `traffic_ranking_datas?...rankingCategory=airport_pairs`.
- busiest **airline in a country**: the country's `state-specific` sync →
  `rankingCategory=aircraft_operators`.
- busiest **destination from an airport**: the airport's `airport` sync →
  `rankingCategory=airports` (an airport's data ranks *destination airports*, NOT
  pairs — `airport_pairs` is not populated on an airport sync).

> **Verified:** `aircraft_operators` on Estonia's sync (Y2D) → **airBaltic** #1
> (~27 daily). `airport_pairs` on British Airways Group's sync → Glasgow /
> Edinburgh ⟷ London Heathrow at the top (~21 daily). `airports` on Hamburg's
> sync (Y2D 2024) → **Munich** #1 (~13 daily).

**Both extremes at once** ("the airports with the highest AND lowest
punctuality"): fetch the ranking descending with a large `itemsPerPage`, then the
first row is the highest and the last row is the lowest — no second call needed
(verified Q1/2026: **Ibiza** highest / **Paris Le Bourget** lowest).

**Yearly / quarter questions:** there is no "year" filter — pick the sync at the
**end of the period** and read `Y2D`. For "2025" use a late-Dec-2025 sync
(`syncDate` ≈ `2025-12-31`) + `dateRange=Y2D` (`avgValue` = 2025 daily average).
For "Q1/2026" use a sync ≈ `2026-03-31`. (The exact daily average a user sees
depends on which day's sync they read — annual figures are vintage-sensitive.)

### 3. Per-entity figure (existing path, now date-aware)
Resolve entity → its sync (optionally for a date) → `<metric>_networks`
filtered by `<prefix>.sync.id`. Supports per-question fan-out over several named
entities.

## Endpoints

**Dimensions:** `/countries` (`name`, `iso2`), `/airports` (`name`, `code`),
`/air_navigation_service_providers` (`name`, `code`), `/aircraft_operators`
(`name`, `code`).

**Syncs:** `/syncs` — `country.id`, `airport.id`,
`airNavigationServiceProvider.id`, `aircraftOperator.id`, `dataType`,
`syncDate[before|after|strictly_before|strictly_after]`, `order[syncDate]`.

**Metrics** (28 GET paths total; each metric has base / `_networks` / `_charts`
and — traffic/delay/punctuality only — `_rankings` / `_ranking_datas`):
- Traffic: `/traffic`, `/traffic_networks`, `/traffic_charts`,
  `/traffic_rankings`, `/traffic_ranking_datas`
- ATFM delay: `/delays`, `/delay_networks`, `/delay_charts`,
  `/delay_rankings`, `/delay_ranking_datas`
- CO2: `/co2s`, `/co2_networks`, `/co2_charts`  *(no rankings)*
- Punctuality: `/punctualities`, `/punctualities_networks`,
  `/punctualities_charts`, `/punctualities_rankings`, `/punctualities_ranking_datas`
- Billing: `/billeds`, `/billed_networks`, `/billed_charts`

**Content:** `/news`, `/situation_reports`.

**Enums** (from OpenAPI): `dateRange` = `DY|WK|MM|Y2D`; `rankingType` =
`top|top_prev`; `rankingCategory` = see list above.

## Fetching the OpenAPI spec

`/api/docs` returns an HTML shell (a `{data: ...}` wrapper), **not** the raw
spec. To get the machine-readable spec, request it with
`Accept: application/vnd.openapi+json` from a browser context, or download it via
the interactive docs page. The spec we validated against is API Platform version
5.0.0; re-verify `dataType`/`rankingCategory` strings on a refresh — API Platform
derives them from the entity graph and they can shift between versions.
