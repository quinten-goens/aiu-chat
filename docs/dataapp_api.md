# EUROCONTROL Data App API — reference

Authoritative notes for the `dataapp` query path. Base URL:
`https://api-data-app.eurocontrol.int/api` (config `AIU_DATAAPP_BASE`).

- **Public, read-only, no auth.** GET only.
- It is an **API Platform** (Symfony) API. Responses are `{ "meta": {...}, "data": [...] }`
  (collections) with `meta.totalItems`, `meta.currentPage`, `meta.itemsPerPage`.
- The full OpenAPI 3.1 spec is served at `/api/docs` with header
  `Accept: application/vnd.openapi+json` (or `/api/docs.jsonopenapi`).

## The 3-hop query pattern (important)

You usually cannot ask for "traffic in France" directly. The metric endpoints
are keyed by a **sync** (a per-stakeholder, per-date snapshot), so resolving a
named entity to data takes up to three hops:

1. **Resolve the entity → id.** Filter the dimension endpoint by name/code:
   - `GET /api/countries?name=France` → `data[0].id`, `iso2`, `icao`
   - `GET /api/airports?code=EGLL` (or `?name=...`) → id
   - `GET /api/air_navigation_service_providers?code=DSNA` → id
   - `GET /api/aircraft_operators?code=...` → id
2. **Find the relevant sync.** Filter `syncs` by the entity id + `dataType`,
   newest first:
   - `GET /api/syncs?country.id=7&dataType=country&order[syncDate]=desc&itemsPerPage=1`
   - `dataType` matches the stakeholder kind: `country`, `airport`,
     `air-navigation-service-provider`, `aircraft-operator`, `network`.
     (Confirm exact strings from live `syncs` data — e.g. observed
     `aircraft-operator`.)
   - A sync has: `id`, `syncDate`, `dataType`, `code`, and the linked
     `country` / `airport` / `airNavigationServiceProvider` / `aircraftOperator`.
3. **Query the metric** filtering by the sync id (note the nested filter syntax):
   - `GET /api/traffic_networks?traffic.sync.id=<syncId>`
   - `GET /api/delay_networks?delay.sync.id=<syncId>`
   - `GET /api/co2_networks?co2.sync.id=<syncId>`
   - `GET /api/punctualities_networks?punctuality.sync.id=<syncId>`

For **network-wide** figures (no specific stakeholder), use `dataType=network`
syncs and the `*_networks` endpoints directly with `networkType` / `dateRange`
filters.

## Endpoints

**Dimensions (resolve name/code → id):**
`/api/countries` (filters: `name`, `iso2`), `/api/airports` (`name`, `code`),
`/api/air_navigation_service_providers` (`name`, `code`),
`/api/aircraft_operators` (`name`, `code`).

**Syncs:** `/api/syncs` — filters: `country.id`, `airport.id`,
`airNavigationServiceProvider.id`, `aircraftOperator.id`, `dataType`,
`syncDate[before|after|strictly_before|strictly_after]`, `order[syncDate]`.

**Metrics** (each has base / `_charts` / `_networks` / `_rankings` variants):
- Traffic: `/api/traffic`, `/api/traffic_networks`, `/api/traffic_charts`,
  `/api/traffic_rankings`, `/api/traffic_ranking_datas`
- ATFM delay: `/api/delays`, `/api/delay_networks`, `/api/delay_charts`,
  `/api/delay_rankings`, `/api/delay_ranking_datas`
- CO2: `/api/co2s`, `/api/co2_networks`, `/api/co2_charts`
- Punctuality: `/api/punctualities`, `/api/punctualities_networks`,
  `/api/punctualities_charts`, `/api/punctualities_rankings`,
  `/api/punctualities_ranking_datas`
- Billing: `/api/billeds`, `/api/billed_networks`, `/api/billed_charts`

Common metric filters: `*.sync.id`, `*.sync.dataType`, `*.rankingCategory`,
`dateRange`, `networkType`, and `*.sync.syncDate[before|after]`.

**Content:** `/api/news` (published news), `/api/situation_reports` (network
situation reports).

## Pagination / shaping

- `itemsPerPage`, `currentPage`. Default page size is small; set `itemsPerPage`
  explicitly. `syncs` is huge (~460k items) — always filter and order it.
- `order[<field>]` for sorting (e.g. `order[syncDate]=desc`, `order[name]=asc`).

> Verify `dataType` strings and metric→sync filter names against live responses
> during ingestion; API Platform names them from the entity graph and they can
> shift between versions (current: 5.0.0).
