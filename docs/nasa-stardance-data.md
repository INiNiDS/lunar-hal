# NASA / Stardance data integration (Stage 4A)

Enrichment sources connected through the `lnai-data` source-adapter layer.
The Gaia backbone is never replaced by NASA rows: adapters declare
`enters_stellar_backbone = false` in their provenance, and enrichment only
adds host-star features.

## Adapters & contracts

* `SourceAdapter`, `SourceAuth`, `SourceProvenance` — `lnai-data/src/sources/mod.rs`.
* `NasaExoplanetAdapter` — anonymous TAP sync against `PSCompPars`
  (`exoplanetarchive.ipac.caltech.edu/TAP/sync`). No API key required; the
  query contract is frozen in `adql_query()`; parsing is deterministic so CI
  replays the recorded fixture
  (`tests/fixtures/nasa_exoplanet_pscomppars_sample.csv`) without network.
* Crossmatch v1 (`crossmatch.rs`): ID-first (exact stable source id),
  epoch-aware positional fallback (proper-motion propagation to a common
  epoch) and an ambiguity metric (`sep_nearest / sep_second`) for diagnostics.

## Registrations checklist

| Source | Registration | Key? | Required? |
|---|---|---|---|
| Gaia DR3 backbone | gea.esac.esa.int account | `GAIA_USERNAME`/`GAIA_PASSWORD` | optional — anonymous sync works, auth unlocks large async jobs |
| NASA Exoplanet Archive TAP | none | none | no |
| MAST (TIC metadata spike) | MyST account | `MAST_API_TOKEN` | optional — only protected/EAP products need it |
| IRSA (2MASS/WISE spike) | none | none | anonymous TAP |
| JPL Horizons scene provider | none | none | anonymous API |

Public endpoints used by the default pipeline run **fully anonymously**;
no key from this table blocks stage-4 execution. Values live only in
`.env` (gitignored) as backend-process environment variables — never in Git,
Notion, logs or frontend payloads. Hackatime keys are personal developer
bookkeeping and must not enter `.env` at all.

## Before/after model impact

Per plan item 4A-7 an improvement claim requires a before/after evaluation
report. When NASA-derived features are introduced into training specs, attach
both reports under `ai/*/target/ai-reports/` and reference them from
`provenance.json → impact_report_reference`. Enrichment fixtures used for
such comparisons live next to the adapter tests.

## Provenance

Each collection directory can carry `provenance.json`
(`DatasetProvenanceV1`): adapter ids, catalog versions, endpoint URLs and
auth *mode* (names of required env variables, never values). Idempotent
registration keeps re-runs clean.

## Stage 4A completion state (2026-08-27)

All remaining substeps of 4A are implemented in `ai/lnai-data`:

| Plan item | Implementation | Verification |
|---|---|---|
| п.1 registrations | `lnaicli auth-status` prints the anonymous/env matrix with presence-only flags | manual |
| п.4/6 enriched fixture 100–1000 + source manifest | recorded real PSCompPars dump (500 rows, SHA pinned), `SourceManifestV1`, `provenance.json` via `lnaicli enrich-fixtures` | `tests/nasa_exoplanet_tap.rs` offline replay |
| п.7 before/after report | `enrich.rs`: deterministic 2-fold closed-form ridge over matched sample; refuses "improved" without data; `lnaicli enrich-report [--gaia-parquet]` writes `enrichment_report.json` (or an honest `not_evaluated`) | unit tests incl. ridge solver + serialization round-trip |
| п.8 MAST TIC spike | `sources/mast.rs` anonymous Mashup cone search (`Mast.Catalogs.Tic.Cone`), frozen query hash, coverage/cross-id stats; live: `lnaicli spike-mast` | fixture replay offline; live run measured 551 rows / 79% pm / 92% GAIA ids |
| п.9 IRSA 2MASS spike | `sources/irsa.rs` TAP sync on `fp_psc`, null-rate metrics (`pm` documented as always-null); live: `lnaicli spike-irsa` | fixture replay; box smoke-test 50 rows JHK 100% |
| п.10 JPL Horizons scene provider | `sources/jpl_horizons.rs` positional parser for frozen QUANTITIES=1,2,20 contract; `enters_stellar_backbone=false`; live: `lnaicli jpl-scenes` | Uranus fixture (3 days) exact-value assertions |

MAST note: `catalogs.mast.stsci.edu` was unreachable from this environment;
the equivalent authenticated-by-default-free path is the STScI Mashup
endpoint, which the adapter and fixtures use. When catalogs API access is
restored, only the endpoint constant needs revisiting.

## Object storage (S3-compatible)

Dataset artifacts are no longer bound to local disk — see
[Data storage on S3-compatible object stores](data-storage-s3.md). Local/dev
default is MinIO via `install/data-minio.compose.yml`.
