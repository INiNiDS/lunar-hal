# lnai-data: canonical dataset pipeline (Stage 4)

Status: implemented and pilot-verified. The Gaia backbone collection,
cleaning, assembly, model views and deterministic splitting now live in the
`lnai-data` crate instead of ad-hoc CLI scripts.

## Pipeline

```
lnaicli collect-data --out-dir data/canonical-v1 \
    [--ra-min 0 --ra-max 360 --mag-limit-g 16 --target-rows-per-shard 200000 \
     --concurrency 4 --only <substr> --retry-failed --verify]

lnaicli build-dataset --out-dir data/canonical-v1 \
    [--batch-shards 2 --keep-parts]
```

* `collect-data` is resumable: `Verified` shards are never re-downloaded;
  `Failed` shards wait for `--retry-failed`; `--verify` recomputes SHA-256 of
  every shard file against the manifest (zero network).
* Row budget overflow (`TOP N+1` returns more rows than budget) triggers
  deterministic adaptive subdivision: first into ≤10° declination bands, then
  by halving the RA extent. Subdivisions are recorded in the manifest via
  `subdivided_into` + `row_limit_hit` on the parent shard.
* Shard files are streamed to unique temp paths and atomically renamed, so an
  interrupted process can never leave half-written inputs behind.
* Bounded worker pool (`--concurrency`) with exponential backoff retries.

## DatasetManifest v1.1 (additive change)

`MANIFEST_SCHEMA_VERSION` bumped `1.0.0 → 1.1.0`. New optional fields with
`#[serde(default)]`:

| field | meaning |
|---|---|
| `ShardState.subdivided_into` | child shard ids created after a row-limit hit |
| `ShardState.row_limit_hit` | parent was too dense and got subdivided |

Old 1.0.0 manifests keep loading unchanged; backward compatibility covered by
`manifest_v1_0_0_json_still_loads_in_v1_1_0`.

## Assembly & views

`build-dataset` produces in `<out_dir>/assembled/`:

* `canonical.parquet` — golden schema columns plus `spatial_tile`/`split`
  labels, deduplicated by stable source ID (lowest RUWE wins, deterministic
  tie-break).
* `view_pinn.parquet`, `view_gnn_kinematics.parquet`, `view_siren.parquet`
  — column-exact subsets per contracted view requirements.
* `gnn_localization_neighbors.parquet` — anchor→neighbor k-NN pairs
  (k ≤ 8) computed deterministically among valid position-bearing rows.

The build is memory-bounded: verified shards are processed in batches of
`--batch-shards` (default 2) — each batch is parsed, cleaned, deduplicated
and flushed to `assembled/parts/canonical_part_XXXXX.parquet` (64k-row groups)
before being dropped, and the final canonical + views are merged with polars'
streaming engine (scan parts → sink), so peak RAM scales with one batch
instead of the whole dataset. Batch progress is printed as
`[batch k/N] shards=… rows=…`. Measured on a release build: ~0.5 GB peak RSS
for a 4M-row dataset, independent of dataset size.

RAM knobs (set as env before running; `build-dataset` installs these defaults
when unset): `POLARS_MAX_THREADS=4`, `POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS=16384`,
`POLARS_INFLIGHT_SINK_MORSEL_LIMIT=4`. For a very constrained host also pass
`--batch-shards 1`. `--keep-parts` preserves the intermediate part files for
debugging; they are removed otherwise. Verified shard RA windows are disjoint
half-open ranges, so cross-batch duplicates cannot occur and the quality
report keeps `unique_ids == total_rows` (same as the in-memory build).

Splitting contract (`SPLIT_SALT = "lnai-split-v1"`): `sha256(salt|source_id)`
mod 1000 → Train 80% / Validation 10% / Test 10%. A ~5% subset of the sky's
15°×15° spatial tiles is a **holdout**: those rows never enter any split
(verified by `split_leakage.rs`).

Coordinates are converted to Galactic Cartesian positions (parsecs) and
velocities (km/s) using the frozen ICRS→Galactic matrix from `clean.rs`;
tests assert both the north galactic pole and galactic centre directions.

## Quality report

`quality_report.json`: total/unique rows, duplicates removed, null rates
(parallax / proper motion / radial velocity), RA/Dec coverage envelope and
position-outlier rate — these are the pilot exit criteria.

## Pilot evidence (RA 30–35°, mag G < 16, ruwe < 1.4)

* 5/5 shards verified; resume run re-downloaded nothing; verify round clean.
* 400 673 canonical rows → train/validation/test/holdout =
  308 907 / 38 633 / 38 579 / 14 554.
* Duplicates: none; position outliers: none; RV null-rate 55.6% (expected:
  radial velocities exist only for brighter stars).
* Manifest finalized: `total_rows=400673`,
  `checksum=3d1c9ec03b20d5cb…`.

The full-sky 360° pass stays **deliberately gated** until the pilot gates
above are signed off (plan stop-condition: no full pass before
resume/row-limit/checksum tests are green). Run it as:

```
lnaicli collect-data --ra-min 0 --ra-max 360
```

Then `build-dataset` + review `quality_report.json` + manifest gap notes.

## Secret policy

Credentials come only from environment variables (names listed in
`.env.example`, values empty there). Gaia auth is optional (anonymous works);
every public TAP endpoint used by default needs no API key. Diagnostic output
goes through `lnai_data::auth::redact`, and `SecretBox` keeps credentials out
of logs/formatters. See `docs/nasa-stardance-data.md` for source-specific
details.
