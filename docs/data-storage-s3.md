# Data storage on S3-compatible object stores

> New requirement added outside the original 4A text (2026-08-27): collected
> artifacts must land in containers backed by MinIO, while remaining portable
> to any S3-compatible provider.

## Design goals

1. **MinIO-first for local/dev**: if you run somewhere "strange" — laptop, CI,
   air-gapped rig — the storage container is MinIO with a single compose file.
2. **Provider-agnostic everywhere else**: DigitalOcean Spaces, AWS S3,
   Cloudflare R2 and friends differ only by endpoint/bucket/keys in `.env`.
   No code changes, no vendor SDK: the client is a minimal AWS Signature V4
   implementation over the workspace-standard blocking `reqwest` + `sha2`.
3. **Secrets stay secret**: keys live in `.env` (gitignored) as process env;
   diagnostics print redacted heads (`INiN…(7)`), never values; provenance
   and manifests carry connection *identity*, not credentials.

## Components

| Piece | Location |
|---|---|
| S3 client + SigV4 | `ai/lnai-data/src/s3.rs` (unit-tested vs RFC 4231 / AWS vectors / live MinIO) |
| Env config & upload/pull sync | `ai/lnai-data/src/storage.rs` |
| CLI verbs | `lnaicli storage-upload / storage-pull / storage-list` |
| Status/diagnostics | `lnaicli auth-status` prints active sink (redacted) |
| Local stack | `install/data-minio.compose.yml` (minio + bucket init) |

## Environment contract

Canonical names (`.env.example`, commented defaults):

```dotenv
S3_ENDPOINT=http://127.0.0.1:9000     # scheme://host[:port], no trailing slash
S3_BUCKET=lunar-hal-data
S3_REGION=us-east-1                   # optional
S3_ACCESS_KEY_ID=                     # required for uploads
S3_SECRET_ACCESS_KEY=                 # paired; never commit real values
S3_PATH_STYLE=true                    # MinIO-style addressing default
```

Legacy aliases keep working (`SPACES_ENDPOINT`, `SPACES_BUCKET`,
`SPACES_REGION`, `SPACES_KEY`, `SPACES_SECRET`) so existing orchestrate.sh /
podman setups run unchanged. `S3_*` wins when both are set. Half-configured
credentials are rejected rather than silently downgraded to anonymous.

Addressing:
* `path_style=true` → `http://endpoint:port/<bucket>/<key>` (MinIO default).
* `path_style=false` → `http://<bucket>.<host>/<key>` (Spaces/AWS style).

## Workflow

### 1. Start the storage container (local)

```sh
podman-compose -f install/data-minio.compose.yml up -d
# console on http://127.0.0.1:9001
```

Defaults mirror `install/podman-compose.yml`: `minioadmin/minioadmin`,
bucket `lunar-hal-data`. Override via `MINIO_ROOT_USER/PASSWORD`.

For a private-bucket-free read setup allow public download:

```sh
podman run --rm --network=host --entrypoint=/bin/sh docker.io/minio/mc:latest -c \
"mc alias set t http://127.0.0.1:9000 minioadmin minioadmin && mc anonymous set download t/lunar-hal-data"
```

Then reads work without credentials; writes always require `S3_ACCESS_KEY_ID`
/ `S3_SECRET_ACCESS_KEY`.

### 2. Upload a dataset directory

```sh
lnaicli storage-upload --dataset-dir data/stellar/v1 --prefix stellar/v1
```

Layout preserved exactly: `stellar/v1/manifest.json`,
`stellar/v1/raw/*.parquet`, `stellar/v1/views/*.parquet`, …

### 3. Verify / pull back

```sh
lnaicli storage-list --prefix stellar/v1
lnaicli storage-pull --dest-dir data/restored --prefix stellar/v1
```

Pull is resume-friendly: files whose local size matches remote are skipped.

### Cloud switch example

Point the same variables at a hosted provider (no rebuild):

```dotenv
S3_ENDPOINT=https://sfo3.digitaloceanspaces.com
S3_BUCKET=lunar-hal-data
S3_REGION=sfo3
S3_PATH_STYLE=false
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
```

## Security notes

* SigV4 canonical requests are unit-pinned against an independent Python
  reference + live MinIO; header ordering was caught and fixed by that test
  (SigV4 requires lexicographic SignedHeaders ordering).
* GET/HEAD/LIST sign the empty-payload digest (`e3b0…b855`) — `UNSIGNED-PAYLOAD`
  is rejected there even by MinIO.
* Query parameters (list prefix etc.) must appear both in the signature and on
  the wire URL byte-identically — enforced by code path plus tests.
* Failure bodies are truncated before display; nothing echoes credentials.

## Out of scope / follow-ups

* Testbench UI cards for sink status (backend wiring is trivial once UI slot exists).
* Multipart upload for very large single objects (current max ~5 GiB object cap applies).
* Bucket-level lifecycle/versioning rules — deliberate: dataset versioning is
  manifest-driven, not bucket-policy-driven.
