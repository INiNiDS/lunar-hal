
# LUNAR-HAL → DigitalOcean

Full playbook: from `doctl create droplet` to downloading the finished
`combined_stars.parquet` onto your laptop. The cycle is: **droplet fetches Gaia →
cleans → streams chunks of 3M stars into Spaces → merges → drops under
`/final/` → you `wget` it → delete the droplet.**

---

## Run it anywhere

The same `orchestrate.sh` works in **five** modes via environment variables.
No edits to the script needed — just flip `SINK_KIND` and the endpoints.

| Mode | When to use | Config |
|---|---|---|
| **DO Spaces** (prod) | On the droplet, $5/mo for the bucket | `SINK_KIND=s3`, `SPACES_BUCKET=lunar-hal-data`, `SPACES_REGION=sfo3` |
| **Local folder** | Dev / debug, no cloud at all | `SINK_KIND=local`, `LOCAL_OUT_DIR=./data` |
| **Podman (1 container)** | Isolated run on your box, local sink | `podman build && podman run ...` |
| **Podman + MinIO** | Full local simulation of DO with S3 API + web console | `podman-compose -f install/podman-compose.yml up` |
| **Podman + RustFS** | Same, but written in Rust (alpha) | `podman-compose -f install/podman-compose.rustfs.yml up` |

All other parameters (`TOTAL_STARS`, `CHUNK_SIZE`, `PARALLEL`, `GAIA_*`) are
shared across modes.

### Quick start: local (no containers)

```bash
# one-time: build the CLI
cargo build --release -p lunar-ai-cli

# one-time: fill in your secrets (see .env.example for placeholders)
cp .env.example .env
$EDITOR .env

# orchestrate.sh auto-sources ./.env — no need to export manually
SINK_KIND=local \
LOCAL_OUT_DIR=./data \
TOTAL_STARS=3000000 \
PARALLEL=2 \
./install/orchestrate.sh
```

Files appear as:
```
./data/
├── chunks/
│   ├── chunk-0.parquet
│   ├── chunk-0.sha256
│   ├── chunk-1.parquet
│   ├── ...
│   └── manifest.json
├── staging/
│   └── combined_stars.parquet
└── final/
    ├── combined_stars.parquet
    └── combined_stars.parquet.sha256
```

### Quick start: podman (1 container, local sink)

```bash
# build the image (one-time, ~5 min)
podman build -t lunar-hal -f install/Containerfile .

# run
./install/run-podman.sh
# data ends up in ./data/

# drop into the container shell
./install/run-podman.sh --shell
# now you're root in the container; you can:
#   ./install/orchestrate.sh
#   lnaicli sha256 -i /work/chunks/chunk-0.parquet
```

Env vars work the usual way:
```bash
TOTAL_STARS=3000000 PARALLEL=1 ./install/run-podman.sh
```

### Quick start: podman + MinIO (full DO simulation)

```bash
# brings up MinIO (S3 + web console) + the pipeline container
podman-compose -f install/podman-compose.yml up

# in another terminal:
#   MinIO console:    http://localhost:9001  (minioadmin / minioadmin)
#   download the result:
#   wget http://localhost:9000/lunar-hal-data/final/combined_stars.parquet
#   or via mc:
#   podman run --rm -it --network host docker.io/minio/mc \
#     mc alias set local http://localhost:9000 minioadmin minioadmin && \
#     mc cp local/lunar-hal-data/final/combined_stars.parquet ./
```

This is **ideal for CI** — no real Gaia / Spaces credentials in the pipeline,
everything runs in the CI runner.

### Quick start: podman + RustFS (experimental alternative)

[RustFS](https://github.com/rustfs/rustfs) is an S3-compatible object store
written in Rust, trending on GitHub. For our case (single-node sink under
compose) **single-node mode works**, distributed is still in beta. Suitable
for simulating DO locally.

```bash
# bring up RustFS + lunar
podman-compose -f install/podman-compose.rustfs.yml up

# in another terminal:
#   RustFS console:  http://localhost:9001  (rustfsadmin / rustfsadmin)
#   download the result:
#   wget http://localhost:9000/lunar-hal-data/final/combined_stars.parquet
```

**RustFS vs MinIO at a glance:**

| | MinIO | RustFS |
|---|---|---|
| Status | AGPL, maintenance mode | Apache 2.0, alpha/beta |
| Single-node | ✅ | ✅ |
| Distributed | ✅ | 🚧 Under testing |
| S3 API | reference | 100% compatible |
| Healthcheck | `:9000/minio/health/ready` | `:9000/health` |
| Credentials env | `MINIO_ROOT_USER/PASSWORD` | `RUSTFS_ACCESS_KEY/SECRET_KEY` (default `rustfsadmin`) |
| Container UID | root | `10001:10001` → bind-mount into `./data` needs `chown 10001:10001` (we use a named vol to avoid the hassle) |
| 4 KB objects | baseline | reportedly ~2.3× faster |
| Large sequential reads | ~53 Gbit/s | ~23 Gbit/s (for now) |

**Verdict for our workload:**
- We push ~500 MB chunks (Parquet), not 4 KB objects — so the 2.3× headline
  number doesn't apply. Sequential-read perf also favours MinIO.
- RustFS is worth a spin **for the "all-Rust" aesthetic and Apache 2.0**;
  not for raw performance.
- On prod with real DO Spaces none of this matters — we talk to the S3 API of
  Spaces, not MinIO/RustFS.

> ⚠️ **Don't put RustFS in front of critical data in prod** — alpha/beta.
> For local dev / CI simulation: absolutely fine.

### Tools required per mode

| Mode | s3cmd | parallel | curl/wget | docker/podman | MinIO/RustFS |
|---|---|---|---|---|---|
| DO Spaces | ✓ (installed by cloud-init) | ✓ | ✓ | — | — |
| Native local | — | ✓ | — | — | — |
| Podman | inside the image | inside | inside | ✓ | — |
| Podman + MinIO/RustFS | inside | inside | inside | ✓ | via compose |

---

## 0. Prerequisites (one-time)

```bash
# locally: install doctl if you don't have it
brew install doctl        # macOS
# or: snap install doctl  # Linux

doctl auth init           # paste the API token from cloud.digitalocean.com → API
doctl account get         # confirm auth
```

Create an SSH key for the droplet (if you don't have one):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/lunar-hal -C "lunar-hal"
doctl compute ssh-key import lunar-hal --public-key-file ~/.ssh/lunar-hal.pub
```

---

## 1. Create a Spaces bucket

Via the UI or CLI:
```bash
# bucket in sfo3 (same region as the droplet → free intra-region egress)
doctl spaces create lunar-hal-data --region sfo3

# access key (one-time; the secret is shown only here)
doctl spaces access-key create --name lunar-hal-pipeline
# → save the Access Key and Secret Key — you'll need them below
```

---

## 2. Create the droplet with cloud-init

```bash
# pull cloud-init from the repo
curl -fsSL https://raw.githubusercontent.com/ininids/lunar-hal/main/install/cloud-init.yaml \
  -o /tmp/lunar-cloud-init.yaml

# grab your key fingerprint:
SSH_FP=$(doctl compute ssh-key list --format fingerprint --no-header | head -1)

doctl compute droplet create lunar-hal-pipeline \
  --size c-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --region sfo3 \
  --ssh-keys "$SSH_FP" \
  --user-data-file /tmp/lunar-cloud-init.yaml \
  --tag-names lunar-hal \
  --enable-monitoring \
  --wait

# ssh in
IP=$(doctl compute droplet list --tag-name lunar-hal --format PublicIPv4 --no-header | head -1)
ssh -i ~/.ssh/lunar-hal root@"$IP"
```

`cloud-init` installs Rust, clones the repo, builds `lnaicli`, and installs
`s3cmd / jq / parallel`. **This takes ~10–15 min** (cloud-init + cargo build).
Tail the progress:
```bash
tail -f /var/log/cloud-init-output.log
tail -f /var/log/lunar-hal-ready.log   # when it appears — ready
```

---

## 3. Put the secrets on the droplet

SSH in and create `/opt/lunar-hal/.env`:
```bash
ssh -i ~/.ssh/lunar-hal root@<DROPLET_IP>
sudo -i
cat > /opt/lunar-hal/.env <<'EOF'
# --- DO Spaces ---
export SPACES_KEY="<paste access key from step 1>"
export SPACES_SECRET="<paste secret key from step 1>"
export SPACES_BUCKET="lunar-hal-data"
export SPACES_REGION="sfo3"

# --- ESA Gaia (https://gea.esac.esa.int → Login → sign up) ---
export GAIA_USER="<your gaia username>"
export GAIA_PASS="<your gaia password>"

# --- run parameters (optional overrides) ---
export TOTAL_STARS=30000000
export CHUNK_SIZE=3000000
export PARALLEL=3
export MAX_RUWE=1.4
EOF
chmod 600 /opt/lunar-hal/.env
. /opt/lunar-hal/.env
```

---

## 4. Run the pipeline

```bash
cd /opt/lunar-hal
./install/orchestrate.sh
```

What happens:
1. Builds a plan: 10 chunks across RA = `[0..36), [36..72), …, [324..360)`.
2. Launches **3 parallel** Gaia queries (in the background).
3. As each chunk finishes, it gets cleaned → written to `chunks/`.
4. Finished chunks are uploaded **atomically** to
   `s3://lunar-hal-data/chunks/chunk-NN.parquet` (via `.tmp` + rename).
5. Once all 10 are done — `combine` →
   `s3://lunar-hal-data/final/combined_stars.parquet`.
6. A `manifest.json` with the chunk list and SHA256 sums is generated.

### Watch progress

```bash
# in a sibling tmux pane / ssh session
watch -n 5 'ls /opt/lunar-hal/run/chunks/*.parquet 2>/dev/null | wc -l
            echo "---"
            tail -n 20 /opt/lunar-hal/run/logs/chunk-*.log 2>/dev/null'

# or live for a specific chunk
tail -f /opt/lunar-hal/run/logs/chunk-0.log
```

### Control

| Action | Command |
|---|---|
| Pause (lets the current query finish) | `kill -STOP $(pidof orchestrate.sh); sleep 1; kill -STOP $(pgrep -f lnaicli)` |
| Resume | `kill -CONT $(pidof orchestrate.sh); kill -CONT $(pgrep -f lnaicli)` |
| Abort | `Ctrl-C` in the `lunar` tmux session |
| Re-run only failed chunks | delete `chunks/chunk-NN.parquet` and `.sha256`, then re-run |

### Idempotency

`process_chunk` skips a chunk if `chunk-NN.parquet` + `chunk-NN.sha256` already
exist. You can safely re-run `orchestrate.sh` — already-done chunks are not
re-fetched.

---

## 5. Download the result

```bash
# from your laptop — egress is free (1 TB/month included with Spaces)
wget https://lunar-hal-data.sfo3.digitaloceanspaces.com/final/combined_stars.parquet
sha256sum combined_stars.parquet
# compare with the digest stored in Spaces
doctl spaces list lunar-hal-data/final/
```

---

## 6. Tear down the droplet (IMPORTANT — otherwise it keeps billing ~$0.05/hr)

```bash
# locally
doctl compute droplet delete lunar-hal-pipeline

# you can keep Spaces — 250 GB is $5/mo
# if you downloaded everything and don't need it anymore:
doctl spaces delete lunar-hal-data
```

---

## Other volumes

| I want | `TOTAL_STARS` | `CHUNK_SIZE` | `PARALLEL` | Expected time |
|---|---|---|---|---|
| 3M (smoke test) | 3 000 000 | 3 000 000 | 1 | ~10–15 min |
| 10M | 10 000 000 | 2 000 000 | 3 | ~30–40 min |
| **30M (default)** | **30 000 000** | **3 000 000** | **3** | **~60–90 min** |
| 60M | 60 000 000 | 3 000 000 | 3 | ~2–2.5 h |
| 100M | 100 000 000 | 5 000 000 | 3 | ~3–4 h |

> ⚠️ **Hard ceiling from Gaia**: DR3 has ~33M stars with `parallax > 0`. If
> you need 100M, either relax `MAX_RUWE` (>1.4) or merge with DR2.

---

## What's in the repo

```
install/
├── README.md                  # ← you are here
├── cloud-init.yaml            # droplet bootstrap (DO)
├── orchestrate.sh             # main pipeline (DO / native / podman)
├── Containerfile              # image for podman/docker
├── podman-compose.yml         # MinIO + lunar-hal one-command stack (stable default)
├── podman-compose.rustfs.yml  # RustFS + lunar-hal (experimental)
└── run-podman.sh              # handy wrapper: SINK=local|minio|rustfs

ai/lunar-ai-cli/src/main.rs   # CLI (with --ra-min/--ra-max support)
                                # + exponential backoff
                                # + SHA256 in clean
```

---

## Troubleshooting

**`s3cmd` complains with 403** — check `--signature-v4` in `orchestrate.sh`.
Some DO Spaces keys require v4; the script already uses v4. If you still hit
403, double-check `SPACES_KEY` and `SPACES_SECRET` and that the bucket region
matches `SPACES_REGION`.

**`lnaicli fetch` is stuck on `Job phase: QUEUED` for hours** — Gaia is
overloaded. Run overnight (UTC 02:00–08:00 is the quietest). Or reduce
`PARALLEL=1`.

**`cargo build` in cloud-init ran out of memory** — likely not enough RAM. On
`c-2vcpu-4gb` polars eats a lot during compilation. Use `c-4` ($72/mo) or
`gd-2vcpu-8gb` for the build, then recreate the smaller one.

**Want to stream straight from Gaia into Spaces** — the current implementation
writes the CSV to the droplet's disk, then uploads to Spaces. You could pipe
`reqwest::Response → hyper::Body → S3 multipart upload` and skip the disk, but
that's outside the scope of this iteration.