#!/usr/bin/env bash
# Runs orchestrate.sh inside a podman container.
# Default: local sink in ./data/. Switchable to MinIO or RustFS.
#
# Usage:
#   ./install/run-podman.sh                       # local sink
#   SINK=minio ./install/run-podman.sh            # sink into MinIO (compose must be up)
#   SINK=rustfs ./install/run-podman.sh           # sink into RustFS (compose must be up)
#   ./install/run-podman.sh --shell               # drop into the container shell
#
# Requires a built image: podman build -t lunar-hal -f install/Containerfile .

set -euo pipefail

# Auto-source .env from the repo root for GAIA_USER/GAIA_PASS/SPACES_*
# and run parameters (TOTAL_STARS, CHUNK_SIZE, JOBS, SINK_KIND).
# These match the envs that install/podman-compose.yml injects via env_file.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
if [[ -f "$_REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$_REPO_ROOT/.env"
  set +a
fi

IMAGE="${IMAGE:-lunar-hal}"
WORKDIR_LOCAL="${WORKDIR_LOCAL:-$PWD/data}"
SINK="${SINK:-local}"   # local | minio | rustfs

mkdir -p "$WORKDIR_LOCAL"

# Base envs — always forwarded from .env to the container.
ENV_ARGS=(
  -e "GAIA_USER=$GAIA_USER"
  -e "GAIA_PASS=$GAIA_PASS"
  -e "TOTAL_STARS=$TOTAL_STARS"
  -e "CHUNK_SIZE=$CHUNK_SIZE"
  -e "JOBS=$JOBS"
)

case "$SINK" in
  local)
    ENV_ARGS+=(
      -e "SINK_KIND=local"
      -e "LOCAL_OUT_DIR=/work"
    )
    EXTRA_NET=""
    ;;
  minio)
    # Assumes MinIO is up via `podman-compose -f install/podman-compose.yml up` and
    # port 9000 is published to the host. Since we are using host network mode,
    # the container reaches MinIO on loopback (overrides SPACES_ENDPOINT from .env).
    ENV_ARGS+=(
      -e "SINK_KIND=s3"
      -e "SPACES_KEY=${SPACES_KEY:-minioadmin}"
      -e "SPACES_SECRET=${SPACES_SECRET:-minioadmin}"
      -e "SPACES_BUCKET=${SPACES_BUCKET:-lunar-hal-data}"
      -e "SPACES_REGION=${SPACES_REGION:-us-east-1}"
      -e "SPACES_ENDPOINT=127.0.0.1:9000"
    )
    EXTRA_NET="--network=host"
    ;;
  rustfs)
    # Assumes RustFS is up via
    # `podman-compose -f install/podman-compose.rustfs.yml up`.
    ENV_ARGS+=(
      -e "SINK_KIND=s3"
      -e "SPACES_KEY=${SPACES_KEY:-rustfsadmin}"
      -e "SPACES_SECRET=${SPACES_SECRET:-rustfsadmin}"
      -e "SPACES_BUCKET=${SPACES_BUCKET:-lunar-hal-data}"
      -e "SPACES_REGION=${SPACES_REGION:-us-east-1}"
      -e "SPACES_ENDPOINT=127.0.0.1:9000"
    )
    EXTRA_NET="--network=host"
    ;;
  *)
    echo "Unknown SINK: $SINK (use 'local', 'minio', or 'rustfs')" >&2
    exit 2
    ;;
esac

if [[ "${1:-}" == "--shell" ]]; then
  exec podman run --rm -it \
    $EXTRA_NET \
    -v "$WORKDIR_LOCAL:/work" \
    -e "LNAICLI=/usr/local/bin/lnaicli" \
    "${ENV_ARGS[@]}" \
    "$IMAGE" \
    bash
fi

exec podman run --rm \
  $EXTRA_NET \
  -v "$WORKDIR_LOCAL:/work" \
  "${ENV_ARGS[@]}" \
  "$IMAGE"
