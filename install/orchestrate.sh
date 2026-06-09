#!/usr/bin/env bash
# lunar-hal orchestrator
#
# Splits the sky into N RA-sliced chunks, fetches Gaia in parallel, cleans each
# chunk into Parquet, and **streams every chunk to the sink as soon as it is
# ready** (DO Spaces / local folder / MinIO / RustFS). At the end merges all
# chunks into a single file under /final/.
#
# Supports two sinks via SINK_KIND:
#   - s3    : DO Spaces or any S3-compatible store (MinIO, RustFS) via s3cmd
#   - local : a local directory (via cp)
#
# Idempotent: chunks that are already locally produced (with their .sha256) are
# skipped on re-run; their pre-computed artefacts are simply re-pushed to sink.

set -euo pipefail

###############################################################################
#  Auto-source .env (repo root) so users don't have to remember to export    #
#  GAIA_USER/GAIA_PASS/SPACES_* themselves.                                   #
###############################################################################
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
if [[ -f "$_REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$_REPO_ROOT/.env"
  set +a
fi

###############################################################################
#  Configuration via env                                                       #
###############################################################################
TOTAL_STARS="${TOTAL_STARS:-30000000}"
CHUNK_SIZE="${CHUNK_SIZE:-3000000}"
JOBS="${JOBS:-3}"
MAX_RUWE="${MAX_RUWE:-1.4}"
WORKDIR="${WORKDIR:-/opt/lunar-hal/run}"

# --- sink --------------------------------------------------------------------
SINK_KIND="${SINK_KIND:-s3}"                  # "s3" | "local"
SPACES_BUCKET="${SPACES_BUCKET:-lunar-hal-data}"
SPACES_REGION="${SPACES_REGION:-sfo3}"
SPACES_PREFIX="${SPACES_PREFIX:-chunks}"
FINAL_NAME="${FINAL_NAME:-combined_stars.parquet}"
SPACES_ENDPOINT="${SPACES_ENDPOINT:-}"        # if set → MinIO / RustFS / custom
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-}"            # required when SINK_KIND=local

# --- CLI search -------------------------------------------------------------
if [[ -z "${LNAICLI:-}" ]]; then
  if command -v lnaicli >/dev/null 2>&1; then
    LNAICLI="lnaicli"
  elif [[ -f "$_REPO_ROOT/target/release/lnaicli" ]]; then
    LNAICLI="$_REPO_ROOT/target/release/lnaicli"
  else
    LNAICLI="$WORKDIR/target/release/lnaicli"
  fi
fi

###############################################################################
#  Logging / helpers                                                           #
###############################################################################
log()  { printf '\033[1;36m[%(%H:%M:%S)T] %s\033[0m\n' -1 "$*"; }
warn() { printf '\033[1;33m[%(%H:%M:%S)T] WARN: %s\033[0m\n' -1 "$*" >&2; }
fail() { printf '\033[1;31m[%(%H:%M:%S)T] ERROR: %s\033[0m\n' -1 "$*" >&2; exit 1; }

require() { command -v "$1" >/dev/null 2>&1 || fail "missing required tool: $1"; }

# ---------- sink abstraction -------------------------------------------------
sink_put() {
  # $1 = local path, $2 = sink key (e.g. "chunks/chunk-0.parquet")
  local local_path="$1" remote_key="$2" dest tmp
  if [[ "$SINK_KIND" == "local" ]]; then
    [[ -n "$LOCAL_OUT_DIR" ]] || fail "SINK_KIND=local but LOCAL_OUT_DIR is empty"
    dest="$LOCAL_OUT_DIR/$remote_key"
    mkdir -p "$(dirname "$dest")"
    tmp="${dest}.tmp.$$"
    cp "$local_path" "$tmp"
    mv -f "$tmp" "$dest"
  else
    local bucket_url="s3://${SPACES_BUCKET}/${remote_key}"
    s3 put "$local_path" "${bucket_url}.tmp" >/dev/null
    s3 mv  "${bucket_url}.tmp" "${bucket_url}"  >/dev/null
  fi
}

s3() {
  # Universal s3cmd wrapper. Works with DO Spaces, MinIO, and RustFS.
  local host host_bucket ssl_opts
  if [[ -n "$SPACES_ENDPOINT" ]]; then
    # MinIO / RustFS / custom endpoint: path-style URL.
    # Both must point to the endpoint to prevent s3cmd from reverting to DNS-style %(bucket)s.s3.amazonaws.com
    host="$SPACES_ENDPOINT"
    host_bucket="$SPACES_ENDPOINT"
    # Local S3 simulations usually run plain HTTP (no SSL)
    ssl_opts=( "--no-ssl" )
  else
    # DO Spaces: virtual-host-style (requires HTTPS)
    host="${SPACES_REGION}.digitaloceanspaces.com"
    host_bucket="%(bucket)s.${SPACES_REGION}.digitaloceanspaces.com"
    ssl_opts=( "--ssl" "--no-check-certificate" )
  fi
  s3cmd \
    --access_key="$SPACES_KEY" \
    --secret_key="$SPACES_SECRET" \
    --host="$host" \
    ${host_bucket:+--host-bucket="$host_bucket"} \
    "${ssl_opts[@]}" \
    "$@"
}

# ---------- paths ------------------------------------------------------------
mkdir -p "$WORKDIR/chunks" "$WORKDIR/logs" "$WORKDIR/staging"
cd "$WORKDIR"

require jq
require sha256sum
require "$LNAICLI"
require bash

if [[ "$SINK_KIND" == "local" ]]; then
  log "Sink: local → $LOCAL_OUT_DIR"
else
  log "Sink: s3   → s3://${SPACES_BUCKET}/${SPACES_PREFIX}/ (${SPACES_ENDPOINT:-DO Spaces $SPACES_REGION})"
  require s3cmd
fi

###############################################################################
#  Build the chunk plan across RA                                              #
###############################################################################
NUM_CHUNKS=$(( (TOTAL_STARS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
RA_STEP=$(awk -v n="$NUM_CHUNKS" 'BEGIN { printf "%.10f", 360.0 / n }')
log "Config: total=$TOTAL_STARS chunk=$CHUNK_SIZE → $NUM_CHUNKS chunks, parallel=$JOBS"

> chunks/plan.tsv
for ((i=0; i<NUM_CHUNKS; i++)); do
  ra_min=$(awk -v i="$i" -v s="$RA_STEP" 'BEGIN { printf "%.6f", i * s }')
  ra_max=$(awk -v i="$i" -v s="$RA_STEP" 'BEGIN { v = (i+1) * s; if (v > 360) v = 360; printf "%.6f", v }')
  printf '%s|%s|%s\n' "$i" "$ra_min" "$ra_max" >> chunks/plan.tsv
done
log "Plan:"
sed 's/^/    /' chunks/plan.tsv

###############################################################################
#  Process a single chunk                                                      #
###############################################################################
process_chunk() {
  local i="$1" ra_min="$2" ra_max="$3"
  local csv="staging/chunk-${i}.csv"
  local parq="chunks/chunk-${i}.parquet"
  local logf="logs/chunk-${i}.log"
  local sha t0 t1

  # Idempotency: if the chunk is already produced locally, skip and just push.
  if [[ -f "$parq" && -f "chunks/chunk-${i}.sha256" ]]; then
    log "[chunk-$i] already produced locally, pushing to sink"
    sha=$(awk '{print $1}' "chunks/chunk-${i}.sha256")
    sink_put "$parq"         "${SPACES_PREFIX}/chunk-${i}.parquet" || return 1
    sink_put "chunks/chunk-${i}.sha256" "${SPACES_PREFIX}/chunk-${i}.sha256" || return 1
    return 0
  fi

  log "[chunk-$i] RA=[$ra_min, $ra_max] → $parq"
  t0=$(date +%s)

  set +e
  {
    echo "=== FETCH ==="
    "$LNAICLI" fetch \
      --username "$GAIA_USER" \
      --password "$GAIA_PASS" \
      --max-rows "$CHUNK_SIZE" \
      --ra-min "$ra_min" \
      --ra-max "$ra_max" \
      --max-ruwe "$MAX_RUWE" \
      --output "$csv"

    echo "=== CLEAN ==="
    "$LNAICLI" clean \
      --input "$csv" \
      --output "$parq" \
      --print-sha256
  } >"$logf" 2>&1
  rc=$?
  set -e

  # Fallback Strategy: If the main query fails, split the RA range in half.
  if [[ $rc -ne 0 || ! -s "$parq" ]]; then
    warn "[chunk-$i] FAILED (rc=$rc). Initiating automatic fallback split..."
    rm -f "$csv" "$parq" # Clear failed partial artifacts

    # Calculate midpoint of RA range
    local ra_mid
    ra_mid=$(awk -v min="$ra_min" -v max="$ra_max" 'BEGIN { printf "%.6f", min + (max - min) / 2 }')
    log "[chunk-$i] Splitting into sub-chunks: Part A=[$ra_min, $ra_mid] and Part B=[$ra_mid, $ra_max]"

    local parq_a="chunks/chunk-${i}a.parquet"
    local parq_b="chunks/chunk-${i}b.parquet"
    local csv_a="staging/chunk-${i}a.csv"
    local csv_b="staging/chunk-${i}b.csv"
    local logf_a="logs/chunk-${i}a.log"
    local logf_b="logs/chunk-${i}b.log"

    # Process Part A (first half)
    log "[chunk-${i}a] Fetching/cleaning Part A [$ra_min, $ra_mid]..."
    set +e
    {
      "$LNAICLI" fetch \
        --username "$GAIA_USER" \
        --password "$GAIA_PASS" \
        --max-rows "$CHUNK_SIZE" \
        --ra-min "$ra_min" \
        --ra-max "$ra_mid" \
        --max-ruwe "$MAX_RUWE" \
        --output "$csv_a"
      "$LNAICLI" clean --input "$csv_a" --output "$parq_a"
    } >"$logf_a" 2>&1
    rc_a=$?
    set -e
    rm -f "$csv_a"

    if [[ $rc_a -ne 0 || ! -s "$parq_a" ]]; then
      warn "[chunk-${i}a] Part A failed. See log: $logf_a"
      return 1
    fi

    # Process Part B (second half)
    log "[chunk-${i}b] Fetching/cleaning Part B [$ra_mid, $ra_max]..."
    set +e
    {
      "$LNAICLI" fetch \
        --username "$GAIA_USER" \
        --password "$GAIA_PASS" \
        --max-rows "$CHUNK_SIZE" \
        --ra-min "$ra_mid" \
        --ra-max "$ra_max" \
        --max-ruwe "$MAX_RUWE" \
        --output "$csv_b"
      "$LNAICLI" clean --input "$csv_b" --output "$parq_b"
    } >"$logf_b" 2>&1
    rc_b=$?
    set -e
    rm -f "$csv_b"

    if [[ $rc_b -ne 0 || ! -s "$parq_b" ]]; then
      warn "[chunk-${i}b] Part B failed. See log: $logf_b"
      return 1
    fi

    # Merge Part A and Part B back into the main chunk parquet file
    log "[chunk-$i] Combining parts A and B into $parq..."
    "$LNAICLI" combine --inputs "$parq_a" "$parq_b" --output "$parq"

    # Cleanup sub-chunk temporary files
    rm -f "$parq_a" "$parq_b" "$logf_a" "$logf_b"
  fi

  sha=$(sha256sum "$parq" | awk '{print $1}')
  echo "$sha  chunk-${i}.parquet" > "chunks/chunk-${i}.sha256"
  rm -f "$csv"

  # Stream straight to the sink — no waiting for the rest of the chunks.
  sink_put "$parq"                          "${SPACES_PREFIX}/chunk-${i}.parquet"  || return 1
  sink_put "chunks/chunk-${i}.sha256"       "${SPACES_PREFIX}/chunk-${i}.sha256"   || return 1

  t1=$(date +%s)
  log "[chunk-$i] done + pushed in $((t1 - t0))s, sha256=${sha:0:12}…"
  return 0
}

export -f process_chunk log warn sink_put s3
export LNAICLI WORKDIR GAIA_USER GAIA_PASS CHUNK_SIZE MAX_RUWE
export SINK_KIND LOCAL_OUT_DIR SPACES_BUCKET SPACES_REGION SPACES_PREFIX SPACES_ENDPOINT SPACES_KEY SPACES_SECRET JOBS

###############################################################################
#  Run the worker pool                                                         #
###############################################################################
log "Starting $NUM_CHUNKS chunks, max $JOBS in parallel..."

if command -v parallel >/dev/null 2>&1; then
  log "Using GNU parallel"
  # Escaped pipe '\|' avoids Regex alternation empty splitting, PARALLEL_SHELL ensures bash
  # shellcheck disable=SC2002
  cat chunks/plan.tsv | PARALLEL_SHELL=bash parallel --colsep '\|' -j "$JOBS" --halt now,fail=1 \
    'process_chunk {1} {2} {3}'
else
  log "Using xargs -P"
  awk -F'|' '{print $1, $2, $3}' chunks/plan.tsv \
    | xargs -n 3 -P "$JOBS" -I{} bash -c "
        set -- {};
        process_chunk "$1" "$2" "$3"
      "
fi

# Verify
ready=$(ls chunks/chunk-*.parquet 2>/dev/null | wc -l)
[[ "$ready" -eq "$NUM_CHUNKS" ]] || fail "only $ready of $NUM_CHUNKS chunks ready, see logs/chunk-*.log"
log "All $NUM_CHUNKS chunks ready"

###############################################################################
#  Manifest                                                                    #
###############################################################################
MANIFEST="chunks/manifest.json"
{
  echo "{"
  echo "  \"generated_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
  echo "  \"total_target_rows\": $TOTAL_STARS,"
  echo "  \"chunk_size\": $CHUNK_SIZE,"
  echo "  \"chunks\": $NUM_CHUNKS,"
  echo "  \"ra_step_deg\": $RA_STEP,"
  echo "  \"max_ruwe\": $MAX_RUWE,"
  echo "  \"sink\": \"$SINK_KIND\","
  echo "  \"files\": ["
  first=1
  while IFS='|' read -r idx ra_min ra_max; do
    [[ $first -eq 0 ]] && echo ","
    first=0
    sha=$(awk '{print $1}' "chunks/chunk-${idx}.sha256")
    printf '    {"i": %s, "ra_min": %s, "ra_max": %s, "sha256": "%s", "key": "%s/chunk-%s.parquet"}' \
      "$idx" "$ra_min" "$ra_max" "$sha" "$SPACES_PREFIX" "$idx"
  done < chunks/plan.tsv
  echo ""
  echo "  ]"
  echo "}"
} > "$MANIFEST"
sink_put "$MANIFEST" "${SPACES_PREFIX}/manifest.json"
log "Manifest pushed"

###############################################################################
#  Final merge                                                                 #
###############################################################################
log "Merging $NUM_CHUNKS chunks into staging/${FINAL_NAME}..."
chunk_args=()
for ((i=0; i<NUM_CHUNKS; i++)); do
  chunk_args+=( "chunks/chunk-${i}.parquet" )
done
"$LNAICLI" combine --inputs "${chunk_args[@]}" --output "staging/${FINAL_NAME}"

final_sha=$(sha256sum "staging/${FINAL_NAME}" | awk '{print $1}')
log "Final file: staging/${FINAL_NAME} (sha256=${final_sha:0:12}…)"

sink_put "staging/${FINAL_NAME}"        "final/${FINAL_NAME}"
echo "$final_sha  ${FINAL_NAME}" > "staging/${FINAL_NAME}.sha256"
sink_put "staging/${FINAL_NAME}.sha256" "final/${FINAL_NAME}.sha256"

cat <<EOF

================================================================================
  DONE 🚀
================================================================================
Sink:     ${SINK_KIND}
File:    $([ "$SINK_KIND" = "local" ] && echo "$LOCAL_OUT_DIR/final/${FINAL_NAME}" || echo "s3://${SPACES_BUCKET}/final/${FINAL_NAME}")
SHA256:  ${final_sha}
Size:    $(du -h "staging/${FINAL_NAME}" | awk '{print $1}')
Chunks:  ${NUM_CHUNKS}
EOF
if [[ "$SINK_KIND" == "s3" && -z "$SPACES_ENDPOINT" ]]; then
cat <<EOF
Download: wget https://${SPACES_BUCKET}.${SPACES_REGION}.digitaloceanspaces.com/final/${FINAL_NAME}
EOF
fi
cat <<'EOF'
================================================================================
EOF