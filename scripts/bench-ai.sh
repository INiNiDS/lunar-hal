#!/usr/bin/env bash
# Stage 3 performance foundation:
#   * CPU criterion benches (preprocessing/k-NN, PINN/SIREN/GNN forward)
#   * report-only baseline with natural-noise estimate over N identical runs
#   * `--gpu` additionally compiles the synchronized CUDA harness
#     (nightly/manual GPU runner only; still report-only).
#
# Usage: scripts/bench-ai.sh [--runs N] [--gpu]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUNS=3
GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU=1; shift ;;
    --runs) RUNS="$2"; shift 2 ;;
    --runs=*) RUNS="${1#--runs=}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export LUNAR_AI_REPORT_DIR="${LUNAR_AI_REPORT_DIR:-$ROOT/target/ai-reports}"
export LUNAR_AI_GIT_REV="${LUNAR_AI_GIT_REV:-$(git -C "$ROOT" rev-parse --short HEAD)}"

echo "== [bench-ai] CPU criterion benches =="
cargo bench -p lnai-training --bench pinn_forward --bench gnn_graph \
  --bench siren_texture --bench localization

if [[ "$GPU" == "1" ]]; then
  echo "== [bench-ai] synchronized GPU harness (report-only) =="
  cargo bench -p lnai-training --features gpu-harness --bench gpu_synchronized
fi

echo "== [bench-ai] baseline noise estimate ($RUNS runs) =="
cargo run -q -p lnai-training --bin ai-baseline -- --runs "$RUNS"

echo "== [bench-ai] reports in $LUNAR_AI_REPORT_DIR =="
ls -1 "$LUNAR_AI_REPORT_DIR"
