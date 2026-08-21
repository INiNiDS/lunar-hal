#!/usr/bin/env bash
# Stage 3 E2E foundation: one command that produces the full AI report set.
#
#   * regenerates the frozen stellar-e2e-v1 fixture deterministically and
#     verifies its checksum/manifest (tasks 1-2)
#   * runs the correctness suites: PINN per-target + Stefan-Boltzmann
#     residual, GNN oracle/chained/baseline, position rollout, the
#     PINN->GNN chain contract and leakage proofs (tasks 4-6)
#
# Reports land in $LUNAR_AI_REPORT_DIR (default: target/ai-reports).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LUNAR_AI_REPORT_DIR="${LUNAR_AI_REPORT_DIR:-$ROOT/target/ai-reports}"
export LUNAR_AI_GIT_REV="${LUNAR_AI_GIT_REV:-$(git -C "$ROOT" rev-parse --short HEAD)}"

echo "== [e2e-ai] regenerate frozen fixture (must be byte-identical) =="
cargo run -q -p lnai-training --example gen-fixture

echo "== [e2e-ai] correctness suites =="
cargo test -p lnai-training \
  --test fixture_leakage \
  --test pinn_accuracy \
  --test gnn_oracle \
  --test position_rollout \
  --test pinn_gnn_chain \
  -- --nocapture

echo "== [e2e-ai] reports in $LUNAR_AI_REPORT_DIR =="
ls -1 "$LUNAR_AI_REPORT_DIR"
