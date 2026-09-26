#!/usr/bin/env bash
# Stage 3 contract/surrogate checks; this does not evaluate trained serving weights.
#
#   * regenerates the 256-row synthetic stellar-e2e-v1 fixture and checks its
#     checksum/manifest
#   * runs contract, synthetic-surrogate, seeded-untrained, and leakage suites
#
# Reports land in $LUNAR_AI_REPORT_DIR (default: target/ai-reports).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LUNAR_AI_REPORT_DIR="${LUNAR_AI_REPORT_DIR:-$ROOT/target/ai-reports}"
export LUNAR_AI_GIT_REV="${LUNAR_AI_GIT_REV:-$(git -C "$ROOT" rev-parse --short HEAD)}"

echo "== [e2e-ai] regenerate frozen fixture (must be byte-identical) =="
cargo run -q -p lnai-training --example gen-fixture

echo "== [e2e-ai] contract and surrogate suites (not checkpoint quality) =="
cargo test -p lnai-training \
  --test fixture_leakage \
  --test pinn_accuracy \
  --test gnn_oracle \
  --test position_rollout \
  --test pinn_gnn_chain \
  -- --nocapture

echo "== [e2e-ai] reports in $LUNAR_AI_REPORT_DIR =="
ls -1 "$LUNAR_AI_REPORT_DIR"
