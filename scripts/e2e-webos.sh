#!/usr/bin/env bash
set -euo pipefail

# Run E2E only against an explicitly isolated stack. Supplying a stack command
# is optional: it receives fresh Gallery/scene/state directories and is stopped
# automatically when Playwright finishes.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/lunar-webos-e2e.XXXXXX")"
stack_pid=""

cleanup() {
  local exit_code=$?
  if [[ -n "$stack_pid" ]] && kill -0 "$stack_pid" 2>/dev/null; then
    kill "$stack_pid" 2>/dev/null || true
    wait "$stack_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp_root"
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

export LUNAR_GALLERY_DIR="${LUNAR_GALLERY_DIR:-$tmp_root/gallery}"
export LUNAR_SCENES_DIR="${LUNAR_SCENES_DIR:-$tmp_root/scenes}"
export LUNAR_FRONTEND_STATE_DIR="${LUNAR_FRONTEND_STATE_DIR:-$tmp_root/frontend-state}"
mkdir -p "$LUNAR_GALLERY_DIR" "$LUNAR_SCENES_DIR" "$LUNAR_FRONTEND_STATE_DIR"

if [[ -n "${LUNAR_E2E_STACK_COMMAND:-}" ]]; then
  (cd "$repo_root" && bash -lc "$LUNAR_E2E_STACK_COMMAND") &
  stack_pid=$!
fi

: "${LUNAR_E2E_BASE_URL:?Set LUNAR_E2E_BASE_URL to the isolated lunar-testbench URL.}"
health_url="${LUNAR_E2E_HEALTH_URL:-$LUNAR_E2E_BASE_URL}"
for _ in $(seq 1 60); do
  if curl --fail --silent --show-error --max-time 2 "$health_url" >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent --show-error --max-time 5 "$health_url" >/dev/null

cd "$repo_root/testbench/lunar-testbench"
exec npx playwright test --config playwright.config.ts "$@"
