#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Building release binaries on host..."
cargo build --release --bin lunar-backend --bin lunar-start-backend

echo "==> Building lunar-frontend WASM..."
(cd crates/lunar-frontend && dx build --platform web --release)

echo "==> Building lunar-testbench WASM..."
(cd testbench/lunar-testbench && dx build --platform web --release)

echo "==> Staging artifacts into deploy/dist..."
rm -rf deploy/dist
mkdir -p deploy/dist/frontend deploy/dist/testbench

cp target/release/lunar-backend deploy/dist/
cp target/release/lunar-start-backend deploy/dist/
cp -r target/dx/lunar-frontend/release/web/public/* deploy/dist/frontend/
cp -r target/dx/lunar-testbench/release/web/public/* deploy/dist/testbench/

echo "==> Artifacts staged successfully in deploy/dist."
