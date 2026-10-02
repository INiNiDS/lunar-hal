#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_NAME="${HEROKU_APP:-lunar-hal-demo}"
MODE="all" # all | frontend | backend | skip-build

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--frontend|--frontend-only)
      MODE="frontend"
      shift
      ;;
    -t|--testbench|--testbench-only)
      MODE="testbench"
      shift
      ;;
    -w|--web|--ui|--frontends)
      MODE="web"
      shift
      ;;
    -b|--backend|--backend-only)
      MODE="backend"
      shift
      ;;
    -s|--skip-build)
      MODE="skip-build"
      shift
      ;;
    -a|--app)
      APP_NAME="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -w, --web         Rebuild BOTH frontends: lunar-frontend + testbench WASM (skips backend)"
      echo "  -f, --frontend    Rebuild only lunar-frontend WASM"
      echo "  -t, --testbench   Rebuild only lunar-testbench WASM"
      echo "  -b, --backend     Rebuild only backend binaries (lunar-backend, lunar-start-backend)"
      echo "  -s, --skip-build  Skip build steps, only repackage container and deploy deploy/dist"
      echo "  -a, --app <name>  Heroku app name (default: lunar-hal-demo)"
      echo "  -h, --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

echo "=========================================="
echo " Deploying to Heroku: $APP_NAME"
echo " Mode: $MODE"
echo "=========================================="

mkdir -p deploy/dist/frontend deploy/dist/testbench

case "$MODE" in
  all)
    echo "==> [1/4] Full build: backend, frontend, testbench..."
    ./scripts/package-release.sh
    ;;
  web)
    echo "==> [1/4] Building BOTH frontends (lunar-frontend + testbench WASM)..."
    echo "  -> Building lunar-frontend..."
    (cd crates/lunar-frontend && dx build --platform web --release)
    rm -rf deploy/dist/frontend/*
    cp -r target/dx/lunar-frontend/release/web/public/* deploy/dist/frontend/

    echo "  -> Building lunar-testbench..."
    (cd testbench/lunar-testbench && dx build --platform web --release)
    rm -rf deploy/dist/testbench/*
    cp -r target/dx/lunar-testbench/release/web/public/* deploy/dist/testbench/
    ;;
  frontend)
    echo "==> [1/4] Building lunar-frontend WASM only..."
    (cd crates/lunar-frontend && dx build --platform web --release)
    rm -rf deploy/dist/frontend/*
    cp -r target/dx/lunar-frontend/release/web/public/* deploy/dist/frontend/
    ;;
  testbench)
    echo "==> [1/4] Building lunar-testbench WASM only..."
    (cd testbench/lunar-testbench && dx build --platform web --release)
    rm -rf deploy/dist/testbench/*
    cp -r target/dx/lunar-testbench/release/web/public/* deploy/dist/testbench/
    ;;
  backend)
    echo "==> [1/4] Building backend binaries only..."
    cargo build --release --bin lunar-backend --bin lunar-start-backend
    cp target/release/lunar-backend deploy/dist/
    cp target/release/lunar-start-backend deploy/dist/
    ;;
  skip-build)
    echo "==> [1/4] Skipping compilation (using existing deploy/dist)..."
    ;;
esac

if [ ! -f "deploy/dist/lunar-backend" ] || [ ! -f "deploy/dist/lunar-start-backend" ]; then
  echo "Error: Backend binaries missing in deploy/dist. Run without -f or run full build first." >&2
  exit 1
fi

if [ ! -d "deploy/dist/frontend" ] || [ -z "$(ls -A deploy/dist/frontend 2>/dev/null)" ]; then
  echo "Error: Frontend assets missing in deploy/dist/frontend. Run full build first." >&2
  exit 1
fi

if [ ! -d "deploy/dist/testbench" ] || [ -z "$(ls -A deploy/dist/testbench 2>/dev/null)" ]; then
  echo "Error: Testbench assets missing in deploy/dist/testbench. Run full build first." >&2
  exit 1
fi

echo "==> [2/4] Building container image with Podman..."
podman build --format docker -t "registry.heroku.com/$APP_NAME/web" .

echo "==> [3/4] Authenticating with Heroku container registry..."
HEROKU_TOKEN="$(heroku auth:token)"
echo "$HEROKU_TOKEN" | podman login --username=_ --password-stdin registry.heroku.com

echo "==> [4/4] Pushing image (v2s2 format) and releasing..."
podman push --format v2s2 "registry.heroku.com/$APP_NAME/web"
heroku container:release web -a "$APP_NAME"

echo ""
echo "=========================================="
echo " Deployment completed successfully!"
APP_URL="$(heroku info -s -a "$APP_NAME" 2>/dev/null | grep web_url | cut -d= -f2 || true)"
if [ -n "$APP_URL" ]; then
  echo " App URL: $APP_URL"
else
  echo " App URL: https://$APP_NAME.herokuapp.com"
fi
echo "=========================================="
