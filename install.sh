#!/usr/bin/env bash
# ==============================================================================
# LUNAR-HAL Installation & Environment Setup Script
# ==============================================================================
set -euo pipefail

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}✦ ========================================================== ✦${NC}"
echo -e "${CYAN}${BOLD}   LUNAR-HAL — Stellar Astrophysics & Kinematics Platform    ${NC}"
echo -e "${CYAN}${BOLD}   Environment Verification & Build Setup                    ${NC}"
echo -e "${CYAN}${BOLD}✦ ========================================================== ✦${NC}\n"

# 1. Check Rust toolchain
echo -e "${BOLD}[1/6] Checking Rust toolchain...${NC}"
if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    RUST_VER=$(rustc --version)
    echo -e "  ${GREEN}✓${NC} Found Rust: ${RUST_VER}"
else
    echo -e "  ${RED}✗ Rust toolchain not found.${NC}"
    echo -e "    Please install Rust via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# 2. Check Node.js and npm
echo -e "\n${BOLD}[2/6] Checking Node.js & npm...${NC}"
if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    NODE_VER=$(node --version)
    NPM_VER=$(npm --version)
    echo -e "  ${GREEN}✓${NC} Found Node.js: ${NODE_VER}, npm: ${NPM_VER}"
else
    echo -e "  ${YELLOW}! Node.js/npm not found. Testbench UI CSS compilation may be skipped.${NC}"
fi

# 3. Check Dioxus CLI (dx)
echo -e "\n${BOLD}[3/6] Checking Dioxus CLI (dx)...${NC}"
if command -v dx >/dev/null 2>&1; then
    DX_VER=$(dx --version 2>&1 | head -n1 || echo "unknown")
    echo -e "  ${GREEN}✓${NC} Found Dioxus CLI: ${DX_VER}"
else
    echo -e "  ${YELLOW}! Dioxus CLI (dx) not found in PATH.${NC}"
    echo -e "    You can install it with: cargo install dioxus-cli --version 0.7.1"
fi

# 4. Prepare required runtime directories & .env
echo -e "\n${BOLD}[4/6] Initializing runtime directories & configuration...${NC}"
mkdir -p ai_data data scenes docs/assets/covers models/covers target
if [ ! -f .env ] && [ -f .env.example ]; then
    echo -e "  ${CYAN}→${NC} Creating .env from .env.example..."
    cp .env.example .env
fi
echo -e "  ${GREEN}✓${NC} Workspace directories verified."

# 5. Build Tailwind / CSS for Testbench (if npm available)
echo -e "\n${BOLD}[5/6] Building UI assets...${NC}"
if command -v npm >/dev/null 2>&1 && [ -d "testbench/lunar-testbench" ]; then
    echo -e "  ${CYAN}→${NC} Installing & compiling CSS in testbench/lunar-testbench..."
    (cd testbench/lunar-testbench && npm ci --silent && npm run build:css --silent) || {
        echo -e "  ${YELLOW}! CSS build skipped or encountered warnings.${NC}"
    }
    echo -e "  ${GREEN}✓${NC} UI assets ready."
else
    echo -e "  ${YELLOW}! Skipping CSS build (npm or testbench folder missing).${NC}"
fi

# 6. Build Core Binaries
echo -e "\n${BOLD}[6/6] Compiling core workspace binaries...${NC}"
cargo build -p lunar-backend -p lunar-start -p lunar-start-backend -p lunar-ai-cli

echo -e "\n${GREEN}${BOLD}✦ ========================================================== ✦${NC}"
echo -e "${GREEN}${BOLD}   LUNAR-HAL is ready to launch!                             ${NC}"
echo -e "${GREEN}${BOLD}✦ ========================================================== ✦${NC}"
echo -e "\nQuick start options:"
echo -e "  ${BOLD}1. Launch full managed suite (backend + frontend + testbench):${NC}"
echo -e "     cargo run -p lunar-start"
echo -e "\n  ${BOLD}2. Run backend manually:${NC}"
echo -e "     cargo run -p lunar-backend"
echo -e "\n  ${BOLD}3. Run WebOS Testbench:${NC}"
echo -e "     cargo run -p lunar-testbench --bin lunar-testbench"
echo -e "\n  ${BOLD}4. Launch Dioxus Frontend directly:${NC}"
echo -e "     cd crates/lunar-frontend && dx serve --platform web\n"