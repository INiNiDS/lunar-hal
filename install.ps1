# ==============================================================================
# LUNAR-HAL Installation & Environment Setup Script (Windows PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "✦ ========================================================== ✦" -ForegroundColor Cyan
Write-Host "   LUNAR-HAL — Stellar Astrophysics & Kinematics Platform    " -ForegroundColor Cyan
Write-Host "   Environment Verification & Build Setup (Windows)          " -ForegroundColor Cyan
Write-Host "✦ ========================================================== ✦`n" -ForegroundColor Cyan

# 1. Check Rust toolchain
Write-Host "[1/6] Checking Rust toolchain..." -ForegroundColor White
if ((Get-Command cargo -ErrorAction SilentlyContinue) -and (Get-Command rustc -ErrorAction SilentlyContinue)) {
    $rustVer = rustc --version
    Write-Host "  ✓ Found Rust: $rustVer" -ForegroundColor Green
} else {
    Write-Host "  ✗ Rust toolchain not found." -ForegroundColor Red
    Write-Host "    Please install Rust via https://rustup.rs" -ForegroundColor Yellow
    exit 1
}

# 2. Check Node.js and npm
Write-Host "`n[2/6] Checking Node.js & npm..." -ForegroundColor White
if ((Get-Command node -ErrorAction SilentlyContinue) -and (Get-Command npm -ErrorAction SilentlyContinue)) {
    $nodeVer = node --version
    $npmVer = npm --version
    Write-Host "  ✓ Found Node.js: $nodeVer, npm: $npmVer" -ForegroundColor Green
} else {
    Write-Host "  ! Node.js/npm not found. Testbench UI CSS compilation may be skipped." -ForegroundColor Yellow
}

# 3. Check Dioxus CLI (dx) & version compatibility
Write-Host "`n[3/6] Checking Dioxus CLI (dx) compatibility..." -ForegroundColor White
$crateDioxusVer = "0.7.9"
if (Test-Path "Cargo.lock") {
    $lockContent = Get-Content "Cargo.lock" -Raw
    if ($lockContent -match 'name = "dioxus"[\r\n]+version = "([0-9]+\.[0-9]+\.[0-9]+)"') {
        $crateDioxusVer = $matches[1]
    }
}
$crateMajorMinor = ($crateDioxusVer -split '\.')[0..1] -join '.'

if (Get-Command dx -ErrorAction SilentlyContinue) {
    $dxRaw = (dx --version 2>&1 | Select-Object -First 1)
    if ($dxRaw -match '([0-9]+\.[0-9]+\.[0-9]+)') {
        $dxVerNum = $matches[1]
        $dxMajorMinor = ($dxVerNum -split '\.')[0..1] -join '.'

        if ($dxVerNum -eq $crateDioxusVer) {
            Write-Host "  ✓ Found Dioxus CLI: $dxRaw (matches Cargo.lock: $crateDioxusVer)" -ForegroundColor Green
        } elseif ($dxMajorMinor -eq $crateMajorMinor) {
            Write-Host "  ! Dioxus CLI patch difference detected:" -ForegroundColor Yellow
            Write-Host "    Installed CLI: $dxVerNum"
            Write-Host "    Project crate: $crateDioxusVer"
            Write-Host "    Note: Dioxus works best when the CLI and crate versions match exactly." -ForegroundColor Cyan
            Write-Host "    To align CLI: cargo install dioxus-cli --version $crateDioxusVer --locked"
            Write-Host "    (or update project crate: cargo update -p dioxus)"
        } else {
            Write-Host "  ✗ Dioxus CLI version mismatch (Incompatible!):" -ForegroundColor Red
            Write-Host "    Installed CLI: $dxVerNum"
            Write-Host "    Project crate: $crateDioxusVer"
            Write-Host "    Different major/minor versions are incompatible."
            Write-Host "    Please install matching CLI: cargo install dioxus-cli --version $crateDioxusVer --locked"
        }
    } else {
        Write-Host "  ✓ Found Dioxus CLI: $dxRaw" -ForegroundColor Green
    }
} else {
    Write-Host "  ! Dioxus CLI (dx) not found in PATH." -ForegroundColor Yellow
    Write-Host "    Required version for this project: $crateDioxusVer"
    Write-Host "    Install with: cargo install dioxus-cli --version $crateDioxusVer --locked"
}

# 4. Prepare required runtime directories & .env
Write-Host "`n[4/6] Initializing runtime directories & configuration..." -ForegroundColor White
$dirs = @("ai_data", "data", "scenes", "target")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
if ((-not (Test-Path ".env")) -and (Test-Path ".env.example")) {
    Write-Host "  → Creating .env from .env.example..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
}
Write-Host "  ✓ Workspace directories verified." -ForegroundColor Green

# 5. Build Tailwind / CSS for Testbench (if npm available)
Write-Host "`n[5/6] Building UI assets..." -ForegroundColor White
if ((Get-Command npm -ErrorAction SilentlyContinue) -and (Test-Path "testbench/lunar-testbench")) {
    Write-Host "  → Installing & compiling CSS in testbench/lunar-testbench..." -ForegroundColor Cyan
    Push-Location "testbench/lunar-testbench"
    try {
        npm ci --silent
        npm run build:css --silent
        Write-Host "  ✓ UI assets ready." -ForegroundColor Green
    } catch {
        Write-Host "  ! CSS build skipped or encountered warnings." -ForegroundColor Yellow
    } finally {
        Pop-Location
    }
} else {
    Write-Host "  ! Skipping CSS build (npm or testbench folder missing)." -ForegroundColor Yellow
}

# 6. Build Core Binaries
Write-Host "`n[6/6] Compiling core workspace binaries in release mode..." -ForegroundColor White
cargo build --release -p lunar-backend -p lunar-start -p lunar-start-backend -p lunar-ai-cli

Write-Host "`n✦ ========================================================== ✦" -ForegroundColor Green
Write-Host "   LUNAR-HAL is ready to launch!                             " -ForegroundColor Green
Write-Host "✦ ========================================================== ✦" -ForegroundColor Green
Write-Host "`nQuick start options:"
Write-Host "  1. Launch WebOS managed suite (Testbench UI + Service Manager):"
Write-Host "     cargo run --release -p lunar-start" -ForegroundColor Cyan
Write-Host "     -> Open in browser: http://127.0.0.1:16180" -ForegroundColor Yellow
Write-Host "`n  2. Run backend manually:"
Write-Host "     cargo run --release -p lunar-backend" -ForegroundColor Cyan
Write-Host "`n  3. Run WebOS Testbench directly:"
Write-Host "     cargo run --release -p lunar-testbench --bin lunar-testbench" -ForegroundColor Cyan
Write-Host "`n  4. Launch Dioxus Frontend directly (default port 8080):"
Write-Host "     cd crates/lunar-frontend; dx serve --platform web`n" -ForegroundColor Cyan
