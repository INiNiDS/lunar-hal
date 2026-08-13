# Testing matrix

Lunar-HAL is checked by independent platform and responsibility layers. Do not replace this matrix with a single `cargo test --all-features`: web, desktop, and Android use incompatible Dioxus feature sets.

| Layer | Owner | Command | Prerequisites | Merge gate |
| --- | --- | --- | --- | --- |
| Formatting | workspace | `cargo fmt --all -- --check` | Rustfmt | yes |
| Core/domain | `lunar-structures`, `lunar-stellar-core` | `cargo test -p lunar-stellar-core` | Rust | yes |
| Launcher/config | `lunar-start`, `lunar-start-backend` | `cargo test -p lunar-start -p lunar-start-backend` | Rust | yes |
| WebOS state/API | `lunar-testbench` | `cargo test -p lunar-testbench --bin lunar-testbench` | generated `assets/tailwind.css` | yes |
| Web frontend type-check | `lunar-frontend` | `cargo check -p lunar-frontend --target wasm32-unknown-unknown --no-default-features --features web` | wasm target | yes |
| Desktop frontend type-check | `lunar-frontend` | `cargo check -p lunar-frontend --no-default-features --features desktop` | desktop WebView development libraries | yes |
| Testbench CSS | `lunar-testbench` | `cd testbench/lunar-testbench && npm ci && npm run build:css` | Node.js 20+ | yes |
| Browser E2E | WebOS | `npm run test:e2e` | running isolated WebOS stack and Playwright Chromium | release gate |
| Visual/responsive | WebOS | `npm run test:visual` | E2E stack, approved snapshots | release gate |
| Android smoke | `lunar-frontend` | `dx build --platform android` | SDK, NDK, adb, emulator/device | release gate |

## Local baseline

```bash
cargo fmt --all -- --check
cargo check -p lunar-utils -p lunar-structures -p lunar-stellar-core -p lunar-backend -p lunar-start -p lunar-start-backend
cargo test -p lunar-stellar-core -p lunar-start -p lunar-start-backend
cd testbench/lunar-testbench
npm ci
npm run build:css
cargo test -p lunar-testbench --bin lunar-testbench
```

The generated `assets/tailwind.css` is an output of `npm run build:css`; change `input.css`, `assets/main.css`, `tailwind.config.js`, or RSX classes instead of hand-editing it.

## Browser stack and artifacts

Browser tests require `LUNAR_E2E_BASE_URL`, the URL of a testbench instance whose managed services use isolated scene and gallery directories. `scripts/e2e-webos.sh` refuses to run against an unspecified URL. On failure, retain the backend, start-backend, frontend, and testbench logs plus Playwright screenshots/traces.

Every responsive browser scenario resizes a **WebOS window client area**, not merely the browser viewport. Required sizes are: `320x480`, `480x640`, `720x520`, `860x600`, `1180x760`, and maximized desktop sizes. Each case checks that controls stay inside the app window, `scrollWidth <= clientWidth` for the page root, and Fill-mode iframe/canvas content has no second page scrollbar.
