<div align="center">

# ✦ LUNAR-HAL

### Technical platform for generating, exploring, visualizing, and cataloguing stars

[![Rust](https://img.shields.io/badge/Language-Rust-ea4a31?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Dioxus](https://img.shields.io/badge/Frontend-Dioxus-000000?style=for-the-badge&logo=rust&logoColor=5fcbf2)](https://dioxuslabs.com)
[![Local AI](https://img.shields.io/badge/AI-Local-8A2BE2?style=for-the-badge&logo=openai&logoColor=white)](#architecture)

</div>

## Overview

Lunar-HAL is a local-first stellar laboratory. It combines PINN, GNN, and SIREN pipelines with an interactive star map, persistent star scenes, camera controls, sector streaming, and technical inspection tools. The project is focused exclusively on scientific visualization and technical workflows.

## Core capabilities

| Capability | Description |
| :--- | :--- |
| **Star scenes** | Create, load, inspect, and delete persistent collections of generated stars. |
| **Local AI pipeline** | Generate stellar parameters, metadata, sector distributions, and SIREN textures locally. |
| **Interactive map** | Navigate with pan and zoom, select stars, stream sectors, and inspect pipeline results. |
| **Multi-platform frontend** | Run the same frontend as a web, desktop, or Android target. |

## Architecture

- **`lunar-stellar-core`** — framework-agnostic stellar-scene client: camera, selection, sector cache, validation, and backend API client.
- **`lunar-backend`** — AI inference and authoritative saved star scenes.
- **`lunar-frontend`** — Dioxus renderer for the interactive star map.
- **`lunar-start` / `lunar-start-backend`** — managed service configuration and platform-aware frontend launch.

## Quick start

Install Rust and the [Dioxus prerequisites](https://dioxuslabs.com/learn/0.7/getting_started/), then run the backend and web frontend in separate terminals:

```bash
cargo run -p lunar-backend
dx serve --platform web
```

The managed frontend service supports `web`, `desktop`, and `android` platforms through its configuration schema. Web uses a public URL; desktop and Android run natively.

## WebOS and documentation

`lunar-testbench` is a floating-window WebOS for the technical tools. It starts applications from the desktop, keeps only opened windows in the Dock, and embeds the real web frontend in Sandbox. Window bodies use per-app `Scroll`/`Fill` modes and container queries, so page layout responds to the window client area rather than browser width.

- [Architecture](docs/architecture.md)
- [HTTP API](docs/api.md)
- [Frontend platforms](docs/frontend-platforms.md)
- [Gallery storage](docs/gallery-storage.md)
- [WebOS lifecycle](docs/webos-lifecycle.md)
- [Testing matrix](docs/testing.md)

## Development checks

```bash
cargo fmt --all -- --check
cargo check -p lunar-utils -p lunar-structures -p lunar-stellar-core -p lunar-backend -p lunar-start -p lunar-start-backend
cargo test -p lunar-stellar-core -p lunar-start -p lunar-start-backend
cd testbench/lunar-testbench
npm ci
npm run build:css
cargo test -p lunar-testbench --bin lunar-testbench
```
