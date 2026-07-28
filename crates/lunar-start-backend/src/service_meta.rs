//! Static service manifest consumed by the `lunar-testbench` WebOS shell.
//!
//! This is the "source of truth" the frontend uses to know which dock apps a
//! given service unlocks (`provides`) and which other services it needs
//! running first (`depends_on`). Kept static/hand-maintained for now; if
//! services become dynamically discoverable this can move into `LauncherConfig`.

use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct ServiceMeta {
    pub name: &'static str,
    pub title: &'static str,
    pub icon: &'static str,
    pub description: &'static str,
    pub kind: &'static str,
    pub url: Option<&'static str>,
    pub provides: &'static [&'static str],
    pub depends_on: &'static [&'static str],
}

pub fn all() -> Vec<ServiceMeta> {
    vec![
        ServiceMeta {
            name: "backend",
            title: "Backend",
            icon: "\u{1f9e0}",
            description: "Core inference backend (PINN / GNN / SIREN, pipeline, description).",
            kind: "binary",
            url: Some("http://127.0.0.1:25255"),
            provides: &["models", "pipeline", "siren_gallery", "backend_api"],
            depends_on: &[],
        },
        ServiceMeta {
            name: "testbench-backend",
            title: "Testbench Backend",
            icon: "\u{1f9ea}",
            description: "Training/validation job orchestration and system snapshot API.",
            kind: "binary",
            url: Some("http://127.0.0.1:25256"),
            provides: &["dashboard", "training", "validation", "datasets"],
            depends_on: &[],
        },
        ServiceMeta {
            name: "testbench",
            title: "Testbench (WebOS shell)",
            icon: "\u{1f5a5}\u{fe0f}",
            description: "The Dioxus web shell hosting all testbench apps (this app itself).",
            kind: "dx-serve",
            url: None,
            provides: &["sandbox"],
            depends_on: &[],
        },
        ServiceMeta {
            name: "frontend",
            title: "Frontend",
            icon: "\u{1f30c}",
            description: "Public-facing lunar frontend site (separate from the testbench).",
            kind: "dx-serve",
            url: None,
            provides: &[],
            depends_on: &[],
        },
    ]
}
