use std::collections::HashMap;
use std::path::PathBuf;

// ── ServiceKind ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum ServiceKind {
    /// Launch a precompiled binary from `target/release/<bin_name>`.
    Binary {
        bin_name: String,
    },
    /// Launch via `dx serve` (Dioxus dev-server).
    DxServe {
        crate_name: String,
        crate_subdir: String,
        default_port: String,
    },
    /// Launch via `cargo run` (or `cargo watch` if watch mode is enabled).
    CargoRun {
        bin_name: String,
        cargo_args: Vec<String>,
    },
}

// ── ServiceConfig ────────────────────────────────────────────────────────────

/// Configuration for an individual service.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Display name of the service (e.g., backend, frontend, testbench).
    pub name: String,
    /// Execution method/kind for the service.
    pub kind: ServiceKind,
    /// Additional command-line arguments for the service binary.
    pub extra_args: Vec<String>,
    /// Arguments passed to `cargo build` (e.g., `--features cuda`).
    pub build_args: Vec<String>,
}

impl ServiceConfig {
    pub fn new(name: impl Into<String>, kind: ServiceKind) -> Self {
        Self {
            name: name.into(),
            kind,
            extra_args: Vec::new(),
            build_args: Vec::new(),
        }
    }

    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }

    pub fn with_build_args(mut self, args: Vec<String>) -> Self {
        self.build_args = args;
        self
    }
}

// ── LauncherConfig ────────────────────────────────────────────────────────────────

/// Global launcher configuration managing workspace services and build options.
#[derive(Clone, Debug)]
pub struct LauncherConfig {
    /// Workspace root directory.
    pub workspace: PathBuf,
    /// List of configured services to manage.
    pub services: Vec<ServiceConfig>,
    /// Enable watch mode for automatic hot-reloading.
    pub watch: bool,
    /// Build release binaries before starting services.
    pub build_release: bool,
    /// Global arguments passed to all managed services.
    pub global_args: Vec<String>,
    /// Environment variables set prior to service startup.
    pub env: HashMap<String, String>,
    /// Run in headless mode (without GUI/interactive UI).
    pub headless: bool,
    /// Maximum log capacity retained in memory.
    pub max_logs: usize,
}

impl LauncherConfig {
    pub fn new(workspace: PathBuf) -> Self {
        Self {
            workspace,
            services: Vec::new(),
            watch: false,
            build_release: false,
            global_args: Vec::new(),
            env: HashMap::new(),
            headless: false,
            max_logs: 3000,
        }
    }

    pub fn with_service(mut self, service: ServiceConfig) -> Self {
        self.services.push(service);
        self
    }

    pub fn with_services(mut self, services: Vec<ServiceConfig>) -> Self {
        self.services = services;
        self
    }

    pub fn watch(mut self, watch: bool) -> Self {
        self.watch = watch;
        self
    }

    pub fn build_release(mut self, build: bool) -> Self {
        self.build_release = build;
        self
    }

    pub fn global_args(mut self, args: Vec<String>) -> Self {
        self.global_args = args;
        self
    }

    pub fn set_env(&mut self, key: &str, value: &str) {
        self.env.insert(key.to_string(), value.to_string());
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn headless(mut self, headless: bool) -> Self {
        self.headless = headless;
        self
    }

    pub fn max_logs(mut self, max: usize) -> Self {
        self.max_logs = max;
        self
    }

    pub fn apply_env(&self) {
        for (k, v) in &self.env {
            // SAFETY: `set_var` is unsafe in multi-threaded contexts in newer Rust editions.
            // Ensure this is invoked sequentially during single-threaded startup initialization.
            unsafe { std::env::set_var(k, v) };
        }
    }

    pub fn all_build_args(&self) -> Vec<String> {
        self.services
            .iter()
            .flat_map(|s| s.build_args.iter().cloned())
            .collect()
    }
}

/// Resolves the absolute path to the workspace root directory.
pub fn workspace_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent();
        while let Some(dir) = cur {
            if dir.join("Cargo.toml").exists() && dir.join("crates").exists() {
                return dir.to_path_buf();
            }
            cur = dir.parent();
        }
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("Cargo.toml").exists() && cwd.join("crates").exists() {
        return cwd;
    }
    cwd
}

// ── Presets ─────────────────────────────────────────────────────────────────────────

/// Presets for commonly used configurations (backend, frontend, testbench, etc.).
pub mod presets {
    use super::*;

    impl ServiceConfig {
        pub fn backend() -> Self {
            Self::new(
                "backend",
                ServiceKind::Binary {
                    bin_name: "lunar-backend".to_string(),
                },
            )
        }

        pub fn testbench_backend() -> Self {
            Self::new(
                "testbench-backend",
                ServiceKind::Binary {
                    bin_name: "lunar-testbench-backend".to_string(),
                },
            )
        }

        pub fn start_backend() -> Self {
            Self::new(
                "start-backend",
                ServiceKind::Binary {
                    bin_name: "lunar-start-backend".to_string(),
                },
            )
        }

        pub fn testbench() -> Self {
            Self::new(
                "testbench",
                ServiceKind::DxServe {
                    crate_name: "lunar-testbench".to_string(),
                    crate_subdir: "testbench".to_string(),
                    default_port: "16180".to_string(),
                },
            )
        }

        pub fn frontend() -> Self {
            Self::new(
                "frontend",
                ServiceKind::DxServe {
                    crate_name: "lunar-frontend".to_string(),
                    crate_subdir: "crates".to_string(),
                    default_port: "8080".to_string(),
                },
            )
        }

        pub fn frontend_desktop() -> Self {
            Self::new(
                "frontend-desktop",
                ServiceKind::CargoRun {
                    bin_name: "lunar-frontend".to_string(),
                    cargo_args: vec![
                        "--no-default-features".to_string(),
                        "--features".to_string(),
                        "desktop".to_string(),
                    ],
                },
            )
        }
    }

    impl LauncherConfig {
        pub fn for_testbench() -> Self {
            Self::new(workspace_root()).with_service(ServiceConfig::testbench())
        }

        pub fn for_testbench_full() -> Self {
            Self::new(workspace_root())
                .with_service(ServiceConfig::testbench_backend())
                .with_service(ServiceConfig::testbench())
        }

        /// Full WebOS preset: every backend service the `lunar-testbench` WebOS shell
        /// can drive apps against, plus the shell itself and the public frontend.
        ///
        /// NOTE: `lunar-game-backend` (`crates/lunar-game-backend`) is a library-only
        /// crate with no standalone binary target today — it's consumed in-process by
        /// `lunar-frontend`, not launched as its own service. If/when it grows a bin
        /// target, add a `ServiceConfig::game_backend()` preset and include it here.
        pub fn for_webos() -> Self {
            Self::new(workspace_root())
                .with_service(ServiceConfig::backend())
                .with_service(ServiceConfig::testbench_backend())
                .with_service(ServiceConfig::testbench())
                .with_service(ServiceConfig::frontend())
        }

        pub fn for_start_full() -> Self {
            Self::new(workspace_root())
                .with_service(ServiceConfig::start_backend())
                .with_service(ServiceConfig::testbench())
        }

        pub fn for_all() -> Self {
            Self::new(workspace_root())
                .with_service(ServiceConfig::backend())
                .with_service(ServiceConfig::testbench_backend())
                .with_service(ServiceConfig::testbench())
                .with_service(ServiceConfig::frontend())
                .build_release(true)
        }

        pub fn for_backend() -> Self {
            Self::new(workspace_root()).with_service(ServiceConfig::backend())
        }

        pub fn for_frontend() -> Self {
            Self::new(workspace_root()).with_service(ServiceConfig::frontend())
        }

        pub fn for_frontend_desktop() -> Self {
            Self::new(workspace_root())
                .with_service(ServiceConfig::frontend_desktop())
                .build_release(true)
        }
    }
}
