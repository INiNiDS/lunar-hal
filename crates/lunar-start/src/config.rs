use std::collections::HashMap;
use std::path::PathBuf;

use crate::service_settings::ServiceConfigValues;
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
    /// Environment variables scoped specifically to this service process.
    pub env: HashMap<String, String>,
    /// Additional command-line arguments for the service binary at runtime.
    pub extra_args: Vec<String>,
    /// Arguments passed to `cargo build` (e.g., `--features cuda`).
    pub build_args: Vec<String>,
}

impl ServiceConfig {
    pub fn new(name: impl Into<String>, kind: ServiceKind) -> Self {
        Self {
            name: name.into(),
            kind,
            env: HashMap::new(),
            extra_args: Vec::new(),
            build_args: Vec::new(),
        }
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn with_env_map(mut self, env: HashMap<String, String>) -> Self {
        self.env.extend(env);
        self
    }

    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }

    pub fn with_build_args(mut self, args: Vec<String>) -> Self {
        self.build_args = args;
        self
    }

    /// Replaces this service's runtime/build values with a complete structured snapshot.
    pub fn apply_values(&mut self, values: ServiceConfigValues) {
        self.env = values.env;
        self.extra_args = values.extra_args;
        self.build_args = values.build_args;
    }

    /// Resolves final effective environment variables by merging global launcher environment
    /// with service-specific environment variables (service overrides global).
    pub fn effective_env(&self, launcher_env: &HashMap<String, String>) -> HashMap<String, String> {
        let mut merged = launcher_env.clone();
        merged.extend(self.env.clone());
        merged
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
    /// Environment variables shared globally across all services (e.g., `RUST_LOG`).
    pub env: HashMap<String, String>,
    /// Run in headless mode (without GUI/interactive UI).
    pub headless: bool,
    /// Maximum log capacity retained in memory.
    pub max_logs: usize,
}

impl LauncherConfig {
    pub fn apply_env(&self) {
        for (k, v) in &self.env {
            // SAFETY: Executed sequentially during startup initialization.
            unsafe { std::env::set_var(k, v) };
        }
    }
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
    use crate::service_settings::{BackendSettings, FrontendSettings, TestbenchBackendSettings};

    impl ServiceConfig {
        pub fn backend() -> Self {
            let mut cfg = Self::new(
                "backend",
                ServiceKind::Binary {
                    bin_name: "lunar-backend".to_string(),
                },
            );
            cfg.apply_values(BackendSettings::default().to_values());
            cfg
        }

        pub fn testbench_backend() -> Self {
            let mut cfg = Self::new(
                "testbench-backend",
                ServiceKind::Binary {
                    bin_name: "lunar-testbench-backend".to_string(),
                },
            );
            cfg.apply_values(TestbenchBackendSettings::default().to_values());
            cfg
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
            let defaults = FrontendSettings::default();
            let mut cfg = Self::new(
                "frontend",
                ServiceKind::DxServe {
                    crate_name: "lunar-frontend".to_string(),
                    crate_subdir: "crates".to_string(),
                    default_port: defaults.port.to_string(),
                },
            );
            cfg.apply_values(defaults.to_values());
            cfg
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

        /// Full WebOS preset: services the already-running `lunar-testbench`
        /// WebOS shell can control. The shell itself is intentionally excluded
        /// so it cannot start, stop, or restart its own hosting process.
        pub fn for_webos() -> Self {
            Self::new(workspace_root())
                .with_service(ServiceConfig::backend())
                .with_service(ServiceConfig::testbench_backend())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applying_values_replaces_argument_snapshots() {
        let mut config = ServiceConfig::new(
            "test",
            ServiceKind::Binary {
                bin_name: "test".to_string(),
            },
        )
        .with_extra_args(vec!["old-runtime".to_string()])
        .with_build_args(vec!["old-build".to_string()]);

        config.apply_values(ServiceConfigValues {
            env: HashMap::from([("SERVICE_ONLY".to_string(), "yes".to_string())]),
            extra_args: vec!["new-runtime".to_string()],
            build_args: vec!["new-build".to_string()],
        });

        assert_eq!(config.extra_args, ["new-runtime"]);
        assert_eq!(config.build_args, ["new-build"]);
        assert_eq!(config.env["SERVICE_ONLY"], "yes");
    }

    #[test]
    fn service_environment_overrides_global_environment() {
        let service = ServiceConfig::new(
            "test",
            ServiceKind::Binary {
                bin_name: "test".to_string(),
            },
        )
        .with_env("SHARED", "service");
        let global = HashMap::from([
            ("SHARED".to_string(), "global".to_string()),
            ("GLOBAL_ONLY".to_string(), "yes".to_string()),
        ]);

        let effective = service.effective_env(&global);
        assert_eq!(effective["SHARED"], "service");
        assert_eq!(effective["GLOBAL_ONLY"], "yes");
    }

    #[test]
    fn presets_use_real_service_defaults() {
        let backend = ServiceConfig::backend();
        assert_eq!(backend.env["LUNAR_BACKEND_PORT"], "25255");
        assert!(!backend.env.contains_key("DATABASE_URL"));

        let testbench = ServiceConfig::testbench_backend();
        assert_eq!(testbench.env["LUNAR_TESTBENCH_BACKEND_PORT"], "25256");

        let frontend = ServiceConfig::frontend();
        assert_eq!(frontend.env["LUNAR_DX_BIN"], "dx");
        assert_eq!(&frontend.extra_args[..2], ["--port", "8080"]);
    }
}
