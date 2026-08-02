//! LogBackend — A backend process manager that spawns services and streams their output logs.
//!
//! The backend handles:
//! - Process spawning (executables, `dx serve`, `cargo watch`)
//! - Reading stdout/stderr streams for each process
//! - Forwarding log lines as structured [`LogEvent`] instances
//! - Process health status tracking (starting / running / stopped / failed)
//! - Process lifecycle management (start / stop / restart)

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use tokio::sync::mpsc;

use crate::ansi::clean_line;
use crate::config::{LauncherConfig, ServiceConfig, ServiceKind};

/// Coarse severity classification for a [`LogEvent`], derived from its text content.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// Case-insensitive substring check without heap allocation.
fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    haystack.as_bytes().windows(needle.len()).any(|window| {
        window
            .iter()
            .zip(needle.bytes())
            .all(|(&a, b)| a.eq_ignore_ascii_case(&b))
    })
}

/// Classifies a single cleaned log line into a [`LogLevel`].
fn classify_level(text: &str, is_stderr: bool) -> LogLevel {
    if contains_ignore_case(text, "error")
        || contains_ignore_case(text, "panicked at")
        || contains_ignore_case(text, "fatal")
    {
        LogLevel::Error
    } else if contains_ignore_case(text, "warn") || is_stderr {
        LogLevel::Warn
    } else {
        LogLevel::Info
    }
}

// ── LogEvent ─────────────────────────────────────────────────────────────────

/// A single log entry produced by a service and emitted by the backend.
#[derive(Clone, Debug, Serialize)]
pub struct LogEvent {
    /// Formatted elapsed time since backend launch (MM:SS or HH:MM:SS).
    pub timestamp: String,
    /// Identifier name of the originating service.
    pub service: String,
    /// Stripped, plain-text log content (ANSI escape sequences removed).
    pub text: String,
    /// Indicates whether the entry was received via stderr.
    pub is_stderr: bool,
    /// Coarse severity classification (`info` / `warn` / `error`).
    pub level: LogLevel,
}

// ── ServiceStatus ────────────────────────────────────────────────────────────

/// Represents the runtime execution state of a service.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ServiceStatus {
    Starting,
    Running,
    Stopped { reason: String },
    Failed { reason: String },
}

/// Active runtime state of a service, tracking configuration, execution status, and process ID.
#[derive(Clone, Debug)]
pub struct ServiceRuntime {
    pub config: ServiceConfig,
    pub status: ServiceStatus,
    pub pid: Option<u32>,
}


/// Log processing and execution management engine.
pub struct LogBackend {
    workspace: PathBuf,
    watch: bool,
    build_release: bool,
    global_args: Vec<String>,
    global_env: HashMap<String, String>,
    services: Vec<ServiceRuntime>,
    children: HashMap<String, Child>,
    log_tx: mpsc::Sender<LogEvent>,
    start_time: Instant,
}

impl LogBackend {
    pub fn new(config: &LauncherConfig, log_tx: mpsc::Sender<LogEvent>) -> Self {
        Self {
            workspace: config.workspace.clone(),
            watch: config.watch,
            build_release: config.build_release,
            global_args: config.global_args.clone(),
            global_env: config.env.clone(),
            services: config
                .services
                .iter()
                .map(|sc| ServiceRuntime {
                    config: sc.clone(),
                    status: ServiceStatus::Stopped {
                        reason: "Pending".to_string(),
                    },
                    pid: None,
                })
                .collect(),
            children: HashMap::new(),
            log_tx,
            start_time: Instant::now(),
        }
    }

    pub fn start_time(&self) -> Instant {
        self.start_time
    }

    pub fn services(&self) -> &[ServiceRuntime] {
        &self.services
    }

    pub fn service(&self, name: &str) -> Option<&ServiceRuntime> {
        self.services.iter().find(|service| service.config.name == name)
    }

    /// Replaces the configuration of a service that has no active process.
    pub fn replace_service_config(&mut self, config: ServiceConfig) -> Result<()> {
        if self.is_running(&config.name) {
            anyhow::bail!("service '{}' is running", config.name);
        }
        let runtime = self
            .services
            .iter_mut()
            .find(|service| service.config.name == config.name)
            .with_context(|| format!("unknown service '{}'", config.name))?;
        if matches!(runtime.status, ServiceStatus::Starting | ServiceStatus::Running) {
            anyhow::bail!("service '{}' is not stopped", config.name);
        }
        runtime.config = config;
        Ok(())
    }

    pub fn is_running(&self, name: &str) -> bool {
        self.children.contains_key(name)
    }

    pub fn any_running(&self) -> bool {
        !self.children.is_empty()
    }

    pub fn needs_build(&self) -> bool {
        self.build_release && !self.watch
    }

    /// Spawns and starts all configured services.
    pub async fn start_all(&mut self) {
        eprintln!(
            "[lns] Starting {} service(s) from {}",
            self.services.len(),
            self.workspace.display()
        );

        for i in 0..self.services.len() {
            let name = self.services[i].config.name.clone();
            let kind = self.services[i].config.kind.clone();
            eprintln!("[lns] [{name}] starting {}", service_command(&kind));

            self.services[i].status = ServiceStatus::Starting;
            if let Err(e) = self.spawn_service_by_index(i).await {
                let reason = format!("Failed to spawn: {e:#}");
                eprintln!("[lns] [{name}] ERROR: {reason}");
                self.services[i].status = ServiceStatus::Failed { reason };
            }
        }
    }

    /// Terminates all currently active child services.
    pub async fn stop_all(&mut self) {
        let names: Vec<String> = self.children.keys().cloned().collect();
        for name in names {
            self.stop(&name).await;
        }
    }

    /// Restarts a specific service by its identifier name.
    pub async fn restart(&mut self, name: &str) {
        self.stop(name).await;
        if let Some(idx) = self.services.iter().position(|s| s.config.name == name) {
            self.services[idx].status = ServiceStatus::Starting;
            self.services[idx].pid = None;
            if let Err(e) = self.spawn_service_by_index(idx).await {
                self.services[idx].status = ServiceStatus::Failed {
                    reason: format!("Failed to restart: {e:#}"),
                };
            }
        }
    }

    /// Stops an active service by name.
    pub async fn stop(&mut self, name: &str) {
        if let Some(mut child) = self.children.remove(name) {
            eprintln!("[lns] [{name}] stopping");
            let _ = child.kill().await;
            let _ = child.wait().await;
            eprintln!("[lns] [{name}] stopped");
        }
        if let Some(s) = self.services.iter_mut().find(|s| s.config.name == name) {
            s.status = ServiceStatus::Stopped {
                reason: "Stopped".to_string(),
            };
            s.pid = None;
        }
    }

    /// Starts a previously stopped service by name.
    pub async fn start(&mut self, name: &str) {
        if let Some(idx) = self.services.iter().position(|s| s.config.name == name) {
            self.services[idx].status = ServiceStatus::Starting;
            if let Err(e) = self.spawn_service_by_index(idx).await {
                self.services[idx].status = ServiceStatus::Failed {
                    reason: format!("Failed to spawn: {e:#}"),
                };
            }
        }
    }

    /// Polls running child processes, updating statuses for terminated instances.
    pub fn poll(&mut self) {
        let mut exited = Vec::new();
        for (name, child) in &mut self.children {
            match child.try_wait() {
                Ok(Some(status)) => {
                    exited.push((
                        name.clone(),
                        ServiceStatus::Stopped {
                            reason: format!("Exited: {status}"),
                        },
                    ));
                }
                Ok(None) => {}
                Err(e) => {
                    exited.push((
                        name.clone(),
                        ServiceStatus::Failed {
                            reason: format!("Error: {e}"),
                        },
                    ));
                }
            }
        }
        for (name, status) in exited {
            eprintln!("[lns] [{name}] status update: {status:?}");
            self.children.remove(&name);
            if let Some(s) = self.services.iter_mut().find(|s| s.config.name == name) {
                s.status = status;
                s.pid = None;
            }
        }
    }

    async fn spawn_service_by_index(&mut self, idx: usize) -> Result<()> {
        let name = self.services[idx].config.name.clone();

        // Clean up pre-existing child process if still registered
        if self.children.contains_key(&name) {
            self.stop(&name).await;
        }

        let kind = self.services[idx].config.kind.clone();
        let extra_args = self.services[idx].config.extra_args.clone();
        let build_args = self.services[idx].config.build_args.clone();
        let env = self.services[idx].config.effective_env(&self.global_env);

        let mut combined_args = extra_args;
        combined_args.extend(self.global_args.iter().cloned());

        let mut child = match &kind {
            ServiceKind::DxServe {
                crate_name,
                crate_subdir,
                default_port,
            } => spawn_dx_serve(
                &self.workspace,
                crate_name,
                crate_subdir,
                default_port,
                &combined_args,
                &env,
            )?,

            ServiceKind::Binary { bin_name } => {
                if self.watch {
                    spawn_cargo_watch(
                        &self.workspace,
                        bin_name,
                        true,
                        &[],
                        &build_args,
                        &combined_args,
                        &env,
                    )?
                } else {
                    spawn_binary(&self.workspace, bin_name, &combined_args, &env)?
                }
            }

            ServiceKind::CargoRun {
                bin_name,
                cargo_args,
            } => {
                if self.watch {
                    spawn_cargo_watch(
                        &self.workspace,
                        bin_name,
                        true,
                        cargo_args,
                        &build_args,
                        &combined_args,
                        &env,
                    )?
                } else {
                    spawn_cargo_run(
                        &self.workspace,
                        bin_name,
                        cargo_args,
                        &build_args,
                        &combined_args,
                        &env,
                    )?
                }
            }
        };

        let pid = child.id();
        self.services[idx].status = ServiceStatus::Running;
        self.services[idx].pid = pid;
        match pid {
            Some(pid) => eprintln!("[lns] [{name}] started (pid {pid})"),
            None => eprintln!("[lns] [{name}] started"),
        }

        // Spawn background tasks to stream stdout and stderr concurrently
        if let Some(stdout) = child.stdout.take() {
            read_output(
                BufReader::new(stdout),
                name.clone(),
                false,
                self.start_time,
                self.log_tx.clone(),
            );
        }
        if let Some(stderr) = child.stderr.take() {
            read_output(
                BufReader::new(stderr),
                name.clone(),
                true,
                self.start_time,
                self.log_tx.clone(),
            );
        }

        self.children.insert(name, child);
        Ok(())
    }
}

/// Human-readable command summary used in launcher diagnostics.
fn service_command(kind: &ServiceKind) -> String {
    match kind {
        ServiceKind::Binary { bin_name } => format!("target/release/{bin_name}"),
        ServiceKind::DxServe {
            crate_name,
            default_port,
            ..
        } => format!("dx serve ({crate_name}, port {default_port})"),
        ServiceKind::CargoRun {
            bin_name,
            cargo_args,
        } => {
            let args = cargo_args.join(" ");
            format!("cargo run --bin {bin_name} {args}").trim().to_string()
        }
    }
}

/// Builds command for executing a precompiled release binary located at `target/release/<name>`.
pub fn build_binary_cmd(
    ws: &Path,
    name: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> tokio::process::Command {
    let bin = ws.join("target").join("release").join(name);
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(extra_args)
        .envs(env)
        .current_dir(ws)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Spawns a precompiled release binary.
fn spawn_binary(
    ws: &Path,
    name: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<Child> {
    build_binary_cmd(ws, name, extra_args, env)
        .spawn()
        .with_context(|| format!("failed to spawn binary at target/release/{name}"))
}

/// Builds command for `cargo run --bin <name> <cargo_args> -- <extra_args>`.
pub fn build_cargo_run_cmd(
    ws: &Path,
    bin_name: &str,
    cargo_args: &[String],
    build_args: &[String],
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> tokio::process::Command {
    let mut args = vec!["run".to_string(), "--bin".to_string(), bin_name.to_string()];
    args.extend(build_args.iter().cloned());
    args.extend(cargo_args.iter().cloned());

    if !extra_args.is_empty() {
        args.push("--".to_string());
        args.extend(extra_args.iter().cloned());
    }

    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(&args)
        .envs(env)
        .current_dir(ws)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Spawns `cargo run`.
fn spawn_cargo_run(
    ws: &Path,
    bin_name: &str,
    cargo_args: &[String],
    build_args: &[String],
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<Child> {
    build_cargo_run_cmd(ws, bin_name, cargo_args, build_args, extra_args, env)
        .spawn()
        .with_context(|| format!("failed to spawn `cargo run --bin {bin_name}`"))
}

/// Builds command for `cargo watch -- cargo run --bin <name>`.
pub fn build_cargo_watch_cmd(
    ws: &Path,
    bin_name: &str,
    use_release: bool,
    cargo_run_args: &[String],
    build_args: &[String],
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> tokio::process::Command {
    let mut args = vec![
        "watch".to_string(),
        "--".to_string(),
        "cargo".to_string(),
        "run".to_string(),
    ];

    if use_release {
        args.push("--release".to_string());
    }
    args.extend(build_args.iter().cloned());
    args.extend(cargo_run_args.iter().cloned());
    args.push("--bin".to_string());
    args.push(bin_name.to_string());

    if !extra_args.is_empty() {
        args.push("--".to_string());
        args.extend(extra_args.iter().cloned());
    }

    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(&args)
        .envs(env)
        .current_dir(ws)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

/// Spawns `cargo watch`.
fn spawn_cargo_watch(
    ws: &Path,
    bin_name: &str,
    use_release: bool,
    cargo_run_args: &[String],
    build_args: &[String],
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<Child> {
    build_cargo_watch_cmd(
        ws,
        bin_name,
        use_release,
        cargo_run_args,
        build_args,
        extra_args,
        env,
    )
    .spawn()
    .context("failed to spawn `cargo watch`")
}

/// Builds command for `dx serve` inside targeted crate directory.
pub fn build_dx_serve_cmd(
    ws: &Path,
    crate_name: &str,
    crate_subdir: &str,
    default_port: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> tokio::process::Command {
    let user_port = extra_args.iter().any(|a| a == "--port" || a == "-p");
    let dx_bin = env
        .get("LUNAR_DX_BIN")
        .map(String::as_str)
        .unwrap_or("dx");

    let mut cmd = tokio::process::Command::new(dx_bin);
    cmd.arg("serve")
        .envs(env)
        .current_dir(ws.join(crate_subdir).join(crate_name))
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if !user_port {
        cmd.args(["--port", default_port]);
    }
    cmd.args(extra_args);
    cmd
}

/// Spawns `dx serve`.
fn spawn_dx_serve(
    ws: &Path,
    crate_name: &str,
    crate_subdir: &str,
    default_port: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<Child> {
    let dx_bin = env
        .get("LUNAR_DX_BIN")
        .map(String::as_str)
        .unwrap_or("dx")
        .to_string();
    build_dx_serve_cmd(ws, crate_name, crate_subdir, default_port, extra_args, env)
        .spawn()
        .with_context(|| {
            format!(
                "failed to run `{dx_bin} serve`. \
                 Check if dioxus-cli is installed or set LUNAR_DX_BIN."
            )
        })
}

/// Asynchronously streams line-buffered output from a reader and dispatches formatted `LogEvent` items over `mpsc`.
fn read_output<R>(
    mut reader: BufReader<R>,
    service_name: String,
    is_stderr: bool,
    start_time: Instant,
    tx: mpsc::Sender<LogEvent>,
) where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {
                    let text = clean_line(line.trim_end());
                    if text.is_empty() {
                        continue;
                    }

                    let total_secs = start_time.elapsed().as_secs();
                    let hours = total_secs / 3600;
                    let minutes = (total_secs % 3600) / 60;
                    let seconds = total_secs % 60;

                    let timestamp = if hours > 0 {
                        format!("{hours:02}:{minutes:02}:{seconds:02}")
                    } else {
                        format!("{minutes:02}:{seconds:02}")
                    };

                    let level = classify_level(&text, is_stderr);
                    let event = LogEvent {
                        timestamp,
                        service: service_name.clone(),
                        text,
                        is_stderr,
                        level,
                    };

                    if tx.send(event).await.is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
}

/// Asynchronously triggers a workspace-wide `cargo build --release` command.
pub async fn cargo_build(ws: &Path, build_args: &[String]) -> Result<()> {
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(["build", "--release"]);
    cmd.args(build_args);
    cmd.current_dir(ws)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd
        .status()
        .await
        .context("failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed with status: {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn extract_envs(cmd: &tokio::process::Command) -> HashMap<String, String> {
        cmd.as_std()
            .get_envs()
            .filter_map(|(k, v)| {
                let key = k.to_str()?.to_string();
                let val = v?.to_str()?.to_string();
                Some((key, val))
            })
            .collect()
    }

    fn extract_args(cmd: &tokio::process::Command) -> Vec<String> {
        cmd.as_std()
            .get_args()
            .map(|a| a.to_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn test_build_binary_cmd_applies_service_env_only() {
        let ws = Path::new("/workspace");
        let mut env = HashMap::new();
        env.insert("PORT".to_string(), "8080".to_string());
        env.insert("SERVICE_NAME".to_string(), "backend".to_string());

        let extra_args = vec!["--port".to_string(), "8080".to_string()];
        let cmd = build_binary_cmd(ws, "backend", &extra_args, &env);

        let env_map = extract_envs(&cmd);
        assert_eq!(env_map.get("PORT"), Some(&"8080".to_string()));
        assert_eq!(env_map.get("SERVICE_NAME"), Some(&"backend".to_string()));
        assert_eq!(env_map.get("FRONTEND_VAR"), None);

        let args = extract_args(&cmd);
        assert_eq!(args, vec!["--port", "8080"]);
    }

    #[test]
    fn test_build_dx_serve_cmd_applies_service_env_and_ports() {
        let ws = Path::new("/workspace");
        let mut env = HashMap::new();
        env.insert("LUNAR_API_URL".to_string(), "http://localhost:8080".to_string());
        env.insert("LUNAR_DX_BIN".to_string(), "/opt/dioxus/dx".to_string());

        let extra_args = vec!["--platform".to_string(), "web".to_string()];
        let cmd = build_dx_serve_cmd(ws, "app", "frontend", "3000", &extra_args, &env);

        assert_eq!(cmd.as_std().get_program(), "/opt/dioxus/dx");
        let env_map = extract_envs(&cmd);
        assert_eq!(
            env_map.get("LUNAR_API_URL"),
            Some(&"http://localhost:8080".to_string())
        );
        assert_eq!(env_map.get("PORT"), None);

        let args = extract_args(&cmd);
        assert_eq!(args, vec!["serve", "--port", "3000", "--platform", "web"]);
    }

    #[test]
    fn test_build_cargo_watch_cmd_applies_env_and_extra_args() {
        let ws = Path::new("/workspace");
        let mut env = HashMap::new();
        env.insert("RUST_LOG".to_string(), "debug".to_string());

        let cargo_run_args = vec!["--features".to_string(), "mock".to_string()];
        let build_args = vec!["--offline".to_string()];
        let extra_args = vec!["--verbose".to_string()];

        let cmd = build_cargo_watch_cmd(
            ws,
            "testbench",
            true,
            &cargo_run_args,
            &build_args,
            &extra_args,
            &env,
        );

        let env_map = extract_envs(&cmd);
        assert_eq!(env_map.get("RUST_LOG"), Some(&"debug".to_string()));

        let args = extract_args(&cmd);
        assert_eq!(
            args,
            vec![
                "watch", "--", "cargo", "run", "--release", "--offline", "--features", "mock",
                "--bin", "testbench", "--", "--verbose"
            ]
        );
    }

    #[test]
    fn test_build_cargo_run_cmd_applies_env() {
        let ws = Path::new("/workspace");
        let mut env = HashMap::new();
        env.insert("DATABASE_URL".to_string(), "postgres://localhost/db".to_string());

        let cargo_args = vec!["--features".to_string(), "postgres".to_string()];
        let build_args = vec![];
        let extra_args = vec!["--migrate".to_string()];

        let cmd = build_cargo_run_cmd(ws, "server", &cargo_args, &build_args, &extra_args, &env);

        let env_map = extract_envs(&cmd);
        assert_eq!(
            env_map.get("DATABASE_URL"),
            Some(&"postgres://localhost/db".to_string())
        );

        let args = extract_args(&cmd);
        assert_eq!(
            args,
            vec!["run", "--bin", "server", "--features", "postgres", "--", "--migrate"]
        );
    }
}
