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

// ── LogLevel ─────────────────────────────────────────────────────────────────

/// Coarse severity classification for a [`LogEvent`], derived from its text
/// content (and, as a fallback, whether it arrived on stderr).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// Classifies a single cleaned log line into a [`LogLevel`].
///
/// Looks for common `ERROR`/`WARN`/panic markers (case-insensitive). Lines
/// from stderr that don't otherwise match are treated as `Warn` rather than
/// `Info`, since stderr output is usually noteworthy.
fn classify_level(text: &str, is_stderr: bool) -> LogLevel {
    let lower = text.to_ascii_lowercase();
    if lower.contains("error") || lower.contains("panicked at") || lower.contains("fatal") {
        LogLevel::Error
    } else if lower.contains("warn") {
        LogLevel::Warn
    } else if is_stderr {
        LogLevel::Warn
    } else {
        LogLevel::Info
    }
}

// ── LogEvent ─────────────────────────────────────────────────────────────────

/// A single log entry produced by a service and emitted by the backend.
#[derive(Clone, Debug, Serialize)]
pub struct LogEvent {
    /// Formatted elapsed time since backend launch (MM:SS).
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
///
/// Serializes as a tagged object, e.g. `{ "kind": "stopped", "reason": "Pending" }`,
/// rather than relying on `format!("{:?}", status)` (which is fragile for consumers).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ServiceStatus {
    Starting,
    Running,
    Stopped { reason: String },
    Failed { reason: String },
}

// ── ServiceRuntime ───────────────────────────────────────────────────────────

/// Active runtime state of a service, tracking configuration, execution status, and process ID.
#[derive(Clone, Debug)]
pub struct ServiceRuntime {
    pub config: ServiceConfig,
    pub status: ServiceStatus,
    pub pid: Option<u32>,
}

// ── LogBackend ───────────────────────────────────────────────────────────────

/// Log processing and execution management engine.
///
/// Constructed using a [`LauncherConfig`] and a channel for emitting [`LogEvent`] instances.
/// The backend manages process lifecycle, stream interception, status polling,
/// and log translation.
pub struct LogBackend {
    workspace: PathBuf,
    watch: bool,
    build_release: bool,
    global_args: Vec<String>,
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
                    reason: format!("Failed to restart: {e}"),
                };
            }
        }
    }

    /// Stops a active service by name.
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
                    reason: format!("Failed to spawn: {e}"),
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
                            reason: format!("Exited with code: {status}"),
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
            eprintln!("[lns] [{name}] {status:?}");
            self.children.remove(&name);
            if let Some(s) = self.services.iter_mut().find(|s| s.config.name == name) {
                s.status = status;
                s.pid = None;
            }
        }
    }

    // ── Spawn logic ──────────────────────────────────────────────────────────

    async fn spawn_service_by_index(&mut self, idx: usize) -> Result<()> {
        let name = self.services[idx].config.name.clone();
        let kind = self.services[idx].config.kind.clone();
        let extra_args = self.services[idx].config.extra_args.clone();
        let build_args = self.services[idx].config.build_args.clone();

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
                    )?
                } else {
                    spawn_binary(&self.workspace, bin_name, &combined_args)?
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
                    )?
                } else {
                    spawn_binary(&self.workspace, bin_name, &combined_args)?
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

// ── Spawn functions ──────────────────────────────────────────────────────────

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

/// Spawns a precompiled release binary located at `target/release/<name>`.
fn spawn_binary(ws: &Path, name: &str, extra_args: &[String]) -> Result<Child> {
    let bin = ws.join("target").join("release").join(name);
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(extra_args)
        .current_dir(ws)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    cmd.spawn()
        .with_context(|| format!("failed to spawn {name}"))
}

/// Spawns `cargo watch -- cargo run --bin <name>`.
fn spawn_cargo_watch(
    ws: &Path,
    bin_name: &str,
    use_release: bool,
    cargo_run_args: &[String],
    build_args: &[String],
    extra_args: &[String],
) -> Result<Child> {
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
        .current_dir(ws)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    cmd.spawn().context("failed to spawn `cargo watch`")
}

/// Spawns `dx serve` inside the targeted crate directory.
fn spawn_dx_serve(
    ws: &Path,
    crate_name: &str,
    crate_subdir: &str,
    default_port: &str,
    extra_args: &[String],
) -> Result<Child> {
    let user_port = extra_args.iter().any(|a| a == "--port" || a == "-p");
    let dx_bin = std::env::var("LUNAR_DX_BIN").unwrap_or_else(|_| "dx".to_string());

    let mut cmd = tokio::process::Command::new(&dx_bin);
    cmd.arg("serve")
        .current_dir(ws.join(crate_subdir).join(crate_name))
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if !user_port {
        cmd.args(["--port", default_port]);
    }
    cmd.args(extra_args);

    cmd.spawn().with_context(|| {
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
                    let elapsed = start_time.elapsed().as_secs();
                    let timestamp = format!("{:02}:{:02}", elapsed / 60, elapsed % 60);
                    let level = classify_level(&text, is_stderr);
                    let _ = tx
                        .send(LogEvent {
                            timestamp,
                            service: service_name.clone(),
                            text,
                            is_stderr,
                            level,
                        })
                        .await;
                }
                Err(_) => break,
            }
        }
    });
}

// ── Build ────────────────────────────────────────────────────────────────────

/// Synchronously triggers a workspace-wide `cargo build --release` command.
pub fn cargo_build(ws: &Path, build_args: &[String]) -> Result<()> {
    let mut cmd = std::process::Command::new("cargo");
    cmd.args(["build", "--release"]);
    cmd.args(build_args);
    cmd.current_dir(ws)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd.status().context("failed to run cargo build")?;
    if !status.success() {
        anyhow::bail!("cargo build failed with status: {status}");
    }
    Ok(())
}
