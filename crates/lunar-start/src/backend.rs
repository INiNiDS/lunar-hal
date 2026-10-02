
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::Child;
use tokio::sync::mpsc;

use crate::ansi::clean_line;
use crate::config::{LauncherConfig, ServiceConfig, ServiceKind};
use crate::service_settings::FrontendLaunchConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    haystack.as_bytes().windows(needle.len()).any(|window| {
        window
            .iter()
            .zip(needle.bytes())
            .all(|(&a, b)| a.eq_ignore_ascii_case(&b))
    })
}

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


#[derive(Clone, Debug, Serialize)]
pub struct LogEvent {
    pub timestamp: String,
    pub service: String,
    pub text: String,
    pub is_stderr: bool,
    pub level: LogLevel,
}


#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ServiceStatus {
    Starting,
    Running,
    Stopped { reason: String },
    Failed { reason: String },
}

#[derive(Clone, Debug)]
pub struct ServiceRuntime {
    pub config: ServiceConfig,
    pub status: ServiceStatus,
    pub pid: Option<u32>,
}

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
        self.services
            .iter()
            .find(|service| service.config.name == name)
    }

    pub fn replace_service_config(&mut self, config: ServiceConfig) -> Result<()> {
        if self.is_running(&config.name) {
            anyhow::bail!("service '{}' is running", config.name);
        }
        let runtime = self
            .services
            .iter_mut()
            .find(|service| service.config.name == config.name)
            .with_context(|| format!("unknown service '{}'", config.name))?;
        if matches!(
            runtime.status,
            ServiceStatus::Starting | ServiceStatus::Running
        ) {
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

    pub async fn stop_all(&mut self) {
        let names: Vec<String> = self.children.keys().cloned().collect();
        for name in names {
            self.stop(&name).await;
        }
    }

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
            } => {
                if crate_name == "lunar-frontend" {
                    spawn_frontend_dx_serve(
                        &self.workspace,
                        crate_name,
                        crate_subdir,
                        &combined_args,
                        &env,
                    )?
                } else {
                    spawn_dx_serve(
                        &self.workspace,
                        crate_name,
                        crate_subdir,
                        default_port,
                        &combined_args,
                        &env,
                    )?
                }
            }

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
        self.services[idx].pid = pid;
        match pid {
            Some(pid) => eprintln!("[lns] [{name}] started (pid {pid})"),
            None => eprintln!("[lns] [{name}] started"),
        }

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
        if let Some((host, port, path)) = readiness_target(&kind, &env, &combined_args) {
            let name = self.services[idx].config.name.clone();
            let deadline = Instant::now() + Duration::from_secs(90);
            loop {
                if probe_http(&host, port, path).await
                    || (path != "/" && probe_http(&host, port, "/").await)
                    || (path == "/" && probe_http(&host, port, "/testbench").await)
                {
                    break;
                }
                if let Some(status) = self
                    .children
                    .get_mut(&name)
                    .expect("spawned child")
                    .try_wait()?
                {
                    self.children.remove(&name);
                    self.services[idx].pid = None;
                    anyhow::bail!("{name} exited before HTTP readiness: {status}");
                }
                if Instant::now() >= deadline {
                    self.stop(&name).await;
                    anyhow::bail!(
                        "{name} did not answer HTTP {path} on {host}:{port} within 90 seconds"
                    );
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        }
        self.services[idx].status = ServiceStatus::Running;
        Ok(())
    }
}

fn readiness_target(
    kind: &ServiceKind,
    env: &HashMap<String, String>,
    extra_args: &[String],
) -> Option<(String, u16, &'static str)> {
    let (host, port, path) = match kind {
        ServiceKind::Binary { bin_name } if bin_name == "lunar-backend" => (
            env.get("LUNAR_BACKEND_HOST")
                .map(String::as_str)
                .unwrap_or("127.0.0.1"),
            env.get("LUNAR_BACKEND_PORT")?.parse().ok()?,
            "/version",
        ),
        ServiceKind::Binary { bin_name } if bin_name == "lunar-testbench-backend" => (
            "127.0.0.1",
            env.get("LUNAR_TESTBENCH_BACKEND_PORT")?.parse().ok()?,
            "/jobs",
        ),
        ServiceKind::Binary { bin_name } if bin_name == "lunar-start-backend" => (
            "127.0.0.1",
            env.get("LUNAR_START_PORT")
                .and_then(|v| v.parse().ok())
                .unwrap_or(16181),
            "/health",
        ),
        ServiceKind::DxServe { crate_name, .. } if crate_name == "lunar-frontend" => (
            "127.0.0.1",
            FrontendLaunchConfig::from_values(env, extra_args)
                .ok()?
                .port?,
            "/",
        ),
        ServiceKind::DxServe {
            crate_name,
            default_port,
            ..
        } if crate_name == "lunar-testbench" => (
            "127.0.0.1",
            extra_args
                .windows(2)
                .find(|args| matches!(args[0].as_str(), "--port" | "-p"))
                .map(|args| args[1].as_str())
                .unwrap_or(default_port)
                .parse()
                .ok()?,
            "/testbench",
        ),
        ServiceKind::DxServe { default_port, .. } => (
            "127.0.0.1",
            extra_args
                .windows(2)
                .find(|args| matches!(args[0].as_str(), "--port" | "-p"))
                .map(|args| args[1].as_str())
                .unwrap_or(default_port)
                .parse()
                .ok()?,
            "/",
        ),
        _ => return None,
    };
    let host = if host.parse::<IpAddr>().is_ok_and(|ip| ip.is_unspecified()) {
        "127.0.0.1"
    } else {
        host
    };
    Some((host.to_string(), port, path))
}

async fn probe_http(host: &str, port: u16, path: &str) -> bool {
    let probe = async {
        let mut stream = TcpStream::connect((host, port)).await?;
        stream
            .write_all(
                format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n")
                    .as_bytes(),
            )
            .await?;
        let mut status = [0u8; 12];
        stream.read_exact(&mut status).await?;
        Ok::<bool, std::io::Error>(
            status.starts_with(b"HTTP/1.1 2")
                || status.starts_with(b"HTTP/1.0 2")
                || status.starts_with(b"HTTP/1.1 3")
                || status.starts_with(b"HTTP/1.0 3"),
        )
    };
    tokio::time::timeout(Duration::from_secs(2), probe)
        .await
        .is_ok_and(|result| result.unwrap_or(false))
}

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
            format!("cargo run --bin {bin_name} {args}")
                .trim()
                .to_string()
        }
    }
}

pub fn resolve_binary_path(ws: &Path, name: &str) -> PathBuf {
    let release_bin = ws.join("target").join("release").join(name);
    let debug_bin = ws.join("target").join("debug").join(name);
    if release_bin.exists() {
        release_bin
    } else if debug_bin.exists() {
        debug_bin
    } else {
        release_bin
    }
}

pub fn build_binary_cmd(
    ws: &Path,
    name: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> tokio::process::Command {
    let bin = resolve_binary_path(ws, name);
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(extra_args)
        .envs(env)
        .current_dir(ws)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd
}

fn spawn_binary(
    ws: &Path,
    name: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<Child> {
    let release_bin = ws.join("target").join("release").join(name);
    let debug_bin = ws.join("target").join("debug").join(name);

    if !release_bin.exists() && !debug_bin.exists() {
        eprintln!(
            "[lns] [{name}] Precompiled binary not found in target/release or target/debug; compiling and launching via cargo run..."
        );
        return spawn_cargo_run(ws, name, &[], &[], extra_args, env);
    }

    let bin = if release_bin.exists() {
        release_bin
    } else {
        debug_bin
    };

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(extra_args)
        .envs(env)
        .current_dir(ws)
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    cmd.spawn()
        .with_context(|| format!("failed to spawn binary at {}", bin.display()))
}

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

pub fn build_frontend_dx_serve_cmd(
    ws: &Path,
    crate_name: &str,
    crate_subdir: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<tokio::process::Command> {
    let launch = FrontendLaunchConfig::from_values(env, extra_args)
        .map_err(|error| anyhow::anyhow!("{}: {}", error.field, error.message))?;
    let dx_bin = env.get("LUNAR_DX_BIN").map(String::as_str).unwrap_or("dx");
    let mut cmd = tokio::process::Command::new(dx_bin);
    cmd.arg("serve")
        .args(launch.dx_args())
        .envs(env)
        .current_dir(ws.join(crate_subdir).join(crate_name))
        .kill_on_drop(true)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    Ok(cmd)
}

fn spawn_frontend_dx_serve(
    ws: &Path,
    crate_name: &str,
    crate_subdir: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> Result<Child> {
    let dx_bin = env
        .get("LUNAR_DX_BIN")
        .map(String::as_str)
        .unwrap_or("dx")
        .to_string();
    build_frontend_dx_serve_cmd(ws, crate_name, crate_subdir, extra_args, env)?
        .spawn()
        .with_context(|| {
            format!(
                "failed to run `{dx_bin} serve` for lunar-frontend. \
                 Check if dioxus-cli is installed or set LUNAR_DX_BIN."
            )
        })
}

pub fn build_dx_serve_cmd(
    ws: &Path,
    crate_name: &str,
    crate_subdir: &str,
    default_port: &str,
    extra_args: &[String],
    env: &HashMap<String, String>,
) -> tokio::process::Command {
    let user_port = extra_args.iter().any(|a| a == "--port" || a == "-p");
    let dx_bin = env.get("LUNAR_DX_BIN").map(String::as_str).unwrap_or("dx");

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

pub async fn cargo_build(ws: &Path, build_args: &[String]) -> Result<()> {
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.args(["build", "--release"]);
    cmd.args(build_args);
    cmd.current_dir(ws)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd.status().await.context("failed to run cargo build")?;

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
        env.insert(
            "LUNAR_API_URL".to_string(),
            "http://localhost:8080".to_string(),
        );
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
    fn test_build_frontend_dx_serve_cmd_uses_platform_matrix() {
        let ws = Path::new("/workspace");
        let mut web_env = HashMap::new();
        web_env.insert("LUNAR_FRONTEND_PLATFORM".to_string(), "web".to_string());
        web_env.insert("LUNAR_FRONTEND_PORT".to_string(), "8088".to_string());
        let web =
            build_frontend_dx_serve_cmd(ws, "lunar-frontend", "crates", &[], &web_env).unwrap();
        assert_eq!(
            extract_args(&web),
            vec!["serve", "--platform", "web", "--port", "8088"]
        );

        let mut desktop_env = HashMap::new();
        desktop_env.insert("LUNAR_FRONTEND_PLATFORM".to_string(), "desktop".to_string());
        let desktop =
            build_frontend_dx_serve_cmd(ws, "lunar-frontend", "crates", &[], &desktop_env).unwrap();
        assert_eq!(
            extract_args(&desktop),
            vec!["serve", "--platform", "desktop"]
        );
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
                "watch",
                "--",
                "cargo",
                "run",
                "--release",
                "--offline",
                "--features",
                "mock",
                "--bin",
                "testbench",
                "--",
                "--verbose"
            ]
        );
    }

    #[test]
    fn test_build_cargo_run_cmd_applies_env() {
        let ws = Path::new("/workspace");
        let mut env = HashMap::new();
        env.insert(
            "DATABASE_URL".to_string(),
            "postgres://localhost/db".to_string(),
        );

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
            vec![
                "run",
                "--bin",
                "server",
                "--features",
                "postgres",
                "--",
                "--migrate"
            ]
        );
    }

    #[tokio::test]
    async fn http_readiness_rejects_unready_responses() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            for (path, response) in [
                ("/version", "503 Service Unavailable"),
                ("/version", "200 OK"),
            ] {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0u8; 256];
                let count = stream.read(&mut request).await.unwrap();
                assert!(
                    String::from_utf8_lossy(&request[..count])
                        .starts_with(&format!("GET {path} HTTP/1.1"))
                );
                stream
                    .write_all(
                        format!("HTTP/1.1 {response}\r\nContent-Length: 0\r\n\r\n").as_bytes(),
                    )
                    .await
                    .unwrap();
            }
        });
        assert!(!probe_http("127.0.0.1", port, "/version").await);
        assert!(probe_http("127.0.0.1", port, "/version").await);
        server.await.unwrap();
    }

    #[test]
    fn readiness_uses_listener_not_public_browser_url() {
        let mut env = crate::service_settings::FrontendSettings::default()
            .to_values()
            .env;
        env.insert(
            "LUNAR_FRONTEND_PUBLIC_URL".into(),
            "https://frontend.example.org/app".into(),
        );
        let kind = ServiceKind::DxServe {
            crate_name: "lunar-frontend".into(),
            crate_subdir: "crates".into(),
            default_port: "8080".into(),
        };
        assert_eq!(
            readiness_target(&kind, &env, &[]),
            Some(("127.0.0.1".into(), 8080, "/"))
        );
    }

    #[test]
    fn readiness_uses_testbench_base_path() {
        let kind = ServiceKind::DxServe {
            crate_name: "lunar-testbench".into(),
            crate_subdir: "testbench".into(),
            default_port: "16180".into(),
        };
        assert_eq!(
            readiness_target(&kind, &HashMap::new(), &[]),
            Some(("127.0.0.1".into(), 16180, "/testbench"))
        );
    }
}
