use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(name = "lunar-start", about = "Lunar-HAL launcher")]
struct Cli {
    /// Build release binaries before launching services
    #[arg(short, long)]
    build: bool,

    /// Core backend host (env: LUNAR_BACKEND_HOST)
    #[arg(long)]
    backend_host: Option<String>,

    /// Core backend port (env: LUNAR_BACKEND_PORT)
    #[arg(long)]
    backend_port: Option<u16>,

    /// Testbench host (env: LUNAR_TESTBENCH_HOST)
    #[arg(long)]
    testbench_host: Option<String>,

    /// Testbench port (env: LUNAR_TESTBENCH_PORT)
    #[arg(long)]
    testbench_port: Option<u16>,

    /// Models directory path (env: LUNAR_MODELS_DIR)
    #[arg(long)]
    models_dir: Option<String>,

    /// World directory path (env: LUNAR_WORLDS_DIR)
    #[arg(long)]
    worlds_dir: Option<String>,

    /// Run environment (env: LUNAR_ENV)
    #[arg(long)]
    env_mode: Option<String>,

    /// Services to launch: backend, frontend, frontend-desktop, testbench, testbench-backend, or all/run.
    /// Extra args in (...) groups are forwarded to the preceding service.
    /// Trailing args after `--` are forwarded to all services.
    /// Example: lunar-start backend (--port 25255) frontend -- --host 127.0.0.1
    #[arg(trailing_var_arg(true))]
    services: Vec<String>,
}

struct ServiceParser {
    services: Vec<(String, Vec<String>)>,
    pending: Vec<String>,
    in_paren: bool,
}

impl ServiceParser {
    fn new() -> Self {
        Self {
            services: Vec::new(),
            pending: Vec::new(),
            in_paren: false,
        }
    }

    fn flush_pending_to_last(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        if let Some((_, args)) = self.services.last_mut() {
            args.append(&mut self.pending);
        } else {
            self.pending.clear();
        }
    }

    fn parse_token(&mut self, token: &str) {
        if let Some(inner) = token.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
            self.handle_isolated_paren_block(inner);
        } else if self.in_paren {
            self.handle_token_inside_paren(token);
        } else {
            self.handle_token_outside_paren(token);
        }
    }

    fn handle_isolated_paren_block(&mut self, inner: &str) {
        if !inner.is_empty() {
            self.pending
                .extend(inner.split_whitespace().map(String::from));
        }
        self.flush_pending_to_last();
    }

    fn handle_token_inside_paren(&mut self, token: &str) {
        if token == ")" {
            self.in_paren = false;
            self.flush_pending_to_last();
        } else if let Some(rest) = token.strip_suffix(')') {
            if !rest.is_empty() {
                self.pending.push(rest.to_string());
            }
            self.in_paren = false;
            self.flush_pending_to_last();
        } else {
            self.pending.push(token.to_string());
        }
    }

    fn handle_token_outside_paren(&mut self, token: &str) {
        if token == "(" {
            self.in_paren = true;
            self.pending.clear();
        } else if let Some(rest) = token.strip_prefix('(') {
            self.in_paren = true;
            self.pending.clear();
            if !rest.is_empty() {
                self.pending.push(rest.to_string());
            }
        } else {
            self.services.push((token.to_string(), Vec::new()));
        }
    }

    fn finish(mut self) -> Vec<(String, Vec<String>)> {
        self.flush_pending_to_last();
        self.services
    }
}

fn parse_services(tokens: &[String]) -> Vec<(String, Vec<String>)> {
    let mut parser = ServiceParser::new();
    for token in tokens {
        parser.parse_token(token);
    }
    parser.finish()
}

fn workspace_root() -> PathBuf {
    let exe = std::env::current_exe().ok();
    if let Some(exe) = exe {
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

fn cargo_build(ws: &Path) -> Result<()> {
    let status = std::process::Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(ws)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("cargo build failed with status: {status}");
    }
    Ok(())
}

fn cargo_build_desktop(ws: &Path) -> Result<()> {
    let status = std::process::Command::new("cargo")
        .args([
            "build",
            "--release",
            "--no-default-features",
            "--features",
            "desktop",
            "-p",
            "lunar-frontend",
        ])
        .current_dir(ws)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to run cargo build for desktop frontend")?;

    if !status.success() {
        anyhow::bail!("cargo build (desktop frontend) failed with status: {status}");
    }
    Ok(())
}

fn spawn_binary(ws: &Path, name: &str, extra_args: &[String]) -> Result<std::process::Child> {
    let bin = ws.join("target").join("release").join(name);
    let child = std::process::Command::new(&bin)
        .args(extra_args)
        .current_dir(ws)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("failed to spawn {name}"))?;
    Ok(child)
}

fn spawn_dx_serve(
    ws: &Path,
    crate_name: &str,
    port: &str,
    extra_args: &[String],
) -> Result<std::process::Child> {
    let user_port = extra_args.iter().any(|a| a == "--port" || a == "-p");
    let actual_port = extra_args
        .windows(2)
        .find(|w| w[0] == "--port" || w[0] == "-p")
        .map(|w| w[1].as_str())
        .unwrap_or(port);

    let mut cmd = std::process::Command::new("dx");
    cmd.arg("serve")
        .current_dir(ws.join("crates").join(crate_name))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if !user_port {
        cmd.args(["--port", port]);
    }
    cmd.args(extra_args);

    let child = cmd
        .spawn()
        .context("failed to run `dx serve` — install dioxus-cli: cargo install dioxus-cli")?;
    println!("[lunar-start] {crate_name} served at http://127.0.0.1:{actual_port}");
    Ok(child)
}

struct ServiceMap {
    bin_name: &'static str,
    is_dx: bool,
    default_port: &'static str,
}

fn map_service_name(name: &str) -> Result<ServiceMap> {
    match name {
        "backend" => Ok(ServiceMap {
            bin_name: "lunar-backend",
            is_dx: false,
            default_port: "",
        }),
        "testbench-backend" => Ok(ServiceMap {
            bin_name: "lunar-testbench-backend",
            is_dx: false,
            default_port: "",
        }),
        "testbench" => Ok(ServiceMap {
            bin_name: "lunar-testbench",
            is_dx: true,
            default_port: "16180",
        }),
        "frontend" => Ok(ServiceMap {
            bin_name: "lunar-frontend",
            is_dx: true,
            default_port: "8080",
        }),
        "frontend-desktop" => Ok(ServiceMap {
            bin_name: "lunar-frontend",
            is_dx: false,
            default_port: "",
        }),
        other => anyhow::bail!("unknown service: {other}"),
    }
}

unsafe fn setup_env(cli: &Cli) {
    unsafe {
        if let Some(host) = &cli.backend_host {
            std::env::set_var("LUNAR_BACKEND_HOST", host);
        }
        if let Some(port) = cli.backend_port {
            std::env::set_var("LUNAR_BACKEND_PORT", port.to_string());
        }
        if let Some(host) = &cli.testbench_host {
            std::env::set_var("LUNAR_TESTBENCH_HOST", host);
        }
        if let Some(port) = cli.testbench_port {
            std::env::set_var("LUNAR_TESTBENCH_PORT", port.to_string());
        }
        if let Some(dir) = &cli.models_dir {
            std::env::set_var("LUNAR_MODELS_DIR", dir);
        }
        if let Some(dir) = &cli.worlds_dir {
            std::env::set_var("LUNAR_WORLDS_DIR", dir);
        }
        if let Some(mode) = &cli.env_mode {
            std::env::set_var("LUNAR_ENV", mode);
        }
    }
}

fn run_single_service(ws: &Path, name: &str, args: &[String], build_desktop: bool) -> Result<()> {
    let mapping = map_service_name(name)?;
    if name == "frontend-desktop" && build_desktop {
        cargo_build_desktop(ws)?;
    }

    let mut child = if mapping.is_dx {
        spawn_dx_serve(ws, mapping.bin_name, mapping.default_port, args)?
    } else {
        spawn_binary(ws, mapping.bin_name, args)?
    };
    child.wait()?;
    Ok(())
}

fn launch_services(
    ws: &Path,
    services: &[(String, Vec<String>)],
    global_args: &[String],
) -> Vec<(String, std::process::Child)> {
    let mut children = Vec::new();
    for (name, extra_args) in services {
        let mapping = match map_service_name(name) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[lunar-start] failed to map {name}: {e}");
                continue;
            }
        };

        let mut combined_args = extra_args.clone();
        combined_args.extend(global_args.iter().cloned());

        let result = if mapping.is_dx {
            spawn_dx_serve(ws, mapping.bin_name, mapping.default_port, &combined_args)
        } else {
            spawn_binary(ws, mapping.bin_name, &combined_args)
        };

        match result {
            Ok(child) => {
                println!("[lunar-start] launched {name} (pid {})", child.id());
                children.push((name.clone(), child));
            }
            Err(e) => eprintln!("[lunar-start] failed to launch {name}: {e}"),
        }
    }
    children
}

async fn monitor_children(mut children: Vec<(String, std::process::Child)>) -> Result<()> {
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[lunar-start] shutting down...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    })
    .context("failed to set Ctrl+C handler")?;

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        let mut i = 0;
        while i < children.len() {
            let (name, child) = &mut children[i];
            match child.try_wait() {
                Ok(Some(status)) => {
                    println!("[lunar-start] {name} exited with {status}");
                    children.swap_remove(i);
                }
                Ok(None) => i += 1,
                Err(e) => {
                    eprintln!("[lunar-start] error waiting for {name}: {e}");
                    children.swap_remove(i);
                }
            }
        }
        if children.is_empty() {
            println!("[lunar-start] all services exited");
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    for (name, mut child) in children {
        let _ = child.kill();
        let _ = child.wait();
        println!("[lunar-start] stopped {name}");
    }

    Ok(())
}

fn split_global_args(services: &[String]) -> (Vec<String>, Vec<String>) {
    if let Some(idx) = services.iter().position(|t| t == "--") {
        let (left, right) = services.split_at(idx);
        let global = right[1..].to_vec();
        (left.to_vec(), global)
    } else {
        (Vec::from(services), Vec::new())
    }
}

fn handle_early_exit(
    parsed: &[(String, Vec<String>)],
    build_flag: bool,
    ws: &Path,
) -> Result<bool> {
    if parsed.is_empty() {
        if build_flag {
            cargo_build(ws)?;
            return Ok(true);
        }
        use clap::CommandFactory;
        Cli::command().print_help()?;
        println!();
        return Ok(true);
    }

    if parsed.iter().any(|(name, _)| name == "build") {
        cargo_build(ws)?;
        return Ok(true);
    }

    Ok(false)
}

fn resolve_services(parsed: Vec<(String, Vec<String>)>) -> (Vec<(String, Vec<String>)>, bool) {
    let is_all = parsed
        .iter()
        .any(|(name, _)| name == "all" || name == "run");

    let services = if is_all {
        vec![
            ("backend".to_string(), Vec::new()),
            ("testbench-backend".to_string(), Vec::new()),
            ("testbench".to_string(), Vec::new()),
            ("frontend".to_string(), Vec::new()),
        ]
    } else {
        parsed
    };

    (services, is_all)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let ws = workspace_root();

    unsafe {
        setup_env(&cli);
    }

    let (service_tokens, global_args) = split_global_args(&cli.services);
    let parsed = parse_services(&service_tokens);

    if handle_early_exit(&parsed, cli.build, &ws)? {
        return Ok(());
    }

    let (services, is_all) = resolve_services(parsed);

    if is_all || cli.build {
        cargo_build(&ws)?;
    }

    if services.len() == 1 && !is_all {
        let (name, extra_args) = &services[0];
        let mut combined_args = extra_args.clone();
        combined_args.extend(global_args);

        run_single_service(&ws, name, &combined_args, cli.build || is_all)?;
        return Ok(());
    }

    let children = launch_services(&ws, &services, &global_args);
    if children.is_empty() {
        anyhow::bail!("no services were launched");
    }

    monitor_children(children).await?;

    Ok(())
}
