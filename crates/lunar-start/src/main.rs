use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{Context, Result};
use clap::Parser;
use nah::{have_duplicate_code, high_complexity};

#[derive(Parser)]
#[command(name = "lunar-start", about = "Lunar-HAL launcher")]
struct Cli {
    /// Build release binaries before launching services
    #[arg(short, long)]
    build: bool,

    /// Services to launch: backend, frontend, testbench, testbench-backend, or all/run.
    /// Extra args in (...) groups are forwarded to the preceding service.
    /// Example: lunar-start backend (--port 25255) frontend (--port 8080)
    #[arg(trailing_var_arg(true))]
    services: Vec<String>,
}

#[have_duplicate_code]
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

fn cargo_build(ws: &PathBuf) -> Result<()> {
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

fn spawn_binary(ws: &PathBuf, name: &str, extra_args: &[String]) -> Result<std::process::Child> {
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

fn spawn_dx_serve(ws: &PathBuf, crate_name: &str, port: &str, extra_args: &[String]) -> Result<std::process::Child> {
    let user_port = extra_args.iter().any(|a| a == "--port" || a == "-p");
    let actual_port = extra_args
        .windows(2)
        .find(|w| w[0] == "--port" || w[0] == "-p")
        .and_then(|w| Some(w[1].as_str()))
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
        .context(
            "failed to run `dx serve` — install dioxus-cli: cargo install dioxus-cli",
        )?;
    println!("[lunar-start] {crate_name} served at http://127.0.0.1:{actual_port}");
    Ok(child)
}

/// Parse tokens into (service_name, extra_args_for_that_service) pairs.
/// Each `(...)` group attaches to the immediately preceding service name.
#[high_complexity]
fn parse_services(tokens: &[String]) -> Vec<(String, Vec<String>)> {
    let mut services: Vec<(String, Vec<String>)> = Vec::new();
    let mut in_paren = false;
    let mut pending: Vec<String> = Vec::new();

    fn flush_to_last(pending: &mut Vec<String>, services: &mut Vec<(String, Vec<String>)>) {
        if pending.is_empty() {
            return;
        }
        if let Some((_, args)) = services.last_mut() {
            args.append(pending);
        } else {
            pending.clear();
        }
    }

    for token in tokens {
        if token == "(" && !in_paren {
            in_paren = true;
            pending.clear();
        } else if token == ")" && in_paren {
            in_paren = false;
            flush_to_last(&mut pending, &mut services);
        } else if let Some(inner) = token.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
            if !inner.is_empty() {
                pending.extend(inner.split_whitespace().map(String::from));
            }
            flush_to_last(&mut pending, &mut services);
        } else if in_paren {
            if let Some(rest) = token.strip_suffix(')') {
                if !rest.is_empty() {
                    pending.push(rest.to_string());
                }
                in_paren = false;
                flush_to_last(&mut pending, &mut services);
            } else {
                pending.push(token.clone());
            }
        } else if let Some(rest) = token.strip_prefix('(') {
            in_paren = true;
            pending.clear();
            if !rest.is_empty() {
                pending.push(rest.to_string());
            }
        } else {
            services.push((token.clone(), Vec::new()));
        }
    }

    services
}

#[high_complexity]
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let ws = workspace_root();

    let parsed = parse_services(&cli.services);

    if parsed.is_empty() {
        if cli.build {
            cargo_build(&ws)?;
            return Ok(());
        }
        use clap::CommandFactory;
        Cli::command().print_help()?;
        println!();
        return Ok(());
    }

    if parsed.iter().any(|(name, _)| name == "build") {
        cargo_build(&ws)?;
        return Ok(());
    }

    let is_all = parsed.iter().any(|(name, _)| name == "all" || name == "run");

    if is_all || cli.build {
        cargo_build(&ws)?;
    }

    let services: Vec<(String, Vec<String>)> = if is_all {
        vec![
            ("lunar-backend".to_string(), Vec::new()),
            ("lunar-testbench-backend".to_string(), Vec::new()),
            ("lunar-testbench".to_string(), Vec::new()),
            ("lunar-frontend".to_string(), Vec::new()),
        ]
    } else {
        parsed
    };

    if services.len() == 1 && !is_all {
        let (name, extra_args) = &services[0];
        match name.as_str() {
            "backend" => {
                let mut child = spawn_binary(&ws, "lunar-backend", extra_args)?;
                child.wait()?;
            }
            "testbench-backend" => {
                let mut child = spawn_binary(&ws, "lunar-testbench-backend", extra_args)?;
                child.wait()?;
            }
            "testbench" => {
                let mut child = spawn_dx_serve(&ws, "lunar-testbench", "16180", extra_args)?;
                child.wait()?;
            }
            "frontend" => {
                let mut child = spawn_dx_serve(&ws, "lunar-frontend", "8080", extra_args)?;
                child.wait()?;
            }
            other => anyhow::bail!("unknown service: {other}"),
        }
        return Ok(());
    }

    let mut children: Vec<(String, std::process::Child)> = Vec::new();

    for (name, extra_args) in &services {
        let (bin_name, is_dx) = match name.as_str() {
            "backend" => ("lunar-backend", false),
            "testbench-backend" => ("lunar-testbench-backend", false),
            "testbench" => ("lunar-testbench", true),
            "frontend" => ("lunar-frontend", true),
            other => {
                eprintln!("[lunar-start] unknown service: {other}");
                continue;
            }
        };

        let result = if is_dx {
            let port = if bin_name == "lunar-testbench" { "16180" } else { "8080" };
            spawn_dx_serve(&ws, bin_name, port, extra_args)
        } else {
            spawn_binary(&ws, bin_name, extra_args)
        };

        match result {
            Ok(child) => {
                println!("[lunar-start] launched {name} (pid {})", child.id());
                children.push((name.to_string(), child));
            }
            Err(e) => eprintln!("[lunar-start] failed to launch {name}: {e}"),
        }
    }

    if children.is_empty() {
        anyhow::bail!("no services were launched");
    }

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
