//! lns — Lunar launcher binary.
//!
//! Launches lunar-testbench and lunar-testbench-backend, streams logs to stdout.

use clap::Parser;
use lunar_start::prelude::*;

#[derive(Parser)]
#[command(name = "lns", about = "Lunar launcher — starts testbench + testbench-backend")]
struct Cli {
    /// Watch for file changes and restart (requires cargo-watch).
    #[arg(short, long)]
    watch: bool,

    /// Build release binaries before launching.
    #[arg(short, long)]
    build: bool,

    /// Run environment (env: LUNAR_ENV).
    #[arg(long, env = "LUNAR_ENV")]
    env_mode: Option<String>,

    /// Core backend host (env: LUNAR_BACKEND_HOST).
    #[arg(long, env = "LUNAR_BACKEND_HOST")]
    backend_host: Option<String>,

    /// Core backend port (env: LUNAR_BACKEND_PORT).
    #[arg(long, env = "LUNAR_BACKEND_PORT")]
    backend_port: Option<u16>,

    /// Testbench host (env: LUNAR_TESTBENCH_HOST).
    #[arg(long, env = "LUNAR_TESTBENCH_HOST")]
    testbench_host: Option<String>,

    /// Testbench port (env: LUNAR_TESTBENCH_PORT).
    #[arg(long, env = "LUNAR_TESTBENCH_PORT")]
    testbench_port: Option<u16>,

    /// Models directory (env: LUNAR_MODELS_DIR).
    #[arg(long, env = "LUNAR_MODELS_DIR")]
    models_dir: Option<String>,

    /// Worlds directory (env: LUNAR_WORLDS_DIR).
    #[arg(long, env = "LUNAR_WORLDS_DIR")]
    worlds_dir: Option<String>,

    /// Dioxus CLI binary name (env: LUNAR_DX_BIN, default: dx).
    #[arg(long, env = "LUNAR_DX_BIN", default_value = "dx")]
    dx_bin: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut config = LauncherConfig::for_start_full();

    config.watch = cli.watch;
    config.build_release = cli.build;

    if let Some(env) = &cli.env_mode {
        config.set_env("LUNAR_ENV", env);
    }
    if let Some(host) = &cli.backend_host {
        config.set_env("LUNAR_BACKEND_HOST", host);
    }
    if let Some(port) = cli.backend_port {
        config.set_env("LUNAR_BACKEND_PORT", &port.to_string());
    }
    if let Some(host) = &cli.testbench_host {
        config.set_env("LUNAR_TESTBENCH_HOST", host);
    }
    if let Some(port) = cli.testbench_port {
        config.set_env("LUNAR_TESTBENCH_PORT", &port.to_string());
    }
    if let Some(dir) = &cli.models_dir {
        config.set_env("LUNAR_MODELS_DIR", dir);
    }
    if let Some(dir) = &cli.worlds_dir {
        config.set_env("LUNAR_WORLDS_DIR", dir);
    }
    config.set_env("LUNAR_DX_BIN", &cli.dx_bin);

    let launcher = Launcher::new(config);
    launcher.run().await
}
