
use clap::Parser;
use lunar_start::prelude::*;

#[derive(Parser)]
#[command(
    name = "lns",
    about = "Lunar launcher — starts testbench + testbench-backend"
)]
struct Cli {
    #[arg(short, long)]
    watch: bool,

    #[arg(short, long)]
    build: bool,

    #[arg(long, env = "LUNAR_ENV")]
    env_mode: Option<String>,

    #[arg(long, env = "LUNAR_BACKEND_HOST")]
    backend_host: Option<String>,

    #[arg(long, env = "LUNAR_BACKEND_PORT")]
    backend_port: Option<u16>,

    #[arg(long, env = "LUNAR_TESTBENCH_HOST")]
    testbench_host: Option<String>,

    #[arg(long, env = "LUNAR_TESTBENCH_PORT")]
    testbench_port: Option<u16>,

    #[arg(long, env = "LUNAR_MODELS_DIR")]
    models_dir: Option<String>,

    #[arg(long, env = "LUNAR_SCENES_DIR")]
    scenes_dir: Option<String>,

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
    } else {
        let ws_models = config.workspace.join("models");
        if ws_models.is_dir() {
            config.set_env("LUNAR_MODELS_DIR", &ws_models.display().to_string());
        }
    }
    if let Some(dir) = &cli.scenes_dir {
        config.set_env("LUNAR_SCENES_DIR", dir);
    }
    config.set_env("LUNAR_DX_BIN", &cli.dx_bin);

    let launcher = Launcher::new(config);
    launcher.run().await
}
