use std::path::PathBuf;

pub const DEFAULT_BACKEND_HOST: &str = "127.0.0.1";
pub const DEFAULT_BACKEND_PORT: u16 = 25255;
pub const DEFAULT_TESTBENCH_HOST: &str = "127.0.0.1";
pub const DEFAULT_TESTBENCH_PORT: u16 = 25256;
pub const DEFAULT_START_BACKEND_HOST: &str = "127.0.0.1";
pub const DEFAULT_START_BACKEND_PORT: u16 = 16181;

fn parse_port(value: Option<&str>, default: u16) -> u16 {
    value
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(default)
}

/// Resolves a CLI `--port`/`-p` value before ordered environment fallbacks.
pub fn resolve_port(args: &[String], env_values: &[Option<&str>], default: u16) -> u16 {
    let cli_port = args
        .windows(2)
        .find(|window| window[0] == "--port" || window[0] == "-p")
        .and_then(|window| window[1].parse::<u16>().ok());

    cli_port
        .or_else(|| {
            env_values
                .iter()
                .find_map(|value| value.and_then(|value| value.parse::<u16>().ok()))
        })
        .unwrap_or(default)
}

pub fn get_host() -> String {
    std::env::var("LUNAR_BACKEND_HOST").unwrap_or_else(|_| DEFAULT_BACKEND_HOST.to_string())
}

pub fn get_port() -> u16 {
    parse_port(
        std::env::var("LUNAR_BACKEND_PORT").ok().as_deref(),
        DEFAULT_BACKEND_PORT,
    )
}

pub fn get_testbench_host() -> String {
    std::env::var("LUNAR_TESTBENCH_HOST")
        .or_else(|_| std::env::var("LUNAR_BACKEND_HOST"))
        .unwrap_or_else(|_| DEFAULT_TESTBENCH_HOST.to_string())
}

pub fn get_testbench_port() -> u16 {
    parse_port(
        std::env::var("LUNAR_TESTBENCH_PORT").ok().as_deref(),
        DEFAULT_TESTBENCH_PORT,
    )
}

pub fn get_url() -> String {
    format!("http://{}:{}", get_host(), get_port())
}

pub fn get_testbench_url() -> String {
    format!("http://{}:{}", get_testbench_host(), get_testbench_port())
}

/// Host for the `lunar-start-backend` service manager (see `crates/lunar-start-backend`).
pub fn get_start_backend_host() -> String {
    std::env::var("LUNAR_START_HOST").unwrap_or_else(|_| DEFAULT_START_BACKEND_HOST.to_string())
}

/// Port for the `lunar-start-backend` service manager.
pub fn get_start_backend_port() -> u16 {
    parse_port(
        std::env::var("LUNAR_START_PORT").ok().as_deref(),
        DEFAULT_START_BACKEND_PORT,
    )
}

/// Base URL for the `lunar-start-backend` service manager REST + SSE API.
pub fn get_start_backend_url() -> String {
    format!(
        "http://{}:{}",
        get_start_backend_host(),
        get_start_backend_port()
    )
}

pub fn get_lunar_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LUNAR_MODELS_DIR") {
        PathBuf::from(dir)
    } else {
        dirs::data_dir()
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
            .join("lunar")
            .join("models")
    }
}

pub fn get_worlds_dir() -> PathBuf {
    let standard_storage = dirs::data_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("lunar")
        .join("worlds");

    std::env::var("LUNAR_WORLDS_DIR")
        .map(PathBuf::from)
        .unwrap_or(standard_storage)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_parser_uses_valid_value() {
        assert_eq!(parse_port(Some("32000"), 1), 32000);
    }

    #[test]
    fn port_parser_falls_back_for_missing_or_invalid_value() {
        assert_eq!(parse_port(None, DEFAULT_BACKEND_PORT), DEFAULT_BACKEND_PORT);
        assert_eq!(
            parse_port(Some("not-a-port"), DEFAULT_TESTBENCH_PORT),
            DEFAULT_TESTBENCH_PORT
        );
        assert_eq!(
            parse_port(Some("70000"), DEFAULT_START_BACKEND_PORT),
            DEFAULT_START_BACKEND_PORT
        );
    }

    #[test]
    fn resolver_uses_cli_then_ordered_env_fallbacks() {
        let args = vec![
            "binary".to_string(),
            "--port".to_string(),
            "28000".to_string(),
        ];
        assert_eq!(
            resolve_port(
                &args,
                &[Some("26000"), Some("27000")],
                DEFAULT_TESTBENCH_PORT
            ),
            28000
        );

        let no_cli = vec!["binary".to_string()];
        assert_eq!(
            resolve_port(
                &no_cli,
                &[Some("26000"), Some("27000")],
                DEFAULT_TESTBENCH_PORT,
            ),
            26000
        );
        assert_eq!(
            resolve_port(
                &no_cli,
                &[Some("invalid"), Some("27000")],
                DEFAULT_TESTBENCH_PORT,
            ),
            27000
        );
    }

    #[test]
    fn service_defaults_stay_distinct() {
        assert_eq!(DEFAULT_BACKEND_PORT, 25255);
        assert_eq!(DEFAULT_TESTBENCH_PORT, 25256);
        assert_eq!(DEFAULT_START_BACKEND_PORT, 16181);
    }
}
