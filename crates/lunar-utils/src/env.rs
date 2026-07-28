use std::path::PathBuf;

pub fn get_host() -> String {
    std::env::var("LUNAR_BACKEND_HOST").unwrap_or_else(|_| "127.0.0.1".to_string())
}

pub fn get_port() -> u16 {
    std::env::var("LUNAR_BACKEND_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25255)
}

pub fn get_testbench_host() -> String {
    std::env::var("LUNAR_TESTBENCH_HOST")
        .or_else(|_| std::env::var("LUNAR_BACKEND_HOST"))
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}

pub fn get_testbench_port() -> u16 {
    std::env::var("LUNAR_TESTBENCH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25256)
}

pub fn get_url() -> String {
    format!("{{http://{}}}:{}", get_host(), get_port())
}

pub fn get_testbench_url() -> String {
    format!("{{http://{}}}:{}", get_testbench_host(), get_testbench_port())
}

/// Host for the `lunar-start-backend` service manager (see `crates/lunar-start-backend`).
pub fn get_start_backend_host() -> String {
    std::env::var("LUNAR_START_HOST").unwrap_or_else(|_| "127.0.0.1".to_string())
}

/// Port for the `lunar-start-backend` service manager (matches its own `LUNAR_START_PORT`
/// default, see `crates/lunar-start-backend/src/main.rs`).
pub fn get_start_backend_port() -> u16 {
    std::env::var("LUNAR_START_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16181)
}

/// Base URL for the `lunar-start-backend` service manager REST + SSE API.
pub fn get_start_backend_url() -> String {
    format!("http://{}:{}", get_start_backend_host(), get_start_backend_port())
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
