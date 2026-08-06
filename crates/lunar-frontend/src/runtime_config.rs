//! Runtime configuration for the standalone stellar frontend.
//!
//! The launcher supplies `LUNAR_BACKEND_URL`. Web and desktop default to the
//! host loopback address, while Android defaults to the emulator bridge. A
//! physical device must receive an explicit reachable host URL.

use lunar_utils::env::{
    DEFAULT_ANDROID_BACKEND_URL, DEFAULT_BACKEND_URL, get_frontend_backend_url,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub backend_url: String,
}

impl RuntimeConfig {
    pub fn from_environment() -> Self {
        let default_url = if cfg!(feature = "android") {
            DEFAULT_ANDROID_BACKEND_URL
        } else {
            DEFAULT_BACKEND_URL
        };

        // A web bundle and Android APK need the launcher-provided build-time
        // value. Native desktop runs can additionally receive it at process
        // startup. Runtime process configuration takes precedence when both
        // sources exist.
        let backend_url = std::env::var("LUNAR_BACKEND_URL")
            .ok()
            .or_else(|| option_env!("LUNAR_BACKEND_URL").map(str::to_owned))
            .unwrap_or_else(|| get_frontend_backend_url(default_url));
        Self { backend_url }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_uses_a_backend_url() {
        assert!(RuntimeConfig::from_environment().backend_url.starts_with("http"));
    }
}
