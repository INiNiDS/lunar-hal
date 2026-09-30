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
        #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
        if let Some(url) = Self::resolve_browser_backend_url() {
            return Self { backend_url: url };
        }

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

    #[cfg(all(target_family = "wasm", not(target_os = "wasi")))]
    pub fn resolve_browser_backend_url() -> Option<String> {
        let compile_time_env = option_env!("LUNAR_BACKEND_URL");
        if let Some(explicit) = compile_time_env {
            if !explicit.is_empty()
                && !explicit.contains("127.0.0.1")
                && !explicit.contains("localhost")
            {
                return Some(explicit.to_string());
            }
        }

        let window = web_sys::window()?;
        let location = window.location();
        if let Ok(origin) = location.origin() {
            if !origin.contains("localhost")
                && !origin.contains("127.0.0.1")
                && !origin.trim().is_empty()
            {
                return Some(origin.trim_end_matches('/').to_string());
            }
        }
        let protocol = location.protocol().ok()?;
        let hostname = location.hostname().ok()?;

        Self::resolve_from_location(&protocol, &hostname)
    }

    /// Resolves a browser-addressable backend URL for a given protocol and hostname,
    /// validating that the scheme is HTTP or HTTPS and the host is non-empty.
    pub fn resolve_from_location(protocol: &str, hostname: &str) -> Option<String> {
        if protocol != "http:" && protocol != "https:" && protocol != "http" && protocol != "https"
        {
            return None;
        }
        if hostname.trim().is_empty() {
            return None;
        }
        let scheme = protocol.trim_end_matches(':');
        Some(format!("{}://{}:25255", scheme, hostname.trim()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_runtime_uses_a_backend_url() {
        assert!(
            RuntimeConfig::from_environment()
                .backend_url
                .starts_with("http")
        );
    }

    #[test]
    fn remote_host_location_resolves_correctly() {
        assert_eq!(
            RuntimeConfig::resolve_from_location("http:", "remote-host"),
            Some("http://remote-host:25255".to_string())
        );
        assert_eq!(
            RuntimeConfig::resolve_from_location("https", "192.168.1.100"),
            Some("https://192.168.1.100:25255".to_string())
        );
        assert_eq!(
            RuntimeConfig::resolve_from_location("ftp:", "remote-host"),
            None
        );
        assert_eq!(RuntimeConfig::resolve_from_location("http:", "   "), None);
    }
}
