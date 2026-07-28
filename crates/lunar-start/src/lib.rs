//! Lunar Start — launcher library with log-translating backend.

pub mod ansi;
pub mod backend;
pub mod config;
pub mod launcher;

pub use backend::{LogBackend, LogEvent, LogLevel, ServiceRuntime, ServiceStatus};
pub use config::{LauncherConfig, ServiceConfig, ServiceKind};
pub use launcher::Launcher;

pub use config::presets;

pub mod prelude {
    pub use crate::{
        Launcher, LauncherConfig, LogBackend, LogEvent, LogLevel, ServiceConfig, ServiceKind,
        ServiceRuntime, ServiceStatus,
    };
}
