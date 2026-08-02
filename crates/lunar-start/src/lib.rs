//! Lunar Start — launcher library with log-translating backend.

pub mod ansi;
pub mod backend;
pub mod config;
pub mod launcher;
pub mod service_settings;

pub use backend::{LogBackend, LogEvent, LogLevel, ServiceRuntime, ServiceStatus};
pub use config::{LauncherConfig, ServiceConfig, ServiceKind};
pub use launcher::Launcher;
pub use service_settings::{
    BackendSettings, ComputeBackend, FieldType, FrontendSettings, ServiceConfigField,
    ServiceConfigSchema, ServiceConfigValues, TestbenchBackendSettings,
};

pub use config::presets;

pub mod prelude {
    pub use crate::{
        Launcher, LauncherConfig, LogBackend, LogEvent, LogLevel, ServiceConfig, ServiceKind,
        ServiceRuntime, ServiceStatus, ServiceConfigField, ServiceConfigSchema,
        ServiceConfigValues, BackendSettings, ComputeBackend, FieldType, FrontendSettings,
        TestbenchBackendSettings,
    };
}
