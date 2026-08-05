//! Lunar Start — launcher library with log-translating backend.

pub mod ansi;
pub mod backend;
pub mod config;
pub mod launcher;
pub mod service_settings;
pub mod validation;

pub use backend::{LogBackend, LogEvent, LogLevel, ServiceRuntime, ServiceStatus};
pub use config::{LauncherConfig, ServiceConfig, ServiceKind};
pub use launcher::Launcher;
pub use service_settings::{
    BackendSettings, ComputeBackend, FieldType, FrontendSettings, ServiceConfigField,
    ServiceConfigSchema, ServiceConfigValues, TestbenchBackendSettings,
};
pub use validation::{ValidationResult, service_port, validate_service_config};

pub use config::presets;

pub mod prelude {
    pub use crate::{
        BackendSettings, ComputeBackend, FieldType, FrontendSettings, Launcher, LauncherConfig,
        LogBackend, LogEvent, LogLevel, ServiceConfig, ServiceConfigField, ServiceConfigSchema,
        ServiceConfigValues, ServiceKind, ServiceRuntime, ServiceStatus, TestbenchBackendSettings,
        ValidationResult, service_port, validate_service_config,
    };
}
