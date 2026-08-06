use std::collections::HashMap;

use lunar_utils::env::{
    DEFAULT_BACKEND_HOST, DEFAULT_BACKEND_PORT, DEFAULT_TESTBENCH_HOST, DEFAULT_TESTBENCH_PORT,
};
use serde::{Deserialize, Serialize};

/// Supported schema field input types for UI rendering and validation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    String,
    Port,
    Path,
    Boolean,
    Select { options: Vec<String> },
    StringList,
}

/// Metadata and validation rules for an individual configuration field.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ServiceConfigField {
    pub key: String,
    pub label: String,
    pub description: Option<String>,
    pub field_type: FieldType,
    pub default_value: String,
    /// Build parameters are applied while compiling; other editable fields are runtime values.
    pub is_build_param: bool,
    pub required: bool,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub allowed_values: Option<Vec<String>>,
    pub read_only: bool,
}

/// Complete schema defining configuration options available for a service.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ServiceConfigSchema {
    pub service: String,
    pub fields: Vec<ServiceConfigField>,
}

/// Values that can be applied to a service process/build.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct ServiceConfigValues {
    pub env: HashMap<String, String>,
    pub extra_args: Vec<String>,
    pub build_args: Vec<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ComputeBackend {
    #[default]
    Wgpu,
    Cuda,
    Cpu,
    Metal,
    Rocm,
}

impl ComputeBackend {
    pub fn feature_name(self) -> &'static str {
        match self {
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
        }
    }
}

fn field(
    key: &str,
    label: &str,
    description: &str,
    field_type: FieldType,
    default_value: impl Into<String>,
) -> ServiceConfigField {
    ServiceConfigField {
        key: key.to_string(),
        label: label.to_string(),
        description: Some(description.to_string()),
        field_type,
        default_value: default_value.into(),
        is_build_param: false,
        required: false,
        min: None,
        max: None,
        allowed_values: None,
        read_only: false,
    }
}

fn optional_env(env: &mut HashMap<String, String>, key: &str, value: &Option<String>) {
    if let Some(value) = value.as_deref().filter(|value| !value.trim().is_empty()) {
        env.insert(key.to_string(), value.to_string());
    }
}

/// Typed settings for `lunar-backend`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BackendSettings {
    pub host: String,
    pub port: u16,
    /// `None` lets lunar-utils resolve the platform data directory.
    pub models_dir: Option<String>,
    /// `None` lets lunar-utils resolve the platform data directory.
    pub scenes_dir: Option<String>,
    pub env_mode: Option<String>,
    pub compute_backend: ComputeBackend,
    pub siren: bool,
    pub extra_args: Vec<String>,
    pub build_args: Vec<String>,
}

impl Default for BackendSettings {
    fn default() -> Self {
        Self {
            host: DEFAULT_BACKEND_HOST.to_string(),
            port: DEFAULT_BACKEND_PORT,
            models_dir: None,
            scenes_dir: None,
            env_mode: None,
            compute_backend: ComputeBackend::Wgpu,
            siren: true,
            extra_args: Vec::new(),
            build_args: Vec::new(),
        }
    }
}

impl BackendSettings {
    pub fn schema() -> ServiceConfigSchema {
        let defaults = Self::default();
        let mut host = field(
            "LUNAR_BACKEND_HOST",
            "Host",
            "Address used by lunar-backend when binding its HTTP server.",
            FieldType::String,
            defaults.host,
        );
        host.required = true;

        let mut port = field(
            "LUNAR_BACKEND_PORT",
            "Port",
            "HTTP port used by lunar-backend.",
            FieldType::Port,
            defaults.port.to_string(),
        );
        port.required = true;
        port.min = Some(1.0);
        port.max = Some(65535.0);

        let models_dir = field(
            "LUNAR_MODELS_DIR",
            "Models directory",
            "Leave empty to use the platform data directory: lunar/models.",
            FieldType::Path,
            "",
        );
        let scenes_dir = field(
            "LUNAR_SCENES_DIR",
            "Star scenes directory",
            "Leave empty to use the platform data directory: lunar/scenes.",
            FieldType::Path,
            "",
        );
        let env_mode = field(
            "LUNAR_ENV",
            "Environment",
            "Optional environment name such as development or production.",
            FieldType::String,
            "",
        );

        let compute_options = vec![
            "wgpu".to_string(),
            "cuda".to_string(),
            "cpu".to_string(),
            "metal".to_string(),
            "rocm".to_string(),
        ];
        let mut compute_backend = field(
            "COMPUTE_BACKEND",
            "Compute backend",
            "Cargo feature used to compile lunar-backend.",
            FieldType::Select {
                options: compute_options.clone(),
            },
            defaults.compute_backend.feature_name(),
        );
        compute_backend.required = true;
        compute_backend.allowed_values = Some(compute_options);
        compute_backend.is_build_param = true;

        let mut siren = field(
            "SIREN",
            "SIREN",
            "Compile lunar-backend with the siren feature.",
            FieldType::Boolean,
            defaults.siren.to_string(),
        );
        siren.is_build_param = true;

        let extra_args = field(
            "EXTRA_ARGS",
            "Additional process arguments",
            "Advanced arguments passed to lunar-backend.",
            FieldType::StringList,
            "",
        );
        let mut build_args = field(
            "BUILD_ARGS",
            "Additional build arguments",
            "Advanced arguments passed to cargo build.",
            FieldType::StringList,
            "",
        );
        build_args.is_build_param = true;

        ServiceConfigSchema {
            service: "backend".to_string(),
            fields: vec![
                host,
                port,
                models_dir,
                scenes_dir,
                env_mode,
                compute_backend,
                siren,
                extra_args,
                build_args,
            ],
        }
    }

    pub fn to_values(&self) -> ServiceConfigValues {
        let mut env = HashMap::new();
        env.insert("LUNAR_BACKEND_HOST".to_string(), self.host.clone());
        env.insert("LUNAR_BACKEND_PORT".to_string(), self.port.to_string());
        optional_env(&mut env, "LUNAR_MODELS_DIR", &self.models_dir);
        optional_env(&mut env, "LUNAR_SCENES_DIR", &self.scenes_dir);
        optional_env(&mut env, "LUNAR_ENV", &self.env_mode);

        let mut build_args = Vec::new();
        // lunar-backend defaults to `wgpu,siren`; emit feature flags only when that changes.
        if self.compute_backend != ComputeBackend::Wgpu || !self.siren {
            let mut features = vec![self.compute_backend.feature_name()];
            if self.siren {
                features.push("siren");
            }
            build_args.extend([
                "--no-default-features".to_string(),
                "--features".to_string(),
                features.join(","),
            ]);
        }
        build_args.extend(self.build_args.clone());

        ServiceConfigValues {
            env,
            extra_args: self.extra_args.clone(),
            build_args,
        }
    }
}

/// Typed settings for `lunar-testbench-backend`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TestbenchBackendSettings {
    /// The process currently binds to this fixed address; expose it read-only until the binary supports it.
    pub bind_host: String,
    pub port: u16,
    pub client_host: String,
    pub workspace_root: Option<String>,
    pub env_mode: Option<String>,
    pub extra_args: Vec<String>,
    pub build_args: Vec<String>,
}

impl Default for TestbenchBackendSettings {
    fn default() -> Self {
        Self {
            bind_host: DEFAULT_TESTBENCH_HOST.to_string(),
            port: DEFAULT_TESTBENCH_PORT,
            client_host: DEFAULT_TESTBENCH_HOST.to_string(),
            workspace_root: None,
            env_mode: None,
            extra_args: Vec::new(),
            build_args: Vec::new(),
        }
    }
}

impl TestbenchBackendSettings {
    pub fn schema() -> ServiceConfigSchema {
        let defaults = Self::default();
        let mut bind_host = field(
            "BIND_HOST",
            "Process host",
            "Currently fixed by lunar-testbench-backend.",
            FieldType::String,
            defaults.bind_host,
        );
        bind_host.read_only = true;

        let mut port = field(
            "LUNAR_TESTBENCH_BACKEND_PORT",
            "Port",
            "Process port; the launcher also mirrors it to LUNAR_TESTBENCH_PORT.",
            FieldType::Port,
            defaults.port.to_string(),
        );
        port.required = true;
        port.min = Some(1.0);
        port.max = Some(65535.0);

        let mut client_host = field(
            "LUNAR_TESTBENCH_HOST",
            "Client host",
            "Host used by lunar-testbench when addressing the job API.",
            FieldType::String,
            defaults.client_host,
        );
        client_host.required = true;

        let workspace_root = field(
            "LUNAR_WORKSPACE_ROOT",
            "Workspace root",
            "Leave empty to discover the Lunar workspace automatically.",
            FieldType::Path,
            "",
        );
        let env_mode = field(
            "LUNAR_ENV",
            "Environment",
            "Optional environment name such as development or production.",
            FieldType::String,
            "",
        );
        let extra_args = field(
            "EXTRA_ARGS",
            "Additional process arguments",
            "Advanced arguments passed to lunar-testbench-backend.",
            FieldType::StringList,
            "",
        );
        let mut build_args = field(
            "BUILD_ARGS",
            "Additional build arguments",
            "Advanced arguments passed to cargo build.",
            FieldType::StringList,
            "",
        );
        build_args.is_build_param = true;

        ServiceConfigSchema {
            service: "testbench-backend".to_string(),
            fields: vec![
                bind_host,
                port,
                client_host,
                workspace_root,
                env_mode,
                extra_args,
                build_args,
            ],
        }
    }

    pub fn to_values(&self) -> ServiceConfigValues {
        let mut env = HashMap::new();
        env.insert(
            "LUNAR_TESTBENCH_BACKEND_PORT".to_string(),
            self.port.to_string(),
        );
        env.insert("LUNAR_TESTBENCH_PORT".to_string(), self.port.to_string());
        env.insert("LUNAR_TESTBENCH_HOST".to_string(), self.client_host.clone());
        optional_env(&mut env, "LUNAR_WORKSPACE_ROOT", &self.workspace_root);
        optional_env(&mut env, "LUNAR_ENV", &self.env_mode);

        ServiceConfigValues {
            env,
            extra_args: self.extra_args.clone(),
            build_args: self.build_args.clone(),
        }
    }
}

/// The single supported launch target for the managed frontend service.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FrontendPlatform {
    #[default]
    Web,
    Desktop,
    Android,
}

impl FrontendPlatform {
    pub const ALL: [&'static str; 3] = ["web", "desktop", "android"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::Desktop => "desktop",
            Self::Android => "android",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "web" => Some(Self::Web),
            "desktop" => Some(Self::Desktop),
            "android" => Some(Self::Android),
            _ => None,
        }
    }

    pub fn default_backend_url(self) -> String {
        match self {
            Self::Web | Self::Desktop => {
                format!("http://{DEFAULT_BACKEND_HOST}:{DEFAULT_BACKEND_PORT}")
            }
            // Android emulators resolve host-loopback through this address.
            Self::Android => format!("http://10.0.2.2:{DEFAULT_BACKEND_PORT}"),
        }
    }
}

pub const DEFAULT_FRONTEND_PORT: u16 = 8080;

/// Structured frontend launch data. It is the only code path that translates
/// configured platform, port, and custom arguments into `dx serve` arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontendLaunchConfig {
    pub platform: FrontendPlatform,
    pub port: Option<u16>,
    pub extra_args: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontendLaunchError {
    pub field: &'static str,
    pub message: String,
}

impl FrontendLaunchConfig {
    pub fn from_values(
        env: &HashMap<String, String>,
        extra_args: &[String],
    ) -> Result<Self, FrontendLaunchError> {
        let raw_platform = env
            .get("LUNAR_FRONTEND_PLATFORM")
            .map(String::as_str)
            .unwrap_or("web");
        let platform = FrontendPlatform::parse(raw_platform).ok_or_else(|| FrontendLaunchError {
            field: "LUNAR_FRONTEND_PLATFORM",
            message: format!("Unsupported frontend platform: {raw_platform}"),
        })?;

        if let Some(arg) = extra_args.iter().find(|arg| {
            matches!(arg.as_str(), "--platform" | "--port" | "-p")
                || arg.starts_with("--platform=")
                || arg.starts_with("--port=")
        }) {
            return Err(FrontendLaunchError {
                field: "EXTRA_ARGS",
                message: format!("{arg} is managed by the platform and web-port fields"),
            });
        }

        let port = match platform {
            FrontendPlatform::Web => {
                let raw_port = env
                    .get("LUNAR_FRONTEND_PORT")
                    .ok_or_else(|| FrontendLaunchError {
                        field: "LUNAR_FRONTEND_PORT",
                        message: "A web frontend requires a port".to_string(),
                    })?;
                let port = raw_port.parse::<u16>().ok().filter(|port| *port != 0).ok_or_else(|| {
                    FrontendLaunchError {
                        field: "LUNAR_FRONTEND_PORT",
                        message: "Web port must be an integer from 1 to 65535".to_string(),
                    }
                })?;
                Some(port)
            }
            FrontendPlatform::Desktop | FrontendPlatform::Android => {
                if env.contains_key("LUNAR_FRONTEND_PORT") {
                    return Err(FrontendLaunchError {
                        field: "LUNAR_FRONTEND_PORT",
                        message: "Port is only configured for the web platform".to_string(),
                    });
                }
                None
            }
        };

        Ok(Self {
            platform,
            port,
            extra_args: extra_args.to_vec(),
        })
    }

    pub fn dx_args(&self) -> Vec<String> {
        let mut args = vec!["--platform".to_string(), self.platform.as_str().to_string()];
        if let Some(port) = self.port {
            args.extend(["--port".to_string(), port.to_string()]);
        }
        args.extend(self.extra_args.clone());
        args
    }

    pub fn public_url(&self) -> Option<String> {
        self.port.map(|port| format!("http://127.0.0.1:{port}"))
    }
}

/// Typed settings for the single `lunar-frontend` service.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrontendSettings {
    pub dx_bin: String,
    pub platform: FrontendPlatform,
    pub port: u16,
    /// An explicit host URL is required for a physical Android device; the
    /// Android emulator default is supplied by `FrontendPlatform`.
    pub backend_url: Option<String>,
    pub env_mode: Option<String>,
    pub extra_args: Vec<String>,
    pub build_args: Vec<String>,
}

impl Default for FrontendSettings {
    fn default() -> Self {
        Self {
            dx_bin: "dx".to_string(),
            platform: FrontendPlatform::Web,
            port: DEFAULT_FRONTEND_PORT,
            backend_url: None,
            env_mode: None,
            extra_args: Vec::new(),
            build_args: Vec::new(),
        }
    }
}

impl FrontendSettings {
    pub fn schema() -> ServiceConfigSchema {
        let defaults = Self::default();
        let mut dx_bin = field(
            "LUNAR_DX_BIN",
            "Dioxus CLI",
            "Executable name or path used to run dx serve.",
            FieldType::Path,
            defaults.dx_bin,
        );
        dx_bin.required = true;

        let platform_options = FrontendPlatform::ALL
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        let mut platform = field(
            "LUNAR_FRONTEND_PLATFORM",
            "Platform",
            "Web publishes a URL; desktop and Android run as native Dioxus targets.",
            FieldType::Select {
                options: platform_options.clone(),
            },
            defaults.platform.as_str(),
        );
        platform.required = true;
        platform.allowed_values = Some(platform_options);

        let mut port = field(
            "LUNAR_FRONTEND_PORT",
            "Web port",
            "Used and validated only when platform is web.",
            FieldType::Port,
            defaults.port.to_string(),
        );
        port.min = Some(1.0);
        port.max = Some(65535.0);

        let backend_url = field(
            "LUNAR_BACKEND_URL",
            "Backend URL",
            "Override the backend URL. Android defaults to 10.0.2.2; use a reachable host URL for a physical device.",
            FieldType::String,
            defaults.platform.default_backend_url(),
        );

        let mut crate_name = field(
            "CRATE",
            "Crate",
            "Managed Dioxus crate name.",
            FieldType::String,
            "lunar-frontend",
        );
        crate_name.read_only = true;
        let mut crate_subdir = field(
            "CRATE_SUBDIR",
            "Working subdirectory",
            "Workspace-relative directory containing the managed frontend crate.",
            FieldType::Path,
            "crates/lunar-frontend",
        );
        crate_subdir.read_only = true;

        let env_mode = field(
            "LUNAR_ENV",
            "Environment",
            "Optional environment name such as development or production.",
            FieldType::String,
            "",
        );
        let extra_args = field(
            "EXTRA_ARGS",
            "Additional Dioxus arguments",
            "Advanced arguments. Platform and web port are managed above.",
            FieldType::StringList,
            "",
        );
        let mut build_args = field(
            "BUILD_ARGS",
            "Additional build arguments",
            "Advanced build arguments retained for the service configuration.",
            FieldType::StringList,
            "",
        );
        build_args.is_build_param = true;

        ServiceConfigSchema {
            service: "frontend".to_string(),
            fields: vec![
                dx_bin,
                platform,
                port,
                backend_url,
                crate_name,
                crate_subdir,
                env_mode,
                extra_args,
                build_args,
            ],
        }
    }

    pub fn to_values(&self) -> ServiceConfigValues {
        let mut env = HashMap::new();
        env.insert("LUNAR_DX_BIN".to_string(), self.dx_bin.clone());
        env.insert(
            "LUNAR_FRONTEND_PLATFORM".to_string(),
            self.platform.as_str().to_string(),
        );
        env.insert(
            "LUNAR_BACKEND_URL".to_string(),
            self.backend_url
                .clone()
                .unwrap_or_else(|| self.platform.default_backend_url()),
        );
        if self.platform == FrontendPlatform::Web {
            env.insert("LUNAR_FRONTEND_PORT".to_string(), self.port.to_string());
        }
        optional_env(&mut env, "LUNAR_ENV", &self.env_mode);

        ServiceConfigValues {
            env,
            extra_args: self.extra_args.clone(),
            build_args: self.build_args.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_defaults_match_runtime() {
        let settings = BackendSettings::default();
        let values = settings.to_values();

        assert_eq!(settings.port, 25255);
        assert_eq!(values.env["LUNAR_BACKEND_HOST"], "127.0.0.1");
        assert_eq!(values.env["LUNAR_BACKEND_PORT"], "25255");
        assert!(!values.env.contains_key("DATABASE_URL"));
        assert!(!values.env.contains_key("HOST"));
        assert!(values.build_args.is_empty());
    }

    #[test]
    fn backend_non_default_features_generate_build_arguments() {
        let settings = BackendSettings {
            compute_backend: ComputeBackend::Cuda,
            siren: true,
            ..BackendSettings::default()
        };
        assert_eq!(
            settings.to_values().build_args,
            ["--no-default-features", "--features", "cuda,siren"]
        );
    }

    #[test]
    fn testbench_defaults_match_runtime_and_mirror_client_port() {
        let settings = TestbenchBackendSettings::default();
        let values = settings.to_values();

        assert_eq!(settings.port, 25256);
        assert_eq!(values.env["LUNAR_TESTBENCH_BACKEND_PORT"], "25256");
        assert_eq!(values.env["LUNAR_TESTBENCH_PORT"], "25256");
        assert!(!values.env.contains_key("MOCK_MODE"));
    }

    #[test]
    fn frontend_defaults_are_web_and_use_structured_runtime_values() {
        let settings = FrontendSettings::default();
        let values = settings.to_values();

        assert_eq!(values.env["LUNAR_DX_BIN"], "dx");
        assert_eq!(values.env["LUNAR_FRONTEND_PLATFORM"], "web");
        assert_eq!(values.env["LUNAR_FRONTEND_PORT"], "8080");
        assert_eq!(values.env["LUNAR_BACKEND_URL"], "http://127.0.0.1:25255");
        assert!(values.extra_args.is_empty());
    }

    #[test]
    fn frontend_launch_arguments_are_platform_specific() {
        let web = FrontendLaunchConfig::from_values(&FrontendSettings::default().to_values().env, &[]).unwrap();
        assert_eq!(web.dx_args(), ["--platform", "web", "--port", "8080"]);
        assert_eq!(web.public_url().as_deref(), Some("http://127.0.0.1:8080"));

        let desktop_values = FrontendSettings { platform: FrontendPlatform::Desktop, ..FrontendSettings::default() }.to_values();
        let desktop = FrontendLaunchConfig::from_values(&desktop_values.env, &desktop_values.extra_args).unwrap();
        assert_eq!(desktop.dx_args(), ["--platform", "desktop"]);
        assert_eq!(desktop.public_url(), None);

        let android_values = FrontendSettings { platform: FrontendPlatform::Android, ..FrontendSettings::default() }.to_values();
        assert_eq!(android_values.env["LUNAR_BACKEND_URL"], "http://10.0.2.2:25255");
    }

    fn schema_keys(schema: &ServiceConfigSchema) -> Vec<&str> {
        schema
            .fields
            .iter()
            .map(|field| field.key.as_str())
            .collect()
    }

    #[test]
    fn backend_schema_contains_the_complete_supported_parameter_set() {
        let schema = BackendSettings::schema();
        assert_eq!(
            schema_keys(&schema),
            [
                "LUNAR_BACKEND_HOST",
                "LUNAR_BACKEND_PORT",
                "LUNAR_MODELS_DIR",
                "LUNAR_SCENES_DIR",
                "LUNAR_ENV",
                "COMPUTE_BACKEND",
                "SIREN",
                "EXTRA_ARGS",
                "BUILD_ARGS",
            ]
        );
        assert_eq!(
            schema.fields[1].default_value,
            DEFAULT_BACKEND_PORT.to_string()
        );
    }

    #[test]
    fn testbench_schema_contains_the_complete_supported_parameter_set() {
        let schema = TestbenchBackendSettings::schema();
        assert_eq!(
            schema_keys(&schema),
            [
                "BIND_HOST",
                "LUNAR_TESTBENCH_BACKEND_PORT",
                "LUNAR_TESTBENCH_HOST",
                "LUNAR_WORKSPACE_ROOT",
                "LUNAR_ENV",
                "EXTRA_ARGS",
                "BUILD_ARGS",
            ]
        );
        assert!(schema.fields[0].read_only);
        assert_eq!(
            schema.fields[1].default_value,
            DEFAULT_TESTBENCH_PORT.to_string()
        );
    }

    #[test]
    fn frontend_schema_contains_the_complete_platform_parameter_set() {
        let schema = FrontendSettings::schema();
        assert_eq!(
            schema_keys(&schema),
            [
                "LUNAR_DX_BIN",
                "LUNAR_FRONTEND_PLATFORM",
                "LUNAR_FRONTEND_PORT",
                "LUNAR_BACKEND_URL",
                "CRATE",
                "CRATE_SUBDIR",
                "LUNAR_ENV",
                "EXTRA_ARGS",
                "BUILD_ARGS",
            ]
        );
        assert_eq!(schema.fields[1].allowed_values.as_ref().unwrap(), &vec!["web", "desktop", "android"]);
        assert!(schema.fields[4].read_only);
        assert!(schema.fields[5].read_only);
        assert!(!schema.fields[1].read_only);
    }
}
