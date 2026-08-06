use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::service_settings::{FrontendLaunchConfig, FrontendPlatform};
use crate::{ServiceConfig, ServiceKind};
use lunar_utils::env::{get_lunar_models_dir, get_scenes_dir};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationResult {
    pub ok: bool,
    pub field_errors: HashMap<String, String>,
}

impl ValidationResult {
    fn add(&mut self, field: impl Into<String>, message: impl Into<String>) {
        self.field_errors
            .entry(field.into())
            .or_insert_with(|| message.into());
        self.ok = false;
    }

    pub fn valid() -> Self {
        Self {
            ok: true,
            field_errors: HashMap::new(),
        }
    }
}

fn parse_port_value(value: &str) -> Result<u16, ()> {
    value
        .parse::<u16>()
        .map_err(|_| ())
        .and_then(|port| if port == 0 { Err(()) } else { Ok(port) })
}

fn parse_port(config: &ServiceConfig) -> Option<(&'static str, Result<u16, ()>)> {
    match config.name.as_str() {
        "backend" => Some((
            "LUNAR_BACKEND_PORT",
            config
                .env
                .get("LUNAR_BACKEND_PORT")
                .ok_or(())
                .and_then(|value| parse_port_value(value)),
        )),
        "testbench-backend" => Some((
            "LUNAR_TESTBENCH_BACKEND_PORT",
            config
                .env
                .get("LUNAR_TESTBENCH_BACKEND_PORT")
                .ok_or(())
                .and_then(|value| parse_port_value(value)),
        )),
        "frontend" => FrontendLaunchConfig::from_values(&config.env, &config.extra_args)
            .ok()
            .and_then(|launch| launch.port.map(|port| ("LUNAR_FRONTEND_PORT", Ok(port)))),
        _ => None,
    }
}

pub fn service_port(config: &ServiceConfig) -> Option<u16> {
    parse_port(config).and_then(|(_, value)| value.ok())
}

fn validate_required_host(
    result: &mut ValidationResult,
    config: &ServiceConfig,
    field: &'static str,
) {
    if config
        .env
        .get(field)
        .is_none_or(|value| value.trim().is_empty())
    {
        result.add(field, "Host is required");
    }
}

fn validate_models_dir(result: &mut ValidationResult, config: &ServiceConfig) {
    let path = config
        .env
        .get("LUNAR_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(get_lunar_models_dir);
    if !path.is_dir() {
        result.add("LUNAR_MODELS_DIR", "Models directory does not exist");
    }
}

fn validate_writable_dir(result: &mut ValidationResult, config: &ServiceConfig) {
    let path = config
        .env
        .get("LUNAR_SCENES_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(get_scenes_dir);
    let path = path.as_path();
    if let Err(error) = fs::create_dir_all(path) {
        result.add(
            "LUNAR_SCENES_DIR",
            format!("Cannot create scenes directory: {error}"),
        );
        return;
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or_default();
    let probe = path.join(format!(".lunar-write-probe-{}-{stamp}", std::process::id()));
    match OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(_) => {
            let _ = fs::remove_file(probe);
        }
        Err(error) => result.add(
            "LUNAR_SCENES_DIR",
            format!("Star scenes directory is not writable: {error}"),
        ),
    }
}

fn validate_workspace(result: &mut ValidationResult, config: &ServiceConfig, workspace: &Path) {
    if config.name != "testbench-backend" {
        return;
    }
    let root = config
        .env
        .get("LUNAR_WORKSPACE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.to_path_buf());
    let valid = root.join("Cargo.toml").is_file()
        && root.join("crates").is_dir()
        && root.join("target").is_dir();
    if !valid {
        result.add(
            "LUNAR_WORKSPACE_ROOT",
            "Workspace must contain Cargo.toml, crates, and target",
        );
    }
}

fn validate_binary(result: &mut ValidationResult, config: &ServiceConfig, workspace: &Path) {
    let ServiceKind::Binary { bin_name } = &config.kind else {
        return;
    };
    let mut binary = workspace.join("target").join("release").join(bin_name);
    if cfg!(windows) {
        binary.set_extension("exe");
    }
    if !binary.is_file() {
        result.add(
            "binary",
            format!("Release binary does not exist: {}", binary.display()),
        );
    }
}

fn executable_exists(command: &str) -> bool {
    let path = Path::new(command);
    if path.components().count() > 1 || path.is_absolute() {
        return path.is_file();
    }
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| {
            let candidate = dir.join(command);
            if candidate.is_file() {
                return true;
            }
            cfg!(windows) && dir.join(format!("{command}.exe")).is_file()
        })
    })
}

fn validate_dx(result: &mut ValidationResult, config: &ServiceConfig) {
    if config.name != "frontend" {
        return;
    }
    let dx = config
        .env
        .get("LUNAR_DX_BIN")
        .map(String::as_str)
        .unwrap_or("dx");
    if dx.trim().is_empty() || !executable_exists(dx) {
        result.add("LUNAR_DX_BIN", format!("Dioxus CLI was not found: {dx}"));
    }
}

fn validate_android_prerequisites(result: &mut ValidationResult) {
    let sdk = std::env::var("ANDROID_SDK_ROOT")
        .or_else(|_| std::env::var("ANDROID_HOME"))
        .ok()
        .filter(|value| Path::new(value).is_dir());
    if sdk.is_none() {
        result.add(
            "ANDROID_SDK_ROOT",
            "Android SDK is required for the android frontend platform",
        );
    }

    let ndk = std::env::var("ANDROID_NDK_HOME")
        .ok()
        .filter(|value| Path::new(value).is_dir());
    if ndk.is_none() {
        result.add(
            "ANDROID_NDK_HOME",
            "Android NDK is required for the android frontend platform",
        );
    }

    let adb = std::env::var("ADB").unwrap_or_else(|_| "adb".to_string());
    if adb.trim().is_empty() || !executable_exists(&adb) {
        result.add("ADB", "adb was not found in PATH");
        return;
    }
    let has_device = Command::new(&adb)
        .arg("devices")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .is_some_and(|output| output.lines().any(|line| line.ends_with("\tdevice")));
    if !has_device {
        result.add(
            "ANDROID_DEVICE",
            "No reachable Android device or emulator was reported by adb",
        );
    }
}

fn validate_frontend_launch(result: &mut ValidationResult, config: &ServiceConfig) {
    validate_dx(result, config);
    let launch = match FrontendLaunchConfig::from_values(&config.env, &config.extra_args) {
        Ok(launch) => launch,
        Err(error) => {
            result.add(error.field, error.message);
            return;
        }
    };
    if launch.platform == FrontendPlatform::Android {
        validate_android_prerequisites(result);
    }
}

fn validate_backend_features(result: &mut ValidationResult, config: &ServiceConfig) {
    if config.name != "backend" {
        return;
    }
    let allowed = ["wgpu", "cuda", "cpu", "metal", "rocm", "siren"];
    let mut features = Vec::new();
    for pair in config.build_args.windows(2) {
        if pair[0] == "--features" || pair[0] == "-F" {
            features.extend(
                pair[1]
                    .split(',')
                    .flat_map(str::split_whitespace)
                    .filter(|value| !value.is_empty()),
            );
        }
    }
    if let Some(feature) = features.iter().find(|feature| !allowed.contains(feature)) {
        result.add(
            "COMPUTE_BACKEND",
            format!("Unsupported backend feature: {feature}"),
        );
        return;
    }
    let compute_count = features
        .iter()
        .filter(|feature| ["wgpu", "cuda", "cpu", "metal", "rocm"].contains(feature))
        .count();
    if compute_count > 1 {
        result.add(
            "COMPUTE_BACKEND",
            "Exactly one compute backend can be selected",
        );
    }
}

pub fn validate_service_config(
    config: &ServiceConfig,
    workspace: &Path,
    other_configs: &[ServiceConfig],
) -> ValidationResult {
    let mut result = ValidationResult::valid();

    if let Some((field, port)) = parse_port(config) {
        match port {
            Ok(port) => {
                if other_configs
                    .iter()
                    .filter(|other| other.name != config.name)
                    .any(|other| service_port(other) == Some(port))
                {
                    result.add(
                        field,
                        format!("Port {port} is already used by another service"),
                    );
                }
            }
            Err(()) => result.add(field, "Port must be an integer from 1 to 65535"),
        }
    }

    match config.name.as_str() {
        "backend" => {
            validate_required_host(&mut result, config, "LUNAR_BACKEND_HOST");
            validate_models_dir(&mut result, config);
            validate_writable_dir(&mut result, config);
            validate_backend_features(&mut result, config);
        }
        "testbench-backend" => {
            validate_required_host(&mut result, config, "LUNAR_TESTBENCH_HOST");
            validate_workspace(&mut result, config, workspace);
        }
        "frontend" => validate_frontend_launch(&mut result, config),
        _ => result.add("service", "Unsupported configurable service"),
    }

    validate_binary(&mut result, config, workspace);
    result.ok = result.field_errors.is_empty();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ServiceConfig;

    fn temp_workspace() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("lunar-validation-{}-{stamp}", std::process::id()));
        fs::create_dir_all(root.join("crates")).unwrap();
        fs::create_dir_all(root.join("target/release")).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        root
    }

    fn add_binary(workspace: &Path, name: &str) {
        fs::write(workspace.join("target/release").join(name), b"test").unwrap();
    }

    #[test]
    fn rejects_required_host_invalid_port_and_port_conflict() {
        let workspace = temp_workspace();
        add_binary(&workspace, "lunar-backend");
        let mut backend = ServiceConfig::backend();
        let models = workspace.join("models");
        fs::create_dir_all(&models).unwrap();
        backend
            .env
            .insert("LUNAR_MODELS_DIR".into(), models.display().to_string());
        backend.env.insert("LUNAR_BACKEND_HOST".into(), " ".into());
        backend
            .env
            .insert("LUNAR_BACKEND_PORT".into(), "25256".into());
        let peer = ServiceConfig::testbench_backend();

        let result = validate_service_config(&backend, &workspace, &[peer]);
        assert!(!result.ok);
        assert!(result.field_errors.contains_key("LUNAR_BACKEND_HOST"));
        assert!(result.field_errors.contains_key("LUNAR_BACKEND_PORT"));
        backend.env.insert("LUNAR_BACKEND_PORT".into(), "0".into());
        let zero = validate_service_config(&backend, &workspace, &[]);
        assert!(zero.field_errors.contains_key("LUNAR_BACKEND_PORT"));
        fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn validates_backend_paths_features_and_binary() {
        let workspace = temp_workspace();
        let models = workspace.join("models");
        let scenes = workspace.join("scenes");
        fs::create_dir_all(&models).unwrap();
        add_binary(&workspace, "lunar-backend");
        let mut backend = ServiceConfig::backend();
        backend
            .env
            .insert("LUNAR_MODELS_DIR".into(), models.display().to_string());
        backend
            .env
            .insert("LUNAR_SCENES_DIR".into(), scenes.display().to_string());

        let result = validate_service_config(&backend, &workspace, &[]);
        assert_eq!(result, ValidationResult::valid());
        assert!(scenes.is_dir());
        fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn reports_missing_release_binary_and_dx() {
        let workspace = temp_workspace();
        let mut backend = ServiceConfig::backend();
        let models = workspace.join("models");
        fs::create_dir_all(&models).unwrap();
        backend
            .env
            .insert("LUNAR_MODELS_DIR".into(), models.display().to_string());
        let mut frontend = ServiceConfig::frontend();
        frontend.env.insert(
            "LUNAR_DX_BIN".into(),
            workspace.join("missing-dx").display().to_string(),
        );

        let backend_result = validate_service_config(&backend, &workspace, &[]);
        let frontend_result = validate_service_config(&frontend, &workspace, &[]);
        assert!(backend_result.field_errors.contains_key("binary"));
        assert!(frontend_result.field_errors.contains_key("LUNAR_DX_BIN"));
        fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn reports_invalid_model_scenes_and_workspace_paths() {
        let workspace = temp_workspace();
        add_binary(&workspace, "lunar-backend");
        add_binary(&workspace, "lunar-testbench-backend");

        let mut backend = ServiceConfig::backend();
        let missing_models = workspace.join("missing-models");
        let scenes_file = workspace.join("scenes-file");
        fs::write(&scenes_file, b"not a directory").unwrap();
        backend.env.insert(
            "LUNAR_MODELS_DIR".into(),
            missing_models.display().to_string(),
        );
        backend
            .env
            .insert("LUNAR_SCENES_DIR".into(), scenes_file.display().to_string());
        let backend_result = validate_service_config(&backend, &workspace, &[]);
        assert!(backend_result.field_errors.contains_key("LUNAR_MODELS_DIR"));
        assert!(backend_result.field_errors.contains_key("LUNAR_SCENES_DIR"));

        let mut testbench = ServiceConfig::testbench_backend();
        testbench.env.insert(
            "LUNAR_WORKSPACE_ROOT".into(),
            workspace.join("missing-workspace").display().to_string(),
        );
        let testbench_result = validate_service_config(&testbench, &workspace, &[]);
        assert!(
            testbench_result
                .field_errors
                .contains_key("LUNAR_WORKSPACE_ROOT")
        );
        fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn rejects_unknown_or_multiple_compute_features() {
        let workspace = temp_workspace();
        add_binary(&workspace, "lunar-backend");
        let mut backend = ServiceConfig::backend();
        let models = workspace.join("models");
        fs::create_dir_all(&models).unwrap();
        backend
            .env
            .insert("LUNAR_MODELS_DIR".into(), models.display().to_string());
        backend.build_args = vec!["--features".into(), "cuda,rocm".into()];
        let multiple = validate_service_config(&backend, &workspace, &[]);
        assert!(multiple.field_errors.contains_key("COMPUTE_BACKEND"));

        backend.build_args = vec!["--features".into(), "quantum".into()];
        let unknown = validate_service_config(&backend, &workspace, &[]);
        assert!(unknown.field_errors.contains_key("COMPUTE_BACKEND"));
        fs::remove_dir_all(workspace).unwrap();
    }
}
