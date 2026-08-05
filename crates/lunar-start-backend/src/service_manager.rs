use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use lunar_start::{
    BackendSettings, FrontendSettings, LauncherConfig, LogBackend, LogEvent, ServiceConfig,
    ServiceConfigSchema, ServiceConfigValues, ServiceRuntime, ServiceStatus,
    TestbenchBackendSettings, ValidationResult, validate_service_config,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, broadcast, mpsc};

/// Max number of buffered log lines retained per service for the `/services/{name}/logs`
/// and `/services/{name}/stats` endpoints.
const MAX_LOGS_PER_SERVICE: usize = 2000;

pub type LogRingBuffers = Arc<Mutex<HashMap<String, VecDeque<LogEvent>>>>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ServiceConfigState {
    pub defaults: ServiceConfigValues,
    pub saved: ServiceConfigValues,
    pub effective: Option<ServiceConfigValues>,
}

#[derive(Clone, Debug)]
struct ManagedConfig {
    base: ServiceConfig,
    state: ServiceConfigState,
}

impl ManagedConfig {
    fn new(config: ServiceConfig) -> Self {
        let defaults = values_from_config(&config);
        Self {
            base: config,
            state: ServiceConfigState {
                defaults: defaults.clone(),
                saved: defaults,
                effective: None,
            },
        }
    }

    fn with_values(&self, values: ServiceConfigValues) -> ServiceConfig {
        let mut config = self.base.clone();
        config.apply_values(values);
        config
    }

    fn saved_config(&self) -> ServiceConfig {
        self.with_values(self.state.saved.clone())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    UnknownService(String),
    StateConflict(String),
    Validation(ValidationResult),
    Runtime(String),
}

fn values_from_config(config: &ServiceConfig) -> ServiceConfigValues {
    ServiceConfigValues {
        env: config.env.clone(),
        extra_args: config.extra_args.clone(),
        build_args: config.build_args.clone(),
    }
}

fn status_allows_edit(status: &ServiceStatus) -> bool {
    matches!(
        status,
        ServiceStatus::Stopped { .. } | ServiceStatus::Failed { .. }
    )
}

#[derive(Clone)]
pub struct ServiceManager {
    workspace: PathBuf,
    global_env: HashMap<String, String>,
    log_tx: broadcast::Sender<LogEvent>,
    backend: Arc<Mutex<LogBackend>>,
    configs: Arc<Mutex<HashMap<String, ManagedConfig>>>,
    operations: Arc<Mutex<()>>,
    logs: LogRingBuffers,
}

impl ServiceManager {
    pub fn new(config: &LauncherConfig) -> Result<Self> {
        let (mpsc_tx, mut mpsc_rx) = mpsc::channel::<LogEvent>(500);
        let (bcast_tx, _) = broadcast::channel::<LogEvent>(500);
        let logs: LogRingBuffers = Arc::new(Mutex::new(HashMap::new()));

        let bcast_tx_clone = bcast_tx.clone();
        let logs_clone = Arc::clone(&logs);
        tokio::spawn(async move {
            while let Some(log) = mpsc_rx.recv().await {
                {
                    let mut guard = logs_clone.lock().await;
                    let buf = guard
                        .entry(log.service.clone())
                        .or_insert_with(VecDeque::new);
                    buf.push_back(log.clone());
                    if buf.len() > MAX_LOGS_PER_SERVICE {
                        buf.pop_front();
                    }
                }
                let _ = bcast_tx_clone.send(log);
            }
        });

        let backend = Arc::new(Mutex::new(LogBackend::new(config, mpsc_tx)));
        let configs = Arc::new(Mutex::new(
            config
                .services
                .iter()
                .cloned()
                .map(|service| (service.name.clone(), ManagedConfig::new(service)))
                .collect::<HashMap<_, _>>(),
        ));

        let backend_clone = Arc::clone(&backend);
        let configs_clone = Arc::clone(&configs);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            loop {
                interval.tick().await;
                let inactive = {
                    let mut backend = backend_clone.lock().await;
                    backend.poll();
                    backend
                        .services()
                        .iter()
                        .filter(|runtime| {
                            matches!(
                                runtime.status,
                                ServiceStatus::Stopped { .. } | ServiceStatus::Failed { .. }
                            )
                        })
                        .map(|runtime| runtime.config.name.clone())
                        .collect::<Vec<_>>()
                };
                if !inactive.is_empty() {
                    let mut configs = configs_clone.lock().await;
                    for name in inactive {
                        if let Some(managed) = configs.get_mut(&name) {
                            managed.state.effective = None;
                        }
                    }
                }
            }
        });

        Ok(Self {
            workspace: config.workspace.clone(),
            global_env: config.env.clone(),
            log_tx: bcast_tx,
            backend,
            configs,
            operations: Arc::new(Mutex::new(())),
            logs,
        })
    }

    pub fn log_sender(&self) -> broadcast::Sender<LogEvent> {
        self.log_tx.clone()
    }

    pub fn logs(&self) -> LogRingBuffers {
        Arc::clone(&self.logs)
    }

    pub async fn services(&self) -> Vec<ServiceRuntime> {
        self.backend.lock().await.services().to_vec()
    }

    pub fn config_schema(&self, name: &str) -> Result<ServiceConfigSchema, ConfigError> {
        match name {
            "backend" => Ok(BackendSettings::schema()),
            "testbench-backend" => Ok(TestbenchBackendSettings::schema()),
            "frontend" => Ok(FrontendSettings::schema()),
            _ => Err(ConfigError::UnknownService(name.to_string())),
        }
    }

    pub async fn config_state(&self, name: &str) -> Result<ServiceConfigState, ConfigError> {
        self.configs
            .lock()
            .await
            .get(name)
            .map(|managed| managed.state.clone())
            .ok_or_else(|| ConfigError::UnknownService(name.to_string()))
    }

    async fn candidate_and_peers(
        &self,
        name: &str,
        values: ServiceConfigValues,
    ) -> Result<(ServiceConfig, Vec<ServiceConfig>), ConfigError> {
        let configs = self.configs.lock().await;
        let managed = configs
            .get(name)
            .ok_or_else(|| ConfigError::UnknownService(name.to_string()))?;
        let candidate = managed.with_values(values);
        let peers = configs
            .iter()
            .filter(|(other_name, _)| other_name.as_str() != name)
            .map(|(_, other)| other.saved_config())
            .collect();
        Ok((candidate, peers))
    }

    pub async fn validate(
        &self,
        name: &str,
        values: ServiceConfigValues,
    ) -> Result<ValidationResult, ConfigError> {
        let (candidate, peers) = self.candidate_and_peers(name, values).await?;
        let mut candidate_for_validation = candidate.clone();
        candidate_for_validation.env = candidate.effective_env(&self.global_env);
        let peers_for_validation = peers
            .into_iter()
            .map(|mut peer| {
                peer.env = peer.effective_env(&self.global_env);
                peer
            })
            .collect::<Vec<_>>();
        Ok(validate_service_config(
            &candidate_for_validation,
            &self.workspace,
            &peers_for_validation,
        ))
    }

    async fn ensure_editable(&self, name: &str) -> Result<(), ConfigError> {
        let backend = self.backend.lock().await;
        let runtime = backend
            .service(name)
            .ok_or_else(|| ConfigError::UnknownService(name.to_string()))?;
        if !status_allows_edit(&runtime.status) || backend.is_running(name) {
            return Err(ConfigError::StateConflict(format!(
                "service '{name}' must be stopped before changing configuration"
            )));
        }
        Ok(())
    }

    pub async fn save_config(
        &self,
        name: &str,
        values: ServiceConfigValues,
    ) -> Result<ServiceConfigState, ConfigError> {
        let _operation = self.operations.lock().await;
        self.ensure_editable(name).await?;
        let validation = self.validate(name, values.clone()).await?;
        if !validation.ok {
            return Err(ConfigError::Validation(validation));
        }

        let mut configs = self.configs.lock().await;
        let managed = configs
            .get_mut(name)
            .ok_or_else(|| ConfigError::UnknownService(name.to_string()))?;
        managed.state.saved = values;
        Ok(managed.state.clone())
    }

    async fn saved_candidate(&self, name: &str) -> Result<ServiceConfig, ConfigError> {
        let configs = self.configs.lock().await;
        configs
            .get(name)
            .map(ManagedConfig::saved_config)
            .ok_or_else(|| ConfigError::UnknownService(name.to_string()))
    }

    async fn set_effective(&self, name: &str, values: Option<ServiceConfigValues>) {
        if let Some(managed) = self.configs.lock().await.get_mut(name) {
            managed.state.effective = values;
        }
    }

    pub async fn start(&self, name: &str) -> Result<(), ConfigError> {
        let _operation = self.operations.lock().await;
        self.ensure_editable(name).await?;
        let candidate = self.saved_candidate(name).await?;
        let validation = self.validate(name, values_from_config(&candidate)).await?;
        if !validation.ok {
            return Err(ConfigError::Validation(validation));
        }

        let effective_values = values_from_config(&candidate);
        let started = {
            let mut backend = self.backend.lock().await;
            backend
                .replace_service_config(candidate)
                .map_err(|error| ConfigError::Runtime(error.to_string()))?;
            backend.start(name).await;
            backend
                .service(name)
                .is_some_and(|runtime| matches!(runtime.status, ServiceStatus::Running))
        };
        if started {
            self.set_effective(name, Some(effective_values)).await;
            Ok(())
        } else {
            Err(ConfigError::Runtime(format!(
                "service '{name}' failed to start"
            )))
        }
    }

    pub async fn stop(&self, name: &str) -> Result<(), ConfigError> {
        let _operation = self.operations.lock().await;
        {
            let mut backend = self.backend.lock().await;
            if backend.service(name).is_none() {
                return Err(ConfigError::UnknownService(name.to_string()));
            }
            backend.stop(name).await;
        }
        self.set_effective(name, None).await;
        Ok(())
    }

    /// Validates a replacement before stopping the old process. Invalid values leave it untouched.
    pub async fn restart(
        &self,
        name: &str,
        replacement: Option<ServiceConfigValues>,
    ) -> Result<(), ConfigError> {
        let _operation = self.operations.lock().await;
        let values = match replacement {
            Some(values) => values,
            None => self.config_state(name).await?.saved,
        };
        let validation = self.validate(name, values.clone()).await?;
        let candidate = self.candidate_and_peers(name, values.clone()).await?.0;
        if !validation.ok {
            return Err(ConfigError::Validation(validation));
        }

        {
            let mut configs = self.configs.lock().await;
            let managed = configs
                .get_mut(name)
                .ok_or_else(|| ConfigError::UnknownService(name.to_string()))?;
            managed.state.saved = values.clone();
        }

        let started = {
            let mut backend = self.backend.lock().await;
            if backend.service(name).is_none() {
                return Err(ConfigError::UnknownService(name.to_string()));
            }
            backend.stop(name).await;
            backend
                .replace_service_config(candidate)
                .map_err(|error| ConfigError::Runtime(error.to_string()))?;
            backend.start(name).await;
            backend
                .service(name)
                .is_some_and(|runtime| matches!(runtime.status, ServiceStatus::Running))
        };
        if started {
            self.set_effective(name, Some(values)).await;
            Ok(())
        } else {
            self.set_effective(name, None).await;
            Err(ConfigError::Runtime(format!(
                "service '{name}' failed to restart"
            )))
        }
    }

    pub async fn start_all(&self) -> Result<(), ConfigError> {
        let names: Vec<String> = self.configs.lock().await.keys().cloned().collect();
        for name in names {
            self.start(&name).await?;
        }
        Ok(())
    }

    pub async fn stop_all(&self) {
        let names: Vec<String> = self.configs.lock().await.keys().cloned().collect();
        for name in names {
            let _ = self.stop(&name).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_workspace() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("lunar-config-store-{}-{stamp}", std::process::id()));
        fs::create_dir_all(root.join("crates")).unwrap();
        fs::create_dir_all(root.join("target/release")).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        root
    }

    fn add_binary(root: &Path, name: &str) {
        fs::write(root.join("target/release").join(name), b"test").unwrap();
    }

    #[tokio::test]
    async fn stores_values_independently_and_rejects_invalid_updates() {
        let root = temp_workspace();
        add_binary(&root, "lunar-backend");
        add_binary(&root, "lunar-testbench-backend");
        let models = root.join("models");
        let worlds = root.join("worlds");
        fs::create_dir_all(&models).unwrap();
        let mut backend = ServiceConfig::backend();
        backend
            .env
            .insert("LUNAR_MODELS_DIR".into(), models.display().to_string());
        backend
            .env
            .insert("LUNAR_WORLDS_DIR".into(), worlds.display().to_string());
        let config = LauncherConfig::new(root.clone())
            .with_service(backend)
            .with_service(ServiceConfig::testbench_backend());
        let manager = ServiceManager::new(&config).unwrap();

        let original_testbench = manager.config_state("testbench-backend").await.unwrap();
        let mut backend_values = manager.config_state("backend").await.unwrap().saved;
        backend_values
            .env
            .insert("LUNAR_BACKEND_PORT".into(), "26000".into());
        let saved = manager
            .save_config("backend", backend_values)
            .await
            .unwrap();
        assert_eq!(saved.saved.env["LUNAR_BACKEND_PORT"], "26000");
        assert_eq!(
            manager.config_state("testbench-backend").await.unwrap(),
            original_testbench
        );

        let mut invalid = saved.saved;
        invalid
            .env
            .insert("LUNAR_BACKEND_PORT".into(), "not-a-port".into());
        let error = manager.save_config("backend", invalid).await.unwrap_err();
        assert!(matches!(error, ConfigError::Validation(_)));
        assert_eq!(
            manager.config_state("backend").await.unwrap().saved.env["LUNAR_BACKEND_PORT"],
            "26000"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_stopped_or_failed_services_are_editable() {
        assert!(status_allows_edit(&ServiceStatus::Stopped {
            reason: "x".into()
        }));
        assert!(status_allows_edit(&ServiceStatus::Failed {
            reason: "x".into()
        }));
        assert!(!status_allows_edit(&ServiceStatus::Starting));
        assert!(!status_allows_edit(&ServiceStatus::Running));
    }
}
