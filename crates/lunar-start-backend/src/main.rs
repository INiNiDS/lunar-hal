use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use lunar_start::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};

mod service_manager;
mod service_meta;

use service_manager::{ConfigError, LogRingBuffers, ServiceConfigState, ServiceManager};

#[derive(Clone, Serialize)]
struct ServiceInfo {
    name: String,
    status: ServiceStatus,
    pid: Option<u32>,
}

#[derive(Clone)]
struct AppState {
    manager: ServiceManager,
    log_tx: broadcast::Sender<LogEvent>,
    logs: LogRingBuffers,
    start_time: Instant,
}

type SharedState = State<Arc<AppState>>;

#[derive(Serialize)]
struct ApiErrorBody {
    error: String,
    message: String,
    field_errors: HashMap<String, String>,
}

struct ApiError {
    status: StatusCode,
    body: ApiErrorBody,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

fn manager_api_error(error: ConfigError) -> ApiError {
    match error {
        ConfigError::UnknownService(message) => ApiError {
            status: StatusCode::NOT_FOUND,
            body: ApiErrorBody {
                error: "unknown_service".to_string(),
                message: format!("Unknown service: {message}"),
                field_errors: HashMap::new(),
            },
        },
        ConfigError::StateConflict(message) => ApiError {
            status: StatusCode::CONFLICT,
            body: ApiErrorBody {
                error: "state_conflict".to_string(),
                message,
                field_errors: HashMap::new(),
            },
        },
        ConfigError::Validation(validation) => {
            let port_conflict = validation
                .field_errors
                .values()
                .any(|message| message.contains("already used"));
            ApiError {
                status: if port_conflict {
                    StatusCode::CONFLICT
                } else {
                    StatusCode::BAD_REQUEST
                },
                body: ApiErrorBody {
                    error: "validation_failed".to_string(),
                    message: "Service configuration is invalid".to_string(),
                    field_errors: validation.field_errors,
                },
            }
        }
        ConfigError::Runtime(message) => ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ApiErrorBody {
                error: "start_failed".to_string(),
                message,
                field_errors: HashMap::new(),
            },
        },
    }
}

async fn list_services(state: SharedState) -> Json<Vec<ServiceInfo>> {
    let services = state
        .manager
        .services()
        .await
        .iter()
        .map(|service| ServiceInfo {
            name: service.config.name.clone(),
            status: service.status.clone(),
            pid: service.pid,
        })
        .collect();
    Json(services)
}

#[derive(Serialize)]
struct HealthResponse {
    ok: bool,
    version: &'static str,
    uptime_ms: u128,
}

async fn health(state: SharedState) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
        uptime_ms: state.start_time.elapsed().as_millis(),
    })
}

async fn services_meta() -> Json<Vec<service_meta::ServiceMeta>> {
    Json(service_meta::all())
}

#[derive(Deserialize)]
struct TailQuery {
    tail: Option<usize>,
}

async fn service_logs(
    state: SharedState,
    Path(name): Path<String>,
    Query(query): Query<TailQuery>,
) -> Json<Vec<LogEvent>> {
    let logs = state.logs.lock().await;
    let empty = VecDeque::new();
    let buffer = logs.get(&name).unwrap_or(&empty);
    let tail = query.tail.unwrap_or(200).min(buffer.len());
    let start = buffer.len() - tail;
    Json(buffer.iter().skip(start).cloned().collect())
}

#[derive(Serialize)]
struct ServiceStats {
    name: String,
    warn_count: usize,
    err_count: usize,
    log_count: usize,
}

async fn service_stats(state: SharedState, Path(name): Path<String>) -> Json<ServiceStats> {
    let logs = state.logs.lock().await;
    let empty = VecDeque::new();
    let buffer = logs.get(&name).unwrap_or(&empty);
    Json(ServiceStats {
        name,
        warn_count: buffer
            .iter()
            .filter(|log| log.level == LogLevel::Warn)
            .count(),
        err_count: buffer
            .iter()
            .filter(|log| log.level == LogLevel::Error)
            .count(),
        log_count: buffer.len(),
    })
}

async fn get_service_config_schema(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<ServiceConfigSchema>, ApiError> {
    state
        .manager
        .config_schema(&name)
        .map(Json)
        .map_err(manager_api_error)
}

async fn get_service_config(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<ServiceConfigState>, ApiError> {
    state
        .manager
        .config_state(&name)
        .await
        .map(Json)
        .map_err(manager_api_error)
}

async fn put_service_config(
    state: SharedState,
    Path(name): Path<String>,
    Json(values): Json<ServiceConfigValues>,
) -> Result<Json<ServiceConfigState>, ApiError> {
    state
        .manager
        .save_config(&name, values)
        .await
        .map(Json)
        .map_err(manager_api_error)
}

async fn validate_service_config_request(
    state: SharedState,
    Path(name): Path<String>,
    Json(values): Json<ServiceConfigValues>,
) -> Result<Json<ValidationResult>, ApiError> {
    state
        .manager
        .validate(&name, values)
        .await
        .map(Json)
        .map_err(manager_api_error)
}

#[derive(Default, Deserialize)]
struct StartServiceRequest {
    config: Option<ServiceConfigValues>,
}

#[derive(Serialize)]
struct ServiceActionResponse {
    service: ServiceInfo,
    config: ServiceConfigState,
}

async fn service_action_response(
    state: &Arc<AppState>,
    name: &str,
) -> Result<Json<ServiceActionResponse>, ApiError> {
    let runtime = state
        .manager
        .services()
        .await
        .into_iter()
        .find(|runtime| runtime.config.name == name)
        .ok_or_else(|| manager_api_error(ConfigError::UnknownService(name.to_string())))?;
    let config = state
        .manager
        .config_state(name)
        .await
        .map_err(manager_api_error)?;
    Ok(Json(ServiceActionResponse {
        service: ServiceInfo {
            name: runtime.config.name,
            status: runtime.status,
            pid: runtime.pid,
        },
        config,
    }))
}

async fn post_start_service(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(request): Json<StartServiceRequest>,
) -> Result<Json<ServiceActionResponse>, ApiError> {
    if let Some(values) = request.config {
        state
            .manager
            .save_config(&name, values)
            .await
            .map_err(manager_api_error)?;
    }
    state
        .manager
        .start(&name)
        .await
        .map_err(manager_api_error)?;
    service_action_response(&state, &name).await
}

async fn post_restart_service(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(request): Json<StartServiceRequest>,
) -> Result<Json<ServiceActionResponse>, ApiError> {
    state
        .manager
        .restart(&name, request.config)
        .await
        .map_err(manager_api_error)?;
    service_action_response(&state, &name).await
}

async fn post_stop_service(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<ServiceActionResponse>, ApiError> {
    state.manager.stop(&name).await.map_err(manager_api_error)?;
    service_action_response(&state, &name).await
}

// Compatibility wrappers for the old GET control API.
async fn legacy_start_service(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<&'static str>, ApiError> {
    state
        .manager
        .start(&name)
        .await
        .map_err(manager_api_error)?;
    Ok(Json("ok"))
}

async fn legacy_stop_service(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<&'static str>, ApiError> {
    state.manager.stop(&name).await.map_err(manager_api_error)?;
    Ok(Json("ok"))
}

async fn legacy_restart_service(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<&'static str>, ApiError> {
    state
        .manager
        .restart(&name, None)
        .await
        .map_err(manager_api_error)?;
    Ok(Json("ok"))
}

async fn legacy_start_all(state: SharedState) -> Result<Json<&'static str>, ApiError> {
    state.manager.start_all().await.map_err(manager_api_error)?;
    Ok(Json("ok"))
}

async fn legacy_stop_all(state: SharedState) -> Json<&'static str> {
    state.manager.stop_all().await;
    Json("ok")
}

async fn sse_logs(
    state: SharedState,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let receiver = state.log_tx.subscribe();
    let stream =
        tokio_stream::wrappers::BroadcastStream::new(receiver).map(|result| match result {
            Ok(event) => {
                Ok(Event::default().data(serde_json::to_string(&event).unwrap_or_default()))
            }
            Err(_) => Ok(Event::default().data("")),
        });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn build_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/services", get(list_services))
        .route("/services/meta", get(services_meta))
        .route("/services/{name}/logs", get(service_logs))
        .route("/services/{name}/stats", get(service_stats))
        .route(
            "/services/{name}/config/schema",
            get(get_service_config_schema),
        )
        .route(
            "/services/{name}/config",
            get(get_service_config).put(put_service_config),
        )
        .route(
            "/services/{name}/validate",
            post(validate_service_config_request),
        )
        .route("/services/{name}/start", post(post_start_service))
        .route("/services/{name}/restart", post(post_restart_service))
        .route("/services/{name}/stop", post(post_stop_service))
        .route("/start", get(legacy_start_all))
        .route("/stop", get(legacy_stop_all))
        .route("/start/{name}", get(legacy_start_service))
        .route("/stop/{name}", get(legacy_stop_service))
        .route("/restart/{name}", get(legacy_restart_service))
        .route("/logs", get(sse_logs))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .with_state(state)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port = std::env::var("LUNAR_START_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(16181);

    let mut config = LauncherConfig::for_webos();
    if let Ok(environment) = std::env::var("LUNAR_ENV") {
        config.set_env("LUNAR_ENV", &environment);
    }
    let manager = ServiceManager::new(&config)?;
    let state = Arc::new(AppState {
        log_tx: manager.log_sender(),
        logs: manager.logs(),
        manager,
        start_time: Instant::now(),
    });

    println!("lunar-start-backend listening on 127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    axum::serve(listener, build_app(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tower::ServiceExt;

    fn temp_workspace() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("lunar-config-api-{}-{stamp}", std::process::id()));
        fs::create_dir_all(root.join("crates")).unwrap();
        fs::create_dir_all(root.join("target/release")).unwrap();
        fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::write(root.join("target/release/lunar-backend"), b"test").unwrap();
        root
    }

    fn test_state(root: &std::path::Path) -> Arc<AppState> {
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
        let config = LauncherConfig::new(root.to_path_buf()).with_service(backend);
        let manager = ServiceManager::new(&config).unwrap();
        Arc::new(AppState {
            log_tx: manager.log_sender(),
            logs: manager.logs(),
            manager,
            start_time: Instant::now(),
        })
    }

    async fn json_body(response: Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn exposes_schema_and_independent_config_state() {
        let root = temp_workspace();
        let app = build_app(test_state(&root));

        let schema = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/services/backend/config/schema")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(schema.status(), StatusCode::OK);
        let schema_body = json_body(schema).await;
        assert_eq!(schema_body["service"], "backend");
        assert!(schema_body["fields"].as_array().unwrap().len() >= 9);

        let config = app
            .oneshot(
                Request::builder()
                    .uri("/services/backend/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(config.status(), StatusCode::OK);
        let config_body = json_body(config).await;
        assert_eq!(config_body["saved"]["env"]["LUNAR_BACKEND_PORT"], "25255");
        assert!(config_body["effective"].is_null());
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn validates_and_saves_config_with_structured_errors() {
        let root = temp_workspace();
        let state = test_state(&root);
        let app = build_app(Arc::clone(&state));
        let mut values = state.manager.config_state("backend").await.unwrap().saved;
        values.env.insert("LUNAR_BACKEND_PORT".into(), "0".into());
        let body = serde_json::to_vec(&values).unwrap();

        let validate = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/services/backend/validate")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(validate.status(), StatusCode::OK);
        let validation = json_body(validate).await;
        assert_eq!(validation["ok"], false);
        assert!(validation["field_errors"]["LUNAR_BACKEND_PORT"].is_string());

        let save = app
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/services/backend/config")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(save.status(), StatusCode::BAD_REQUEST);
        let error = json_body(save).await;
        assert_eq!(error["error"], "validation_failed");
        assert!(error["field_errors"]["LUNAR_BACKEND_PORT"].is_string());
        assert_eq!(
            state
                .manager
                .config_state("backend")
                .await
                .unwrap()
                .saved
                .env["LUNAR_BACKEND_PORT"],
            "25255"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn saves_valid_config_through_put() {
        let root = temp_workspace();
        let state = test_state(&root);
        let mut values = state.manager.config_state("backend").await.unwrap().saved;
        values
            .env
            .insert("LUNAR_BACKEND_PORT".into(), "26000".into());
        let response = build_app(Arc::clone(&state))
            .oneshot(
                Request::builder()
                    .method("PUT")
                    .uri("/services/backend/config")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&values).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = json_body(response).await;
        assert_eq!(body["saved"]["env"]["LUNAR_BACKEND_PORT"], "26000");
        assert_eq!(
            state
                .manager
                .config_state("backend")
                .await
                .unwrap()
                .saved
                .env["LUNAR_BACKEND_PORT"],
            "26000"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn post_start_and_stop_return_status_and_effective_config() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_workspace();
        let binary = root.join("target/release/lunar-backend");
        fs::write(&binary, "#!/bin/sh\nsleep 30\n").unwrap();
        let mut permissions = fs::metadata(&binary).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&binary, permissions).unwrap();
        let state = test_state(&root);
        let app = build_app(Arc::clone(&state));

        let start = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/services/backend/start")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(start.status(), StatusCode::OK);
        let started = json_body(start).await;
        assert_eq!(started["service"]["status"]["kind"], "running");
        assert!(started["config"]["effective"].is_object());

        // Invalid replacement config must not stop the current process.
        let mut invalid = state.manager.config_state("backend").await.unwrap().saved;
        invalid.env.insert("LUNAR_BACKEND_PORT".into(), "0".into());
        let invalid_restart = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/services/backend/restart")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "config": invalid }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_restart.status(), StatusCode::BAD_REQUEST);
        assert!(
            state
                .manager
                .services()
                .await
                .iter()
                .any(|service| service.config.name == "backend"
                    && matches!(service.status, lunar_start::ServiceStatus::Running))
        );

        // A valid replacement is applied before the service is started again.
        let mut replacement = state.manager.config_state("backend").await.unwrap().saved;
        replacement
            .env
            .insert("LUNAR_BACKEND_PORT".into(), "26001".into());
        let restart = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/services/backend/restart")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "config": replacement }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(restart.status(), StatusCode::OK);
        let restarted = json_body(restart).await;
        assert_eq!(restarted["service"]["status"]["kind"], "running");
        assert_eq!(
            restarted["config"]["effective"]["env"]["LUNAR_BACKEND_PORT"],
            "26001"
        );

        let stop = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/services/backend/stop")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(stop.status(), StatusCode::OK);
        let stopped = json_body(stop).await;
        assert_eq!(stopped["service"]["status"]["kind"], "stopped");
        assert!(stopped["config"]["effective"].is_null());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn maps_state_and_port_conflicts_to_http_409() {
        let state = manager_api_error(ConfigError::StateConflict("running".to_string()));
        assert_eq!(state.status, StatusCode::CONFLICT);

        let validation = manager_api_error(ConfigError::Validation(ValidationResult {
            ok: false,
            field_errors: HashMap::from([(
                "SERVE_PORT".to_string(),
                "Port 8080 is already used by another service".to_string(),
            )]),
        }));
        assert_eq!(validation.status, StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn unknown_service_returns_structured_not_found() {
        let root = temp_workspace();
        let response = build_app(test_state(&root))
            .oneshot(
                Request::builder()
                    .uri("/services/unknown/config")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let error = json_body(response).await;
        assert_eq!(error["error"], "unknown_service");
        fs::remove_dir_all(root).unwrap();
    }
}
