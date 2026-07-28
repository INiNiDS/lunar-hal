use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::get,
};
use lunar_start::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex};
use tokio_stream::StreamExt;
use tower_http::cors::{Any, CorsLayer};

mod service_manager;
mod service_meta;

use service_manager::{LogRingBuffers, ServiceManager};

#[derive(Clone, Serialize)]
struct ServiceInfo {
    name: String,
    status: ServiceStatus,
    pid: Option<u32>,
}

#[derive(Clone)]
struct AppState {
    manager: Arc<Mutex<LogBackend>>,
    log_tx: broadcast::Sender<LogEvent>,
    logs: LogRingBuffers,
    start_time: Instant,
}

type SharedState = State<Arc<AppState>>;

async fn list_services(state: SharedState) -> Json<Vec<ServiceInfo>> {
    let be = state.manager.lock().await;
    let services: Vec<ServiceInfo> = be
        .services()
        .iter()
        .map(|s| ServiceInfo {
            name: s.config.name.clone(),
            status: s.status.clone(),
            pid: s.pid,
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

/// Cheap liveness probe for the "black room" boot phase — the frontend polls
/// this instead of `GET /services` to decide when to ignite the first lamp.
async fn health(state: SharedState) -> Json<HealthResponse> {
    Json(HealthResponse {
        ok: true,
        version: env!("CARGO_PKG_VERSION"),
        uptime_ms: state.start_time.elapsed().as_millis(),
    })
}

/// Static manifest describing each managed service: display metadata, which
/// dock apps it unlocks (`provides`), and which other services it needs.
async fn services_meta() -> Json<Vec<service_meta::ServiceMeta>> {
    Json(service_meta::all())
}

#[derive(Deserialize)]
struct TailQuery {
    tail: Option<usize>,
}

/// Returns the last `tail` (default 200) buffered log lines for a service, so
/// a newly opened log window isn't empty until the next SSE event arrives.
async fn service_logs(
    state: SharedState,
    Path(name): Path<String>,
    Query(q): Query<TailQuery>,
) -> Json<Vec<LogEvent>> {
    let logs = state.logs.lock().await;
    let empty = VecDeque::new();
    let buf = logs.get(&name).unwrap_or(&empty);
    let tail = q.tail.unwrap_or(200).min(buf.len());
    let start = buf.len() - tail;
    Json(buf.iter().skip(start).cloned().collect())
}

#[derive(Serialize)]
struct ServiceStats {
    name: String,
    warn_count: usize,
    err_count: usize,
    log_count: usize,
}

/// Aggregated warn/error counts for a service's buffered logs, used for LED
/// badges on the service rack.
async fn service_stats(state: SharedState, Path(name): Path<String>) -> Json<ServiceStats> {
    let logs = state.logs.lock().await;
    let empty = VecDeque::new();
    let buf = logs.get(&name).unwrap_or(&empty);
    let warn_count = buf.iter().filter(|l| l.level == LogLevel::Warn).count();
    let err_count = buf.iter().filter(|l| l.level == LogLevel::Error).count();
    Json(ServiceStats {
        name,
        warn_count,
        err_count,
        log_count: buf.len(),
    })
}

async fn start_service(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<&'static str>, StatusCode> {
    let mut be = state.manager.lock().await;
    be.start(&name).await;
    Ok(Json("ok"))
}

async fn stop_service(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<&'static str>, StatusCode> {
    let mut be = state.manager.lock().await;
    be.stop(&name).await;
    Ok(Json("ok"))
}

async fn restart_service(
    state: SharedState,
    Path(name): Path<String>,
) -> Result<Json<&'static str>, StatusCode> {
    let mut be = state.manager.lock().await;
    be.restart(&name).await;
    Ok(Json("ok"))
}

async fn start_all(state: SharedState) -> Result<Json<&'static str>, StatusCode> {
    let mut be = state.manager.lock().await;
    be.start_all().await;
    Ok(Json("ok"))
}

async fn stop_all(state: SharedState) -> Result<Json<&'static str>, StatusCode> {
    let mut be = state.manager.lock().await;
    be.stop_all().await;
    Ok(Json("ok"))
}

async fn sse_logs(
    state: SharedState,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let rx = state.log_tx.subscribe();

    let stream = tokio_stream::wrappers::BroadcastStream::new(rx).map(|res| match res {
        Ok(event) => {
            let json = serde_json::to_string(&event).unwrap_or_default();
            Ok(Event::default().data(json))
        }
        Err(_) => Ok(Event::default().data("")),
    });

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port: u16 = std::env::var("LUNAR_START_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(16181);

    let mut config = LauncherConfig::for_webos();

    if let Ok(env_val) = std::env::var("LUNAR_ENV") {
        config.set_env("LUNAR_ENV", &env_val);
    }
    config.apply_env();

    let mgr = ServiceManager::new(&config)?;
    let (log_tx, backend, logs) = mgr.into_parts();
    let state = Arc::new(AppState {
        log_tx,
        manager: backend,
        logs,
        start_time: Instant::now(),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/services", get(list_services))
        .route("/services/meta", get(services_meta))
        .route("/services/{name}/logs", get(service_logs))
        .route("/services/{name}/stats", get(service_stats))
        .route("/start", get(start_all))
        .route("/stop", get(stop_all))
        .route("/start/{name}", get(start_service))
        .route("/stop/{name}", get(stop_service))
        .route("/restart/{name}", get(restart_service))
        .route("/logs", get(sse_logs))
        .layer(cors)
        .with_state(state);

    println!("lunar-start-backend listening on 127.0.0.1:{port}");

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}
