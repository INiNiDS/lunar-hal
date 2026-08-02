use std::sync::Arc;

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use lunar_utils::env::{resolve_port, DEFAULT_TESTBENCH_HOST, DEFAULT_TESTBENCH_PORT};
use tower_http::cors::{Any, CorsLayer};

pub mod jobs;
pub mod system;

use crate::jobs::JobRegistry;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<JobRegistry>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let backend_port_env = std::env::var("LUNAR_TESTBENCH_BACKEND_PORT").ok();
    let client_port_env = std::env::var("LUNAR_TESTBENCH_PORT").ok();
    let port = resolve_port(
        &args,
        &[backend_port_env.as_deref(), client_port_env.as_deref()],
        DEFAULT_TESTBENCH_PORT,
    );
    let host = DEFAULT_TESTBENCH_HOST;

    let registry = JobRegistry::new();
    let state = AppState { registry };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/system/snapshot", get(system::system_snapshot))
        .route("/jobs", get(jobs::list_jobs))
        .route("/jobs/get", get(jobs::get_job_by_query).post(jobs::get_job))
        .route(
            "/jobs/cancel",
            post(jobs::cancel_job).get(jobs::cancel_job_by_query),
        )
        .route("/jobs/train", post(jobs::start_train))
        .route("/jobs/validate", post(jobs::start_validate))
        .with_state(state)
        .layer(cors);

    println!("lunar-testbench-backend listening on {host}:{port}");
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
