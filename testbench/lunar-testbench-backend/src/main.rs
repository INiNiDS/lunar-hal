use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
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
    let port: u16 = {
        let args: Vec<String> = std::env::args().collect();
        let from_args = args
            .windows(2)
            .find(|w| w[0] == "--port" || w[0] == "-p")
            .and_then(|w| w[1].parse().ok());
        from_args
            .or_else(|| {
                std::env::var("LUNAR_TESTBENCH_BACKEND_PORT")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(25256)
    };

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

    println!("lunar-testbench-backend listening on 127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
