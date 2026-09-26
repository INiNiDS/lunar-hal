use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use lunar_utils::env::{DEFAULT_TESTBENCH_HOST, DEFAULT_TESTBENCH_PORT, resolve_port};
use tower_http::cors::{Any, CorsLayer};

use lunar_testbench_backend::jobs::JobRegistry;
use lunar_testbench_backend::{AppState, ai_jobs, data_jobs, db, jobs, reports, system};

pub async fn cancel_job_by_path(
    axum::extract::State(state): axum::extract::State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<axum::Json<()>, String> {
    state
        .registry
        .cancel(&id)
        .map(|_| axum::Json(()))
        .map_err(|e| e.to_string())
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

    let ws = jobs::workspace_root();
    let db_path = ws.join("data").join("testbench.db");
    let db = db::Database::open(db_path)?;
    let registry = JobRegistry::new(db);
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
        .route(
            "/jobs/cancel/{id}",
            post(cancel_job_by_path).get(cancel_job_by_path),
        )
        // Stage 5 canonical typed routes (task 10).
        .route("/jobs/training", post(ai_jobs::start_training))
        .route("/jobs/evaluation", post(ai_jobs::start_evaluation))
        .route("/jobs/benchmark", post(ai_jobs::start_benchmark))
        .route("/jobs/events/{id}", get(ai_jobs::job_typed_events))
        // Stage 10 reports routes.
        .route(
            "/reports",
            get(reports::list_reports).post(reports::create_report),
        )
        .route("/reports/{id}", get(reports::get_report))
        // Compatibility aliases: old routes stay, but spawn through the
        // same typed spec path (no separate implementation).
        .route("/jobs/train", post(jobs::start_train))
        .route("/jobs/validate", post(jobs::start_validate))
        .route("/data/status", get(data_jobs::get_status))
        .route("/data/collect", post(data_jobs::start_collect))
        .route("/data/verify", post(data_jobs::start_verify))
        .route("/data/build", post(data_jobs::start_build))
        .with_state(state)
        .layer(cors);

    println!("lunar-testbench-backend listening on {host}:{port}");
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
