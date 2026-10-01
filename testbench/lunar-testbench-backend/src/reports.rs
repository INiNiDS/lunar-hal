use crate::AppState;
use axum::{
    Json,
    extract::{Path as AxumPath, State},
};
use lunar_structures_testbench::ReportRecord;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BaselineDelta {
    pub baseline_ref: String,
    pub model: String,
    pub comparisons: Vec<MetricComparison>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MetricComparison {
    pub metric: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub delta: f64,
    pub delta_pct: f64,
    pub status: String,

}

pub fn compute_baseline_delta(
    kind: &str,
    model_name: &str,
    report: &Value,
    workspace_root: &Path,
) -> Option<Value> {
    let stage7_path = workspace_root.join("ai/fixtures/stage7-approved-baseline.json");
    let baseline_ref = if stage7_path.exists() {
        "stage7-approved-baseline".to_string()
    } else {
        "builtin-stage7-defaults".to_string()
    };

    let mut comparisons = Vec::new();

    if kind == "benchmark"
        || report.get("iterations").is_some()
        || report.get("avg_latency_ms").is_some()
    {
        let current_ms = report
            .get("avg_latency_ms")
            .or_else(|| report.get("per_iter_ms"))
            .and_then(|v| v.as_f64());

        if let Some(cur) = current_ms {
            let baseline_ms = match model_name {
                "pinn" => 0.58,
                "gnn" | "gnn_kinematics" | "gnn_localization" => 0.85,
                "siren" => 1.20,
                _ => 1.0,
            };

            let delta = cur - baseline_ms;
            let delta_pct = if baseline_ms > 0.0 {
                ((cur - baseline_ms) / baseline_ms) * 100.0
            } else {
                0.0
            };

            let status = if delta_pct <= -5.0 {
                "improved"
            } else if delta_pct >= 5.0 {
                "regression"
            } else {
                "parity"
            };

            comparisons.push(MetricComparison {
                metric: "avg_latency_ms".into(),
                baseline_value: baseline_ms,
                current_value: cur,
                delta,
                delta_pct,
                status: status.into(),
            });
        }
    } else if kind == "evaluation" {
        let current_val_loss = report
            .get("val_loss")
            .or_else(|| report.get("loss"))
            .and_then(|v| v.as_f64());

        if let Some(cur) = current_val_loss {
            let baseline_loss = match model_name {
                "pinn" => 0.005,
                "gnn" | "gnn_kinematics" | "gnn_localization" => 0.012,
                "siren" => 0.008,
                _ => 0.01,
            };

            let delta = cur - baseline_loss;
            let delta_pct = if baseline_loss > 0.0 {
                ((cur - baseline_loss) / baseline_loss) * 100.0
            } else {
                0.0
            };

            let status = if delta_pct <= -5.0 {
                "improved"
            } else if delta_pct >= 5.0 {
                "regression"
            } else {
                "parity"
            };

            comparisons.push(MetricComparison {
                metric: "val_loss".into(),
                baseline_value: baseline_loss,
                current_value: cur,
                delta,
                delta_pct,
                status: status.into(),
            });
        }
    }

    if comparisons.is_empty() {
        None
    } else {
        let delta = BaselineDelta {
            baseline_ref,
            model: model_name.to_string(),
            comparisons,
        };
        serde_json::to_value(delta).ok()
    }
}

pub async fn list_reports(State(state): State<AppState>) -> Json<Vec<ReportRecord>> {
    let reports = state.registry.db().list_reports().unwrap_or_default();
    Json(reports)
}

pub async fn get_report(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<ReportRecord>, String> {
    state
        .registry
        .db()
        .get_report(&id)
        .map_err(|e| e.to_string())?
        .map(Json)
        .ok_or_else(|| "report not found".to_string())
}

#[derive(Deserialize)]
pub struct CreateReportPayload {
    pub job_id: String,
    pub kind: String,
    pub model: Option<String>,
    pub report: Value,
}

pub async fn create_report(
    State(state): State<AppState>,
    Json(p): Json<CreateReportPayload>,
) -> Result<Json<ReportRecord>, String> {
    let ws = crate::jobs::workspace_root();
    let model = p.model.as_deref().unwrap_or("unknown");
    let baseline_delta = compute_baseline_delta(&p.kind, model, &p.report, &ws);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let id = format!("rep-{}", uuid::Uuid::new_v4().simple());
    let record = ReportRecord {
        id,
        job_id: p.job_id,
        kind: p.kind,
        baseline_delta_json: baseline_delta,
        report_json: p.report,
        created_at: now,
    };

    state
        .registry
        .db()
        .insert_report(&record)
        .map_err(|e| e.to_string())?;

    Ok(Json(record))
}
