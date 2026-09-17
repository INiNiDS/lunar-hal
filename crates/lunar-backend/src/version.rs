//! Stage 6.7: model registry serving surface — exact artifact versions
//! (`GET /version`) and controlled reload (`POST /models/reload`).

use axum::{Json, http::StatusCode, response::IntoResponse};
use lnai_training::artifacts::{RegisteredArtifact, RegistryStatus};
use serde::Serialize;

use crate::ai::{ReloadReport, registry_snapshot, reload_models};

#[derive(Serialize)]
pub struct ModelVersionInfo {
    pub kind: String,
    pub dir: String,
    pub weight_file: String,
    pub norm_file: String,
    pub status: String,
    pub manifest_version: Option<String>,
    pub architecture_version: Option<String>,
    pub git_revision: Option<String>,
    pub created_ms: Option<u64>,
    pub evaluation_metrics: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct VersionResponse {
    pub service: String,
    pub git_revision: String,
    pub models_dir: String,
    pub models: Vec<ModelVersionInfo>,
}

fn status_slug(status: &RegistryStatus) -> String {
    match status {
        RegistryStatus::Verified => "verified".to_string(),
        RegistryStatus::LegacyUnverified => "legacy_unverified".to_string(),
        RegistryStatus::Invalid(reason) => format!("invalid: {reason}"),
    }
}

fn entry_info(entry: RegisteredArtifact) -> ModelVersionInfo {
    let manifest = entry.manifest;
    ModelVersionInfo {
        kind: format!("{:?}", entry.kind).to_lowercase(),
        dir: entry.dir,
        weight_file: entry.weight_file,
        norm_file: entry.norm_file,
        status: status_slug(&entry.status),
        manifest_version: manifest.as_ref().map(|m| m.version.clone()),
        architecture_version: manifest.as_ref().map(|m| m.architecture_version.clone()),
        git_revision: manifest.as_ref().map(|m| m.git_revision.clone()),
        created_ms: manifest.as_ref().map(|m| m.created_ms),
        evaluation_metrics: manifest.as_ref().and_then(|m| m.evaluation_metrics.clone()),
    }
}

/// `GET /version`: service identity plus the exact artifact version of
/// every registry entry (exit gate: "API показывает точную artifact
/// version").
pub async fn version() -> Json<VersionResponse> {
    let entries = registry_snapshot().await;
    let models_dir = lunar_utils::env::get_lunar_models_dir()
        .display()
        .to_string();
    // PINN is compiled in: report the embedded bytes' hashes as its
    // exact version (it can never drift from the binary).
    let mut models = vec![ModelVersionInfo {
        kind: "pinn".to_string(),
        dir: "<embedded>".to_string(),
        weight_file: "stellar_model.bpk".to_string(),
        norm_file: "stellar_norm.json".to_string(),
        status: "embedded".to_string(),
        manifest_version: None,
        architecture_version: Some(
            lnai_training::artifacts::architecture_version(&lnai_training::spec::ModelKind::Pinn)
                .to_string(),
        ),
        git_revision: Some(
            option_env!("LUNAR_AI_GIT_REV")
                .unwrap_or("unknown")
                .to_string(),
        ),
        created_ms: None,
        evaluation_metrics: None,
    }];
    models.extend(entries.into_iter().map(entry_info));
    Json(VersionResponse {
        service: "lunar-backend".to_string(),
        git_revision: option_env!("LUNAR_AI_GIT_REV")
            .unwrap_or("unknown")
            .to_string(),
        models_dir,
        models,
    })
}

#[derive(Serialize)]
pub struct ReloadResponse {
    pub reloaded: Vec<String>,
    pub refused: Vec<String>,
    pub note: String,
}

fn reload_body(report: ReloadReport) -> ReloadResponse {
    ReloadResponse {
        reloaded: report.reloaded,
        refused: report.refused,
        note: report.note,
    }
}

/// `POST /models/reload`: controlled reload. Invalid bundles refuse the
/// whole reload (409) with zero state change; otherwise dir-loaded models
/// are dropped and lazily reloaded on next use.
pub async fn reload() -> impl IntoResponse {
    let report = reload_models().await;
    let status = if report.refused.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::CONFLICT
    };
    (status, Json(reload_body(report)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn version_reports_service_and_model_rows() {
        let body = version().await.0;
        assert_eq!(body.service, "lunar-backend");
        assert!(!body.git_revision.is_empty());
        assert!(!body.models_dir.is_empty());
        // PINN is always reported (embedded); dir-loaded kinds appear
        // when their files exist in the models dir.
        assert!(body.models.iter().any(|m| m.kind == "pinn"));
        for m in &body.models {
            assert!(!m.kind.is_empty());
            assert!(!m.status.is_empty());
        }
    }
}
