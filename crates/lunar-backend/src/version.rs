
use axum::{Json, http::StatusCode, response::IntoResponse};
use lnai_training::artifacts::{RegisteredArtifact, RegistryStatus};
use lnai_training::spec::ModelKind;
use serde::Serialize;

use crate::ai::{
    LoadedModelIdentity, ReloadReport, loaded_model_identities, registry_snapshot, reload_models,
};

#[derive(Serialize)]
pub struct ModelVersionInfo {
    pub kind: String,
    pub dir: String,
    pub weight_file: String,
    pub norm_file: String,
    pub status: String,
    pub manifest_version: Option<String>,
    pub architecture_version: Option<String>,
    pub model_hash: Option<String>,
    pub norm_hash: Option<String>,
    pub git_revision: Option<String>,
    pub created_ms: Option<u64>,
    pub evaluation_metrics: Option<serde_json::Value>,
    pub release_blockers: Vec<String>,
}

#[derive(Serialize)]
pub struct VersionResponse {
    pub service: String,
    pub git_revision: String,
    pub models_dir: String,
    pub models: Vec<ModelVersionInfo>,
    pub active_models: Vec<LoadedModelIdentity>,
}

fn status_slug(status: &RegistryStatus) -> String {
    match status {
        RegistryStatus::Verified => "verified".to_string(),
        RegistryStatus::ReleaseBlocked(_) => "release_blocked".to_string(),
        RegistryStatus::LegacyUnverified => "legacy_unverified".to_string(),
        RegistryStatus::Invalid(reason) => format!("invalid: {reason}"),
    }
}

fn kind_slug(kind: &ModelKind) -> &'static str {
    match kind {
        ModelKind::Pinn => "pinn",
        ModelKind::GnnKinematics => "gnn_kinematics",
        ModelKind::GnnLocalization => "gnn_localization",
        ModelKind::Siren => "siren",
    }
}

fn entry_info(entry: RegisteredArtifact) -> ModelVersionInfo {
    let manifest = entry.manifest;
    let release_blockers = match &entry.status {
        RegistryStatus::ReleaseBlocked(blockers) => blockers.clone(),
        _ => Vec::new(),
    };
    ModelVersionInfo {
        kind: kind_slug(&entry.kind).to_string(),
        dir: entry.dir,
        weight_file: entry.weight_file,
        norm_file: entry.norm_file,
        status: status_slug(&entry.status),
        manifest_version: manifest.as_ref().map(|m| m.version.clone()),
        architecture_version: manifest.as_ref().map(|m| m.architecture_version.clone()),
        model_hash: manifest.as_ref().map(|m| m.model_hash.clone()),
        norm_hash: manifest.as_ref().map(|m| m.norm_hash.clone()),
        git_revision: manifest.as_ref().map(|m| m.git_revision.clone()),
        created_ms: manifest.as_ref().map(|m| m.created_ms),
        evaluation_metrics: manifest.as_ref().and_then(|m| m.evaluation_metrics.clone()),
        release_blockers,
    }
}

pub async fn version() -> Json<VersionResponse> {
    let models = registry_snapshot()
        .await
        .into_iter()
        .map(entry_info)
        .collect();
    let active_models = loaded_model_identities().await;
    let models_dir = lunar_utils::env::get_lunar_models_dir()
        .display()
        .to_string();
    Json(VersionResponse {
        service: "lunar-backend".to_string(),
        git_revision: option_env!("LUNAR_AI_GIT_REV")
            .unwrap_or("unknown")
            .to_string(),
        models_dir,
        models,
        active_models,
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
        for m in &body.models {
            assert!(!m.kind.is_empty());
            assert!(!m.status.is_empty());
            if matches!(m.status.as_str(), "verified" | "release_blocked") {
                assert!(m.model_hash.is_some());
                assert!(m.norm_hash.is_some());
            }
        }
        for model in &body.active_models {
            assert!(!model.kind.is_empty());
            assert_eq!(model.model_hash.len(), 64);
            assert_eq!(model.norm_hash.len(), 64);
        }
    }
}
