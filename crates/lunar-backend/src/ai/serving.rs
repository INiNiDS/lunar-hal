use lnai_training::artifacts::{
    ArtifactManifestV1, RegisteredArtifact, RegistryStatus, manifest_file_name,
    scan_model_registry, verify_manifest_against_files,
};
use lnai_training::spec::ModelKind;
use lunar_utils::env::get_lunar_models_dir;

use super::gnn::{GNN, get_gnn};
use super::localization::{GNN_LOCALIZATION, localization_enabled};
use super::lore::{LORE_CACHE, get_lore_cache};
use super::pinn::{PINN, load_pinn, loaded_pinn_hashes};
#[cfg(feature = "siren")]
use super::siren::{SIREN_MODEL, get_siren};
use super::types::LoadedModelIdentity;

pub(crate) fn check_serving_manifest(
    models_dir: &std::path::Path,
    kind: &ModelKind,
) -> Result<ArtifactManifestV1, String> {
    let path = models_dir.join(manifest_file_name(kind));
    if !path.exists() {
        return Err(format!(
            "serving bundle for {} has no artifact manifest",
            path.display()
        ));
    }
    let raw =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let manifest: ArtifactManifestV1 =
        serde_json::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?;
    if &manifest.model_kind != kind {
        return Err(format!(
            "serving bundle for {} declares model kind {:?}, expected {:?}",
            path.display(),
            manifest.model_kind,
            kind
        ));
    }
    verify_manifest_against_files(models_dir, &manifest)
        .map_err(|e| format!("serving bundle for {} rejected: {e}", path.display()))?;
    let blockers = manifest.release_blockers();
    if !blockers.is_empty() {
        let allow_unapproved = std::env::var("LUNAR_ALLOW_UNAPPROVED_MODELS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or_else(|_| {
                std::env::var("LUNAR_STRICT_RELEASE").as_deref() != Ok("1") && !cfg!(test)
            });
        if !allow_unapproved {
            return Err(format!(
                "serving bundle for {} is not release-approved; missing evidence: {}",
                path.display(),
                blockers.join(", ")
            ));
        }
        eprintln!(
            "  warning: serving bundle for {} is not release-approved ({}); proceeding under unapproved allowance",
            path.display(),
            blockers.join(", ")
        );
    }
    Ok(manifest)
}

pub async fn loaded_model_identities() -> Vec<LoadedModelIdentity> {
    let mut active = Vec::new();
    if let Some((model_hash, norm_hash)) = loaded_pinn_hashes().await {
        active.push(LoadedModelIdentity {
            kind: "pinn",
            model_hash,
            norm_hash,
        });
    }
    if let Some(gnn) = GNN.read().await.as_ref() {
        active.push(LoadedModelIdentity {
            kind: "gnn_kinematics",
            model_hash: gnn.model_hash.clone(),
            norm_hash: gnn.norm_hash.clone(),
        });
    }
    if localization_enabled() {
        if let Some(loc) = GNN_LOCALIZATION.read().await.as_ref() {
            active.push(LoadedModelIdentity {
                kind: "gnn_localization",
                model_hash: loc.model_hash.clone(),
                norm_hash: loc.norm_hash.clone(),
            });
        }
    }
    #[cfg(feature = "siren")]
    if let Some(siren) = SIREN_MODEL.read().await.as_ref() {
        active.push(LoadedModelIdentity {
            kind: "siren",
            model_hash: siren.model_hash.clone(),
            norm_hash: siren.norm_hash.clone(),
        });
    }
    active
}

pub async fn registry_snapshot() -> Vec<RegisteredArtifact> {
    let models_dir = get_lunar_models_dir();
    scan_model_registry(&models_dir)
}

#[derive(Debug, Clone)]
pub struct ReloadReport {
    pub reloaded: Vec<String>,
    pub refused: Vec<String>,
    pub note: String,
}

pub async fn reload_models() -> ReloadReport {
    let entries = registry_snapshot().await;
    let serving_dir = get_lunar_models_dir().display().to_string();
    let refused: Vec<String> = entries
        .iter()
        .filter(|entry| entry.dir == serving_dir)
        .filter_map(|entry| match &entry.status {
            RegistryStatus::Verified => None,
            RegistryStatus::ReleaseBlocked(blockers) => Some(format!(
                "{:?}: release blocked ({})",
                entry.kind,
                blockers.join(", ")
            )),
            RegistryStatus::LegacyUnverified => {
                Some(format!("{:?}: artifact manifest required", entry.kind))
            }
            RegistryStatus::Invalid(reason) => Some(format!("{:?}: {reason}", entry.kind)),
        })
        .collect();
    if !refused.is_empty() {
        return ReloadReport {
            reloaded: Vec::new(),
            refused,
            note: "reload refused: fix or remove invalid bundles, old models keep serving"
                .to_string(),
        };
    }
    let pinn = match load_pinn(&get_lunar_models_dir()) {
        Ok(pinn) => pinn,
        Err(err) => {
            return ReloadReport {
                reloaded: Vec::new(),
                refused: vec![format!("pinn: {err:#}")],
                note: "reload refused: existing models keep serving".to_string(),
            };
        }
    };
    *PINN.write().await = Some(pinn);
    *GNN.write().await = None;
    *LORE_CACHE.write().await = None;
    *GNN_LOCALIZATION.write().await = None;
    #[cfg(feature = "siren")]
    {
        *SIREN_MODEL.write().await = None;
    }
    let mut reloaded = vec!["pinn: loaded".to_string()];
    reloaded.push(format!(
        "gnn_kinematics: {}",
        if get_gnn().await.is_some() {
            "loaded"
        } else {
            "missing"
        }
    ));
    reloaded.push(format!(
        "lore_cache: {}",
        if get_lore_cache().await.is_some() {
            "loaded"
        } else {
            "missing"
        }
    ));
    #[cfg(feature = "siren")]
    reloaded.push(format!(
        "siren: {}",
        if get_siren().await.is_some() {
            "loaded"
        } else {
            "missing"
        }
    ));
    ReloadReport {
        reloaded,
        refused: Vec::new(),
        note: "PINN swapped after validation; other models refreshed".to_string(),
    }
}
