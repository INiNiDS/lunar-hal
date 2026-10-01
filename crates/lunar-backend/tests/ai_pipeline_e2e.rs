
use lnai_models::{GNN_INPUT_DIM, GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM, GnnHeadKind};
use lnai_training::artifacts::{
    ARTIFACT_MANIFEST_VERSION, ArtifactManifestV1, RegistryStatus, architecture_version,
    manifest_file_name, scan_model_registry, verify_manifest_against_files,
};
use lnai_training::spec::ModelKind;
use std::path::PathBuf;

fn repo_models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn baseline_json() -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../ai/fixtures/stage6-approved-baseline.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("stage6 baseline must exist at {}: {e}", path.display()));
    serde_json::from_str(&raw).expect("stage6 baseline must parse")
}

fn read_flat_manifest(models: &std::path::Path, kind: &ModelKind) -> ArtifactManifestV1 {
    let path = models.join(manifest_file_name(kind));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing flat manifest {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("flat manifest {} must parse: {e}", path.display()))
}

#[test]
fn flat_serving_bundles_pass_the_file_integrity_gate() {
    let models = repo_models_dir();
    assert!(
        models.is_dir(),
        "models dir must exist: {}",
        models.display()
    );
    for kind in [ModelKind::Pinn, ModelKind::GnnKinematics, ModelKind::Siren] {
        let manifest = read_flat_manifest(&models, &kind);
        assert_eq!(manifest.version, ARTIFACT_MANIFEST_VERSION);
        assert_eq!(
            manifest.architecture_version,
            architecture_version(&kind),
            "frozen arch line for {kind:?}"
        );
        verify_manifest_against_files(&models, &manifest)
            .unwrap_or_else(|e| panic!("serving bundle {kind:?} rejected: {e}"));
    }
}

#[test]
fn registry_separates_integrity_from_release_approval() {
    let models = repo_models_dir();
    let entries = scan_model_registry(&models);
    assert!(!entries.is_empty(), "registry must discover bundles");
    for e in &entries {
        assert!(
            !matches!(e.status, RegistryStatus::Invalid(_)),
            "invalid registry entry {}: {:?}",
            e.dir,
            e.status
        );
    }
    for kind in [ModelKind::Pinn, ModelKind::GnnKinematics, ModelKind::Siren] {
        let entry = entries
            .iter()
            .find(|e| e.kind == kind && e.dir == models.display().to_string())
            .unwrap_or_else(|| panic!("flat {kind:?} must be discovered: {entries:?}"));
        let manifest = entry
            .manifest
            .as_ref()
            .expect("flat manifest must be retained");
        let blockers = manifest.release_blockers();
        if blockers.is_empty() {
            assert_eq!(entry.status, RegistryStatus::Verified, "{kind:?}");
        } else {
            assert_eq!(
                entry.status,
                RegistryStatus::ReleaseBlocked(blockers.into_iter().map(str::to_string).collect()),
                "{kind:?} must expose missing release evidence"
            );
        }
    }
}

#[test]
fn serving_head_contract_matches_train_eval_widths() {
    assert_eq!(GNN_INPUT_DIM, 8, "frozen node width");
    assert_eq!(
        GnnHeadKind::from_output_dim(GNN_OUTPUT_DIM),
        Some(GnnHeadKind::Deterministic)
    );
    assert_eq!(
        GnnHeadKind::from_output_dim(GNN_VARIATIONAL_DIM),
        Some(GnnHeadKind::Variational)
    );
    assert_eq!(GnnHeadKind::from_output_dim(4), None);
}

#[test]
fn stage6_baseline_record_does_not_claim_checkpoint_quality() {
    let base = baseline_json();
    assert_eq!(base["stage"], 6);
    assert_eq!(base["release_qualification"], "not_qualified");
    assert_eq!(base["evidence_scope"]["real_checkpoint_evaluation"], false);
    let suites = base["suites"].as_array().expect("suites array");
    assert!(!suites.is_empty());
    for s in suites {
        assert_eq!(
            s["status"], "passed",
            "frozen suite must be green: {}",
            s["file"]
        );
        assert!(s["tests"].as_u64().unwrap_or(0) > 0);
    }
    for gate in [
        "finite_deterministic_holdout",
        "api_artifact_version",
        "chained_report_no_synthetic_noise",
    ] {
        assert!(
            base["exit_gate"][gate].is_string(),
            "exit gate must pin {gate}"
        );
    }
}
