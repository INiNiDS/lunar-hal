#[cfg(test)]
use super::serving::check_serving_manifest;
#[cfg(test)]
use lnai_training::artifacts::{
    ArtifactManifestV1, architecture_version, expected_feature_schema_hash, manifest_file_name,
    norm_file_name, sha256_file_hex, weight_file_name,
};
#[cfg(test)]
use lnai_training::spec::ModelKind;

#[test]
fn serving_manifest_requires_bound_release_evidence() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let models_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target")
        .join(format!("serving-manifest-test-{nonce}"));
    std::fs::create_dir_all(&models_dir).expect("create test model directory");
    let kind = ModelKind::Pinn;
    let weight_path = models_dir.join(weight_file_name(&kind));
    let norm_path = models_dir.join(norm_file_name(&kind));
    std::fs::write(&weight_path, b"test-weights").unwrap();
    std::fs::write(&norm_path, b"test-normalization").unwrap();

    let mut manifest = ArtifactManifestV1::new(
        kind.clone(),
        architecture_version(&kind).to_string(),
        sha256_file_hex(&weight_path).unwrap(),
        sha256_file_hex(&norm_path).unwrap(),
        expected_feature_schema_hash(&kind).unwrap().to_string(),
        "gaia_dr3".into(),
        "dataset-manifest-v1".into(),
        42,
        serde_json::json!({}),
        "test-revision".into(),
        "cpu".into(),
    );
    let manifest_path = models_dir.join(manifest_file_name(&kind));
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let blocked = check_serving_manifest(&models_dir, &kind).unwrap_err();
    assert!(blocked.contains("spatial_holdout_evaluation"), "{blocked}");

    let report_file = format!("{}_spatial_holdout_report.json", kind.slug());
    let report_path = models_dir.join(&report_file);
    let report_bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "passed": true,
        "gate_version": "test-release-gate-v1",
        "model_kind": kind.clone(),
        "architecture_version": manifest.architecture_version.clone(),
        "model_hash": manifest.model_hash.clone(),
        "norm_hash": manifest.norm_hash.clone(),
        "dataset_id": manifest.dataset_id.clone(),
        "dataset_version": manifest.dataset_version.clone(),
        "feature_schema_hash": manifest.feature_schema_hash.clone(),
        "metrics": { "fixture_metric": 0.0 }
    }))
    .unwrap();
    std::fs::write(&report_path, &report_bytes).unwrap();
    let report_sha256 = sha256_file_hex(&report_path).unwrap();
    manifest.evaluation_metrics = Some(serde_json::json!({
        "spatial_holdout": {
            "passed": true,
            "model_hash": manifest.model_hash.clone(),
            "norm_hash": manifest.norm_hash.clone(),
            "dataset_id": manifest.dataset_id.clone(),
            "dataset_version": manifest.dataset_version.clone(),
            "feature_schema_hash": manifest.feature_schema_hash.clone(),
            "report_file": report_file,
            "report_sha256": report_sha256
        }
    }));
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    check_serving_manifest(&models_dir, &kind).expect("bound passing report should be accepted");
    std::fs::write(&report_path, b"tampered report").unwrap();
    let rejected = check_serving_manifest(&models_dir, &kind).unwrap_err();
    assert!(rejected.contains("checksum mismatch"), "{rejected}");
    std::fs::remove_dir_all(models_dir).expect("remove test bundle");
}
