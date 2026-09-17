use crate::spec::ModelKind;
use lunar_utils::time::current_time_ms;
use serde::{Deserialize, Serialize};

/// Version of the artifact manifest structure itself.
pub const ARTIFACT_MANIFEST_VERSION: &str = "1.0.0";

/// Runs `write` against a temp sibling of `final_path`, then atomically
/// renames over it. A kill mid-save can never leave a half-written
/// checkpoint behind (the failure mode that zeroed a 135-epoch run once).
pub fn atomic_write_through<F>(
    final_path: &std::path::Path,
    tmp_ext: &str,
    write: F,
) -> Result<(), String>
where
    F: FnOnce(&std::path::Path) -> Result<(), String>,
{
    let tmp = final_path.with_extension(tmp_ext);
    write(&tmp)?;
    std::fs::rename(&tmp, final_path)
        .map_err(|e| format!("publish {}: {e}", final_path.display()))
}

/// Top-level manifest for a trained model artifact.
/// Used to verify compatibility with a specific dataset, schema, and normalization state.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ArtifactManifestV1 {
    /// Version of this manifest structure
    pub version: String,

    /// Type of the model (e.g., Pinn, GnnLocalization)
    pub model_kind: ModelKind,

    /// Architecture version (e.g., "pinn-v2", "gnn-loc-v1") to prevent loading incompatible graphs
    pub architecture_version: String,

    /// SHA-256 hash of the model weights file (e.g., `model.safetensors`)
    pub model_hash: String,

    /// SHA-256 hash of the normalization snapshot
    pub norm_hash: String,

    /// Hash of the canonical data schema (from `lnai-data/src/schema.rs`)
    pub feature_schema_hash: String,

    /// Identifier of the dataset used for training (e.g., "gaia_dr3")
    pub dataset_id: String,

    /// Version of the dataset manifest used (from `lnai-data/src/manifest.rs`)
    pub dataset_version: String,

    /// Global seed used during training/inference for reproducibility
    pub seed: u64,

    /// JSON representation of the hyperparameters used (from TrainingSpec)
    pub hyperparameters: serde_json::Value,

    /// Git commit hash of the codebase that produced this artifact
    pub git_revision: String,

    /// Backend and device used for training (e.g., "cuda:0", "cpu", "wgpu")
    pub backend_device: String,

    /// Optional evaluation metrics computed at the end of training or during validation
    pub evaluation_metrics: Option<serde_json::Value>,

    /// Creation timestamp (ms since UNIX_EPOCH)
    pub created_ms: u64,

    /// Optional metadata specific to GNN-Localization artifacts
    pub localization_meta: Option<LocalizationArtifactMeta>,
}

impl ArtifactManifestV1 {
    pub fn new(
        model_kind: ModelKind,
        architecture_version: String,
        model_hash: String,
        norm_hash: String,
        feature_schema_hash: String,
        dataset_id: String,
        dataset_version: String,
        seed: u64,
        hyperparameters: serde_json::Value,
        git_revision: String,
        backend_device: String,
    ) -> Self {
        Self {
            version: ARTIFACT_MANIFEST_VERSION.to_string(),
            model_kind,
            architecture_version,
            model_hash,
            norm_hash,
            feature_schema_hash,
            dataset_id,
            dataset_version,
            seed,
            hyperparameters,
            git_revision,
            backend_device,
            evaluation_metrics: None,
            created_ms: current_time_ms(),
            localization_meta: None,
        }
    }

    /// Validates full compatibility of this artifact with the inference context.
    /// Per contract v1 an artifact may only be loaded when the model kind,
    /// architecture version, normalization snapshot, dataset and canonical
    /// schema hashes all match.
    #[allow(clippy::too_many_arguments)]
    pub fn is_compatible_with(
        &self,
        model_kind: &ModelKind,
        architecture_version: &str,
        norm_hash: &str,
        dataset_id: &str,
        dataset_version: &str,
        schema_hash: &str,
    ) -> bool {
        &self.model_kind == model_kind
            && self.architecture_version == architecture_version
            && self.norm_hash == norm_hash
            && self.dataset_id == dataset_id
            && self.dataset_version == dataset_version
            && self.feature_schema_hash == schema_hash
    }
}

/// Specific metadata required for GNN-Localization artifacts.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationArtifactMeta {
    /// Hash of the spatial split / catalog selection function used during data assembly
    pub catalog_selection_hash: String,

    /// Hash of the masking policy (e.g., visibility masks, hidden neighbor ratios)
    pub masking_policy_hash: String,

    /// Radius used for neighborhood extraction (in parsecs)
    pub radius_pc: f32,

    /// Maximum number of slots (neighbors) the model was trained to predict
    pub max_slots: u32,
}

/// Normalization snapshot used to ensure consistent data scaling between training and inference.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NormSnapshotV1 {
    /// Type of normalization (e.g., "standard", "minmax")
    pub kind: String,

    /// Path to the saved normalization file (e.g., JSON or Safetensors)
    pub path: String,

    /// The actual normalization parameters (mean, std, min, max per column)
    pub data: serde_json::Value,

    /// SHA-256 checksum of the `data` payload to detect corruption
    pub checksum: String,
}

/// Error type for artifact loading and compatibility checks.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactError {
    IncompatibleDataset(String),
    IncompatibleSchema(String),
    IncompatibleArchitecture(String),
    ChecksumMismatch(String),
    FileNotFound(String),
    DeserializationError(String),
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArtifactError::IncompatibleDataset(m) => write!(f, "incompatible dataset: {m}"),
            ArtifactError::IncompatibleSchema(m) => write!(f, "incompatible schema: {m}"),
            ArtifactError::IncompatibleArchitecture(m) => {
                write!(f, "incompatible architecture: {m}")
            }
            ArtifactError::ChecksumMismatch(m) => write!(f, "checksum mismatch: {m}"),
            ArtifactError::FileNotFound(m) => write!(f, "file not found: {m}"),
            ArtifactError::DeserializationError(m) => write!(f, "deserialization error: {m}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

/// Well-known weight/norm file names per model kind, shared by CLI workers,
/// Testbench spawn code and parity tests so renames break in one place.
pub fn weight_file_name(model: &ModelKind) -> &'static str {
    match model {
        ModelKind::Pinn => "stellar_model.bpk",
        ModelKind::GnnKinematics => "stellar_gnn_model.bpk",
        ModelKind::GnnLocalization => "stellar_gnn_loc_model.bpk",
        ModelKind::Siren => "stellar_siren_model.bpk",
    }
}

/// Well-known normalization file names per model kind.
pub fn norm_file_name(model: &ModelKind) -> &'static str {
    match model {
        ModelKind::Pinn => "stellar_norm.json",
        ModelKind::GnnKinematics => "stellar_gnn_norm.json",
        ModelKind::GnnLocalization => "stellar_gnn_loc_norm.json",
        ModelKind::Siren => "stellar_siren_norm.json",
    }
}

/// Architecture versions frozen with contracts v1 (Stage 2).
pub fn architecture_version(model: &ModelKind) -> &'static str {
    match model {
        ModelKind::Pinn => "pinn-v1",
        ModelKind::GnnKinematics => "gnn-kinematics-v1",
        ModelKind::GnnLocalization => "gnn-loc-v1",
        ModelKind::Siren => "siren-v1",
    }
}

/// SHA-256 of a file's bytes as lowercase hex.
pub fn sha256_file_hex(path: &std::path::Path) -> Result<String, ArtifactError> {
    let bytes = std::fs::read(path)
        .map_err(|_| ArtifactError::FileNotFound(format!("missing file: {}", path.display())))?;
    Ok(crate::e2e::sha256_hex(&bytes))
}

/// On-disk rendering of a norm snapshot: pretty JSON. Every trainer must
/// write exactly this string to the norm file.
pub fn render_norm_file<T: serde::Serialize>(norm: &T) -> String {
    serde_json::to_string_pretty(norm).unwrap_or_else(|_| "{}".to_string())
}

/// Content hash of a norm snapshot. Stage 6 fix: this MUST be computed
/// over [`render_norm_file`] output (hash what you write). Hashing the
/// compact serialization while writing pretty JSON made every real
/// bundle fail validation.
pub fn hash_norm_rendered(rendered: &str) -> String {
    crate::e2e::sha256_hex(rendered.as_bytes())
}

/// Atomically writes `artifact.json` next to the weights (temp + rename).
/// Stage 5 (task 8): every training run leaves a versioned bundle whose
/// manifest validation gates resume/serve paths.
pub fn write_artifact_bundle(
    output_dir: &std::path::Path,
    manifest: &ArtifactManifestV1,
) -> Result<std::path::PathBuf, ArtifactError> {
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| ArtifactError::DeserializationError(e.to_string()))?;
    crate::runner::write_checkpoint_sidecar(output_dir, "artifact.json", &json)
        .map_err(|e| ArtifactError::ChecksumMismatch(e.to_string()))
}

/// Flat serving-layout manifest file name per model kind, for models dirs
/// that hold several models side by side (where a single `artifact.json`
/// would collide). Training output dirs keep `artifact.json`.
pub fn manifest_file_name(model: &ModelKind) -> &'static str {
    match model {
        ModelKind::Pinn => "stellar_model.artifact.json",
        ModelKind::GnnKinematics => "stellar_gnn_model.artifact.json",
        ModelKind::GnnLocalization => "stellar_gnn_loc_model.artifact.json",
        ModelKind::Siren => "stellar_siren_model.artifact.json",
    }
}

/// Registry status of one discovered model entry (Stage 6.7).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RegistryStatus {
    /// Manifest present, frozen version, matching arch, weight/norm hashes
    /// reproduce from disk.
    Verified,
    /// Weight + norm files present but no manifest: servable, provenance
    /// unverified (pre-Stage-6 deployments).
    LegacyUnverified,
    /// Entry found but failed validation; the reason is carried along and
    /// serving/reload paths must refuse it.
    Invalid(String),
}

/// One model entry in a scanned models directory.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RegisteredArtifact {
    pub kind: ModelKind,
    /// Directory holding the entry (`models/` for flat entries, the run
    /// dir for `artifact.json` layouts).
    pub dir: String,
    pub weight_file: String,
    pub norm_file: String,
    pub manifest: Option<ArtifactManifestV1>,
    pub status: RegistryStatus,
}

/// Loads `dir/artifact.json` and verifies it against itself: frozen
/// manifest version, expected architecture line, and weight/norm SHA-256
/// reproducing from the sibling files.
pub fn discover_bundle(dir: &std::path::Path) -> Result<RegisteredArtifact, ArtifactError> {
    let raw = std::fs::read_to_string(dir.join("artifact.json")).map_err(|_| {
        ArtifactError::FileNotFound(format!("missing artifact.json in {}", dir.display()))
    })?;
    let manifest: ArtifactManifestV1 = serde_json::from_str(&raw)
        .map_err(|e| ArtifactError::DeserializationError(e.to_string()))?;
    verify_manifest_against_files(dir, &manifest)?;
    let (weight_path, norm_path) = bundle_file_paths(dir, &manifest.model_kind);
    Ok(RegisteredArtifact {
        kind: manifest.model_kind.clone(),
        dir: dir.display().to_string(),
        weight_file: weight_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        norm_file: norm_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        manifest: Some(manifest),
        status: RegistryStatus::Verified,
    })
}

/// Resolves a bundle's weight/norm files: canonical per-kind names
/// first, then the legacy flat names (`stellar_model.bpk` /
/// `stellar_norm.json`) that older run dirs use regardless of kind.
fn bundle_file_paths(
    dir: &std::path::Path,
    kind: &ModelKind,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let weight = dir.join(weight_file_name(kind));
    let weight = if weight.exists() {
        weight
    } else {
        dir.join(weight_file_name(&ModelKind::Pinn))
    };
    let norm = dir.join(norm_file_name(kind));
    let norm = if norm.exists() {
        norm
    } else {
        dir.join(norm_file_name(&ModelKind::Pinn))
    };
    (weight, norm)
}

/// Self-verification shared by [`discover_bundle`] and the serving load
/// path: frozen version, architecture line, file hashes.
pub fn verify_manifest_against_files(
    dir: &std::path::Path,
    manifest: &ArtifactManifestV1,
) -> Result<(), ArtifactError> {
    if manifest.version != ARTIFACT_MANIFEST_VERSION {
        return Err(ArtifactError::IncompatibleArchitecture(format!(
            "manifest version {} != {ARTIFACT_MANIFEST_VERSION}",
            manifest.version
        )));
    }
    let expected_arch = architecture_version(&manifest.model_kind);
    if manifest.architecture_version != expected_arch {
        return Err(ArtifactError::IncompatibleArchitecture(format!(
            "arch {} != {expected_arch}",
            manifest.architecture_version
        )));
    }
    let (weight_path, norm_path) = bundle_file_paths(dir, &manifest.model_kind);
    for (path, want) in [
        (weight_path, &manifest.model_hash),
        (norm_path, &manifest.norm_hash),
    ] {
        let actual = sha256_file_hex(&path)?;
        if &actual != want {
            return Err(ArtifactError::ChecksumMismatch(format!(
                "{}: manifest {want}, actual {actual}",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Best-effort kind for an unreadable/invalid manifest: reads just the
/// `model_kind` field; falls back to `Pinn` with the real problem carried
/// in the [`RegistryStatus::Invalid`] message.
fn guess_manifest_kind(dir: &std::path::Path) -> ModelKind {
    std::fs::read_to_string(dir.join("artifact.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .and_then(|v| serde_json::from_value::<ModelKind>(v["model_kind"].clone()).ok())
        .unwrap_or(ModelKind::Pinn)
}

/// Stage 6.7 model registry: scans a models directory for servable
/// entries. Two layouts are recognized per model kind: a run directory
/// holding `artifact.json` (e.g. `models/gnn-v1/`), and the flat serving
/// layout (`stellar_gnn_model.bpk` + [`manifest_file_name`]). Weight+norm
/// files without any manifest are reported as [`RegistryStatus::LegacyUnverified`];
/// nothing here touches disk beyond reading.
pub fn scan_model_registry(models_dir: &std::path::Path) -> Vec<RegisteredArtifact> {
    let mut entries = Vec::new();
    if !models_dir.is_dir() {
        return entries;
    }
    // Run-dir layout: any immediate subdir with artifact.json.
    if let Ok(rd) = std::fs::read_dir(models_dir) {
        let mut subdirs: Vec<_> = rd.flatten().filter(|e| e.path().is_dir()).collect();
        subdirs.sort_by_key(|e| e.file_name());
        for sub in subdirs {
            let dir = sub.path();
            if !dir.join("artifact.json").exists() {
                continue;
            }
            match discover_bundle(&dir) {
                Ok(entry) => entries.push(entry),
                Err(err) => entries.push(RegisteredArtifact {
                    kind: guess_manifest_kind(&dir),
                    dir: dir.display().to_string(),
                    weight_file: String::new(),
                    norm_file: String::new(),
                    manifest: None,
                    status: RegistryStatus::Invalid(err.to_string()),
                }),
            }
        }
    }
    // Flat serving layout, per model kind.
    for kind in [
        ModelKind::Pinn,
        ModelKind::GnnKinematics,
        ModelKind::GnnLocalization,
        ModelKind::Siren,
    ] {
        let manifest_path = models_dir.join(manifest_file_name(&kind));
        if manifest_path.exists() {
            match std::fs::read_to_string(&manifest_path)
                .map_err(|e| ArtifactError::FileNotFound(e.to_string()))
                .and_then(|raw| {
                    serde_json::from_str::<ArtifactManifestV1>(&raw)
                        .map_err(|e| ArtifactError::DeserializationError(e.to_string()))
                })
                .and_then(|manifest| {
                    verify_manifest_against_files(models_dir, &manifest)?;
                    Ok(manifest)
                }) {
                Ok(manifest) => entries.push(RegisteredArtifact {
                    kind: kind.clone(),
                    dir: models_dir.display().to_string(),
                    weight_file: weight_file_name(&kind).to_string(),
                    norm_file: norm_file_name(&kind).to_string(),
                    manifest: Some(manifest),
                    status: RegistryStatus::Verified,
                }),
                Err(err) => entries.push(RegisteredArtifact {
                    kind: kind.clone(),
                    dir: models_dir.display().to_string(),
                    weight_file: weight_file_name(&kind).to_string(),
                    norm_file: norm_file_name(&kind).to_string(),
                    manifest: None,
                    status: RegistryStatus::Invalid(err.to_string()),
                }),
            }
            continue;
        }
        let has_files = models_dir.join(weight_file_name(&kind)).exists()
            && models_dir.join(norm_file_name(&kind)).exists();
        if has_files
            && !entries.iter().any(|e| {
                e.kind == kind
                    && e.status == RegistryStatus::Verified
                    && e.dir == models_dir.display().to_string()
            })
        {
            entries.push(RegisteredArtifact {
                kind: kind.clone(),
                dir: models_dir.display().to_string(),
                weight_file: weight_file_name(&kind).to_string(),
                norm_file: norm_file_name(&kind).to_string(),
                manifest: None,
                status: RegistryStatus::LegacyUnverified,
            });
        }
    }
    entries
}

/// Loads and validates an artifact bundle: manifest must exist, parse, be
/// version-frozen and reference weight/norm files whose SHA-256 matches.
pub fn validate_artifact_bundle(
    output_dir: &std::path::Path,
    expected: &ArtifactManifestV1,
) -> Result<(), ArtifactError> {
    let raw = std::fs::read_to_string(output_dir.join("artifact.json")).map_err(|_| {
        ArtifactError::FileNotFound(format!("missing artifact.json in {}", output_dir.display()))
    })?;
    let manifest: ArtifactManifestV1 = serde_json::from_str(&raw)
        .map_err(|e| ArtifactError::DeserializationError(e.to_string()))?;
    if manifest.version != ARTIFACT_MANIFEST_VERSION {
        return Err(ArtifactError::IncompatibleArchitecture(format!(
            "manifest version {} != {ARTIFACT_MANIFEST_VERSION}",
            manifest.version
        )));
    }
    if manifest != *expected {
        return Err(ArtifactError::ChecksumMismatch(
            "artifact.json differs from the run manifest".to_string(),
        ));
    }
    let (weight_path, norm_path) = bundle_file_paths(output_dir, &manifest.model_kind);
    for (path, want) in [
        (weight_path, &manifest.model_hash),
        (norm_path, &manifest.norm_hash),
    ] {
        let actual = sha256_file_hex(&path)?;
        if &actual != want {
            return Err(ArtifactError::ChecksumMismatch(format!(
                "{}: manifest {want}, actual {actual}",
                path.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::ModelKind;

    fn sample_manifest(model_kind: ModelKind) -> ArtifactManifestV1 {
        ArtifactManifestV1::new(
            model_kind,
            "gnn-loc-v1".into(),
            "model-hash".into(),
            "norm-hash".into(),
            "schema-hash".into(),
            "gaia_dr3".into(),
            "manifest-v1".into(),
            42,
            serde_json::json!({ "epochs": 120 }),
            "git-rev".into(),
            "cuda:0".into(),
        )
    }

    #[test]
    fn artifact_manifest_v1_round_trips_through_json() {
        let manifest = sample_manifest(ModelKind::GnnLocalization);
        let json = serde_json::to_string(&manifest).expect("serialize artifact manifest");
        let back: ArtifactManifestV1 = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, manifest);
    }

    #[test]
    fn compatibility_requires_model_norm_schema_and_dataset_match() {
        let manifest = sample_manifest(ModelKind::GnnLocalization);
        let ok = |m: &ArtifactManifestV1| {
            m.is_compatible_with(
                &ModelKind::GnnLocalization,
                "gnn-loc-v1",
                "norm-hash",
                "gaia_dr3",
                "manifest-v1",
                "schema-hash",
            )
        };
        assert!(ok(&manifest), "identical context must be compatible");

        let mut mismatched = manifest.clone();
        mismatched.model_kind = ModelKind::Pinn;
        assert!(!ok(&mismatched), "model kind must participate");

        let mut mismatched = manifest.clone();
        mismatched.norm_hash = "other-norm".into();
        assert!(!ok(&mismatched), "norm snapshot must participate");

        let mut mismatched = manifest.clone();
        mismatched.feature_schema_hash = "schema-v2".into();
        assert!(!ok(&mismatched), "feature schema must participate");

        let mut mismatched = manifest.clone();
        mismatched.dataset_version = "manifest-v2".into();
        assert!(!ok(&mismatched), "dataset version must participate");

        let mut mismatched = manifest;
        mismatched.architecture_version = "gnn-loc-v2".into();
        assert!(!ok(&mismatched), "architecture version must participate");
    }

    #[test]
    fn artifact_version_is_frozen() {
        assert_eq!(ARTIFACT_MANIFEST_VERSION, "1.0.0");
    }

    #[test]
    fn norm_snapshot_round_trips_through_json() {
        let norm = NormSnapshotV1 {
            kind: "standard".into(),
            path: "runs/pinn/norm.json".into(),
            data: serde_json::json!({ "ra_deg": { "mean": 180.0, "std": 90.0 } }),
            checksum: "cafe".into(),
        };
        let json = serde_json::to_string(&norm).unwrap();
        let back: NormSnapshotV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(back, norm);
    }

    #[test]
    fn weight_and_norm_names_are_stable_per_model_kind() {
        assert_eq!(weight_file_name(&ModelKind::Pinn), "stellar_model.bpk");
        assert_eq!(
            weight_file_name(&ModelKind::GnnKinematics),
            "stellar_gnn_model.bpk"
        );
        assert_eq!(
            weight_file_name(&ModelKind::Siren),
            "stellar_siren_model.bpk"
        );
        assert_eq!(norm_file_name(&ModelKind::Pinn), "stellar_norm.json");
        assert_eq!(
            architecture_version(&ModelKind::Pinn),
            architecture_version(&ModelKind::Pinn)
        );
    }

    #[test]
    fn artifact_bundle_round_trips_through_atomic_write_and_validate() {
        let dir = std::env::temp_dir().join("lnai-training-artifact-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(weight_file_name(&ModelKind::Pinn)), b"weights").unwrap();
        std::fs::write(dir.join(norm_file_name(&ModelKind::Pinn)), b"norm").unwrap();

        let mut manifest = sample_manifest(ModelKind::Pinn);
        manifest.model_hash =
            sha256_file_hex(&dir.join(weight_file_name(&ModelKind::Pinn))).unwrap();
        manifest.norm_hash = sha256_file_hex(&dir.join(norm_file_name(&ModelKind::Pinn))).unwrap();
        write_artifact_bundle(&dir, &manifest).expect("write bundle");
        validate_artifact_bundle(&dir, &manifest).expect("validate bundle");

        let mut tampered = manifest.clone();
        tampered.model_hash = "deadbeef".into();
        assert!(validate_artifact_bundle(&dir, &tampered).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_run_bundle(dir: &std::path::Path, kind: ModelKind, arch: &str) -> ArtifactManifestV1 {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(weight_file_name(&kind)), b"weights").unwrap();
        std::fs::write(dir.join(norm_file_name(&kind)), b"norm").unwrap();
        let mut manifest = sample_manifest(kind);
        manifest.architecture_version = arch.into();
        manifest.model_hash =
            sha256_file_hex(&dir.join(weight_file_name(&manifest.model_kind))).unwrap();
        manifest.norm_hash =
            sha256_file_hex(&dir.join(norm_file_name(&manifest.model_kind))).unwrap();
        write_artifact_bundle(dir, &manifest).expect("write bundle");
        manifest
    }

    #[test]
    fn registry_discovers_run_dir_flat_and_legacy_layouts() {
        let root = tempfile::tempdir().expect("tmpdir");
        let models = root.path();
        // Run-dir layout with a valid bundle.
        write_run_bundle(
            &models.join("gnn-v1"),
            ModelKind::GnnKinematics,
            "gnn-kinematics-v1",
        );
        // Flat serving layout with a valid manifest.
        std::fs::write(models.join(weight_file_name(&ModelKind::Pinn)), b"w").unwrap();
        std::fs::write(models.join(norm_file_name(&ModelKind::Pinn)), b"n").unwrap();
        let mut pinn = sample_manifest(ModelKind::Pinn);
        pinn.architecture_version = "pinn-v1".into();
        pinn.model_hash =
            sha256_file_hex(&models.join(weight_file_name(&ModelKind::Pinn))).unwrap();
        pinn.norm_hash = sha256_file_hex(&models.join(norm_file_name(&ModelKind::Pinn))).unwrap();
        std::fs::write(
            models.join(manifest_file_name(&ModelKind::Pinn)),
            serde_json::to_string_pretty(&pinn).unwrap(),
        )
        .unwrap();
        // Legacy files without any manifest.
        std::fs::write(models.join(weight_file_name(&ModelKind::Siren)), b"w").unwrap();
        std::fs::write(models.join(norm_file_name(&ModelKind::Siren)), b"n").unwrap();

        let entries = scan_model_registry(models);
        let status = |kind: &ModelKind, dir: &str| {
            entries
                .iter()
                .find(|e| &e.kind == kind && e.dir.ends_with(dir))
                .map(|e| e.status.clone())
        };
        assert_eq!(
            status(&ModelKind::GnnKinematics, "gnn-v1"),
            Some(RegistryStatus::Verified)
        );
        assert_eq!(
            status(&ModelKind::Pinn, "tmp"),
            None,
            "flat entries carry the models dir itself"
        );
        assert!(
            entries.iter().any(|e| e.kind == ModelKind::Pinn
                && e.status == RegistryStatus::Verified
                && e.dir == models.display().to_string()),
            "flat manifest verifies: {entries:?}"
        );
        assert!(
            entries.iter().any(|e| e.kind == ModelKind::Siren
                && e.status == RegistryStatus::LegacyUnverified),
            "manifest-less files report legacy: {entries:?}"
        );
    }

    #[test]
    fn registry_flags_tampered_weights_as_invalid() {
        let root = tempfile::tempdir().expect("tmpdir");
        let run = root.path().join("pinn-v1");
        write_run_bundle(&run, ModelKind::Pinn, "pinn-v1");
        std::fs::write(run.join(weight_file_name(&ModelKind::Pinn)), b"tampered").unwrap();

        let entries = scan_model_registry(root.path());
        assert_eq!(entries.len(), 1);
        assert!(
            matches!(entries[0].status, RegistryStatus::Invalid(_)),
            "tampered weights must invalidate: {:?}",
            entries[0].status
        );
        assert!(entries[0].manifest.is_none());
    }

    #[test]
    fn registry_on_missing_dir_is_empty() {
        let entries = scan_model_registry(std::path::Path::new("/nonexistent-models-dir-xyz"));
        assert!(entries.is_empty());
    }

    #[test]
    fn norm_hash_convention_matches_written_pretty_file() {
        use crate::pinn::dataset::NormParams;
        // Hash-what-you-write: the manifest hash must reproduce from the
        // pretty-printed file bytes (compact serialization hashes
        // differently and broke every real bundle once).
        let norm = NormParams {
            x_mean: 1.0,
            x_std: 2.0,
            y_mean: 3.0,
            y_std: 4.0,
            z_mean: 5.0,
            z_std: 6.0,
            bp_rp_mean: 7.0,
            bp_rp_std: 8.0,
            mg_mean: 9.0,
            mg_std: 10.0,
            log_teff_mean: 11.0,
            log_teff_std: 12.0,
            log_rad_mean: 13.0,
            log_rad_std: 14.0,
            log_mass_mean: 15.0,
            log_mass_std: 16.0,
            log_lum_mean: 17.0,
            log_lum_std: 18.0,
        };
        let rendered = render_norm_file(&norm);
        assert_ne!(
            rendered,
            serde_json::to_string(&norm).unwrap(),
            "pretty and compact renderings must differ (else the convention is vacuous)"
        );
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join("stellar_norm.json");
        std::fs::write(&path, &rendered).unwrap();
        assert_eq!(
            sha256_file_hex(&path).unwrap(),
            hash_norm_rendered(&rendered)
        );
    }

    #[test]
    fn atomic_write_publishes_content_and_leaves_no_tmp() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let target = dir.path().join("model.bpk");
        std::fs::write(&target, b"old").expect("seed");
        atomic_write_through(&target, "bpk.tmp", |tmp| {
            std::fs::write(tmp, b"new").map_err(|e| e.to_string())?;
            Ok(())
        })
        .expect("atomic write");
        assert_eq!(std::fs::read(&target).expect("read"), b"new");
        assert!(!target.with_extension("bpk.tmp").exists());
        let _ = std::fs::remove_dir_all(dir);
    }
}
