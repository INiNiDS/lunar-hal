use crate::spec::ModelKind;
use lunar_utils::time::current_time_ms;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Component, Path};

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
    std::fs::rename(&tmp, final_path).map_err(|e| format!("publish {}: {e}", final_path.display()))
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
        !norm_hash.is_empty()
            && !dataset_id.is_empty()
            && !dataset_version.is_empty()
            && !schema_hash.is_empty()
            && !self.git_revision.is_empty()
            && self.git_revision != "unknown"
            && &self.model_kind == model_kind
            && self.architecture_version == architecture_version
            && self.norm_hash == norm_hash
            && self.dataset_id == dataset_id
            && self.dataset_version == dataset_version
            && self.feature_schema_hash == schema_hash
    }

    /// Missing evidence for promotion. `verify_manifest_against_files` only
    /// establishes file integrity, not dataset provenance or model quality.
    pub fn release_blockers(&self) -> Vec<&'static str> {
        let mut blockers = Vec::new();
        if expected_feature_schema_hash(&self.model_kind)
            .is_none_or(|expected| self.feature_schema_hash != expected)
        {
            blockers.push("feature_schema_hash");
        }
        if self.dataset_id.trim().is_empty() {
            blockers.push("dataset_id");
        }
        if self.dataset_version.trim().is_empty() {
            blockers.push("dataset_version");
        }
        if self.git_revision.trim().is_empty() || self.git_revision == "unknown" {
            blockers.push("git_revision");
        }
        if !self.has_bound_spatial_holdout_evaluation() {
            blockers.push("spatial_holdout_evaluation");
        }
        blockers
    }

    fn has_bound_spatial_holdout_evaluation(&self) -> bool {
        let Some(holdout) = self
            .evaluation_metrics
            .as_ref()
            .and_then(|metrics| metrics.get("spatial_holdout"))
        else {
            return false;
        };
        let matches_artifact = |field: &str, expected: &str| {
            holdout.get(field).and_then(serde_json::Value::as_str) == Some(expected)
        };
        let report_file = holdout
            .get("report_file")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let report_hash = holdout
            .get("report_sha256")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        holdout.get("passed").and_then(serde_json::Value::as_bool) == Some(true)
            && matches_artifact("model_hash", &self.model_hash)
            && matches_artifact("norm_hash", &self.norm_hash)
            && matches_artifact("dataset_id", &self.dataset_id)
            && matches_artifact("dataset_version", &self.dataset_version)
            && matches_artifact("feature_schema_hash", &self.feature_schema_hash)
            && is_safe_relative_report_path(report_file)
            && report_hash.len() == 64
            && report_hash.bytes().all(|byte| byte.is_ascii_hexdigit())
    }
}

fn is_safe_relative_report_path(path: &str) -> bool {
    !path.is_empty()
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Stable serving feature identifiers. These values pin both feature meaning
/// and order; a non-empty but unrelated identifier is not sufficient.
pub fn expected_feature_schema_hash(model_kind: &ModelKind) -> Option<&'static str> {
    match model_kind {
        ModelKind::Pinn => Some("pinn-input-x,y,z,bp_rp,g_mag-v1"),
        ModelKind::GnnKinematics => {
            Some("gnn-serving-pinn-derived-log_teff,log_rad,log_mass,log_lum,mg,x,y,z-v1")
        }
        ModelKind::Siren => Some("siren-serving-uv,bp_rp,m_g,log_teff-v1"),
        // Localization remains experimental until a versioned serving
        // schema and trained artifact are published.
        ModelKind::GnnLocalization => None,
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
    let mut file = std::fs::File::open(path)
        .map_err(|_| ArtifactError::FileNotFound(format!("missing file: {}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            ArtifactError::ChecksumMismatch(format!("read {} for hashing: {error}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Fingerprints the exact training input together with any sibling canonical
/// dataset/view manifests. This remains content-addressed when a manually
/// edited parquet no longer agrees with its older manifest checksum.
pub fn dataset_fingerprint(path: &std::path::Path) -> Result<String, ArtifactError> {
    let input_hash = sha256_file_hex(path)?;
    let mut identity = format!("dataset-input-v1\nfile_sha256:{input_hash}\n");
    if let Some(parent) = path.parent() {
        for name in ["manifest.json", "views_manifest.json"] {
            let manifest_path = parent.join(name);
            if manifest_path.is_file() {
                identity.push_str(&format!(
                    "{name}_sha256:{}\n",
                    sha256_file_hex(&manifest_path)?
                ));
            }
        }
    }
    Ok(crate::e2e::sha256_hex(identity.as_bytes()))
}

/// Resolves the build revision at runtime for workers built without an
/// explicit compile-time revision, and marks tracked local modifications.
pub fn current_git_revision() -> String {
    let revision = option_env!("LUNAR_AI_GIT_REV")
        .filter(|revision| !revision.trim().is_empty() && *revision != "unknown")
        .map(str::to_string)
        .or_else(|| {
            std::env::var("LUNAR_AI_GIT_REV")
                .ok()
                .filter(|revision| !revision.trim().is_empty() && revision != "unknown")
        })
        .or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "--verify", "HEAD"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|revision| revision.trim().to_string())
                .filter(|revision| !revision.is_empty())
        });
    let Some(mut revision) = revision else {
        return "unknown".to_string();
    };
    let has_tracked_changes = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    if has_tracked_changes && !revision.ends_with("-dirty") {
        revision.push_str("-dirty");
    }
    revision
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
    /// Manifest and artifact integrity are valid, and all release evidence
    /// is present.
    Verified,
    /// File integrity is valid, but required provenance or evaluation
    /// evidence is missing; serving and promotion must refuse this bundle.
    ReleaseBlocked(Vec<String>),
    /// Weight + norm files present but no manifest; not safe to serve.
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

fn release_status(manifest: &ArtifactManifestV1) -> RegistryStatus {
    let blockers: Vec<String> = manifest
        .release_blockers()
        .into_iter()
        .map(str::to_string)
        .collect();
    if blockers.is_empty() {
        RegistryStatus::Verified
    } else {
        RegistryStatus::ReleaseBlocked(blockers)
    }
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
    let status = release_status(&manifest);
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
        status,
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
    verify_spatial_holdout_report(dir, manifest)?;
    Ok(())
}

fn verify_spatial_holdout_report(
    dir: &Path,
    manifest: &ArtifactManifestV1,
) -> Result<(), ArtifactError> {
    let Some(holdout) = manifest
        .evaluation_metrics
        .as_ref()
        .and_then(|metrics| metrics.get("spatial_holdout"))
    else {
        return Ok(());
    };
    if holdout.get("passed").and_then(serde_json::Value::as_bool) != Some(true) {
        return Ok(());
    }
    let report_file = holdout
        .get("report_file")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !is_safe_relative_report_path(report_file) {
        return Err(ArtifactError::IncompatibleArchitecture(
            "spatial holdout report path must be relative to the artifact directory".into(),
        ));
    }
    let root = dir
        .canonicalize()
        .map_err(|error| ArtifactError::FileNotFound(format!("artifact directory: {error}")))?;
    let report_path = dir.join(report_file);
    let report_path = report_path.canonicalize().map_err(|error| {
        ArtifactError::FileNotFound(format!(
            "spatial holdout report {}: {error}",
            report_path.display()
        ))
    })?;
    if !report_path.starts_with(&root) || !report_path.is_file() {
        return Err(ArtifactError::IncompatibleArchitecture(
            "spatial holdout report must be a regular file inside the artifact directory".into(),
        ));
    }
    let report_hash = holdout
        .get("report_sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let report_bytes = std::fs::read(&report_path).map_err(|error| {
        ArtifactError::FileNotFound(format!("{}: {error}", report_path.display()))
    })?;
    let actual_hash = crate::e2e::sha256_hex(&report_bytes);
    if actual_hash != report_hash {
        return Err(ArtifactError::ChecksumMismatch(format!(
            "spatial holdout report {}: manifest {report_hash}, actual {actual_hash}",
            report_path.display()
        )));
    }
    let report: serde_json::Value = serde_json::from_slice(&report_bytes)
        .map_err(|error| ArtifactError::DeserializationError(error.to_string()))?;
    let report_kind = serde_json::to_value(&manifest.model_kind)
        .map_err(|error| ArtifactError::DeserializationError(error.to_string()))?;
    let bound = report.get("passed").and_then(serde_json::Value::as_bool) == Some(true)
        && report
            .get("gate_version")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|version| !version.trim().is_empty())
        && report
            .get("metrics")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|metrics| !metrics.is_empty())
        && report.get("model_kind") == Some(&report_kind)
        && report
            .get("architecture_version")
            .and_then(serde_json::Value::as_str)
            == Some(manifest.architecture_version.as_str())
        && report.get("model_hash").and_then(serde_json::Value::as_str)
            == Some(manifest.model_hash.as_str())
        && report.get("norm_hash").and_then(serde_json::Value::as_str)
            == Some(manifest.norm_hash.as_str())
        && report.get("dataset_id").and_then(serde_json::Value::as_str)
            == Some(manifest.dataset_id.as_str())
        && report
            .get("dataset_version")
            .and_then(serde_json::Value::as_str)
            == Some(manifest.dataset_version.as_str())
        && report
            .get("feature_schema_hash")
            .and_then(serde_json::Value::as_str)
            == Some(manifest.feature_schema_hash.as_str());
    if !bound {
        return Err(ArtifactError::IncompatibleDataset(
            "spatial holdout report does not bind a passing result to this artifact".into(),
        ));
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
                    if &manifest.model_kind != &kind {
                        return Err(ArtifactError::IncompatibleArchitecture(format!(
                            "manifest kind {:?} != registry kind {:?}",
                            manifest.model_kind, kind
                        )));
                    }
                    verify_manifest_against_files(models_dir, &manifest)?;
                    Ok(manifest)
                }) {
                Ok(manifest) => {
                    let status = release_status(&manifest);
                    entries.push(RegisteredArtifact {
                        kind: kind.clone(),
                        dir: models_dir.display().to_string(),
                        weight_file: weight_file_name(&kind).to_string(),
                        norm_file: norm_file_name(&kind).to_string(),
                        manifest: Some(manifest),
                        status,
                    });
                }
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
            && !entries
                .iter()
                .any(|e| e.kind == kind && e.dir == models_dir.display().to_string())
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
        let report_file = format!("{}_spatial_holdout_report.json", model_kind.slug());
        let schema_hash = expected_feature_schema_hash(&model_kind)
            .unwrap_or("gnn-localization-unreleased-schema-v1");
        let mut manifest = ArtifactManifestV1::new(
            model_kind,
            "gnn-loc-v1".into(),
            "model-hash".into(),
            "norm-hash".into(),
            schema_hash.into(),
            "gaia_dr3".into(),
            "manifest-v1".into(),
            42,
            serde_json::json!({ "epochs": 120 }),
            "git-rev".into(),
            "cuda:0".into(),
        );
        manifest.evaluation_metrics = Some(serde_json::json!({
            "spatial_holdout": {
                "passed": true,
                "model_hash": manifest.model_hash.clone(),
                "norm_hash": manifest.norm_hash.clone(),
                "dataset_id": manifest.dataset_id.clone(),
                "dataset_version": manifest.dataset_version.clone(),
                "feature_schema_hash": manifest.feature_schema_hash.clone(),
                "report_file": report_file,
                "report_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            }
        }));
        manifest
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
        let schema_hash = manifest.feature_schema_hash.clone();
        let ok = |m: &ArtifactManifestV1| {
            m.is_compatible_with(
                &ModelKind::GnnLocalization,
                "gnn-loc-v1",
                "norm-hash",
                "gaia_dr3",
                "manifest-v1",
                &schema_hash,
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

        let mut incomplete = sample_manifest(ModelKind::GnnLocalization);
        incomplete.dataset_version.clear();
        assert!(
            !incomplete.is_compatible_with(
                &ModelKind::GnnLocalization,
                "gnn-loc-v1",
                "norm-hash",
                "gaia_dr3",
                "",
                &schema_hash
            ),
            "matching empty provenance must not be compatible"
        );
    }

    #[test]
    fn file_integrity_does_not_certify_model_quality() {
        let mut manifest = sample_manifest(ModelKind::Pinn);
        manifest.dataset_version.clear();
        manifest.git_revision = "unknown".into();
        manifest.evaluation_metrics = Some(serde_json::json!({"best_val_loss": 0.1}));
        assert!(manifest.release_blockers().contains(&"dataset_version"));
        assert!(manifest.release_blockers().contains(&"git_revision"));
        assert!(
            manifest
                .release_blockers()
                .contains(&"spatial_holdout_evaluation")
        );
    }

    #[test]
    fn spatial_holdout_must_explicitly_pass_before_release() {
        let mut manifest = sample_manifest(ModelKind::Pinn);
        manifest.evaluation_metrics = Some(serde_json::json!({
            "spatial_holdout": {
                "passed": false,
                "model_hash": manifest.model_hash.clone(),
                "norm_hash": manifest.norm_hash.clone(),
                "dataset_id": manifest.dataset_id.clone(),
                "dataset_version": manifest.dataset_version.clone(),
                "feature_schema_hash": manifest.feature_schema_hash.clone(),
                "report_file": "spatial_holdout_report.json",
                "report_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            }
        }));
        assert!(
            manifest
                .release_blockers()
                .contains(&"spatial_holdout_evaluation")
        );

        manifest.evaluation_metrics = Some(serde_json::json!({
            "spatial_holdout": {
                "passed": true,
                "model_hash": manifest.model_hash.clone(),
                "norm_hash": manifest.norm_hash.clone(),
                "dataset_id": manifest.dataset_id.clone(),
                "dataset_version": manifest.dataset_version.clone(),
                "feature_schema_hash": manifest.feature_schema_hash.clone(),
                "report_file": "spatial_holdout_report.json",
                "report_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
            }
        }));
        assert!(
            !manifest
                .release_blockers()
                .contains(&"spatial_holdout_evaluation")
        );
    }

    #[test]
    fn spatial_holdout_evaluation_must_bind_to_this_artifact() {
        let mut manifest = sample_manifest(ModelKind::Pinn);
        manifest.evaluation_metrics.as_mut().unwrap()["spatial_holdout"]["model_hash"] =
            serde_json::Value::String("different-model-hash".into());
        assert!(
            manifest
                .release_blockers()
                .contains(&"spatial_holdout_evaluation")
        );

        manifest = sample_manifest(ModelKind::Pinn);
        manifest.evaluation_metrics.as_mut().unwrap()["spatial_holdout"]["report_sha256"] =
            serde_json::Value::String("not-a-sha256".into());
        assert!(
            manifest
                .release_blockers()
                .contains(&"spatial_holdout_evaluation")
        );
    }

    #[test]
    fn release_gate_rejects_unknown_feature_schema_ids() {
        let mut manifest = sample_manifest(ModelKind::Pinn);
        manifest.feature_schema_hash = "some-nonempty-but-unrelated-schema".into();
        assert!(manifest.release_blockers().contains(&"feature_schema_hash"));

        let localization = sample_manifest(ModelKind::GnnLocalization);
        assert!(
            localization
                .release_blockers()
                .contains(&"feature_schema_hash")
        );
    }

    #[test]
    fn integrity_valid_bundle_without_release_evidence_is_blocked() {
        let root = tempfile::tempdir().expect("tmpdir");
        let models = root.path();
        std::fs::write(models.join(weight_file_name(&ModelKind::Pinn)), b"w").unwrap();
        std::fs::write(models.join(norm_file_name(&ModelKind::Pinn)), b"n").unwrap();
        let mut manifest = sample_manifest(ModelKind::Pinn);
        manifest.architecture_version = "pinn-v1".into();
        manifest.dataset_version.clear();
        manifest.evaluation_metrics = None;
        manifest.model_hash =
            sha256_file_hex(&models.join(weight_file_name(&ModelKind::Pinn))).unwrap();
        manifest.norm_hash =
            sha256_file_hex(&models.join(norm_file_name(&ModelKind::Pinn))).unwrap();
        std::fs::write(
            models.join(manifest_file_name(&ModelKind::Pinn)),
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let entries = scan_model_registry(models);
        let entry = entries
            .iter()
            .find(|entry| entry.kind == ModelKind::Pinn)
            .unwrap();
        assert!(matches!(
            &entry.status,
            RegistryStatus::ReleaseBlocked(blockers)
                if blockers.contains(&"dataset_version".to_string())
                    && blockers.contains(&"spatial_holdout_evaluation".to_string())
        ));
        assert!(
            entry.manifest.is_some(),
            "hash-valid manifest remains inspectable"
        );
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
        {
            let spatial = manifest
                .evaluation_metrics
                .as_mut()
                .and_then(|metrics| metrics.get_mut("spatial_holdout"))
                .unwrap();
            spatial["model_hash"] = serde_json::Value::String(manifest.model_hash.clone());
            spatial["norm_hash"] = serde_json::Value::String(manifest.norm_hash.clone());
            spatial["dataset_version"] =
                serde_json::Value::String(manifest.dataset_version.clone());
            spatial["feature_schema_hash"] =
                serde_json::Value::String(manifest.feature_schema_hash.clone());
        }
        write_spatial_holdout_report(dir, &mut manifest);
        write_artifact_bundle(dir, &manifest).expect("write bundle");
        manifest
    }

    fn write_spatial_holdout_report(dir: &std::path::Path, manifest: &mut ArtifactManifestV1) {
        let report_file = format!("{}_spatial_holdout_report.json", manifest.model_kind.slug());
        let report = serde_json::json!({
            "passed": true,
            "gate_version": "fixture-test-v1",
            "model_kind": manifest.model_kind.clone(),
            "architecture_version": manifest.architecture_version.clone(),
            "model_hash": manifest.model_hash.clone(),
            "norm_hash": manifest.norm_hash.clone(),
            "dataset_id": manifest.dataset_id.clone(),
            "dataset_version": manifest.dataset_version.clone(),
            "feature_schema_hash": manifest.feature_schema_hash.clone(),
            "metrics": { "fixture_metric": 0.0 }
        });
        let report_path = dir.join(&report_file);
        std::fs::write(&report_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        let report_hash = sha256_file_hex(&report_path).unwrap();
        let spatial = manifest
            .evaluation_metrics
            .as_mut()
            .and_then(|metrics| metrics.get_mut("spatial_holdout"))
            .unwrap();
        spatial["report_file"] = serde_json::Value::String(report_file);
        spatial["report_sha256"] = serde_json::Value::String(report_hash);
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
        {
            let spatial = pinn
                .evaluation_metrics
                .as_mut()
                .and_then(|metrics| metrics.get_mut("spatial_holdout"))
                .unwrap();
            spatial["model_hash"] = serde_json::Value::String(pinn.model_hash.clone());
            spatial["norm_hash"] = serde_json::Value::String(pinn.norm_hash.clone());
            spatial["dataset_version"] = serde_json::Value::String(pinn.dataset_version.clone());
            spatial["feature_schema_hash"] =
                serde_json::Value::String(pinn.feature_schema_hash.clone());
        }
        write_spatial_holdout_report(models, &mut pinn);
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
    fn registry_flags_missing_or_tampered_holdout_report_as_invalid() {
        let root = tempfile::tempdir().expect("tmpdir");
        let run = root.path().join("pinn-v1");
        write_run_bundle(&run, ModelKind::Pinn, "pinn-v1");
        let report_file = format!("{}_spatial_holdout_report.json", ModelKind::Pinn.slug());
        std::fs::write(run.join(report_file), b"tampered report").unwrap();

        let entries = scan_model_registry(root.path());
        assert_eq!(entries.len(), 1);
        assert!(
            matches!(entries[0].status, RegistryStatus::Invalid(_)),
            "tampered quality report must invalidate release evidence: {:?}",
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
    fn dataset_fingerprint_tracks_input_bytes_and_sibling_manifests() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target")
            .join(format!("dataset-fingerprint-{nonce}"));
        std::fs::create_dir_all(&root).unwrap();
        let input = root.join("view_pinn.parquet");
        std::fs::write(&input, b"input-v1").unwrap();
        let first = dataset_fingerprint(&input).unwrap();
        std::fs::write(root.join("views_manifest.json"), b"manifest-v1").unwrap();
        let with_manifest = dataset_fingerprint(&input).unwrap();
        assert_ne!(first, with_manifest);
        std::fs::write(&input, b"input-v2").unwrap();
        let with_new_input = dataset_fingerprint(&input).unwrap();
        assert_ne!(with_manifest, with_new_input);
        std::fs::write(root.join("manifest.json"), b"canonical-manifest-v1").unwrap();
        let with_both_manifests = dataset_fingerprint(&input).unwrap();
        assert_ne!(with_new_input, with_both_manifests);
        std::fs::remove_dir_all(root).unwrap();
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
