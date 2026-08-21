use crate::spec::ModelKind;
use lunar_utils::time::current_time_ms;
use serde::{Deserialize, Serialize};

/// Version of the artifact manifest structure itself.
pub const ARTIFACT_MANIFEST_VERSION: &str = "1.0.0";

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
}
