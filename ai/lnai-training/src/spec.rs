//! Stage 5 (task 1/9): shared training specs with per-model capability
//! validation. CLI and Testbench build the same [`TrainingSpec`]; the worker
//! binaries execute it. A universal bag of CLI flags is rejected — each model
//! kind declares exactly which parameters it accepts.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Pinn,
    GnnKinematics,
    GnnLocalization,
    Siren,
}

impl ModelKind {
    pub fn slug(&self) -> &'static str {
        match self {
            ModelKind::Pinn => "pinn",
            ModelKind::GnnKinematics => "gnn_kinematics",
            ModelKind::GnnLocalization => "gnn_localization",
            ModelKind::Siren => "siren",
        }
    }

    pub fn binary_name(&self) -> &'static str {
        match self {
            ModelKind::Pinn => "lnai",
            ModelKind::GnnKinematics => "lnai-gnn",
            ModelKind::GnnLocalization => "lnai-gnn-loc",
            ModelKind::Siren => "lnai-siren",
        }
    }

    /// Accepts both canonical slugs and the legacy Testbench slugs
    /// (`"gnn"` → [`ModelKind::GnnKinematics`]).
    pub fn from_slug_loose(slug: &str) -> Option<Self> {
        match slug {
            "pinn" => Some(ModelKind::Pinn),
            "gnn" | "gnn_kinematics" | "gnn-kinematics" => Some(ModelKind::GnnKinematics),
            "gnn_localization" | "gnn-localization" | "gnn-loc" => Some(ModelKind::GnnLocalization),
            "siren" => Some(ModelKind::Siren),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PinnConfig {
    pub physics_weight: f64,
    pub hidden_dim: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GnnKinematicsConfig {
    pub knn_k: u32,
    pub hidden_dim: u32,
    pub output_dim: u32,
    pub max_group_size: u32,
    pub radius_pc: f32,
    pub physics_weight: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SirenConfig {
    pub texture_size: u32,
    pub hidden_dim: u32,
    pub max_stars: u32,
    pub seed: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationConfig {
    pub radius: f32,
    pub max_slots: u32,
    pub mask_ratio: f32,
    pub loss_weights: LocalizationLossWeights,
    pub seed: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationLossWeights {
    pub existence: f32,
    pub position_nll: f32,
    pub chamfer: f32,
    pub feature: f32,
    pub calibration: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelConfig {
    Pinn(PinnConfig),
    GnnKinematics(GnnKinematicsConfig),
    GnnLocalization(LocalizationConfig),
    Siren(SirenConfig),
}

impl ModelConfig {
    pub fn kind(&self) -> ModelKind {
        match self {
            ModelConfig::Pinn(_) => ModelKind::Pinn,
            ModelConfig::GnnKinematics(_) => ModelKind::GnnKinematics,
            ModelConfig::GnnLocalization(_) => ModelKind::GnnLocalization,
            ModelConfig::Siren(_) => ModelKind::Siren,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrainingSpec {
    pub model: ModelKind,
    pub config: ModelConfig,
    pub dataset_manifest_hash: String,
    /// Explicit dataset path; `None` lets the worker fall back to its
    /// compiled-in default candidates (legacy behaviour).
    pub data_path: Option<String>,
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub val_frac: f32,
    pub output_dir: String,
    pub resume_from: Option<String>,
    pub holdout: Option<String>,
    pub gpu_index: u32,
    pub patience: u32,
    pub grad_accum: u32,
    pub clip_grad_norm: f64,
    /// Explicit global seed; `None` derives a stable seed from
    /// dataset/model/epochs (see `runner::effective_train_seed`).
    pub seed: Option<u64>,
    pub model_file: String,
    pub norm_file: String,
}

impl TrainingSpec {
    /// Per-model capability gate: rejects parameter combinations the model
    /// binary does not implement instead of silently ignoring them.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.model != self.config.kind() {
            errors.push(format!(
                "config kind {:?} does not match model {:?}",
                self.config.kind(),
                self.model
            ));
        }
        if self.epochs == 0 {
            errors.push("epochs must be >= 1".to_string());
        }
        if self.batch_size == 0 {
            errors.push("batch_size must be >= 1".to_string());
        }
        if !(0.0..1.0).contains(&f64::from(self.val_frac)) {
            errors.push(format!("val_frac must be in [0, 1), got {}", self.val_frac));
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            errors.push(format!("lr must be finite and > 0, got {}", self.lr));
        }
        if self.grad_accum == 0 {
            errors.push("grad_accum must be >= 1".to_string());
        }
        match &self.config {
            ModelConfig::Pinn(cfg) => {
                if !cfg.physics_weight.is_finite() || cfg.physics_weight < 0.0 {
                    errors.push("pinn physics_weight must be finite and >= 0".to_string());
                }
                if cfg.hidden_dim == 0 {
                    errors.push("pinn hidden_dim must be >= 1".to_string());
                }
            }
            ModelConfig::GnnKinematics(cfg) => {
                if cfg.knn_k == 0 {
                    errors.push("gnn knn_k must be >= 1".to_string());
                }
                if cfg.hidden_dim == 0 {
                    errors.push("gnn hidden_dim must be >= 1".to_string());
                }
                if cfg.output_dim != 3 {
                    errors.push(format!(
                        "gnn output_dim is frozen to 3 (vx/vy/vz), got {}",
                        cfg.output_dim
                    ));
                }
                if !cfg.physics_weight.is_finite() || cfg.physics_weight < 0.0 {
                    errors.push("gnn physics_weight must be finite and >= 0".to_string());
                }
            }
            ModelConfig::GnnLocalization(_) => {
                errors.push(
                    "gnn_localization training is not implemented in Stage 5 (Stage 8)".to_string(),
                );
            }
            ModelConfig::Siren(cfg) => {
                if cfg.texture_size == 0 {
                    errors.push("siren texture_size must be >= 1".to_string());
                }
                if cfg.max_stars == 0 {
                    errors.push("siren max_stars must be >= 1".to_string());
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Builds the exact worker-binary argv for this spec, byte-compatible
    /// with the pre-Stage-5 binaries (parity gate). The binary path itself
    /// is resolved by the caller; only flags are returned.
    pub fn worker_argv(&self) -> Vec<String> {
        let data = self.data_path.clone().unwrap_or_default();
        let mut argv = vec![
            "--data".to_string(),
            data,
            "--output-dir".to_string(),
            self.output_dir.clone(),
            "--model-file".to_string(),
            self.model_file.clone(),
            "--norm-file".to_string(),
            self.norm_file.clone(),
            "--epochs".to_string(),
            self.epochs.to_string(),
            "--lr".to_string(),
            self.lr.to_string(),
            "--val-frac".to_string(),
            self.val_frac.to_string(),
            "--gpu-index".to_string(),
            self.gpu_index.to_string(),
            "--patience".to_string(),
            self.patience.to_string(),
            "--clip-grad-norm".to_string(),
            self.clip_grad_norm.to_string(),
            "--grad-accum".to_string(),
            self.grad_accum.to_string(),
        ];
        if let Some(resume) = &self.resume_from
            && !resume.is_empty()
        {
            argv.push("--resume-from".to_string());
            argv.push(resume.clone());
        }
        if let Some(holdout) = &self.holdout
            && !holdout.is_empty()
        {
            argv.push("--holdout".to_string());
            argv.push(holdout.clone());
        }
        match &self.config {
            ModelConfig::Pinn(cfg) => {
                argv.push("--batch-size".to_string());
                argv.push(self.batch_size.to_string());
                argv.push("--physics-weight".to_string());
                argv.push(cfg.physics_weight.to_string());
            }
            ModelConfig::GnnKinematics(cfg) => {
                // Legacy lnai-gnn flag name is --max-nodes carrying the batch budget.
                argv.push("--max-nodes".to_string());
                argv.push(self.batch_size.to_string());
                argv.push("--physics-weight".to_string());
                argv.push(cfg.physics_weight.to_string());
                argv.push("--knn-k".to_string());
                argv.push(cfg.knn_k.to_string());
                argv.push("--hidden-dim".to_string());
                argv.push(cfg.hidden_dim.to_string());
                argv.push("--max-group-size".to_string());
                argv.push(cfg.max_group_size.to_string());
                argv.push("--radius-pc".to_string());
                argv.push(cfg.radius_pc.to_string());
            }
            ModelConfig::GnnLocalization(_) => {}
            ModelConfig::Siren(cfg) => {
                argv.push("--batch-size".to_string());
                argv.push(self.batch_size.to_string());
                argv.push("--texture-size".to_string());
                argv.push(cfg.texture_size.to_string());
                argv.push("--max-stars".to_string());
                argv.push(cfg.max_stars.to_string());
                argv.push("--seed".to_string());
                argv.push(cfg.seed.to_string());
            }
        }
        argv
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EvaluationSpec {
    pub model: ModelKind,
    pub artifact_hash: String,
    pub dataset_manifest_hash: String,
    /// Explicit dataset path; `None` keeps the worker default candidates.
    pub data_path: Option<String>,
    pub batch_size: u32,
    pub output_dir: String,
    pub seed: Option<u64>,
}

impl EvaluationSpec {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if matches!(self.model, ModelKind::GnnLocalization) {
            errors.push(
                "gnn_localization evaluation is not implemented in Stage 5 (Stage 8)".to_string(),
            );
        }
        if self.batch_size == 0 {
            errors.push("batch_size must be >= 1".to_string());
        }
        if self.artifact_hash.is_empty() {
            errors.push("artifact_hash must be non-empty".to_string());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Argv for the worker's read-only evaluation mode: same binary, plus
    /// `--evaluate-only`, never an optimizer step.
    pub fn worker_argv(&self, holdout: Option<&str>) -> Vec<String> {
        let mut argv = vec![
            "--data".to_string(),
            self.data_path.clone().unwrap_or_default(),
            "--output-dir".to_string(),
            self.output_dir.clone(),
            "--batch-size".to_string(),
            self.batch_size.to_string(),
            "--evaluate-only".to_string(),
        ];
        if let Some(seed) = self.seed {
            argv.push("--seed".to_string());
            argv.push(seed.to_string());
        }
        if let Some(holdout) = holdout
            && !holdout.is_empty()
        {
            argv.push("--holdout".to_string());
            argv.push(holdout.to_string());
        }
        argv
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BenchmarkSpec {
    pub model: ModelKind,
    pub artifact_hash: String,
    pub iterations: u32,
    pub warmup_iterations: u32,
    pub output_dir: String,
    pub batch_size: u32,
    pub seed: Option<u64>,
}

impl BenchmarkSpec {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.iterations == 0 {
            errors.push("iterations must be >= 1".to_string());
        }
        if self.artifact_hash.is_empty() {
            errors.push("artifact_hash must be non-empty".to_string());
        }
        if self.batch_size == 0 {
            errors.push("batch_size must be >= 1".to_string());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Argv for the worker's benchmark mode: loads the artifact, times
    /// forward passes, writes `benchmark.json`, trains nothing.
    pub fn worker_argv(&self) -> Vec<String> {
        let mut argv = vec![
            "--output-dir".to_string(),
            self.output_dir.clone(),
            "--batch-size".to_string(),
            self.batch_size.to_string(),
            "--benchmark-iters".to_string(),
            self.iterations.to_string(),
            "--benchmark-warmup".to_string(),
            self.warmup_iterations.to_string(),
        ];
        if let Some(seed) = self.seed {
            argv.push("--seed".to_string());
            argv.push(seed.to_string());
        }
        argv
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobKind {
    Train(TrainingSpec),
    Evaluate(EvaluationSpec),
    Benchmark(BenchmarkSpec),
    Custom(String),
}

impl JobKind {
    pub fn model_kind(&self) -> &ModelKind {
        match self {
            JobKind::Train(t) => &t.model,
            JobKind::Evaluate(v) => &v.model,
            JobKind::Benchmark(b) => &b.model,
            JobKind::Custom(_) => {
                static DEFAULT: ModelKind = ModelKind::Pinn;
                &DEFAULT
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_training_spec() -> TrainingSpec {
        TrainingSpec {
            model: ModelKind::GnnLocalization,
            config: ModelConfig::GnnLocalization(LocalizationConfig {
                radius: 25.0,
                max_slots: 16,
                mask_ratio: 0.3,
                loss_weights: LocalizationLossWeights {
                    existence: 1.0,
                    position_nll: 1.0,
                    chamfer: 1.0,
                    feature: 0.5,
                    calibration: 0.25,
                },
                seed: 7,
            }),
            dataset_manifest_hash: "manifest-hash".into(),
            data_path: None,
            epochs: 120,
            batch_size: 2048,
            lr: 5e-4,
            val_frac: 0.1,
            output_dir: "runs/loc".into(),
            resume_from: None,
            holdout: Some("spatial_tile_h".into()),
            gpu_index: 0,
            patience: 20,
            grad_accum: 2,
            clip_grad_norm: 1.0,
            seed: None,
            model_file: "stellar_gnn_loc_model.bpk".into(),
            norm_file: "stellar_gnn_loc_norm.json".into(),
        }
    }

    fn pinn_training_spec() -> TrainingSpec {
        TrainingSpec {
            model: ModelKind::Pinn,
            config: ModelConfig::Pinn(PinnConfig {
                physics_weight: 0.1,
                hidden_dim: 256,
            }),
            dataset_manifest_hash: "manifest-hash".into(),
            data_path: Some("ai_data/clean_stars2.parquet".into()),
            epochs: 50,
            batch_size: 2048,
            lr: 5e-4,
            val_frac: 0.1,
            output_dir: "models".into(),
            resume_from: None,
            holdout: None,
            gpu_index: 0,
            patience: 20,
            grad_accum: 2,
            clip_grad_norm: 1.0,
            seed: Some(42),
            model_file: "stellar_model.bpk".into(),
            norm_file: "stellar_norm.json".into(),
        }
    }

    #[test]
    fn model_kinds_are_frozen() {
        assert_eq!(ModelKind::Pinn.slug(), "pinn");
        assert_eq!(ModelKind::GnnKinematics.slug(), "gnn_kinematics");
        assert_eq!(ModelKind::GnnLocalization.slug(), "gnn_localization");
        assert_eq!(ModelKind::Siren.slug(), "siren");
        for kind in [
            ModelKind::Pinn,
            ModelKind::GnnKinematics,
            ModelKind::GnnLocalization,
            ModelKind::Siren,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: ModelKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn legacy_testbench_slugs_resolve() {
        assert_eq!(
            ModelKind::from_slug_loose("gnn"),
            Some(ModelKind::GnnKinematics)
        );
        assert_eq!(ModelKind::from_slug_loose("pinn"), Some(ModelKind::Pinn));
        assert_eq!(ModelKind::from_slug_loose("siren"), Some(ModelKind::Siren));
        assert_eq!(ModelKind::from_slug_loose("mlp"), None);
    }

    #[test]
    fn training_spec_round_trips_through_json() {
        let spec = sample_training_spec();
        let json = serde_json::to_string(&spec).expect("serialize TrainingSpec");
        let back: TrainingSpec = serde_json::from_str(&json).expect("deserialize TrainingSpec");
        assert_eq!(back, spec);
    }

    #[test]
    fn evaluation_and_benchmark_specs_round_trip() {
        let eval = EvaluationSpec {
            model: ModelKind::Pinn,
            artifact_hash: "artifact".into(),
            dataset_manifest_hash: "manifest".into(),
            data_path: None,
            batch_size: 4096,
            output_dir: "eval".into(),
            seed: None,
        };
        let bench = BenchmarkSpec {
            model: ModelKind::Siren,
            artifact_hash: "artifact-2".into(),
            iterations: 100,
            warmup_iterations: 10,
            output_dir: "bench".into(),
            batch_size: 1024,
            seed: None,
        };
        let eval_job = JobKind::Evaluate(eval);
        let bench_job = JobKind::Benchmark(bench);
        for (spec, expected_kind) in [
            (&eval_job, &ModelKind::Pinn),
            (&bench_job, &ModelKind::Siren),
        ] {
            let json = serde_json::to_string(spec).unwrap();
            let back: JobKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *spec);
            assert_eq!(back.model_kind(), expected_kind);
        }
    }

    #[test]
    fn model_config_uses_tagged_representation() {
        let config = ModelConfig::Siren(SirenConfig {
            texture_size: 256,
            hidden_dim: 128,
            max_stars: 100,
            seed: 42,
        });
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["kind"], "siren");
        let back: ModelConfig = serde_json::from_value(json).unwrap();
        assert_eq!(back, config);
    }

    #[test]
    fn capability_validation_accepts_matching_config() {
        assert!(pinn_training_spec().validate().is_ok());
    }

    #[test]
    fn capability_validation_rejects_mismatched_config() {
        let mut spec = pinn_training_spec();
        spec.config = ModelConfig::Siren(SirenConfig {
            texture_size: 64,
            hidden_dim: 64,
            max_stars: 10,
            seed: 1,
        });
        let errs = spec
            .validate()
            .expect_err("model/config mismatch must fail");
        assert!(errs.iter().any(|e| e.contains("does not match")));
    }

    #[test]
    fn capability_validation_rejects_bad_hyperparameters() {
        let mut spec = pinn_training_spec();
        spec.epochs = 0;
        spec.batch_size = 0;
        spec.val_frac = 1.5;
        spec.lr = f64::NAN;
        let errs = spec.validate().expect_err("bad hypers must fail");
        assert!(errs.len() >= 4, "{errs:?}");
    }

    #[test]
    fn pinn_worker_argv_matches_legacy_order() {
        let argv = pinn_training_spec().worker_argv();
        let get = |flag: &str| -> String {
            argv.windows(2)
                .find(|w| w[0] == flag)
                .unwrap_or_else(|| panic!("missing {flag} in {argv:?}"))[1]
                .clone()
        };
        assert_eq!(get("--batch-size"), "2048");
        assert_eq!(get("--physics-weight"), "0.1");
        assert_eq!(get("--epochs"), "50");
        assert_eq!(get("--model-file"), "stellar_model.bpk");
        assert_eq!(get("--norm-file"), "stellar_norm.json");
        assert!(!argv.iter().any(|a| a == "--max-nodes"));
        assert!(!argv.iter().any(|a| a == "--knn-k"));
        assert!(!argv.iter().any(|a| a == "--texture-size"));
    }

    #[test]
    fn gnn_worker_argv_uses_max_nodes_and_graph_flags() {
        let mut spec = pinn_training_spec();
        spec.model = ModelKind::GnnKinematics;
        spec.config = ModelConfig::GnnKinematics(GnnKinematicsConfig {
            knn_k: 8,
            hidden_dim: 256,
            output_dim: 3,
            max_group_size: 64,
            radius_pc: 50.0,
            physics_weight: 0.05,
        });
        assert!(spec.validate().is_ok());
        let argv = spec.worker_argv();
        let get = |flag: &str| -> String {
            argv.windows(2)
                .find(|w| w[0] == flag)
                .unwrap_or_else(|| panic!("missing {flag} in {argv:?}"))[1]
                .clone()
        };
        assert_eq!(get("--max-nodes"), "2048");
        assert_eq!(get("--knn-k"), "8");
        assert_eq!(get("--hidden-dim"), "256");
        assert_eq!(get("--max-group-size"), "64");
        assert_eq!(get("--radius-pc"), "50");
        assert!(!argv.iter().any(|a| a == "--batch-size"));
        assert!(!argv.iter().any(|a| a == "--texture-size"));
    }

    #[test]
    fn siren_worker_argv_carries_texture_and_seed_flags() {
        let mut spec = pinn_training_spec();
        spec.model = ModelKind::Siren;
        spec.config = ModelConfig::Siren(SirenConfig {
            texture_size: 64,
            hidden_dim: 64,
            max_stars: 5000,
            seed: 42,
        });
        assert!(spec.validate().is_ok());
        let argv = spec.worker_argv();
        let get = |flag: &str| -> String {
            argv.windows(2)
                .find(|w| w[0] == flag)
                .unwrap_or_else(|| panic!("missing {flag} in {argv:?}"))[1]
                .clone()
        };
        assert_eq!(get("--batch-size"), "2048");
        assert_eq!(get("--texture-size"), "64");
        assert_eq!(get("--max-stars"), "5000");
        assert_eq!(get("--seed"), "42");
        assert!(!argv.iter().any(|a| a == "--physics-weight"));
        assert!(!argv.iter().any(|a| a == "--knn-k"));
    }

    #[test]
    fn evaluation_argv_is_read_only_with_optional_holdout() {
        let eval = EvaluationSpec {
            model: ModelKind::Pinn,
            artifact_hash: "artifact".into(),
            dataset_manifest_hash: "manifest".into(),
            data_path: Some("data.parquet".into()),
            batch_size: 512,
            output_dir: "runs/pinn".into(),
            seed: Some(7),
        };
        assert!(eval.validate().is_ok());
        let argv = eval.worker_argv(Some("holdout.parquet"));
        assert!(argv.contains(&"--evaluate-only".to_string()));
        assert!(argv.contains(&"--holdout".to_string()));
        assert!(argv.contains(&"--seed".to_string()));
        // No training flags leak into evaluation.
        assert!(!argv.iter().any(|a| a == "--epochs"));
        assert!(!argv.iter().any(|a| a == "--lr"));
    }

    #[test]
    fn benchmark_argv_trains_nothing() {
        let bench = BenchmarkSpec {
            model: ModelKind::Siren,
            artifact_hash: "artifact".into(),
            iterations: 50,
            warmup_iterations: 5,
            output_dir: "runs/siren".into(),
            batch_size: 1024,
            seed: None,
        };
        assert!(bench.validate().is_ok());
        let argv = bench.worker_argv();
        assert!(argv.contains(&"--benchmark-iters".to_string()));
        assert!(argv.contains(&"--benchmark-warmup".to_string()));
        assert!(!argv.iter().any(|a| a == "--epochs"));
    }
}
