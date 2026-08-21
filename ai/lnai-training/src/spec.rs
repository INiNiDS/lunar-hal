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
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SirenConfig {
    pub texture_size: u32,
    pub hidden_dim: u32,
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrainingSpec {
    pub model: ModelKind,
    pub config: ModelConfig,
    pub dataset_manifest_hash: String,
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
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EvaluationSpec {
    pub model: ModelKind,
    pub artifact_hash: String,
    pub dataset_manifest_hash: String,
    pub batch_size: u32,
    pub output_dir: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BenchmarkSpec {
    pub model: ModelKind,
    pub artifact_hash: String,
    pub iterations: u32,
    pub warmup_iterations: u32,
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
            batch_size: 4096,
            output_dir: "eval".into(),
        };
        let bench = BenchmarkSpec {
            model: ModelKind::Siren,
            artifact_hash: "artifact-2".into(),
            iterations: 100,
            warmup_iterations: 10,
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
        });
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json["kind"], "siren");
        let back: ModelConfig = serde_json::from_value(json).unwrap();
        assert_eq!(back, config);
    }
}
