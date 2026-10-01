
pub use lnai_dto::{
    BenchmarkRequest, DataCollectRequestShim, EvaluationRequest, JobEventDto, ModelKindDto,
    ReportRecord, TrainingRequest,
};

pub use lnai_dto::{ModelKindDto as ModelKindV1, TrainingSpecDto as TrainingSpecV1};

mod lnai_dto {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    pub enum ModelKindDto {
        Pinn,
        Gnn,
        Siren,
        GnnLocalization,
    }

    impl ModelKindDto {
        pub fn slug(&self) -> &'static str {
            match self {
                ModelKindDto::Pinn => "pinn",
                ModelKindDto::Gnn => "gnn",
                ModelKindDto::Siren => "siren",
                ModelKindDto::GnnLocalization => "gnn_localization",
            }
        }

        pub fn label(&self) -> &'static str {
            match self {
                ModelKindDto::Pinn => "PINN",
                ModelKindDto::Gnn => "GNN",
                ModelKindDto::Siren => "SIREN",
                ModelKindDto::GnnLocalization => "GNN-Localization",
            }
        }

        pub fn binary_name(&self) -> &'static str {
            match self {
                ModelKindDto::Pinn => "lnai",
                ModelKindDto::Gnn => "lnai-gnn",
                ModelKindDto::Siren => "lnai-siren",
                ModelKindDto::GnnLocalization => "lnai-gnn",
            }
        }

        pub fn from_slug(s: &str) -> Option<Self> {
            match s {
                "pinn" => Some(Self::Pinn),
                "gnn" | "gnn_kinematics" | "gnn-kinematics" => Some(Self::Gnn),
                "siren" => Some(Self::Siren),
                "gnn_localization" | "gnn-localization" | "localization" => {
                    Some(Self::GnnLocalization)
                }
                _ => None,
            }
        }
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct TrainingSpecDto {
        pub model: ModelKindDto,
        pub epochs: u32,
        pub batch_size: u32,
        pub lr: f64,
        pub physics_weight: f64,
        pub val_frac: f32,
        pub data_path: String,
        pub output_dir: String,
        pub resume_from: Option<String>,
        pub holdout: Option<String>,
        pub gpu_index: u32,
        pub knn_k: Option<u32>,
        pub hidden_dim: Option<u32>,
        pub max_group_size: Option<u32>,
        pub radius_pc: Option<f32>,
        pub texture_size: Option<u32>,
        pub max_stars: Option<u32>,
        pub max_slots: Option<u32>,
        pub mask_ratio: Option<f32>,
        pub seed: Option<u64>,
        pub dataset_manifest_hash: Option<String>,
        pub patience: u32,
        pub grad_accum: u32,
        pub clip_grad_norm: f64,
    }

    pub type TrainingRequest = TrainingSpecDto;

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct EvaluationRequest {
        pub model: ModelKindDto,
        pub data_path: String,
        pub output_dir: String,
        pub holdout: Option<String>,
        pub batch_size: u32,
        pub seed: Option<u64>,
        pub artifact_hash: Option<String>,
        pub dataset_manifest_hash: Option<String>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct BenchmarkRequest {
        pub model: ModelKindDto,
        pub output_dir: String,
        pub batch_size: u32,
        pub iterations: u32,
        pub warmup_iterations: u32,
        pub seed: Option<u64>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct DataCollectRequestShim {
        pub out_dir: String,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    pub struct ReportRecord {
        pub id: String,
        pub job_id: String,
        pub kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub baseline_delta_json: Option<serde_json::Value>,
        pub report_json: serde_json::Value,
        pub created_at: u64,
    }

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    #[serde(tag = "event", rename_all = "snake_case")]
    pub enum JobEventDto {
        Queued,
        Started,
        Progress {
            epoch: u32,
            total_epochs: u32,
        },
        Metric(crate::EpochMetric),
        Checkpoint {
            epoch: u32,
            path: String,
            hash: String,
        },
        Completed {
            exit_code: i32,
        },
        Failed {
            error_summary: String,
            exit_code: i32,
        },
        Cancelled,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn model_slugs_match_worker_contract() {
            assert_eq!(ModelKindDto::Pinn.slug(), "pinn");
            assert_eq!(ModelKindDto::Gnn.slug(), "gnn");
            assert_eq!(ModelKindDto::Siren.slug(), "siren");
            assert_eq!(ModelKindDto::Pinn.binary_name(), "lnai");
            assert_eq!(ModelKindDto::Gnn.binary_name(), "lnai-gnn");
            assert_eq!(ModelKindDto::Siren.binary_name(), "lnai-siren");
            assert_eq!(
                ModelKindDto::from_slug("gnn_kinematics"),
                Some(ModelKindDto::Gnn)
            );
            assert_eq!(ModelKindDto::from_slug("mlp"), None);
        }

        #[test]
        fn typed_requests_round_trip() {
            let train = TrainingSpecDto {
                model: ModelKindDto::Pinn,
                epochs: 10,
                batch_size: 64,
                lr: 5e-4,
                physics_weight: 0.1,
                val_frac: 0.1,
                data_path: "data.parquet".into(),
                output_dir: "out".into(),
                resume_from: None,
                holdout: None,
                gpu_index: 0,
                knn_k: None,
                hidden_dim: None,
                max_group_size: None,
                radius_pc: None,
                texture_size: None,
                max_stars: None,
                max_slots: None,
                mask_ratio: None,
                seed: Some(42),
                dataset_manifest_hash: None,
                patience: 5,
                grad_accum: 2,
                clip_grad_norm: 1.0,
            };
            let json = serde_json::to_string(&train).unwrap();
            let back: TrainingSpecDto = serde_json::from_str(&json).unwrap();
            assert_eq!(back, train);

            let eval = EvaluationRequest {
                model: ModelKindDto::Gnn,
                data_path: "data.parquet".into(),
                output_dir: "out".into(),
                holdout: None,
                batch_size: 64,
                seed: None,
                artifact_hash: None,
                dataset_manifest_hash: None,
            };
            let json = serde_json::to_string(&eval).unwrap();
            let back: EvaluationRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(back, eval);
        }

        #[test]
        fn job_event_dto_uses_typed_tags() {
            let event = JobEventDto::Completed { exit_code: 0 };
            let value = serde_json::to_value(&event).unwrap();
            assert_eq!(value["event"], "completed");
        }
    }
}
