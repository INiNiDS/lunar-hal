//! Stage 5 (task 7): read-only evaluation contract for all model kinds.
//!
//! Evaluation never runs an optimizer step and never rewrites checkpoints.
//! Given an [`EvaluationSpec`](crate::spec::EvaluationSpec) plus an artifact
//! directory it produces a deterministic [`EvaluationReport`] that CLI and
//! Testbench serialize identically.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::report::{ReportKind, ReportV1, RunIdentity};
use crate::spec::{EvaluationSpec, ModelKind};

/// Version of the evaluation report envelope (frozen with contracts v1).
pub const EVALUATION_REPORT_VERSION: &str = "1.0.0";

/// Minimal per-split loss summary. Rich per-target breakdowns live in
/// `metrics/` suites; the worker emits these numbers as typed
/// [`JobEvent::Metric`](crate::events::JobEvent) values, never stdout text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitLosses {
    pub data_loss: f64,
    pub physics_loss: Option<f64>,
}

/// Deterministic evaluation result for one artifact + dataset pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub version: String,
    pub model: ModelKind,
    pub artifact_hash: String,
    pub dataset_manifest_hash: String,
    pub validation: SplitLosses,
    pub holdout: Option<SplitLosses>,
    /// Was the holdout split non-empty (Stage 1 non-empty holdout gate)?
    pub holdout_non_empty: bool,
    pub seed: u64,
}

impl EvaluationReport {
    pub fn new(
        spec: &EvaluationSpec,
        validation: SplitLosses,
        holdout: Option<SplitLosses>,
        holdout_non_empty: bool,
        seed: u64,
    ) -> Self {
        Self {
            version: EVALUATION_REPORT_VERSION.to_string(),
            model: spec.model.clone(),
            artifact_hash: spec.artifact_hash.clone(),
            dataset_manifest_hash: spec.dataset_manifest_hash.clone(),
            validation,
            holdout,
            holdout_non_empty,
            seed,
        }
    }

    /// Converts the report into the shared [`ReportV1`] JSON envelope so
    /// evaluation results land in `target/ai-reports` like every other suite.
    pub fn to_report(&self, identity: RunIdentity) -> ReportV1 {
        let mut report = ReportV1::new(
            ReportKind::Correctness,
            &format!("{}_evaluate", self.model.slug()),
            identity,
        );
        report.add_metric("data_loss", self.validation.data_loss);
        if let Some(phys) = self.validation.physics_loss {
            report.add_metric("physics_loss", phys);
        }
        if let Some(holdout) = &self.holdout {
            report.add_metric("holdout_data_loss", holdout.data_loss);
            if let Some(phys) = holdout.physics_loss {
                report.add_metric("holdout_physics_loss", phys);
            }
        }
        report.add_metric(
            "holdout_non_empty",
            f64::from(u8::from(self.holdout_non_empty)),
        );
        report.add_note(format!(
            "read-only evaluation of artifact {} (seed {seed})",
            self.artifact_hash,
            seed = self.seed
        ));
        report.passed = self.holdout_non_empty;
        report
    }

    /// Atomically writes `evaluation.json` into the artifact directory.
    pub fn write_to_artifact_dir(&self, dir: &Path) -> std::io::Result<std::path::PathBuf> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        crate::runner::write_checkpoint_sidecar(dir, "evaluation.json", &json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::identity_from_env;

    fn spec() -> EvaluationSpec {
        EvaluationSpec {
            model: ModelKind::Pinn,
            artifact_hash: "artifact".into(),
            dataset_manifest_hash: "manifest".into(),
            data_path: None,
            batch_size: 64,
            output_dir: "runs/pinn".into(),
            seed: Some(7),
        }
    }

    #[test]
    fn evaluation_report_round_trips_and_converts_to_report_v1() {
        let report = EvaluationReport::new(
            &spec(),
            SplitLosses {
                data_loss: 0.5,
                physics_loss: Some(0.1),
            },
            Some(SplitLosses {
                data_loss: 0.6,
                physics_loss: None,
            }),
            true,
            7,
        );
        let json = serde_json::to_string(&report).unwrap();
        let back: EvaluationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);

        let envelope = report.to_report(identity_from_env(7));
        assert_eq!(envelope.metric("data_loss"), Some(0.5));
        assert_eq!(envelope.metric("holdout_data_loss"), Some(0.6));
        assert!(envelope.passed);
    }

    #[test]
    fn evaluation_report_version_is_frozen() {
        assert_eq!(EVALUATION_REPORT_VERSION, "1.0.0");
    }
}
