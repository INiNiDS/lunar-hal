//! Stage 5 (task 6): shared training lifecycle — cancellation, progress,
//! checkpoints and deterministic seeding.
//!
//! The runner keeps no model weights: the GPU trainer lives in the worker
//! binary. The library owns the *protocol* — spec validation, deterministic
//! seed derivation, atomic checkpoint bookkeeping and NDJSON event encoding —
//! so CLI and Testbench produce byte-compatible run folders.

use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::spec::{EvaluationSpec, TrainingSpec};

/// Deterministic seed defaults when the spec carries no explicit seed.
pub const DEFAULT_TRAIN_SEED: u64 = 42;

/// Derives the effective training seed for a run.
///
/// Explicit `TrainingSpec::seed` always wins. Otherwise a stable FNV-1a hash
/// of `dataset_manifest_hash + model slug + epochs` keeps old vs new trainer
/// parity reproducible without hidden RNG state.
pub fn effective_train_seed(spec: &TrainingSpec) -> u64 {
    if let Some(seed) = spec.seed {
        return seed;
    }
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in spec
        .dataset_manifest_hash
        .bytes()
        .chain(spec.model.slug().bytes())
        .chain(spec.epochs.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Lifecycle callbacks — one hook per observable training event.
/// Backends implement this to stream [`crate::events::JobEvent`] NDJSON
/// plus human-readable logs.
pub trait ProgressSink {
    fn on_epoch(&mut self, epoch: u32, total_epochs: u32, train_loss: f64, val_loss: f64);
    fn on_checkpoint(&mut self, epoch: u32, path: &str);
    fn on_completed(&mut self, best_val_loss: f64);
    fn on_cancelled(&mut self, epoch: u32);
}

/// Cooperative cancellation flag polled once per epoch.
#[derive(Debug, Default)]
pub struct CancelFlag {
    cancelled: std::sync::atomic::AtomicBool,
}

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Outcome of a (possibly interrupted) training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Completed,
    Cancelled,
    EarlyStopped,
    /// Epoch-watch AI agent issued VERDICT: STOP (checkpoint saved).
    AgentStopped,
}

/// Validates a training spec without touching data or GPU:
/// capability check, epoch/batch sanity, cancel contract.
pub fn validate_training_run(spec: &TrainingSpec) -> Result<(), String> {
    spec.validate().map_err(|errs| errs.join("; "))?;
    if spec.epochs == 0 {
        return Err("epochs must be >= 1".to_string());
    }
    if spec.batch_size == 0 {
        return Err("batch_size must be >= 1".to_string());
    }
    Ok(())
}

/// Read-only evaluation never mutates checkpoints: verifies the artifact
/// directory exists and the spec is capability-valid.
pub fn validate_evaluation_run(spec: &EvaluationSpec) -> Result<(), String> {
    spec.validate().map_err(|errs| errs.join("; "))?;
    let dir = Path::new(&spec.output_dir);
    if !dir.exists() {
        return Err(format!("artifact dir does not exist: {}", spec.output_dir));
    }
    Ok(())
}

/// Atomically persists a small text/JSON checkpoint sidecar
/// (temp file + rename) so interrupted epochs never corrupt the run folder.
pub fn write_checkpoint_sidecar(
    output_dir: &Path,
    file_name: &str,
    contents: &str,
) -> std::io::Result<std::path::PathBuf> {
    std::fs::create_dir_all(output_dir)?;
    let path = output_dir.join(file_name);
    let tmp = output_dir.join(format!("{file_name}.tmp"));
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Appends one NDJSON-encoded event line to the run's `events.ndjson`.
/// Stdout stays human-readable; machines read this file.
pub fn append_event_line(
    output_dir: &Path,
    event: &crate::events::JobEvent,
) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_dir.join("events.ndjson"))?;
    let line = serde_json::to_string(event)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    writeln!(file, "{line}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ModelConfig, ModelKind, PinnConfig};

    fn pinn_spec() -> TrainingSpec {
        TrainingSpec {
            model: ModelKind::Pinn,
            config: ModelConfig::Pinn(PinnConfig {
                physics_weight: 0.1,
                hidden_dim: 256,
                ..Default::default()
            }),
            dataset_manifest_hash: "manifest".into(),
            data_path: None,
            epochs: 10,
            batch_size: 64,
            lr: 5e-4,
            val_frac: 0.1,
            output_dir: "runs/pinn".into(),
            resume_from: None,
            holdout: None,
            gpu_index: 0,
            patience: 5,
            grad_accum: 2,
            clip_grad_norm: 1.0,
            seed: None,
            model_file: "stellar_model.bpk".into(),
            norm_file: "stellar_norm.json".into(),
            max_rows: None,
            tiles: None,
            agent: None,
        }
    }

    #[test]
    fn explicit_seed_wins_over_derived_seed() {
        let mut spec = pinn_spec();
        spec.seed = Some(7);
        assert_eq!(effective_train_seed(&spec), 7);
    }

    #[test]
    fn derived_seed_is_stable_and_model_specific() {
        let a = effective_train_seed(&pinn_spec());
        let b = effective_train_seed(&pinn_spec());
        assert_eq!(a, b);

        let mut gnn = pinn_spec();
        gnn.model = ModelKind::GnnKinematics;
        assert_ne!(effective_train_seed(&gnn), a);
    }

    #[test]
    fn training_validation_rejects_empty_specs() {
        let mut spec = pinn_spec();
        spec.epochs = 0;
        assert!(validate_training_run(&spec).is_err());
        spec.epochs = 10;
        spec.batch_size = 0;
        assert!(validate_training_run(&spec).is_err());
    }

    #[test]
    fn checkpoint_sidecar_is_atomic() {
        let dir = std::env::temp_dir().join("lnai-training-runner-test");
        let _ = std::fs::remove_dir_all(&dir);
        let path = write_checkpoint_sidecar(&dir, "checkpoint.json", "{\"epoch\":3}")
            .expect("write sidecar");
        assert!(path.exists());
        assert!(!dir.join("checkpoint.json.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
