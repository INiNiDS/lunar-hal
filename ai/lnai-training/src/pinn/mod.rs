//! Stage 5 (task 2): PINN dataset/loss/trainer moved verbatim from
//! `ai/lnai` — the single implementation CLI and Testbench share.
//!
//! The trainer writes typed [`JobEvent`](crate::events::JobEvent) NDJSON via
//! [`ProgressSink`](crate::runner::ProgressSink) in addition to
//! human-readable stdout lines; stdout is a log attachment, never the metric
//! protocol (exit gate).

pub mod dataset;
pub mod loss;
pub mod trainer;

pub use dataset::{GpuBatcher, INPUT_DIM, NormParams, StellarDataset, TARGET_DIM};
pub use loss::{compute_data_loss, compute_physics_loss, compute_pinn_loss};
