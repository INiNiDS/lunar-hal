//! Stage 5 (task 5): SIREN dataset/loss/trainer moved verbatim from
//! `ai/lnai-siren` — the single implementation CLI and Testbench share.
//!
//! Stdout stays a human-readable log attachment; typed
//! [`JobEvent`](crate::events::JobEvent) NDJSON is the metric protocol.

pub mod dataset;
pub mod loss;
pub mod trainer;

pub use dataset::{PrefetchBatcher, SirenDataset, SirenNorm, TARGET_DIM};
pub use loss::{compute_data_loss, compute_siren_loss};
