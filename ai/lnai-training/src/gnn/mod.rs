//! Stage 5 (task 4): GNN-Kinematics dataset/loss/trainer moved verbatim
//! from `ai/lnai-gnn` — the single implementation CLI and Testbench share.
//!
//! Stdout stays a human-readable log attachment; typed
//! [`JobEvent`](crate::events::JobEvent) NDJSON is the metric protocol.

pub mod dataset;
pub mod loss;
pub mod trainer;

pub use dataset::{
    DEFAULT_KNN_K, DEFAULT_MAX_GROUP, GnnDataset, GnnNormParams, NODE_FEATURE_DIM,
    PrefetchBatchedBatcher, StarGroup, VELOCITY_DIM,
};
pub use loss::{compute_gnn_loss, compute_gnn_physics_loss};
