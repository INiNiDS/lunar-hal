//! Stage 5: frozen GNN CLI surface (parity gate) plus additive
//! library-owned flags (`--seed`, `--evaluate-only`, `--benchmark-*`).

use clap::Parser;
use lnai_training::gnn::dataset::{DEFAULT_KNN_K, DEFAULT_MAX_GROUP};

#[derive(Parser)]
#[command(
    name = "lnai-gnn",
    about = "Stellar GNN trainer for velocity prediction"
)]
pub struct Args {
    #[arg(long, default_value = "")]
    pub data: String,
    #[arg(long)]
    pub holdout: Option<String>,
    #[arg(long, default_value_t = 200)]
    pub epochs: usize,
    #[arg(long, default_value_t = 4096)]
    pub max_nodes: usize,
    #[arg(long, default_value_t = 8)]
    pub grad_accum: usize,
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 0.05)]
    pub physics_weight: f64,
    #[arg(long, default_value_t = 0.1)]
    pub val_frac: f32,
    #[arg(long, default_value_t = 0)]
    pub gpu_index: usize,
    #[arg(long)]
    pub resume_from: Option<String>,
    #[arg(long, default_value = ".")]
    pub output_dir: String,
    #[arg(long, default_value = "stellar_gnn_model.bpk")]
    pub model_file: String,
    #[arg(long, default_value = "stellar_gnn_norm.json")]
    pub norm_file: String,
    #[arg(long, default_value_t = 1.0)]
    pub clip_grad_norm: f64,
    #[arg(long, default_value_t = 20)]
    pub patience: usize,
    #[arg(long, default_value_t = 256)]
    pub hidden_dim: usize,
    #[arg(long, default_value_t = DEFAULT_KNN_K)]
    pub knn_k: usize,
    #[arg(long, default_value_t = DEFAULT_MAX_GROUP)]
    pub max_group_size: usize,
    #[arg(long, default_value_t = 50.0)]
    pub radius_pc: f32,
    /// Explicit global seed (group build + split + shuffle).
    #[arg(long)]
    pub seed: Option<u64>,
    /// Read-only evaluation: load artifact, report losses, train nothing.
    #[arg(long, default_value_t = false)]
    pub evaluate_only: bool,
    /// Benchmark mode: time forward passes instead of training (0 = off).
    #[arg(long, default_value_t = 0)]
    pub benchmark_iters: u32,
    #[arg(long, default_value_t = 10)]
    pub benchmark_warmup: u32,
}
