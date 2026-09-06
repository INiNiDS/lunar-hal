//! Stage 5: frozen PINN CLI surface (parity gate) plus additive
//! library-owned flags (`--seed`, `--evaluate-only`, `--benchmark-*`).

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "lnai",
    about = "Stellar MLP trainer with Fourier features and PINN loss"
)]
pub struct Args {
    #[arg(long, default_value = "")]
    pub data: String,
    #[arg(long)]
    pub holdout: Option<String>,
    #[arg(long, default_value_t = 200)]
    pub epochs: usize,
    #[arg(long, default_value_t = 2048)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 2)]
    pub grad_accum: usize,
    #[arg(long, default_value_t = 5e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 0.1)]
    pub physics_weight: f64,
    #[arg(long, default_value_t = 0.1)]
    pub val_frac: f32,
    #[arg(long, default_value_t = 0)]
    pub gpu_index: usize,
    #[arg(long)]
    pub resume_from: Option<String>,
    #[arg(long, default_value = ".")]
    pub output_dir: String,
    #[arg(long, default_value = "stellar_model.bpk")]
    pub model_file: String,
    #[arg(long, default_value = "stellar_norm.json")]
    pub norm_file: String,
    #[arg(long, default_value_t = 1.0)]
    pub clip_grad_norm: f64,
    #[arg(long, default_value_t = 20)]
    pub patience: usize,
    /// Explicit global seed (train/val split + batch order). Unset keeps the
    /// legacy derived seed via `runner::effective_train_seed`.
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
