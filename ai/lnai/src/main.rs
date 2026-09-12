//! Stage 5: PINN worker — thin compatibility wrapper around `lnai-training`.
//!
//! Argument surface is frozen (parity gate): every flag accepted before
//! Stage 5 is still accepted with identical semantics. New library-owned
//! flags are purely additive:
//! * `--seed` — explicit global seed (overrides spec derivation)
//! * `--evaluate-only` — read-only evaluation, no optimizer step
//! * `--benchmark-iters/--benchmark-warmup` — forward-pass timing harness

mod args;

use anyhow::Result;
use clap::Parser;
use lnai_training::spec::{ModelConfig, ModelKind, PinnConfig, TrainingSpec};

pub use args::Args;

pub fn spec_from_args(args: &Args) -> TrainingSpec {
    TrainingSpec {
        model: ModelKind::Pinn,
        config: ModelConfig::Pinn(PinnConfig {
            physics_weight: args.physics_weight,
            hidden_dim: 256,
        }),
        dataset_manifest_hash: String::new(),
        data_path: if args.data.is_empty() {
            None
        } else {
            Some(args.data.clone())
        },
        epochs: args.epochs as u32,
        batch_size: args.batch_size as u32,
        lr: args.lr,
        val_frac: args.val_frac,
        output_dir: args.output_dir.clone(),
        resume_from: args.resume_from.clone(),
        holdout: args.holdout.clone(),
        gpu_index: args.gpu_index as u32,
        patience: args.patience as u32,
        grad_accum: args.grad_accum as u32,
        clip_grad_norm: args.clip_grad_norm,
        seed: args.seed,
        model_file: args.model_file.clone(),
        norm_file: args.norm_file.clone(),
        max_rows: args.max_rows,
        tiles: args.tiles.clone(),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let spec = spec_from_args(&args);
    if let Err(errs) = spec.validate() {
        anyhow::bail!("invalid PINN spec: {}", errs.join("; "));
    }
    if args.evaluate_only {
        lnai_training::pinn::trainer::run_evaluate(&spec)?;
        return Ok(());
    }
    if args.benchmark_iters > 0 {
        lnai_training::pinn::trainer::run_benchmark(
            &spec,
            args.benchmark_iters,
            args.benchmark_warmup,
        )?;
        return Ok(());
    }
    lnai_training::pinn::trainer::run_train(&spec)?;
    Ok(())
}
