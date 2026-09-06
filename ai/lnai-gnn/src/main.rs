//! Stage 5: GNN worker — thin compatibility wrapper around `lnai-training`.

mod args;

use anyhow::Result;
use clap::Parser;
use lnai_training::spec::{GnnKinematicsConfig, ModelConfig, ModelKind, TrainingSpec};

pub use args::Args;
pub use lnai_training::gnn::dataset::{DEFAULT_KNN_K, DEFAULT_MAX_GROUP};

pub fn spec_from_args(args: &Args) -> TrainingSpec {
    TrainingSpec {
        model: ModelKind::GnnKinematics,
        config: ModelConfig::GnnKinematics(GnnKinematicsConfig {
            knn_k: args.knn_k as u32,
            hidden_dim: args.hidden_dim as u32,
            output_dim: 3,
            max_group_size: args.max_group_size as u32,
            radius_pc: args.radius_pc,
            physics_weight: args.physics_weight,
        }),
        dataset_manifest_hash: String::new(),
        data_path: if args.data.is_empty() {
            None
        } else {
            Some(args.data.clone())
        },
        epochs: args.epochs as u32,
        batch_size: args.max_nodes as u32,
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
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let spec = spec_from_args(&args);
    if let Err(errs) = spec.validate() {
        anyhow::bail!("invalid GNN spec: {}", errs.join("; "));
    }
    if args.evaluate_only {
        lnai_training::gnn::trainer::run_evaluate(&spec)?;
        return Ok(());
    }
    if args.benchmark_iters > 0 {
        lnai_training::gnn::trainer::run_benchmark(
            &spec,
            args.benchmark_iters,
            args.benchmark_warmup,
        )?;
        return Ok(());
    }
    lnai_training::gnn::trainer::run_train(&spec)?;
    Ok(())
}
