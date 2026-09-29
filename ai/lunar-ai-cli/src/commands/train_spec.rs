use anyhow::{Result, anyhow};
use std::path::Path;

use crate::cli::args_train::CliModel;

pub struct TrainOptions<'a> {
    pub model: CliModel,
    pub data: Option<&'a str>,
    pub resume: Option<&'a str>,
    pub norm: Option<&'a str>,
    pub output_dir: &'a str,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub physics_weight: f64,
    pub val_frac: f32,
    pub gpu_index: usize,
    pub holdout: Option<&'a str>,
    pub lnai_bin: Option<&'a str>,
    pub model_file: &'a str,
    pub norm_file: &'a str,
    pub knn_k: usize,
    pub hidden_dim: usize,
    pub max_group_size: usize,
    pub radius_pc: f32,
    pub texture_size: usize,
    pub max_stars: usize,
    pub seed: Option<u64>,
    pub dataset_manifest_hash: &'a str,
    pub max_rows: Option<u64>,
    pub tiles: Option<String>,
    pub agent_every: Option<u64>,
    pub agent_model: &'a str,
    pub agent_fallback_model: &'a str,
    pub agent_timeout_secs: u64,
    pub agent_log_lines: usize,
    pub agent_dry_run: bool,
}

pub fn training_spec_from_opts(
    opts: &TrainOptions<'_>,
    data_path: &Path,
) -> Result<lnai_training::spec::TrainingSpec> {
    use lnai_training::spec::{
        GnnKinematicsConfig, ModelConfig, ModelKind, PinnConfig, SirenConfig, TrainingSpec,
    };
    let (model, config) = match opts.model {
        CliModel::Pinn => (
            ModelKind::Pinn,
            ModelConfig::Pinn(PinnConfig {
                physics_weight: opts.physics_weight,
                hidden_dim: 256,
                ..Default::default()
            }),
        ),
        CliModel::Gnn => (
            ModelKind::GnnKinematics,
            ModelConfig::GnnKinematics(GnnKinematicsConfig {
                knn_k: opts.knn_k as u32,
                hidden_dim: opts.hidden_dim as u32,
                output_dim: 3,
                max_group_size: opts.max_group_size as u32,
                radius_pc: opts.radius_pc,
                physics_weight: opts.physics_weight,
                kl_weight: 0.0,
            }),
        ),
        CliModel::Siren => (
            ModelKind::Siren,
            ModelConfig::Siren(SirenConfig {
                texture_size: opts.texture_size as u32,
                hidden_dim: 64,
                max_stars: opts.max_stars as u32,
                seed: opts.seed.unwrap_or(42),
            }),
        ),
    };
    let dataset_manifest_hash = if opts.dataset_manifest_hash.trim().is_empty() {
        lnai_training::artifacts::dataset_fingerprint(data_path)?
    } else {
        opts.dataset_manifest_hash.to_string()
    };
    let spec = TrainingSpec {
        model,
        config,
        dataset_manifest_hash,
        data_path: Some(data_path.display().to_string()),
        epochs: opts.epochs as u32,
        batch_size: opts.batch_size as u32,
        lr: opts.lr,
        val_frac: opts.val_frac,
        output_dir: opts.output_dir.to_string(),
        resume_from: opts.resume.map(str::to_string),
        holdout: opts.holdout.map(str::to_string),
        gpu_index: opts.gpu_index as u32,
        patience: 20,
        grad_accum: 2,
        clip_grad_norm: 1.0,
        seed: opts.seed,
        model_file: opts.model_file.to_string(),
        norm_file: opts.norm_file.to_string(),
        max_rows: opts.max_rows,
        tiles: opts.tiles.clone(),
        agent: match opts.model {
            CliModel::Gnn => agent_hook_from_opts(
                opts.agent_every,
                opts.agent_model,
                opts.agent_fallback_model,
                opts.agent_timeout_secs,
                opts.agent_log_lines,
                opts.agent_dry_run,
            ),
            CliModel::Pinn | CliModel::Siren => {
                if opts.agent_every.is_some_and(|n| n > 0) || opts.agent_dry_run {
                    println!("note: epoch-watch agent is GNN-only for now; ignoring agent flags");
                }
                None
            }
        },
    };
    spec.validate()
        .map_err(|errs| anyhow!("invalid training spec: {}", errs.join("; ")))?;
    Ok(spec)
}

pub fn agent_hook_from_opts(
    every: Option<u64>,
    model: &str,
    fallback_model: &str,
    timeout_secs: u64,
    log_lines: usize,
    dry_run: bool,
) -> Option<lnai_training::agent::AgentHookConfig> {
    match every {
        Some(0) | None if !dry_run => None,
        _ => Some(lnai_training::agent::AgentHookConfig {
            every: every.unwrap_or(1).max(1),
            model: model.to_string(),
            fallback_model: fallback_model.to_string(),
            timeout_secs,
            log_lines,
            agent: lnai_training::agent::DEFAULT_AGENT_NAME.to_string(),
            dry_run,
        }),
    }
}
