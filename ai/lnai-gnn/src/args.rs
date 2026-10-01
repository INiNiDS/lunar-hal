
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
    #[arg(long, default_value_t = 3)]
    pub output_dim: u32,
    #[arg(long, default_value_t = 0.0)]
    pub kl_weight: f64,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, default_value_t = false)]
    pub evaluate_only: bool,
    #[arg(long, default_value_t = 0)]
    pub benchmark_iters: u32,
    #[arg(long, default_value_t = 10)]
    pub benchmark_warmup: u32,
    #[arg(long)]
    pub max_rows: Option<u64>,
    #[arg(long)]
    pub tiles: Option<String>,
    #[arg(long)]
    pub agent_every: Option<u64>,
    #[arg(long, default_value = lnai_training::agent::DEFAULT_AGENT_MODEL)]
    pub agent_model: String,
    #[arg(long, default_value = lnai_training::agent::DEFAULT_AGENT_FALLBACK_MODEL)]
    pub agent_fallback_model: String,
    #[arg(long, default_value_t = lnai_training::agent::DEFAULT_AGENT_TIMEOUT_SECS)]
    pub agent_timeout_secs: u64,
    #[arg(long, default_value_t = lnai_training::agent::DEFAULT_AGENT_LOG_LINES)]
    pub agent_log_lines: usize,
    #[arg(long, default_value_t = false)]
    pub agent_dry_run: bool,
}

impl Args {
    pub fn agent_hook(&self) -> Option<lnai_training::agent::AgentHookConfig> {
        match self.agent_every {
            Some(0) | None if !self.agent_dry_run => None,
            _ => Some(lnai_training::agent::AgentHookConfig {
                every: self.agent_every.unwrap_or(1).max(1),
                model: self.agent_model.clone(),
                fallback_model: self.agent_fallback_model.clone(),
                timeout_secs: self.agent_timeout_secs,
                log_lines: self.agent_log_lines,
                agent: lnai_training::agent::DEFAULT_AGENT_NAME.to_string(),
                dry_run: self.agent_dry_run,
            }),
        }
    }
}
