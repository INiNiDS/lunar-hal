use clap::{Args, ValueEnum};

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliModel {
    Pinn,
    Gnn,
    Siren,
}

impl CliModel {
    pub fn worker_binary(&self) -> &'static str {
        match self {
            CliModel::Pinn => "lnai",
            CliModel::Gnn => "lnai-gnn",
            CliModel::Siren => "lnai-siren",
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct TrainArgs {
    #[arg(short, long, value_enum, default_value_t = CliModel::Pinn)]
    pub model: CliModel,

    #[arg(short, long)]
    pub data: Option<String>,

    #[arg(short, long)]
    pub resume: Option<String>,

    #[arg(long)]
    pub norm: Option<String>,

    #[arg(short = 'O', long, default_value = "stellar_model")]
    pub output_dir: String,

    #[arg(long, default_value_t = 200)]
    pub epochs: usize,

    #[arg(long, default_value_t = 4096)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 5e-4)]
    pub lr: f64,

    #[arg(long, default_value_t = 0.1)]
    pub physics_weight: f64,

    #[arg(long, default_value_t = 0.1)]
    pub val_frac: f32,

    #[arg(long, default_value_t = 0)]
    pub gpu_index: usize,

    #[arg(long)]
    pub holdout: Option<String>,

    #[arg(long)]
    pub lnai_bin: Option<String>,

    #[arg(long, default_value = "stellar_model.bpk")]
    pub model_file: String,

    #[arg(long, default_value = "stellar_norm.json")]
    pub norm_file: String,

    #[arg(long, default_value_t = 8)]
    pub knn_k: usize,
    #[arg(long, default_value_t = 256)]
    pub hidden_dim: usize,
    #[arg(long, default_value_t = 64)]
    pub max_group_size: usize,
    #[arg(long, default_value_t = 50.0)]
    pub radius_pc: f32,
    #[arg(long, default_value_t = 64)]
    pub texture_size: usize,
    #[arg(long, default_value_t = 5000)]
    pub max_stars: usize,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub max_rows: Option<u64>,
    #[arg(long)]
    pub tiles: Option<String>,
    #[arg(long)]
    pub agent_every: Option<u64>,
    #[arg(long, default_value = "openrouter/meta/muse-spark-1.3-contributor")]
    pub agent_model: String,
    #[arg(long, default_value = "opencode/muse-spark-1.3-contributor-free")]
    pub agent_fallback_model: String,
    #[arg(long, default_value_t = 300)]
    pub agent_timeout_secs: u64,
    #[arg(long, default_value_t = 30)]
    pub agent_log_lines: usize,
    #[arg(long, default_value_t = false)]
    pub agent_dry_run: bool,
    #[arg(long, default_value = "")]
    pub dataset_manifest_hash: String,
}

#[derive(Args, Debug, Clone)]
pub struct AgentFixArgs {
    #[arg(short, long)]
    pub dir: String,
    #[arg(long, default_value = "openrouter/meta/muse-spark-1.3-contributor")]
    pub model: String,
    #[arg(long)]
    pub message: Option<String>,
    #[arg(long, default_value_t = true)]
    pub auto_approve: bool,
}

#[derive(Args, Debug, Clone)]
pub struct EvaluateArgs {
    #[arg(short, long, value_enum, default_value_t = CliModel::Pinn)]
    pub model: CliModel,
    #[arg(short, long, default_value = "stellar_model")]
    pub output_dir: String,
    #[arg(short, long)]
    pub data: Option<String>,
    #[arg(long)]
    pub holdout: Option<String>,
    #[arg(long, default_value_t = 4096)]
    pub batch_size: usize,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, default_value = "stellar_model.bpk")]
    pub model_file: String,
    #[arg(long, default_value = "stellar_norm.json")]
    pub norm_file: String,
}

#[derive(Args, Debug, Clone)]
pub struct BenchmarkArgs {
    #[arg(short, long, value_enum, default_value_t = CliModel::Pinn)]
    pub model: CliModel,
    #[arg(short, long, default_value = "stellar_model")]
    pub output_dir: String,
    #[arg(long, default_value_t = 100)]
    pub iters: u32,
    #[arg(long, default_value_t = 10)]
    pub warmup: u32,
    #[arg(long, default_value_t = 4096)]
    pub batch_size: usize,
    #[arg(long)]
    pub seed: Option<u64>,
}
