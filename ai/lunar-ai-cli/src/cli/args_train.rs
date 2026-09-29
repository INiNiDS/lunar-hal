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

    /// GNN-Kinematics: k-NN neighbours per node.
    #[arg(long, default_value_t = 8)]
    pub knn_k: usize,
    /// GNN-Kinematics: hidden width.
    #[arg(long, default_value_t = 256)]
    pub hidden_dim: usize,
    /// GNN-Kinematics: spatial group cap.
    #[arg(long, default_value_t = 64)]
    pub max_group_size: usize,
    /// GNN-Kinematics: group radius in parsecs.
    #[arg(long, default_value_t = 50.0)]
    pub radius_pc: f32,
    /// SIREN: texture grid size.
    #[arg(long, default_value_t = 64)]
    pub texture_size: usize,
    /// SIREN: star budget for texture synthesis.
    #[arg(long, default_value_t = 5000)]
    pub max_stars: usize,
    /// Explicit global seed (overrides the derived run seed).
    #[arg(long)]
    pub seed: Option<u64>,
    /// Deterministic systematic sample cap (every k-th row); unset = all rows.
    #[arg(long)]
    pub max_rows: Option<u64>,
    /// Spatial-tile subset, comma-separated (GNN/PINN); unset = all tiles.
    #[arg(long)]
    pub tiles: Option<String>,
    /// Consult the epoch-watch AI agent every N epochs, GNN only
    /// (0/unset = off). Reads `.opencode/agents/gnn-watch.md`.
    #[arg(long)]
    pub agent_every: Option<u64>,
    /// Agent model id (`provider/model`) for epoch watch.
    #[arg(long, default_value = "openrouter/meta/muse-spark-1.3-contributor")]
    pub agent_model: String,
    /// Fallback model id when the primary agent call fails.
    #[arg(long, default_value = "opencode/muse-spark-1.3-contributor-free")]
    pub agent_fallback_model: String,
    /// Per-call agent timeout in seconds (fail-open: training continues).
    #[arg(long, default_value_t = 300)]
    pub agent_timeout_secs: u64,
    /// Event-log tail lines attached to each agent call.
    #[arg(long, default_value_t = 30)]
    pub agent_log_lines: usize,
    /// Print the agent prompt instead of spawning (zero-cost check).
    #[arg(long, default_value_t = false)]
    pub agent_dry_run: bool,
    /// Dataset manifest hash recorded into the artifact (default empty).
    #[arg(long, default_value = "")]
    pub dataset_manifest_hash: String,
}

#[derive(Args, Debug, Clone)]
pub struct AgentFixArgs {
    /// Training output dir to diagnose (events.ndjson, artifact.json).
    #[arg(short, long)]
    pub dir: String,
    /// Model id for the fixer (`provider/model`).
    #[arg(long, default_value = "openrouter/meta/muse-spark-1.3-contributor")]
    pub model: String,
    /// Extra instructions appended to the fixer prompt.
    #[arg(long)]
    pub message: Option<String>,
    /// Auto-approve the fixer's tool calls (else it may stall unread).
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
