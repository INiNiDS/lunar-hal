
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
    #[arg(long, default_value = "mse")]
    pub loss_kind: String,
    #[arg(long, default_value_t = 1.0)]
    pub huber_delta: f32,
    #[arg(long, default_value = "1,1,1,1")]
    pub target_weights: String,
}

impl Args {
    pub fn loss_kind(&self) -> anyhow::Result<lnai_training::spec::PinnLossKind> {
        use lnai_training::spec::PinnLossKind;
        PinnLossKind::from_slug(self.loss_kind.as_str()).ok_or_else(|| {
            anyhow::anyhow!(
                "invalid --loss-kind '{}': expected 'mse' or 'huber'",
                self.loss_kind
            )
        })
    }

    pub fn target_weights_array(&self) -> anyhow::Result<[f32; 4]> {
        let parts: Vec<&str> = self.target_weights.split(',').collect();
        if parts.len() != 4 {
            anyhow::bail!(
                "invalid --target-weights '{}': expected 4 comma-separated floats",
                self.target_weights
            );
        }
        let mut out = [0.0f32; 4];
        for (slot, part) in out.iter_mut().zip(parts) {
            *slot = part.trim().parse::<f32>().map_err(|_| {
                anyhow::anyhow!(
                    "invalid --target-weights '{}': '{}' is not a float",
                    self.target_weights,
                    part.trim()
                )
            })?;
        }
        Ok(out)
    }
}
