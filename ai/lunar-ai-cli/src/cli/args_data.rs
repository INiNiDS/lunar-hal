use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct CollectDataArgs {
    #[arg(short, long, default_value = "data/canonical-v1")]
    pub out_dir: String,
    #[arg(long, default_value_t = 0.0)]
    pub ra_min: f64,
    #[arg(long, default_value_t = 360.0)]
    pub ra_max: f64,
    #[arg(long, default_value_t = 16.0)]
    pub mag_limit_g: f64,
    #[arg(long, default_value = "200000")]
    pub target_rows_per_shard: usize,
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    #[arg(long)]
    pub only: Option<String>,
    #[arg(long, default_value_t = false)]
    pub retry_failed: bool,
    #[arg(long, default_value_t = false)]
    pub verify: bool,
}

#[derive(Args, Debug, Clone)]
pub struct BuildDatasetArgs {
    #[arg(short, long, default_value = "data/canonical-v1")]
    pub out_dir: String,
    #[arg(long, default_value_t = true)]
    pub quality_report: bool,
    #[arg(long, default_value_t = 2)]
    pub batch_shards: usize,
    #[arg(long, default_value_t = false)]
    pub keep_parts: bool,
}

#[derive(Args, Debug, Clone)]
pub struct EnrichStellarArgs {
    #[arg(short, long)]
    pub data: String,
    #[arg(short, long, default_value = "data/stellar-ap-v1")]
    pub out_dir: String,
    #[arg(long, default_value_t = 2000)]
    pub ids_per_query: usize,
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    #[arg(long, default_value_t = 0)]
    pub max_ids: u64,
    #[arg(long, default_value_t = false)]
    pub join_only: bool,
    #[arg(long)]
    pub join_output: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct EnrichFixturesArgs {
    #[arg(short, long, default_value = "data/nasa-enriched-v1")]
    pub out_dir: String,
}

#[derive(Args, Debug, Clone)]
pub struct EnrichReportArgs {
    #[arg(short, long, default_value = "data/nasa-enriched-v1")]
    pub out_dir: String,
    #[arg(long)]
    pub gaia_parquet: Option<String>,
}
