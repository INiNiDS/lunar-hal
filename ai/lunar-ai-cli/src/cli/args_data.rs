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
    /// Restrict work to shards whose id contains this substring.
    #[arg(long)]
    pub only: Option<String>,
    /// Re-attempt shards currently in Failed.
    #[arg(long, default_value_t = false)]
    pub retry_failed: bool,
    /// Re-verify checksums of Verified shards instead of skipping.
    #[arg(long, default_value_t = false)]
    pub verify: bool,
}

#[derive(Args, Debug, Clone)]
pub struct BuildDatasetArgs {
    #[arg(short, long, default_value = "data/canonical-v1")]
    pub out_dir: String,
    #[arg(long, default_value_t = true)]
    pub quality_report: bool,
    /// Verified shards loaded/cleaned per batch; lower this if RAM runs out.
    #[arg(long, default_value_t = 2)]
    pub batch_shards: usize,
    /// Keep assembled/parts/*.parquet after the merge for debugging.
    #[arg(long, default_value_t = false)]
    pub keep_parts: bool,
}

#[derive(Args, Debug, Clone)]
pub struct EnrichStellarArgs {
    /// Canonical parquet to enrich (e.g. assembled/canonical.parquet).
    #[arg(short, long)]
    pub data: String,
    /// Workdir for AP parts + manifest + coverage (default: data/stellar-ap-v1).
    #[arg(short, long, default_value = "data/stellar-ap-v1")]
    pub out_dir: String,
    /// Source IDs per TAP query (URL-length bound).
    #[arg(long, default_value_t = 2000)]
    pub ids_per_query: usize,
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Cap on scanned IDs, 0 = all. Pilot runs use this.
    #[arg(long, default_value_t = 0)]
    pub max_ids: u64,
    /// Skip fetching; join whatever parts exist on disk.
    #[arg(long, default_value_t = false)]
    pub join_only: bool,
    /// Enriched parquet path (default: <out_dir>/enriched.parquet).
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
    /// Optional Gaia canonical parquet (e.g. data/stellar/v1/canonical/stars.parquet).
    #[arg(long)]
    pub gaia_parquet: Option<String>,
}
