use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct FetchArgs {
    #[arg(short, long, default_value = "raw_stars.csv")]
    pub output: String,
    #[arg(short, long)]
    pub username: Option<String>,
    #[arg(short, long)]
    pub password: Option<String>,
    #[arg(long, default_value = "1000000")]
    pub max_rows: usize,
    #[arg(long, default_value_t = 0.0)]
    pub ra_min: f64,
    #[arg(long, default_value_t = 180.0)]
    pub ra_max: f64,
    #[arg(long, default_value_t = 1.4)]
    pub max_ruwe: f64,
    #[arg(long, default_value_t = 10)]
    pub poll_initial_secs: u64,
    #[arg(long, default_value_t = 120)]
    pub poll_max_secs: u64,
}

#[derive(Args, Debug, Clone)]
pub struct FetchGnnArgs {
    #[arg(short, long, default_value = "raw_gnn_stars.csv")]
    pub output: String,
    #[arg(short, long)]
    pub username: Option<String>,
    #[arg(short, long)]
    pub password: Option<String>,
    #[arg(long, default_value = "1000000")]
    pub max_rows: usize,
    #[arg(long, default_value_t = 0.0)]
    pub ra_min: f64,
    #[arg(long, default_value_t = 180.0)]
    pub ra_max: f64,
    #[arg(long, default_value_t = 1.4)]
    pub max_ruwe: f64,
    #[arg(long, default_value_t = 10)]
    pub poll_initial_secs: u64,
    #[arg(long, default_value_t = 120)]
    pub poll_max_secs: u64,
}

#[derive(Args, Debug, Clone)]
pub struct CleanArgs {
    #[arg(short, long, default_value = "raw_stars.csv")]
    pub input: String,
    #[arg(short, long, default_value = "clean_stars.parquet")]
    pub output: String,
    #[arg(long, default_value_t = false)]
    pub print_sha256: bool,
}

#[derive(Args, Debug, Clone)]
pub struct CleanGnnArgs {
    #[arg(short, long, default_value = "raw_gnn_stars.csv")]
    pub input: String,
    #[arg(short, long, default_value = "clean_gnn_stars.parquet")]
    pub output: String,
    #[arg(long, default_value_t = false)]
    pub print_sha256: bool,
}

#[derive(Args, Debug, Clone)]
pub struct Sha256Args {
    #[arg(short, long)]
    pub input: String,
}

#[derive(Args, Debug, Clone)]
pub struct CombineArgs {
    #[arg(short, long, num_args = 2..)]
    pub inputs: Vec<String>,
    #[arg(short, long, default_value = "combined_stars.parquet")]
    pub output: String,
}
