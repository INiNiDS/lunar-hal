use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct SpikeMastArgs {
    #[arg(long, default_value_t = 59.0)]
    pub ra: f64,
    #[arg(long, default_value_t = 6.0)]
    pub dec: f64,
    #[arg(long, default_value_t = 0.2)]
    pub radius: f64,
}

#[derive(Args, Debug, Clone)]
pub struct SpikeIrsaArgs {
    #[arg(long, default_value_t = 172.0)]
    pub ra_min: f64,
    #[arg(long, default_value_t = 172.05)]
    pub ra_max: f64,
    #[arg(long, default_value_t = -58.35)]
    pub dec_min: f64,
    #[arg(long, default_value_t = -58.28)]
    pub dec_max: f64,
    #[arg(long, default_value_t = 50)]
    pub top: usize,
}

#[derive(Args, Debug, Clone)]
pub struct JplScenesArgs {
    #[arg(long, default_value = "799")]
    pub body: String,
    #[arg(long, default_value = "2026-08-27")]
    pub start_time: String,
    #[arg(long, default_value = "2026-08-29")]
    pub stop_time: String,
    #[arg(long, default_value = "1d")]
    pub step_size: String,
    #[arg(short, long, default_value = "500@399")]
    pub center: String,
}
