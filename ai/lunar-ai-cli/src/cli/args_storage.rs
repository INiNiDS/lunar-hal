use clap::Args;

#[derive(Args, Debug, Clone)]
pub struct StorageUploadArgs {
    #[arg(short, long)]
    pub dataset_dir: String,
    #[arg(short, long, default_value = "stellar/v1")]
    pub prefix: String,
}

#[derive(Args, Debug, Clone)]
pub struct StoragePullArgs {
    #[arg(short, long)]
    pub dest_dir: String,
    #[arg(short, long, default_value = "stellar/v1")]
    pub prefix: String,
}

#[derive(Args, Debug, Clone)]
pub struct StorageListArgs {
    #[arg(short, long, default_value = "stellar/v1")]
    pub prefix: String,
}
