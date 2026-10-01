pub mod args_data;
pub mod args_fetch;
pub mod args_spikes;
pub mod args_storage;
pub mod args_train;

use clap::{Parser, Subcommand};

pub use args_data::*;
pub use args_fetch::*;
pub use args_spikes::*;
pub use args_storage::*;
pub use args_train::*;

#[derive(Parser)]
#[command(name = "lnaicli")]
#[command(
    about = "CLI tool to download, process, and combine stellar data from Gaia",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    Fetch(FetchArgs),
    FetchGnn(FetchGnnArgs),
    Clean(CleanArgs),
    CleanGnn(CleanGnnArgs),
    Sha256(Sha256Args),
    Combine(CombineArgs),
    CollectData(CollectDataArgs),
    BuildDataset(BuildDatasetArgs),
    EnrichStellar(EnrichStellarArgs),
    AuthStatus,
    EnrichFixtures(EnrichFixturesArgs),
    EnrichReport(EnrichReportArgs),
    SpikeMast(SpikeMastArgs),
    SpikeIrsa(SpikeIrsaArgs),
    JplScenes(JplScenesArgs),
    StorageUpload(StorageUploadArgs),
    StoragePull(StoragePullArgs),
    StorageList(StorageListArgs),
    Train(TrainArgs),
    AgentFix(AgentFixArgs),
    Evaluate(EvaluateArgs),
    Benchmark(BenchmarkArgs),
}
