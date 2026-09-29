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
    /// Stage 4: shard-wise canonical collection with resume/verify support.
    CollectData(CollectDataArgs),
    /// Stage 4: clean verified shards, assemble canonical parquet + model
    /// views and persist a quality report for the pilot.
    BuildDataset(BuildDatasetArgs),
    /// Stage 4B: fetch Gaia astrophysical_parameters (Teff/R/M/L) for the
    /// source_ids in a canonical parquet and LEFT JOIN them on top.
    EnrichStellar(EnrichStellarArgs),
    /// Stage 4A: which sources run anonymously vs require env secrets,
    /// plus active object-storage sink status (no secret values printed).
    AuthStatus,
    /// Stage 4A (пункт 6): write versioned NASA-enriched fixture artifacts:
    /// parsed rows, source_manifest.json, provenance.json.
    EnrichFixtures(EnrichFixturesArgs),
    /// Stage 4A (пункт 7): deterministic before/after ridge evaluation of
    /// NASA-derived features against Gaia photometry.
    EnrichReport(EnrichReportArgs),
    /// Stage 4A (пункт 8): anonymous MAST TIC cone spike (live request).
    SpikeMast(SpikeMastArgs),
    /// Stage 4A (пункт 9): anonymous IRSA 2MASS TAP spike with null-rate stats.
    SpikeIrsa(SpikeIrsaArgs),
    /// Stage 4A (пункт 10): JPL Horizons scene provider spike (anonymous).
    JplScenes(JplScenesArgs),
    /// Object storage: push every artifact of a dataset directory into the bucket.
    StorageUpload(StorageUploadArgs),
    /// Object storage: pull remote artifacts back into a local directory.
    StoragePull(StoragePullArgs),
    /// Object storage: show effective sink configuration and object counts.
    StorageList(StorageListArgs),
    Train(TrainArgs),
    /// Stage 5: spawn a fixer agent over a stopped training run.
    AgentFix(AgentFixArgs),
    /// Stage 5 (task 9): read-only evaluation through the shared library.
    Evaluate(EvaluateArgs),
    /// Stage 5 (task 9): forward-pass benchmark through the shared library.
    Benchmark(BenchmarkArgs),
}
