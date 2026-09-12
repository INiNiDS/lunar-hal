use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use polars::prelude::*;
use rand::seq::SliceRandom;
use sha2::{Digest, Sha256};
use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "lnaicli")]
#[command(about = "CLI tool to download, process, and combine stellar data from Gaia", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Fetch {
        #[arg(short, long, default_value = "raw_stars.csv")]
        output: String,
        #[arg(short, long)]
        username: Option<String>,
        #[arg(short, long)]
        password: Option<String>,
        #[arg(long, default_value = "1000000")]
        max_rows: usize,
        #[arg(long, default_value_t = 0.0)]
        ra_min: f64,
        #[arg(long, default_value_t = 180.0)]
        ra_max: f64,
        #[arg(long, default_value_t = 1.4)]
        max_ruwe: f64,
        #[arg(long, default_value_t = 10)]
        poll_initial_secs: u64,
        #[arg(long, default_value_t = 120)]
        poll_max_secs: u64,
    },
    FetchGnn {
        #[arg(short, long, default_value = "raw_gnn_stars.csv")]
        output: String,
        #[arg(short, long)]
        username: Option<String>,
        #[arg(short, long)]
        password: Option<String>,
        #[arg(long, default_value = "1000000")]
        max_rows: usize,
        #[arg(long, default_value_t = 0.0)]
        ra_min: f64,
        #[arg(long, default_value_t = 180.0)]
        ra_max: f64,
        #[arg(long, default_value_t = 1.4)]
        max_ruwe: f64,
        #[arg(long, default_value_t = 10)]
        poll_initial_secs: u64,
        #[arg(long, default_value_t = 120)]
        poll_max_secs: u64,
    },
    Clean {
        #[arg(short, long, default_value = "raw_stars.csv")]
        input: String,
        #[arg(short, long, default_value = "clean_stars.parquet")]
        output: String,
        #[arg(long, default_value_t = false)]
        print_sha256: bool,
    },
    CleanGnn {
        #[arg(short, long, default_value = "raw_gnn_stars.csv")]
        input: String,
        #[arg(short, long, default_value = "clean_gnn_stars.parquet")]
        output: String,
        #[arg(long, default_value_t = false)]
        print_sha256: bool,
    },
    Sha256 {
        #[arg(short, long)]
        input: String,
    },
    Combine {
        #[arg(short, long, num_args = 2..)]
        inputs: Vec<String>,
        #[arg(short, long, default_value = "combined_stars.parquet")]
        output: String,
    },
    /// Stage 4: shard-wise canonical collection with resume/verify support.
    CollectData {
        #[arg(short, long, default_value = "data/canonical-v1")]
        out_dir: String,
        #[arg(long, default_value_t = 0.0)]
        ra_min: f64,
        #[arg(long, default_value_t = 360.0)]
        ra_max: f64,
        #[arg(long, default_value_t = 16.0)]
        mag_limit_g: f64,
        #[arg(long, default_value = "200000")]
        target_rows_per_shard: usize,
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Restrict work to shards whose id contains this substring.
        #[arg(long)]
        only: Option<String>,
        /// Re-attempt shards currently in Failed.
        #[arg(long, default_value_t = false)]
        retry_failed: bool,
        /// Re-verify checksums of Verified shards instead of skipping.
        #[arg(long, default_value_t = false)]
        verify: bool,
    },
    /// Stage 4: clean verified shards, assemble canonical parquet + model
    /// views and persist a quality report for the pilot.
    ///
    /// Memory-bounded by design: shards are processed in batches of
    /// --batch-shards and merged with the streaming engine, so peak RAM
    /// scales with one batch, not the whole dataset.
    BuildDataset {
        #[arg(short, long, default_value = "data/canonical-v1")]
        out_dir: String,
        #[arg(long, default_value_t = true)]
        quality_report: bool,
        /// Verified shards loaded/cleaned per batch; lower this if RAM runs out.
        #[arg(long, default_value_t = 2)]
        batch_shards: usize,
        /// Keep assembled/parts/*.parquet after the merge for debugging.
        #[arg(long, default_value_t = false)]
        keep_parts: bool,
    },
    /// Stage 4B: fetch Gaia astrophysical_parameters (Teff/R/M/L) for the
    /// source_ids in a canonical parquet and LEFT JOIN them on top.
    /// PINN targets live here; reruns resume finished chunks.
    EnrichStellar {
        /// Canonical parquet to enrich (e.g. assembled/canonical.parquet).
        #[arg(short, long)]
        data: String,
        /// Workdir for AP parts + manifest + coverage (default: data/stellar-ap-v1).
        #[arg(short, long, default_value = "data/stellar-ap-v1")]
        out_dir: String,
        /// Source IDs per TAP query (URL-length bound).
        #[arg(long, default_value_t = 2000)]
        ids_per_query: usize,
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Cap on scanned IDs, 0 = all. Pilot runs use this.
        #[arg(long, default_value_t = 0)]
        max_ids: u64,
        /// Skip fetching; join whatever parts exist on disk.
        #[arg(long, default_value_t = false)]
        join_only: bool,
        /// Enriched parquet path (default: <out_dir>/enriched.parquet).
        #[arg(long)]
        join_output: Option<String>,
    },
    /// Stage 4A: which sources run anonymously vs require env secrets,
    /// plus active object-storage sink status (no secret values printed).
    AuthStatus,
    /// Stage 4A (пункт 6): write versioned NASA-enriched fixture artifacts:
    /// parsed rows, source_manifest.json, provenance.json.
    EnrichFixtures {
        #[arg(short, long, default_value = "data/nasa-enriched-v1")]
        out_dir: String,
    },
    /// Stage 4A (пункт 7): deterministic before/after ridge evaluation of
    /// NASA-derived features against Gaia photometry. Writes
    /// enrichment_report.json; honestly reports underpowered samples instead
    /// of claiming an improvement.
    EnrichReport {
        #[arg(short, long, default_value = "data/nasa-enriched-v1")]
        out_dir: String,
        /// Optional Gaia canonical parquet (e.g. data/stellar/v1/canonical/stars.parquet).
        #[arg(long)]
        gaia_parquet: Option<String>,
    },
    /// Stage 4A (пункт 8): anonymous MAST TIC cone spike (live request).
    SpikeMast {
        #[arg(long, default_value_t = 59.0)]
        ra: f64,
        #[arg(long, default_value_t = 6.0)]
        dec: f64,
        #[arg(long, default_value_t = 0.2)]
        radius: f64,
    },
    /// Stage 4A (пункт 9): anonymous IRSA 2MASS TAP spike with null-rate stats.
    SpikeIrsa {
        #[arg(long, default_value_t = 172.0)]
        ra_min: f64,
        #[arg(long, default_value_t = 172.05)]
        ra_max: f64,
        #[arg(long, default_value_t = -58.35)]
        dec_min: f64,
        #[arg(long, default_value_t = -58.28)]
        dec_max: f64,
        #[arg(long, default_value_t = 50)]
        top: usize,
    },
    /// Stage 4A (пункт 10): JPL Horizons scene provider spike (anonymous).
    JplScenes {
        #[arg(long, default_value = "799")]
        body: String,
        #[arg(long, default_value = "2026-08-27")]
        start_time: String,
        #[arg(long, default_value = "2026-08-29")]
        stop_time: String,
        #[arg(long, default_value = "1d")]
        step_size: String,
        #[arg(short, long, default_value = "500@399")]
        center: String,
    },
    /// Object storage: push every artifact of a dataset directory into the
    /// configured bucket (any S3-compatible endpoint: MinIO/Spaces/AWS/R2).
    StorageUpload {
        #[arg(short, long)]
        dataset_dir: String,
        #[arg(short, long, default_value = "stellar/v1")]
        prefix: String,
    },
    /// Object storage: pull remote artifacts back into a local directory
    /// (resume-friendly: unchanged files are skipped by size match).
    StoragePull {
        #[arg(short, long)]
        dest_dir: String,
        #[arg(short, long, default_value = "stellar/v1")]
        prefix: String,
    },
    /// Object storage: show effective sink configuration (secrets redacted)
    /// and, when reachable, remote object counts under the prefix.
    StorageList {
        #[arg(short, long, default_value = "stellar/v1")]
        prefix: String,
    },
    Train {
        #[arg(short, long, value_enum, default_value_t = CliModel::Pinn)]
        model: CliModel,

        #[arg(short, long)]
        data: Option<String>,

        #[arg(short, long)]
        resume: Option<String>,

        #[arg(long)]
        norm: Option<String>,

        #[arg(short = 'O', long, default_value = "stellar_model")]
        output_dir: String,

        #[arg(long, default_value_t = 200)]
        epochs: usize,

        #[arg(long, default_value_t = 4096)]
        batch_size: usize,

        #[arg(long, default_value_t = 5e-4)]
        lr: f64,

        #[arg(long, default_value_t = 0.1)]
        physics_weight: f64,

        #[arg(long, default_value_t = 0.1)]
        val_frac: f32,

        #[arg(long, default_value_t = 0)]
        gpu_index: usize,

        #[arg(long)]
        holdout: Option<String>,

        #[arg(long)]
        lnai_bin: Option<String>,

        #[arg(long, default_value = "stellar_model.bpk")]
        model_file: String,

        #[arg(long, default_value = "stellar_norm.json")]
        norm_file: String,

        /// GNN-Kinematics: k-NN neighbours per node.
        #[arg(long, default_value_t = 8)]
        knn_k: usize,
        /// GNN-Kinematics: hidden width.
        #[arg(long, default_value_t = 256)]
        hidden_dim: usize,
        /// GNN-Kinematics: spatial group cap.
        #[arg(long, default_value_t = 64)]
        max_group_size: usize,
        /// GNN-Kinematics: group radius in parsecs.
        #[arg(long, default_value_t = 50.0)]
        radius_pc: f32,
        /// SIREN: texture grid size.
        #[arg(long, default_value_t = 64)]
        texture_size: usize,
        /// SIREN: star budget for texture synthesis.
        #[arg(long, default_value_t = 5000)]
        max_stars: usize,
        /// Explicit global seed (overrides the derived run seed).
        #[arg(long)]
        seed: Option<u64>,
        /// Deterministic systematic sample cap (every k-th row); unset = all rows.
        #[arg(long)]
        max_rows: Option<u64>,
        /// Spatial-tile subset, comma-separated (GNN/PINN); unset = all tiles.
        #[arg(long)]
        tiles: Option<String>,
        /// Dataset manifest hash recorded into the artifact (default empty).
        #[arg(long, default_value = "")]
        dataset_manifest_hash: String,
    },
    /// Stage 5 (task 9): read-only evaluation through the shared library —
    /// loads the artifact, reports losses, runs no optimizer step.
    Evaluate {
        #[arg(short, long, value_enum, default_value_t = CliModel::Pinn)]
        model: CliModel,
        #[arg(short, long, default_value = "stellar_model")]
        output_dir: String,
        #[arg(short, long)]
        data: Option<String>,
        #[arg(long)]
        holdout: Option<String>,
        #[arg(long, default_value_t = 4096)]
        batch_size: usize,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value = "stellar_model.bpk")]
        model_file: String,
        #[arg(long, default_value = "stellar_norm.json")]
        norm_file: String,
    },
    /// Stage 5 (task 9): forward-pass benchmark through the shared library —
    /// times inference, writes `benchmark.json`, trains nothing.
    Benchmark {
        #[arg(short, long, value_enum, default_value_t = CliModel::Pinn)]
        model: CliModel,
        #[arg(short, long, default_value = "stellar_model")]
        output_dir: String,
        #[arg(long, default_value_t = 100)]
        iters: u32,
        #[arg(long, default_value_t = 10)]
        warmup: u32,
        #[arg(long, default_value_t = 4096)]
        batch_size: usize,
        #[arg(long)]
        seed: Option<u64>,
    },
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum CliModel {
    Pinn,
    Gnn,
    Siren,
}

impl CliModel {
    fn worker_binary(&self) -> &'static str {
        match self {
            CliModel::Pinn => "lnai",
            CliModel::Gnn => "lnai-gnn",
            CliModel::Siren => "lnai-siren",
        }
    }
}

fn main() -> Result<()> {
    // Pick up storage/credential variables from .env unless already set
    // (orchestrate.sh/run-podman.sh source it themselves).
    load_dotenv_if_present();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Fetch {
            output,
            username,
            password,
            max_rows,
            ra_min,
            ra_max,
            max_ruwe,
            poll_initial_secs,
            poll_max_secs,
        } => {
            fetch_stellar_data(&FetchOptions {
                output_path: output,
                username: username.as_deref(),
                password: password.as_deref(),
                max_rows,
                ra_min: *ra_min,
                ra_max: *ra_max,
                max_ruwe: *max_ruwe,
                poll_initial_secs: *poll_initial_secs,
                poll_max_secs: *poll_max_secs,
                include_velocities: false,
            })?;
        }
        Commands::FetchGnn {
            output,
            username,
            password,
            max_rows,
            ra_min,
            ra_max,
            max_ruwe,
            poll_initial_secs,
            poll_max_secs,
        } => {
            fetch_stellar_data(&FetchOptions {
                output_path: output,
                username: username.as_deref(),
                password: password.as_deref(),
                max_rows,
                ra_min: *ra_min,
                ra_max: *ra_max,
                max_ruwe: *max_ruwe,
                poll_initial_secs: *poll_initial_secs,
                poll_max_secs: *poll_max_secs,
                include_velocities: true,
            })?;
        }
        Commands::Clean {
            input,
            output,
            print_sha256,
        } => {
            clean_and_transform(input, output, *print_sha256)?;
        }
        Commands::CleanGnn {
            input,
            output,
            print_sha256,
        } => {
            clean_and_transform_gnn(input, output, *print_sha256)?;
        }
        Commands::Sha256 { input } => {
            let h = sha256_file(input)?;
            println!("{}  {}", h, input);
        }
        Commands::Combine { inputs, output } => {
            combine_datasets(inputs, output)?;
        }
        Commands::CollectData {
            out_dir,
            ra_min,
            ra_max,
            mag_limit_g,
            target_rows_per_shard,
            concurrency,
            only,
            retry_failed,
            verify,
        } => {
            run_collect_data(CollectDataArgs {
                out_dir: PathBuf::from(out_dir),
                ra_start_deg: *ra_min,
                ra_end_deg: *ra_max,
                mag_limit_g: *mag_limit_g,
                target_rows_per_shard: *target_rows_per_shard,
                concurrency: *concurrency,
                only: only.clone(),
                retry_failed: *retry_failed,
                verify: *verify,
            })?;
        }
        Commands::BuildDataset {
            out_dir,
            quality_report,
            batch_shards,
            keep_parts,
        } => {
            run_build_dataset(
                &PathBuf::from(out_dir),
                *quality_report,
                *batch_shards,
                *keep_parts,
            )?;
        }
        Commands::AuthStatus => {
            run_auth_status();
        }
        Commands::EnrichStellar {
            data,
            out_dir,
            ids_per_query,
            concurrency,
            max_ids,
            join_only,
            join_output,
        } => {
            run_enrich_stellar(
                &PathBuf::from(data),
                &PathBuf::from(out_dir),
                *ids_per_query,
                *concurrency,
                *max_ids,
                *join_only,
                join_output.clone(),
            )?;
        }
        Commands::EnrichFixtures { out_dir } => {
            run_enrich_fixtures(out_dir)?;
        }
        Commands::EnrichReport {
            out_dir,
            gaia_parquet,
        } => {
            run_enrich_report(&PathBuf::from(out_dir), gaia_parquet.as_deref())?;
        }
        Commands::SpikeMast { ra, dec, radius } => {
            run_spike_mast(*ra, *dec, *radius)?;
        }
        Commands::SpikeIrsa {
            ra_min,
            ra_max,
            dec_min,
            dec_max,
            top,
        } => {
            run_spike_irsa(*ra_min, *ra_max, *dec_min, *dec_max, *top)?;
        }
        Commands::JplScenes {
            body,
            start_time,
            stop_time,
            step_size,
            center,
        } => {
            run_jpl_scenes(body, start_time, stop_time, step_size, center)?;
        }
        Commands::StorageUpload {
            dataset_dir,
            prefix,
        } => {
            run_storage_upload(&PathBuf::from(dataset_dir), prefix)?;
        }
        Commands::StoragePull { dest_dir, prefix } => {
            run_storage_pull(&PathBuf::from(dest_dir), prefix)?;
        }
        Commands::StorageList { prefix } => {
            run_storage_list(prefix)?;
        }
        Commands::Train {
            model,
            data,
            resume,
            norm,
            output_dir,
            epochs,
            batch_size,
            lr,
            physics_weight,
            val_frac,
            gpu_index,
            holdout,
            lnai_bin,
            model_file,
            norm_file,
            knn_k,
            hidden_dim,
            max_group_size,
            radius_pc,
            texture_size,
            max_stars,
            seed,
            dataset_manifest_hash,
            max_rows,
            tiles,
        } => {
            run_train(&TrainOptions {
                model: *model,
                data: data.as_deref(),
                resume: resume.as_deref(),
                norm: norm.as_deref(),
                output_dir,
                epochs: *epochs,
                batch_size: *batch_size,
                lr: *lr,
                physics_weight: *physics_weight,
                val_frac: *val_frac,
                gpu_index: *gpu_index,
                holdout: holdout.as_deref(),
                lnai_bin: lnai_bin.as_deref(),
                model_file,
                norm_file,
                knn_k: *knn_k,
                hidden_dim: *hidden_dim,
                max_group_size: *max_group_size,
                radius_pc: *radius_pc,
                texture_size: *texture_size,
                max_stars: *max_stars,
                seed: *seed,
                dataset_manifest_hash,
                max_rows: *max_rows,
                tiles: tiles.clone(),
            })?;
        }
        Commands::Evaluate {
            model,
            output_dir,
            data,
            holdout,
            batch_size,
            seed,
            model_file,
            norm_file,
        } => {
            run_evaluate_cmd(
                *model,
                output_dir,
                data.as_deref(),
                holdout.as_deref(),
                *batch_size,
                *seed,
                model_file,
                norm_file,
            )?;
        }
        Commands::Benchmark {
            model,
            output_dir,
            iters,
            warmup,
            batch_size,
            seed,
        } => {
            run_benchmark_cmd(*model, output_dir, *iters, *warmup, *batch_size, *seed)?;
        }
    }

    Ok(())
}

fn resolve_data_path(data: Option<&str>) -> Result<PathBuf> {
    if let Some(p) = data {
        return Ok(PathBuf::from(p));
    }
    for candidate in [
        "ai_data/clean_stars2.parquet",
        "ai_data/clean_stars.parquet",
    ] {
        let path = Path::new(candidate);
        if path.exists() {
            println!("Using dataset: {}", path.display());
            return Ok(path.to_path_buf());
        }
    }
    anyhow::bail!(
        "No --data given and no cleaned parquet found in ai_data/. \
         Run 'lnaicli fetch' and 'lnaicli clean' first, or pass --data <path>."
    )
}

fn find_lnai_binary(override_path: Option<&str>, model: CliModel) -> Result<PathBuf> {
    if let Some(p) = override_path {
        let path = PathBuf::from(p);
        if !path.exists() {
            anyhow::bail!("--lnai-bin does not exist: {}", path.display());
        }
        return Ok(path);
    }

    let exe_suffix = env::consts::EXE_SUFFIX;
    let bin_name = format!("{}{}", model.worker_binary(), exe_suffix);

    if model == CliModel::Pinn
        && let Ok(path_env) = env::var("LNAI_BIN")
    {
        let p = PathBuf::from(path_env);
        if p.exists() {
            return Ok(p);
        }
    }

    if let Ok(paths) = env::var("PATH") {
        for dir in paths.split(std::path::MAIN_SEPARATOR_STR) {
            if dir.is_empty() {
                continue;
            }
            let candidate = Path::new(dir).join(&bin_name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
    }

    let workspace_root = env::current_dir().ok();
    if let Some(cwd) = workspace_root {
        for profile in ["release", "debug"] {
            for sub in ["", "ai/lnai"] {
                let candidate = cwd.join("target").join(profile);
                let candidate = if sub.is_empty() {
                    candidate.join(&bin_name)
                } else {
                    candidate.join(sub).join(&bin_name)
                };
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
    }

    anyhow::bail!(
        "Could not find the '{}' binary. Build it with `cargo build -p {} --release` \
         or pass --lnai-bin /path/to/{}.",
        bin_name,
        match model {
            CliModel::Pinn => "lnai",
            CliModel::Gnn => "lnai-gnn",
            CliModel::Siren => "lnai-siren",
        },
        bin_name,
    )
}

fn copy_file_if_present(src: &Path, dst: &Path) -> Result<()> {
    if !src.exists() {
        return Ok(());
    }
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::copy(src, dst)
        .with_context(|| format!("failed to copy {} -> {}", src.display(), dst.display()))?;
    Ok(())
}

struct TrainOptions<'a> {
    model: CliModel,
    data: Option<&'a str>,
    resume: Option<&'a str>,
    norm: Option<&'a str>,
    output_dir: &'a str,
    epochs: usize,
    batch_size: usize,
    lr: f64,
    physics_weight: f64,
    val_frac: f32,
    gpu_index: usize,
    holdout: Option<&'a str>,
    lnai_bin: Option<&'a str>,
    model_file: &'a str,
    norm_file: &'a str,
    knn_k: usize,
    hidden_dim: usize,
    max_group_size: usize,
    radius_pc: f32,
    texture_size: usize,
    max_stars: usize,
    seed: Option<u64>,
    dataset_manifest_hash: &'a str,
    max_rows: Option<u64>,
    tiles: Option<String>,
}

/// Stage 5 (task 9): `lnaicli train` builds one shared [`TrainingSpec`] and
/// delegates flag rendering to the library (`worker_argv`), so CLI and
/// Testbench spawn byte-identical worker commands.
fn training_spec_from_opts(
    opts: &TrainOptions<'_>,
    data_path: &Path,
) -> Result<lnai_training::spec::TrainingSpec> {
    use lnai_training::spec::{
        GnnKinematicsConfig, ModelConfig, ModelKind, PinnConfig, SirenConfig, TrainingSpec,
    };
    let (model, config) = match opts.model {
        CliModel::Pinn => (
            ModelKind::Pinn,
            ModelConfig::Pinn(PinnConfig {
                physics_weight: opts.physics_weight,
                hidden_dim: 256,
            }),
        ),
        CliModel::Gnn => (
            ModelKind::GnnKinematics,
            ModelConfig::GnnKinematics(GnnKinematicsConfig {
                knn_k: opts.knn_k as u32,
                hidden_dim: opts.hidden_dim as u32,
                output_dim: 3,
                max_group_size: opts.max_group_size as u32,
                radius_pc: opts.radius_pc,
                physics_weight: opts.physics_weight,
            }),
        ),
        CliModel::Siren => (
            ModelKind::Siren,
            ModelConfig::Siren(SirenConfig {
                texture_size: opts.texture_size as u32,
                hidden_dim: 64,
                max_stars: opts.max_stars as u32,
                // SIREN texture synthesis predates the global seed flag.
                seed: opts.seed.unwrap_or(42),
            }),
        ),
    };
    let spec = TrainingSpec {
        model,
        config,
        dataset_manifest_hash: opts.dataset_manifest_hash.to_string(),
        data_path: Some(data_path.display().to_string()),
        epochs: opts.epochs as u32,
        batch_size: opts.batch_size as u32,
        lr: opts.lr,
        val_frac: opts.val_frac,
        output_dir: opts.output_dir.to_string(),
        resume_from: opts.resume.map(str::to_string),
        holdout: opts.holdout.map(str::to_string),
        gpu_index: opts.gpu_index as u32,
        patience: 20,
        grad_accum: 2,
        clip_grad_norm: 1.0,
        seed: opts.seed,
        model_file: opts.model_file.to_string(),
        norm_file: opts.norm_file.to_string(),
        max_rows: opts.max_rows,
        tiles: opts.tiles.clone(),
    };
    spec.validate()
        .map_err(|errs| anyhow!("invalid training spec: {}", errs.join("; ")))?;
    Ok(spec)
}

fn run_train(opts: &TrainOptions<'_>) -> Result<()> {
    let data_path = resolve_data_path(opts.data)?;
    let worker = find_lnai_binary(opts.lnai_bin, opts.model)?;

    let output_path = Path::new(opts.output_dir);
    std::fs::create_dir_all(output_path)
        .with_context(|| format!("failed to create output dir {}", output_path.display()))?;
    let out_model = output_path.join(opts.model_file);
    let out_norm = output_path.join(opts.norm_file);

    let mut resume_dir: Option<PathBuf> = None;
    if let Some(resume_path) = opts.resume {
        let resume_pb = PathBuf::from(resume_path);
        let resume_dir_path = if resume_pb.is_dir() {
            resume_pb.clone()
        } else {
            resume_pb
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        };

        let src_model = resume_pb
            .join(opts.model_file)
            .components()
            .collect::<PathBuf>();
        let src_norm = resume_pb.join(opts.norm_file);

        let resolved_model = if src_model.exists() {
            src_model
        } else {
            resume_dir_path.join(opts.model_file)
        };
        let resolved_norm = if src_norm.exists() {
            src_norm
        } else {
            resume_dir_path.join(opts.norm_file)
        };

        if !resolved_model.exists() {
            anyhow::bail!(
                "Resume requested but model file not found: {}",
                resolved_model.display()
            );
        }
        if !resolved_norm.exists() {
            anyhow::bail!(
                "Resume requested but norm file not found: {}",
                resolved_norm.display()
            );
        }

        println!("Staging resume files into output dir:");
        println!("  {} -> {}", resolved_model.display(), out_model.display());
        copy_file_if_present(&resolved_model, &out_model)?;
        println!("  {} -> {}", resolved_norm.display(), out_norm.display());
        copy_file_if_present(&resolved_norm, &out_norm)?;

        if let Some(extra_norm) = opts.norm {
            let extra_norm_path = Path::new(extra_norm);
            if !extra_norm_path.exists() {
                anyhow::bail!("--norm file not found: {}", extra_norm_path.display());
            }
            println!(
                "  {} -> {} (overrides any copied norm)",
                extra_norm_path.display(),
                out_norm.display()
            );
            copy_file_if_present(extra_norm_path, &out_norm)?;
        }

        resume_dir = Some(output_path.to_path_buf());
    } else if let Some(extra_norm) = opts.norm {
        let extra_norm_path = Path::new(extra_norm);
        if !extra_norm_path.exists() {
            anyhow::bail!("--norm file not found: {}", extra_norm_path.display());
        }
        println!(
            "Staging norm into output dir: {} -> {}",
            extra_norm_path.display(),
            out_norm.display()
        );
        copy_file_if_present(extra_norm_path, &out_norm)?;
    }

    let spec = training_spec_from_opts(opts, &data_path)?;

    println!();
    println!("=== lnaicli train ===");
    println!("Model:       {}", spec.model.slug());
    println!("Data:        {}", data_path.display());
    if let Some(rd) = &resume_dir {
        println!("Resume from: {}", rd.display());
    } else {
        println!("Resume from: <none, training from scratch>");
    }
    println!("Output dir:  {}", output_path.display());
    println!("Worker:      {}", worker.display());
    println!(
        "Hyperparams: epochs={}, batch={}, lr={:.2e}, phys_w={}, val_frac={}, gpu={}",
        opts.epochs, opts.batch_size, opts.lr, opts.physics_weight, opts.val_frac, opts.gpu_index
    );
    println!();

    // Stage 5: resume staging tells the worker via --resume-from pointing at
    // the output dir (legacy contract, unchanged).
    let mut worker_spec = spec.clone();
    if resume_dir.is_some() {
        worker_spec.resume_from = Some(output_path.display().to_string());
    }
    let mut cmd = Command::new(&worker);
    for arg in worker_spec.worker_argv() {
        // worker_argv carries `--data ""` when unset; the resolved path wins.
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }

    println!("Running: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn worker at {}", worker.display()))?;
    if !status.success() {
        anyhow::bail!("training worker failed with exit code {:?}", status.code());
    }

    println!();
    println!("Done. Model artifacts:");
    println!("  {}", out_model.display());
    println!("  {}", out_norm.display());
    Ok(())
}

/// Stage 5 (task 9): read-only evaluation via the shared worker protocol.
#[allow(clippy::too_many_arguments)]
fn run_evaluate_cmd(
    model: CliModel,
    output_dir: &str,
    data: Option<&str>,
    holdout: Option<&str>,
    batch_size: usize,
    seed: Option<u64>,
    model_file: &str,
    norm_file: &str,
) -> Result<()> {
    use lnai_training::spec::{EvaluationSpec, ModelKind};
    let kind = match model {
        CliModel::Pinn => ModelKind::Pinn,
        CliModel::Gnn => ModelKind::GnnKinematics,
        CliModel::Siren => ModelKind::Siren,
    };
    let output_path = Path::new(output_dir);
    if !output_path.join(model_file).exists() {
        anyhow::bail!(
            "evaluate: model file not found: {}",
            output_path.join(model_file).display()
        );
    }
    let data_path = resolve_data_path(data)
        .ok()
        .map(|p| p.display().to_string());
    let spec = EvaluationSpec {
        model: kind,
        artifact_hash: sha256_file(&output_path.join(model_file).display().to_string())
            .unwrap_or_default(),
        dataset_manifest_hash: String::new(),
        data_path,
        batch_size: batch_size as u32,
        output_dir: output_dir.to_string(),
        seed,
    };
    spec.validate()
        .map_err(|errs| anyhow!("invalid evaluation spec: {}", errs.join("; ")))?;
    let worker = find_lnai_binary(None, model)?;
    let mut cmd = Command::new(&worker);
    for arg in spec.worker_argv(holdout) {
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }
    // Evaluation reuses the training worker binary in --evaluate-only mode;
    // model/norm file names travel along for artifact lookup.
    cmd.arg("--model-file").arg(model_file);
    cmd.arg("--norm-file").arg(norm_file);
    println!("Running: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn worker at {}", worker.display()))?;
    if !status.success() {
        anyhow::bail!(
            "evaluation worker failed with exit code {:?}",
            status.code()
        );
    }
    Ok(())
}

/// Stage 5 (task 9): forward-pass benchmark via the shared worker protocol.
fn run_benchmark_cmd(
    model: CliModel,
    output_dir: &str,
    iters: u32,
    warmup: u32,
    batch_size: usize,
    seed: Option<u64>,
) -> Result<()> {
    use lnai_training::spec::{BenchmarkSpec, ModelKind};
    let kind = match model {
        CliModel::Pinn => ModelKind::Pinn,
        CliModel::Gnn => ModelKind::GnnKinematics,
        CliModel::Siren => ModelKind::Siren,
    };
    let spec = BenchmarkSpec {
        model: kind,
        artifact_hash: String::new(),
        iterations: iters,
        warmup_iterations: warmup,
        output_dir: output_dir.to_string(),
        batch_size: batch_size as u32,
        seed,
    };
    spec.validate()
        .map_err(|errs| anyhow!("invalid benchmark spec: {}", errs.join("; ")))?;
    let worker = find_lnai_binary(None, model)?;
    let mut cmd = Command::new(&worker);
    for arg in spec.worker_argv() {
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }
    println!("Running: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn worker at {}", worker.display()))?;
    if !status.success() {
        anyhow::bail!("benchmark worker failed with exit code {:?}", status.code());
    }
    Ok(())
}

fn sha256_file(path: &str) -> Result<String> {
    let mut file =
        File::open(path).map_err(|e| anyhow!("Cannot open {} for hashing: {}", path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(hex_encode(&digest))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

struct FetchOptions<'a> {
    output_path: &'a str,
    username: Option<&'a str>,
    password: Option<&'a str>,
    max_rows: &'a usize,
    ra_min: f64,
    ra_max: f64,
    max_ruwe: f64,
    poll_initial_secs: u64,
    poll_max_secs: u64,
    include_velocities: bool,
}

fn fetch_stellar_data(opts: &FetchOptions<'_>) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Some(std::time::Duration::from_secs(7200)))
        .pool_max_idle_per_host(0)
        .cookie_store(true)
        .build()?;

    #[cfg(debug_assertions)]
    {
        let _ = (
            opts.username,
            opts.password,
            opts.max_rows,
            opts.ra_min,
            opts.ra_max,
            opts.max_ruwe,
            opts.poll_initial_secs,
            opts.poll_max_secs,
            opts.include_velocities,
        );
        let url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync";
        let query = "select hostname, ra, dec, sy_dist, st_teff, st_rad, st_mass, st_lum from ps";

        println!("Sending request to NASA Exoplanet Archive (Debug Mode)...");
        println!("Note: NASA Exoplanet Archive does not include bp_rp and g_mag.");

        let response = client
            .get(url)
            .query(&[("query", query), ("format", "csv")])
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "NASA server returned an error status: {}",
                response.status()
            ));
        }

        save_filtered_response(response, opts.output_path)?;
    }

    #[cfg(not(debug_assertions))]
    {
        if opts.username.is_none() || opts.password.is_none() {
            eprintln!("WARNING: No Gaia credentials provided.");
            eprintln!("         Anonymous access may limit result set size.");
            eprintln!("         Use --username and --password to authenticate.");
            eprintln!();
        }

        if let (Some(user), Some(pass)) = (opts.username, opts.password) {
            println!("Authenticating with ESA Gaia Archive as {}...", user);
            let login_url = "https://gea.esac.esa.int/tap-server/login";
            let login_resp = client
                .post(login_url)
                .form(&[("username", user), ("password", pass)])
                .send()?;

            if !login_resp.status().is_success() {
                return Err(anyhow!(
                    "Gaia login failed. Status: {}. Check your credentials.",
                    login_resp.status()
                ));
            }
            println!("Authentication successful!");
        }

        let query = if opts.include_velocities {
            format!(
                "SELECT TOP {} \
        CAST(gs.source_id AS varchar) AS hostname, \
        gs.ra, \
        gs.dec, \
        gs.parallax, \
        gs.pmra, \
        gs.pmdec, \
        gs.radial_velocity, \
        gs.phot_g_mean_mag AS g_mag, \
        gs.bp_rp, \
        ap.teff_gspphot AS st_teff, \
        ap.radius_gspphot AS st_rad, \
        ap.mass_flame AS st_mass, \
        ap.lum_flame AS st_lum \
     FROM gaiadr3.gaia_source gs \
     JOIN gaiadr3.astrophysical_parameters ap USING (source_id) \
     WHERE gs.parallax IS NOT NULL \
       AND gs.parallax > 0 \
       AND gs.ra BETWEEN {} AND {} \
       AND gs.pmra IS NOT NULL \
       AND gs.pmdec IS NOT NULL \
       AND gs.radial_velocity IS NOT NULL \
       AND ap.teff_gspphot IS NOT NULL \
       AND ap.radius_gspphot IS NOT NULL \
       AND ap.mass_flame IS NOT NULL \
       AND gs.bp_rp IS NOT NULL \
       AND gs.phot_g_mean_mag IS NOT NULL \
       AND gs.ruwe < {}",
                opts.max_rows, opts.ra_min, opts.ra_max, opts.max_ruwe
            )
        } else {
            format!(
                "SELECT TOP {} \
        CAST(gs.source_id AS varchar) AS hostname, \
        gs.ra, \
        gs.dec, \
        1000.0/gs.parallax AS sy_dist, \
        gs.phot_g_mean_mag AS g_mag, \
        gs.bp_rp, \
        ap.teff_gspphot AS st_teff, \
        ap.radius_gspphot AS st_rad, \
        ap.mass_flame AS st_mass, \
        ap.lum_flame AS st_lum \
     FROM gaiadr3.gaia_source gs \
     JOIN gaiadr3.astrophysical_parameters ap USING (source_id) \
     WHERE gs.parallax IS NOT NULL \
       AND gs.parallax > 0 \
       AND gs.ra BETWEEN {} AND {} \
       AND ap.teff_gspphot IS NOT NULL \
       AND ap.radius_gspphot IS NOT NULL \
       AND ap.mass_flame IS NOT NULL \
       AND gs.bp_rp IS NOT NULL \
       AND gs.phot_g_mean_mag IS NOT NULL \
       AND gs.ruwe < {}",
                opts.max_rows, opts.ra_min, opts.ra_max, opts.max_ruwe
            )
        };

        let url = "https://gea.esac.esa.int/tap-server/tap/async";
        println!("Submitting asynchronous job to ESA Gaia Archive...");
        println!(
            "Query TOP {} rows, RA=[{}, {}], ruwe<{} ...",
            opts.max_rows, opts.ra_min, opts.ra_max, opts.max_ruwe
        );

        let response = client
            .post(url)
            .form(&[
                ("REQUEST", "doQuery"),
                ("LANG", "ADQL"),
                ("FORMAT", "csv"),
                ("QUERY", &query),
                ("PHASE", "RUN"),
            ])
            .send()?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!(
                "Gaia server rejected job submission. Status: {}\nBody: {}",
                status,
                body
            ));
        }

        let job_url = response.url().clone();
        println!("Job created. Monitoring: {}", job_url);

        let phase_url = format!("{}/phase", job_url);
        let result_url = format!("{}/results/result", job_url);

        let mut backoff = opts.poll_initial_secs.max(1);
        let backoff_max = opts.poll_max_secs.max(backoff);
        let mut attempts: u32 = 0;
        loop {
            let phase_resp = client
                .get(&phase_url)
                .header(reqwest::header::CONNECTION, "close")
                .send();

            let phase = match phase_resp {
                Ok(resp) => match resp.text() {
                    Ok(text) => text.trim().to_uppercase(),
                    Err(_) => {
                        eprintln!(
                            "Warning: failed to read phase (attempt {}), retrying in {}s...",
                            attempts + 1,
                            backoff
                        );
                        std::thread::sleep(std::time::Duration::from_secs(backoff));
                        backoff = (backoff.saturating_mul(2)).min(backoff_max);
                        attempts += 1;
                        if attempts > 200 {
                            return Err(anyhow!("Gaia job aborted: too many failed phase reads"));
                        }
                        continue;
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Warning: network error '{}' (attempt {}), retrying in {}s...",
                        e,
                        attempts + 1,
                        backoff
                    );
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                    backoff = (backoff.saturating_mul(2)).min(backoff_max);
                    attempts += 1;
                    if attempts > 200 {
                        return Err(anyhow!("Gaia job aborted: too many network errors"));
                    }
                    continue;
                }
            };

            attempts = 0;
            backoff = opts.poll_initial_secs.max(1);
            println!("  Job phase: {} (next poll in {}s)", phase, backoff);

            match phase.as_str() {
                "COMPLETED" => {
                    println!("Job completed!");
                    break;
                }
                "ERROR" | "ABORTED" => {
                    return Err(anyhow!("Job failed on server. Phase: {}", phase));
                }
                _ => {
                    std::thread::sleep(std::time::Duration::from_secs(backoff));
                    backoff = (backoff.saturating_mul(2)).min(backoff_max);
                }
            }
        }

        println!("Downloading results from: {}", result_url);
        let response = client
            .get(&result_url)
            .header(reqwest::header::CONNECTION, "close")
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to download results. Status: {}",
                response.status()
            ));
        }

        save_filtered_response(response, opts.output_path)?;
    }

    Ok(())
}

fn save_filtered_response(response: reqwest::blocking::Response, output_path: &str) -> Result<()> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);
    let reader = BufReader::new(response);

    let mut lines = 0usize;
    for line_result in reader.lines() {
        let line = line_result?;
        if !line.trim_start().starts_with('#') {
            writer.write_all(line.as_bytes())?;
            writer.write_all(b"\n")?;
            lines += 1;
        }
    }
    writer.flush()?;
    println!("Raw data saved to: {} ({} lines)", output_path, lines);
    Ok(())
}

fn clean_and_transform(input_path: &str, output_path: &str, print_sha256: bool) -> Result<()> {
    println!("Reading and preprocessing: {}", input_path);

    if !Path::new(input_path).exists() {
        return Err(anyhow!("Input file does not exist: {}", input_path));
    }

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    let has_gaia_cols = df.column("bp_rp").is_ok() && df.column("g_mag").is_ok();

    if has_gaia_cols {
        println!("Detected Gaia columns (bp_rp, g_mag) - photometric data included");
    } else {
        println!("WARNING: bp_rp and g_mag columns not found.");
        println!("         The model requires these for conditional inputs.");
        println!("         Re-fetch data with: lnaicli fetch --username USER --password PASS");
    }

    let lazy_df = df.lazy();

    let mut target_cols = vec![
        PlSmallStr::from_str("ra"),
        PlSmallStr::from_str("dec"),
        PlSmallStr::from_str("sy_dist"),
        PlSmallStr::from_str("st_teff"),
        PlSmallStr::from_str("st_rad"),
        PlSmallStr::from_str("st_mass"),
    ];

    if has_gaia_cols {
        target_cols.push(PlSmallStr::from_str("bp_rp"));
        target_cols.push(PlSmallStr::from_str("g_mag"));
    }

    let selector = Selector::ByName {
        names: Arc::from(target_cols),
        strict: true,
    };

    let mut agg_exprs = vec![
        col("ra").first(),
        col("dec").first(),
        col("sy_dist").first(),
        col("st_teff").first(),
        col("st_rad").first(),
        col("st_mass").first(),
        col("st_lum").first(),
    ];

    if has_gaia_cols {
        agg_exprs.push(col("bp_rp").first());
        agg_exprs.push(col("g_mag").first());
    }

    let mut select_exprs = vec![
        col("hostname"),
        col("x_pc"),
        col("y_pc"),
        col("z_pc"),
        col("st_teff"),
        col("st_rad"),
        col("st_mass"),
        col("st_lum"),
    ];

    if has_gaia_cols {
        select_exprs.push(col("bp_rp"));
        select_exprs.push(col("g_mag"));
    }

    let cleaned_lazy = lazy_df
        .drop_nulls(Some(selector.clone()))
        .group_by([col("hostname")])
        .agg(agg_exprs)
        .with_columns([
            (col("ra") * lit(PI / 180.0)).alias("ra_rad"),
            (col("dec") * lit(PI / 180.0)).alias("dec_rad"),
        ])
        .with_columns([
            (col("sy_dist") * col("dec_rad").cos() * col("ra_rad").cos()).alias("x_pc"),
            (col("sy_dist") * col("dec_rad").cos() * col("ra_rad").sin()).alias("y_pc"),
            (col("sy_dist") * col("dec_rad").sin()).alias("z_pc"),
        ])
        .select(select_exprs);

    let mut final_df = cleaned_lazy.collect()?;

    let tmp_path = format!("{}.tmp", output_path);
    {
        let file = File::create(&tmp_path)?;
        ParquetWriter::new(file).finish(&mut final_df)?;
    }
    std::fs::rename(&tmp_path, output_path)?;

    println!(
        "Done. {} unique stars. Output: {}",
        final_df.height(),
        output_path
    );

    if print_sha256 {
        let h = sha256_file(output_path)?;
        println!("SHA256  {}", h);
    }

    Ok(())
}

fn clean_and_transform_gnn(input_path: &str, output_path: &str, print_sha256: bool) -> Result<()> {
    println!(
        "Reading and preprocessing (GNN velocity mode): {}",
        input_path
    );

    if !Path::new(input_path).exists() {
        return Err(anyhow!("Input file does not exist: {}", input_path));
    }

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(input_path.into()))?
        .finish()?;

    let has_vel_cols = df.column("pmra").is_ok()
        && df.column("pmdec").is_ok()
        && df.column("radial_velocity").is_ok()
        && df.column("parallax").is_ok();

    if !has_vel_cols {
        anyhow::bail!(
            "GNN velocity columns (pmra, pmdec, radial_velocity, parallax) not found. \
             Use 'lnaicli fetch-gnn' to download data with velocities."
        );
    }

    println!("Detected velocity columns (pmra, pmdec, radial_velocity, parallax)");

    let target_cols = vec![
        PlSmallStr::from_str("ra"),
        PlSmallStr::from_str("dec"),
        PlSmallStr::from_str("parallax"),
        PlSmallStr::from_str("pmra"),
        PlSmallStr::from_str("pmdec"),
        PlSmallStr::from_str("radial_velocity"),
        PlSmallStr::from_str("st_teff"),
        PlSmallStr::from_str("st_rad"),
        PlSmallStr::from_str("st_mass"),
        PlSmallStr::from_str("bp_rp"),
        PlSmallStr::from_str("g_mag"),
    ];

    let selector = Selector::ByName {
        names: Arc::from(target_cols),
        strict: true,
    };

    let agg_exprs = vec![
        col("ra").first(),
        col("dec").first(),
        col("parallax").first(),
        col("pmra").first(),
        col("pmdec").first(),
        col("radial_velocity").first(),
        col("st_teff").first(),
        col("st_rad").first(),
        col("st_mass").first(),
        col("st_lum").first(),
        col("bp_rp").first(),
        col("g_mag").first(),
    ];

    // k = 4.74047 km/s per (mas/yr at distance 1/parallax mas)
    // distance_pc = 1000 / parallax_mas
    // v_tangential = k * pm / parallax (km/s)
    //
    // ICRS Cartesian velocity:
    //   vx = vr*cos(d)*cos(a) - v_a*sin(a) - v_d*sin(d)*cos(a)
    //   vy = vr*cos(d)*sin(a) + v_a*cos(a) - v_d*sin(d)*sin(a)
    //   vz = vr*sin(d) + v_d*cos(d)
    // where a=ra_rad, d=dec_rad, vr=radial_velocity,
    //       v_a = k*pmra/parallax, v_d = k*pmdec/parallax

    let k_ast: f64 = 4.74047;

    let cleaned_lazy = df
        .lazy()
        .drop_nulls(Some(selector.clone()))
        .group_by([col("hostname")])
        .agg(agg_exprs)
        .with_columns([
            (col("ra") * lit(PI / 180.0)).alias("ra_rad"),
            (col("dec") * lit(PI / 180.0)).alias("dec_rad"),
        ])
        .with_columns([(lit(1000.0) / col("parallax")).alias("dist_pc")])
        .with_columns([
            (lit(k_ast) * col("pmra") / col("parallax")).alias("v_alpha"),
            (lit(k_ast) * col("pmdec") / col("parallax")).alias("v_delta"),
        ])
        .with_columns([
            (col("dist_pc") * col("dec_rad").cos() * col("ra_rad").cos()).alias("x_pc"),
            (col("dist_pc") * col("dec_rad").cos() * col("ra_rad").sin()).alias("y_pc"),
            (col("dist_pc") * col("dec_rad").sin()).alias("z_pc"),
        ])
        .with_columns([
            (col("radial_velocity") * col("dec_rad").cos() * col("ra_rad").cos()
                - col("v_alpha") * col("ra_rad").sin()
                - col("v_delta") * col("dec_rad").sin() * col("ra_rad").cos())
            .alias("vx"),
            (col("radial_velocity") * col("dec_rad").cos() * col("ra_rad").sin()
                + col("v_alpha") * col("ra_rad").cos()
                - col("v_delta") * col("dec_rad").sin() * col("ra_rad").sin())
            .alias("vy"),
            (col("radial_velocity") * col("dec_rad").sin() + col("v_delta") * col("dec_rad").cos())
                .alias("vz"),
        ])
        .select([
            col("hostname"),
            col("x_pc"),
            col("y_pc"),
            col("z_pc"),
            col("bp_rp"),
            col("g_mag"),
            col("st_teff"),
            col("st_rad"),
            col("st_mass"),
            col("st_lum"),
            col("vx"),
            col("vy"),
            col("vz"),
        ]);

    let mut final_df = cleaned_lazy.collect()?;

    let n = final_df.height();
    println!("Computed Cartesian velocities (vx, vy, vz) for {} stars", n);

    let tmp_path = format!("{}.tmp", output_path);
    {
        let file = File::create(&tmp_path)?;
        ParquetWriter::new(file).finish(&mut final_df)?;
    }
    std::fs::rename(&tmp_path, output_path)?;

    println!(
        "Done. {} unique stars with velocities. Output: {}",
        n, output_path
    );

    if print_sha256 {
        let h = sha256_file(output_path)?;
        println!("SHA256  {}", h);
    }

    Ok(())
}

fn combine_datasets(input_paths: &[String], output_path: &str) -> Result<()> {
    if input_paths.len() < 2 {
        return Err(anyhow!("At least 2 input files required for combining"));
    }

    println!("Combining {} datasets...", input_paths.len());

    let mut dfs = Vec::new();
    let mut total_rows: usize = 0;
    for path in input_paths {
        println!("  Reading: {}", path);
        if !Path::new(path).exists() {
            return Err(anyhow!("Input parquet missing: {}", path));
        }
        let file = File::open(path)?;
        let df = ParquetReader::new(file).finish()?;
        println!("    {} rows", df.height());
        total_rows += df.height();
        dfs.push(df);
    }
    println!("  Total rows to merge: {}", total_rows);

    let mut combined = dfs.remove(0);
    for df in &dfs {
        combined = combined.vstack(df)?;
    }

    let mut indices: Vec<usize> = (0..combined.height()).collect();
    indices.shuffle(&mut rand::rng());
    let idx_ca = UInt32Chunked::from_vec(
        PlSmallStr::from_str("idx"),
        indices.iter().map(|&i| i as u32).collect(),
    );
    combined = combined.take(&idx_ca)?;

    let tmp_path = format!("{}.tmp", output_path);
    {
        let file = File::create(&tmp_path)?;
        ParquetWriter::new(file).finish(&mut combined)?;
    }
    std::fs::rename(&tmp_path, output_path)?;

    println!(
        "Combined: {} rows. Output: {}",
        combined.height(),
        output_path
    );

    Ok(())
}
// ------------------------- Stage 4: lnai-data wiring -------------------------

struct CollectDataArgs {
    out_dir: PathBuf,
    ra_start_deg: f64,
    ra_end_deg: f64,
    mag_limit_g: f64,
    target_rows_per_shard: usize,
    concurrency: usize,
    only: Option<String>,
    retry_failed: bool,
    verify: bool,
}

fn run_collect_data(args: CollectDataArgs) -> Result<()> {
    use lnai_data::collector::{CollectConfig, CollectOptions, run_collection};
    use lnai_data::tap::TapFetcher;

    // Credentials come exclusively from the environment (never argv, never
    // logs); anonymous mode is the default and sufficient for public TAP.
    let gaia_user = env::var("GAIA_USERNAME").ok();
    let gaia_pass = env::var("GAIA_PASSWORD").ok();
    let auth_mode = if gaia_user.is_some() && gaia_pass.is_some() {
        "authenticated (env credentials)"
    } else {
        "anonymous"
    };

    let cfg = CollectConfig {
        out_dir: args.out_dir.clone(),
        ra_start_deg: args.ra_start_deg,
        ra_end_deg: args.ra_end_deg,
        mag_limit_g: args.mag_limit_g,
        max_ruwe: 1.4,
        target_rows_per_shard: args.target_rows_per_shard,
        concurrency: args.concurrency,
        retry_backoff_ms_base: 500,
        retry_max_attempts: 3,
    };
    let fetcher: Arc<dyn lnai_data::collector::ShardFetcher> =
        Arc::new(TapFetcher::anonymous().with_credentials(gaia_user, gaia_pass));

    println!(
        "Collecting RA [{:.3}, {:.3}) deg, mag_g < {:.1}, ruwe < 1.4 into {}\n  shards budget={} rows, workers={}, auth={}",
        args.ra_start_deg,
        args.ra_end_deg,
        args.mag_limit_g,
        args.out_dir.display(),
        args.target_rows_per_shard,
        args.concurrency,
        auth_mode
    );

    let report = run_collection(
        cfg,
        fetcher,
        CollectOptions {
            retry_failed: args.retry_failed,
            verify: args.verify,
            only: args.only,
            test_interrupt_after_n_shards: 0,
        },
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    println!(
        "Done: fetched={} subdivided={} verified-or-skipped failed={}",
        report.fetched, report.subdivided, report.failed
    );
    if report.failed > 0 {
        anyhow::bail!(
            "{} shard(s) remain Failed; re-run with --retry-failed (or fix network) before building",
            report.failed
        );
    }
    Ok(())
}

fn run_enrich_stellar(
    data: &Path,
    out_dir: &Path,
    ids_per_query: usize,
    concurrency: usize,
    max_ids: u64,
    join_only: bool,
    join_output: Option<String>,
) -> Result<()> {
    use lnai_data::stellar_params::{ApEnrichConfig, run_enrich_ap};

    // Credentials come exclusively from the environment (never argv, never
    // logs); anonymous mode is the default and sufficient for public TAP.
    let gaia_user = env::var("GAIA_USERNAME").ok();
    let gaia_pass = env::var("GAIA_PASSWORD").ok();

    let report = run_enrich_ap(&ApEnrichConfig {
        data_path: data.to_path_buf(),
        out_dir: out_dir.to_path_buf(),
        ids_per_query,
        concurrency,
        max_ids,
        join_only,
        join_output: join_output.map(PathBuf::from),
        retry_attempts: 3,
        sync_url: None,
        gaia_user,
        gaia_pass,
    })
    .map_err(|e| anyhow::anyhow!("enrich-stellar: {e}"))?;
    println!(
        "enrich-stellar done: {}/{} rows carry AP params -> {}",
        report.matched_rows, report.canonical_rows, report.output
    );
    Ok(())
}

fn run_build_dataset(
    out_dir: &Path,
    write_quality_report: bool,
    batch_shards: usize,
    keep_parts: bool,
) -> Result<()> {
    use lnai_data::assemble::{StreamingAssembleOptions, assemble_dataset_streaming};
    use lnai_data::clean::CleanPolicy;

    // RAM-guard defaults for the streaming engine (every value can be
    // overridden from the environment; existing env always wins). Fewer
    // engine threads + small sink morsels keep the merge phase at a couple
    // hundred MB regardless of dataset size.
    for (key, value) in [
        ("POLARS_MAX_THREADS", "4"),
        ("POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS", "16384"),
        ("POLARS_INFLIGHT_SINK_MORSEL_LIMIT", "4"),
    ] {
        if env::var_os(key).is_none() {
            // SAFETY: single-threaded startup path, before any engine spawn.
            unsafe { env::set_var(key, value) };
        }
    }

    let manifest_path = out_dir.join("manifest.json");
    let raw = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("missing {}", manifest_path.display()))?;
    let mut manifest: lnai_data::manifest::DatasetManifestV1 =
        serde_json::from_str(&raw).context("manifest.json is not a valid DatasetManifest")?;

    let verified = manifest
        .shards
        .iter()
        .filter(|s| s.status == lnai_data::manifest::ShardStatus::Verified)
        .count();
    println!(
        "Building dataset from {verified} verified shards in batches of {batch_shards} (streaming merge)..."
    );

    let (report, qr) = assemble_dataset_streaming(
        out_dir,
        &mut manifest,
        &CleanPolicy::default(),
        StreamingAssembleOptions {
            batch_shards,
            keep_parts,
        },
    )
    .map_err(|e| anyhow::anyhow!(e))?;
    persist_manifest_atomic(&manifest, &manifest_path)?;

    println!(
        "Assembled canonical parquet: {} rows at {}\n  train/validation/test/holdout: {}/{}/{}/{}",
        report.rows_written,
        report.canonical_path.display(),
        report.train_rows,
        report.validation_rows,
        report.test_rows,
        report.holdout_rows
    );
    for (view, path) in &report.view_paths {
        println!("  view {view:?}: {}", path.display());
    }

    if write_quality_report {
        let json = serde_json::to_string_pretty(&qr).context("serialize quality report")?;
        let qpath = out_dir.join("quality_report.json");
        std::fs::write(&qpath, json).with_context(|| format!("write {}", qpath.display()))?;
        println!("Quality report written: {}", qpath.display());
    } else {
        println!("Quality summary: {qr:?}");
    }

    println!(
        "Manifest finalized: total_rows={}, checksum_prefix={}...",
        manifest.total_rows,
        &manifest.checksum[..16.min(manifest.checksum.len())]
    );
    Ok(())
}

/// Atomic manifest persistence (temp file + rename) so an interrupted run can
/// never corrupt the JSON state between collect/build invocations.
fn persist_manifest_atomic(
    manifest: &lnai_data::manifest::DatasetManifestV1,
    path: &Path,
) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest).context("serialize manifest")?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(tmp, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4A + object storage (S3-compatible, MinIO-first)
// ---------------------------------------------------------------------------

/// Minimal .env loader so `lnaicli` picks up SINK/storage variables even when
/// not sourced by orchestrate.sh. Existing process env always wins; values are
/// never printed.
fn load_dotenv_if_present() {
    let Ok(content) = std::fs::read_to_string(".env") else {
        return;
    };
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim().trim_matches('"').trim_matches('\'');
        if key.is_empty() || env::var_os(key).is_some() {
            continue;
        }
        // SAFETY: single-threaded startup path, before any thread spawn.
        unsafe { env::set_var(key, value) };
    }
}

const NASA_KEY_ENV: &str = "NASA_API_KEY";
const MAST_TOKEN_ENV: &str = "MAST_API_TOKEN";
const GAIA_USER_ENV: &str = "GAIA_USERNAME";
const GAIA_PASS_ENV: &str = "GAIA_PASSWORD";

fn run_auth_status() {
    use lnai_data::sources::{SourceAdapter as _, SourceAuth};

    println!("Source auth matrix (secret values are NEVER printed):");
    for (id, p) in [
        (
            "exoplanet",
            lnai_data::sources::nasa_exoplanet::NasaExoplanetAdapter.provenance(),
        ),
        (
            "mast",
            lnai_data::sources::mast::MastTicAdapter.provenance(),
        ),
        ("irsa", lnai_data::sources::irsa::IrsaAdapter.provenance()),
        (
            "jpl",
            lnai_data::sources::jpl_horizons::JplHorizonsAdapter.provenance(),
        ),
    ] {
        let mode = match &p.auth {
            SourceAuth::Anonymous => "anonymous".to_string(),
            SourceAuth::EnvKeys { requires } => format!("env:{}", requires.join(",")),
        };
        println!(
            "  [{id}] {catalog:<40} mode={mode} backbone={}",
            p.enters_stellar_backbone,
            catalog = p.catalog_name
        );
    }

    // Optional credentials known to the runtime (presence only).
    let optional = [
        (
            NASA_KEY_ENV,
            "api.nasa.gov key-based endpoints (NOT required for Exoplanet TAP)",
        ),
        (MAST_TOKEN_ENV, "protected/EAP MAST products only"),
        (GAIA_USER_ENV, "Gaia user space / long async jobs"),
        (GAIA_PASS_ENV, "Gaia password half"),
    ];
    for (name, purpose) in optional {
        let present = env::var(name)
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false);
        println!(
            "  env {name}={present} — {purpose}",
            present = if present { "<set>" } else { "<unset>" }
        );
    }

    match lnai_data::storage::sink_status() {
        lnai_data::storage::SinkStatus::Unconfigured => {
            println!(
                "  storage sink: unconfigured (set S3_ENDPOINT/S3_BUCKET; MinIO via install/data-minio.compose.yml)"
            );
        }
        lnai_data::storage::SinkStatus::AnonymousRead { endpoint, bucket } => {
            println!("  storage sink: anonymous-read endpoint={endpoint} bucket={bucket}");
        }
        lnai_data::storage::SinkStatus::Authenticated {
            endpoint,
            bucket,
            key_head,
        } => {
            println!(
                "  storage sink: authenticated endpoint={endpoint} bucket={bucket} access_key={key_head}"
            );
        }
    }
}

/// Пункт 6: versioned enriched fixture artifacts from recorded dumps.
fn run_enrich_fixtures(out_dir: &str) -> Result<()> {
    use lnai_data::sources::SourceAdapter as _;
    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../lnai-data/tests/fixtures/nasa_exoplanet_pscomppars_sample.csv"
    ));
    std::fs::create_dir_all(out_dir)?;
    let recs = lnai_data::sources::nasa_exoplanet::parse_pscomppars_csv(FIXTURE)
        .map_err(anyhow::Error::msg)?;

    // 100–1000 objects band enforced by the plan.
    anyhow::ensure!(
        (100..=1000).contains(&recs.len()),
        "fixture row count {} violates the plan band",
        recs.len()
    );

    let adapter_id = lnai_data::sources::nasa_exoplanet::ADAPTER_ID;
    let query_hash = lnai_data::integrity::sha256_hex(
        lnai_data::sources::nasa_exoplanet::adql_query(500).as_bytes(),
    );
    let mut manifest = lnai_data::source_manifest::SourceManifestV1::new();
    manifest.register(
        lnai_data::source_manifest::SourceManifestEntry::from_payload(
            adapter_id,
            lnai_data::sources::nasa_exoplanet::PS_COMPPARS_TAP_SYNC_URL,
            &query_hash,
            // Retrieval session timestamp of THIS fixture recording (2026-08-27).
            1_787_637_000_000,
            FIXTURE,
            recs.len(),
        ),
    );
    manifest
        .save(&Path::new(out_dir).join("source_manifest.json"))
        .map_err(anyhow::Error::msg)?;

    // Enrichment-only rows stay out of the stellar backbone by contract.
    let enriched: Vec<serde_json::Value> = recs
        .iter()
        .map(|r| {
            serde_json::json!({
                "planet_name": r.planet_name,
                "hostname": r.hostname,
                "ra_deg": r.ra_deg,
                "dec_deg": r.dec_deg,
                "teff_k": r.teff_k,
                "radius_rsun": r.radius_rsun,
                "mass_msun": r.mass_msun,
                "enters_stellar_backbone": false,
            })
        })
        .collect();
    let rows_path = Path::new(out_dir).join("nasa_enriched_rows.json");
    std::fs::write(&rows_path, serde_json::to_vec_pretty(&enriched).unwrap())?;

    let mut provenance = lnai_data::provenance::DatasetProvenanceV1::new(out_dir);
    provenance.register(lnai_data::sources::nasa_exoplanet::NasaExoplanetAdapter.provenance());
    provenance.impact_report_reference =
        Some("enrichment_report.json (produced by `lnaicli enrich-report`)".into());
    provenance
        .save(Path::new(out_dir))
        .map_err(anyhow::Error::msg)?;

    // Raw dump copy keeps byte-exact reproducibility alongside hashes.
    std::fs::write(Path::new(out_dir).join("raw_pscomppars.csv"), FIXTURE)?;

    println!(
        "Enriched fixtures written to {out_dir}: {} objects; manifest entries={}, sha256={}",
        recs.len(),
        manifest.entries.len(),
        &manifest.entries[0].payload_sha256[..16]
    );
    Ok(())
}

fn collect_gaia_sample_rows_from_parquet(
    path: &Path,
    max_rows: usize,
) -> Result<Vec<lnai_data::enrich::GaiaSampleRow>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;

    let col_req_f64 = |name: &str| -> Result<Vec<f64>> {
        let series = df
            .column(name)
            .ok()
            .with_context(|| format!("column {name} required in {}", path.display()))?;
        Ok(series.f64()?.into_no_null_iter().collect())
    };
    let col_opt_f64 = |name: &str| -> Vec<Option<f64>> {
        df.column(name)
            .ok()
            .and_then(|series| series.f64().ok().map(|ca| ca.iter().collect()))
            .unwrap_or_default()
    };
    let col_ids = || -> Result<Vec<String>> {
        let series = df
            .column("source_id")
            .map_err(|_| anyhow!("column source_id required"))?;
        Ok(series
            .str()?
            .iter()
            .flatten()
            .map(String::from)
            .collect())
    };

    let ra = col_req_f64("ra_deg")?;
    let dec = col_req_f64("dec_deg")?;
    let ids = col_ids()?;
    let mag_g: Vec<Option<f64>> = col_opt_f64("mag_g");
    let bp: Vec<Option<f64>> = col_opt_f64("mag_bp");
    let rp: Vec<Option<f64>> = col_opt_f64("mag_rp");
    let plx: Vec<Option<f64>> = col_opt_f64("parallax_mas");

    let n = ra.len().min(dec.len()).min(ids.len()).min(max_rows);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        // Incomplete rows are skipped, never imputed (honest-null policy).
        let (Some(g), Some(bp_i), Some(rp_i), Some(parallax)) = (mag_g[i], bp[i], rp[i], plx[i])
        else {
            continue;
        };
        if !g.is_finite() || parallax <= 0.0 || !(bp_i - rp_i).is_finite() {
            continue;
        }
        out.push(lnai_data::enrich::GaiaSampleRow {
            source_id: ids[i].clone(),
            ra_deg: ra[i],
            dec_deg: dec[i],
            epoch_year: 2016.0,
            pm_ra_mas_yr: None,
            pm_dec_mas_yr: None,
            mag_g: g,
            mag_bp: bp_i,
            mag_rp: rp_i,
            parallax_mas: parallax,
        });
    }
    Ok(out)
}

/// Пункт 7: honest before/after report; refuses to claim improvement without data.
fn run_enrich_report(out_dir: &Path, gaia_parquet: Option<&str>) -> Result<()> {
    let rows_raw = std::fs::read(out_dir.join("nasa_enriched_rows.json")).with_context(|| {
        format!(
            "run `lnaicli enrich-fixtures --out-dir {}` first",
            out_dir.display()
        )
    })?;
    let raw: Vec<serde_json::Value> =
        serde_json::from_slice(&rows_raw).context("nasa_enriched_rows.json is valid JSON")?;
    let grab = |v: &serde_json::Value, k: &str| -> Result<f64> {
        v.get(k)
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| anyhow!("row missing numeric field {k}"))
    };
    let nasa_rows = raw
        .into_iter()
        .map(|v| {
            Ok(lnai_data::sources::nasa_exoplanet::NasaExoplanetRecord {
                planet_name: v
                    .get("planet_name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                hostname: v
                    .get("hostname")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                ra_deg: grab(&v, "ra_deg")?,
                dec_deg: grab(&v, "dec_deg")?,
                parallax_mas: None,
                distance_pc: None,
                teff_k: v.get("teff_k").and_then(serde_json::Value::as_f64),
                radius_rsun: v.get("radius_rsun").and_then(serde_json::Value::as_f64),
                mass_msun: v.get("mass_msun").and_then(serde_json::Value::as_f64),
                luminosity_lsun: None,
                discovery_year: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let Some(parquet_path) = gaia_parquet else {
        let note = serde_json::json!({
            "version": "1.0.0",
            "status": "insufficient_data",
            "reason": "no Gaia backbone provided; pass --gaia-parquet pointing at canonical stars.parquet",
            "nasa_rows": nasa_rows.len(),
            "verdict": "not_evaluated"
        });
        let path = out_dir.join("enrichment_report.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&note).unwrap())?;
        println!("Honest no-data report written: {}", path.display());
        return Ok(());
    };

    const TOLERANCE_ARCSEC: f64 = 1.0;
    const MAX_GAIA_ROWS: usize = 400_000;
    let gaia_rows = collect_gaia_sample_rows_from_parquet(Path::new(parquet_path), MAX_GAIA_ROWS)?;
    let (samples, unmatched) = lnai_data::enrich::enrichment_samples_from_records(
        &nasa_rows,
        &gaia_rows,
        TOLERANCE_ARCSEC,
    );

    let report_json = match lnai_data::enrich::evaluate(&samples) {
        Some(report) => {
            let improved = report.verdict == lnai_data::enrich::EnrichmentVerdict::Improved;
            println!(
                "Enrichment evaluation on {}/{} matched samples:\n  before {}\n  after  {}\n  delta {:.1}% verdict={:?}",
                samples.len(),
                nasa_rows.len(),
                report.before,
                report.after,
                report.mae_delta_fraction * 100.0,
                report.verdict
            );
            let _ = improved;
            serde_json::to_string_pretty(&report)?
        }
        None => {
            let note = serde_json::json!({
                "version": "1.0.0",
                "status": "insufficient_matched_samples",
                "matched_complete_samples": samples.len(),
                "unmatched_or_incomplete": unmatched,
                "verdict": "not_evaluated"
            });
            println!(
                "Only {}/{} matched complete samples (<8); wrote not_evaluated report honestly.",
                samples.len(),
                nasa_rows.len()
            );
            serde_json::to_string_pretty(&note)?
        }
    };
    let path = out_dir.join("enrichment_report.json");
    std::fs::write(&path, report_json)?;
    println!("Report written: {}", path.display());
    Ok(())
}

fn run_spike_mast(ra: f64, dec: f64, radius: f64) -> Result<()> {
    let params = lnai_data::sources::mast::ConeParams {
        ra_deg: ra,
        dec_deg: dec,
        radius_deg: radius,
        page_size: 50,
    };
    let (raw, records) =
        lnai_data::sources::mast::fetch_cone(&params).map_err(anyhow::Error::msg)?;
    let with_pm = records.iter().filter(|r| r.pm_ra_mas_yr.is_some()).count();
    let with_gaia = records
        .iter()
        .filter(|r| r.gaia_source_id.is_some())
        .count();
    println!(
        "MAST TIC cone ({ra},{dec},r={radius}): {} metadata rows; pm coverage {:.0}%, cross-id GAIA {:.0}% (query hash {})",
        records.len(),
        100.0 * with_pm as f64 / records.len().max(1) as f64,
        100.0 * with_gaia as f64 / records.len().max(1) as f64,
        &params.query_hash()[..16]
    );
    // Provenance of a spike = request identity + response digest (no secrets).
    println!(
        "response sha256: {}",
        &lnai_data::integrity::sha256_hex(raw.as_bytes())[..16]
    );
    Ok(())
}

fn run_spike_irsa(ra_min: f64, ra_max: f64, dec_min: f64, dec_max: f64, top: usize) -> Result<()> {
    let query = lnai_data::sources::irsa::adql_query(ra_min, ra_max, dec_min, dec_max, top);
    let csv = lnai_data::sources::irsa::fetch_box_csv(&query).map_err(anyhow::Error::msg)?;
    let rows = lnai_data::sources::irsa::parse_two_mass_csv(&csv).map_err(anyhow::Error::msg)?;
    let stats = lnai_data::sources::irsa::coverage_stats(&rows);
    println!(
        "IRSA 2MASS fp_psc box RA[{ra_min},{ra_max}) Dec[{dec_min},{dec_max}): {} rows; full JHK {:.0}%; null-rates j={:.2} h={:.2} k={:.2} pm={:.2}",
        stats.row_count,
        100.0 * stats.full_photometry_fraction,
        stats.null_rate.j_m,
        stats.null_rate.h_m,
        stats.null_rate.k_m,
        stats.null_rate.pm
    );
    Ok(())
}

fn run_jpl_scenes(
    body: &str,
    start_time: &str,
    stop_time: &str,
    step_size: &str,
    center: &str,
) -> Result<()> {
    let req = lnai_data::sources::jpl_horizons::SceneRequest {
        body_code: body.to_string(),
        center: center.to_string(),
        start_time: start_time.to_string(),
        stop_time: stop_time.to_string(),
        step_size: step_size.to_string(),
    };
    let (_raw, rows) =
        lnai_data::sources::jpl_horizons::fetch_scene(&req).map_err(anyhow::Error::msg)?;
    println!(
        "JPL Horizons scene (body={body}, center={center}): {} ephemeris rows, query hash {}",
        rows.len(),
        &req.query_hash()[..16]
    );
    for r in rows.iter().take(3) {
        println!(
            "  {} RA={:.5}deg DEC={:.5} Delta={:.4}AU",
            r.time_utc, r.ra_deg, r.dec_deg, r.delta_au
        );
    }
    println!("scene rows are scene-provider output only: enters_stellar_backbone=false");
    Ok(())
}

fn require_writable_sink() -> Result<()> {
    match lnai_data::storage::sink_status() {
        lnai_data::storage::SinkStatus::Authenticated {
            endpoint, bucket, ..
        } => {
            println!("sink: authenticated → {endpoint}/{bucket}");
            Ok(())
        }
        lnai_data::storage::SinkStatus::AnonymousRead { endpoint, bucket } => Err(anyhow!(
            "only anonymous read available at {endpoint}/{bucket}; set S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY to upload"
        )),
        lnai_data::storage::SinkStatus::Unconfigured => Err(anyhow!(
            "no S3 sink configured: set S3_ENDPOINT/S3_BUCKET (+ S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY) or SPACES_* aliases"
        )),
    }
}

fn run_storage_upload(dataset_dir: &Path, prefix: &str) -> Result<()> {
    require_writable_sink()?;
    let uploaded =
        lnai_data::storage::upload_dataset_dir(dataset_dir, prefix).map_err(|e| anyhow!("{e}"))?;
    let bytes: u64 = uploaded.iter().map(|(_, s)| s).sum();
    println!(
        "Uploaded {} objects ({bytes} bytes) under `{prefix}`",
        uploaded.len()
    );
    Ok(())
}

fn run_storage_pull(dest_dir: &Path, prefix: &str) -> Result<()> {
    let pulled =
        lnai_data::storage::download_dataset_dir(dest_dir, prefix).map_err(|e| anyhow!("{e}"))?;
    let bytes: u64 = pulled.iter().map(|(_, s)| s).sum();
    println!(
        "Pulled {} objects ({bytes} bytes) into {} (unchanged skipped)",
        pulled.len(),
        dest_dir.display()
    );
    Ok(())
}

fn run_storage_list(prefix: &str) -> Result<()> {
    match lnai_data::storage::sink_status() {
        lnai_data::storage::SinkStatus::Unconfigured => {
            anyhow::bail!("no S3 sink configured; see lnaicli auth-status")
        }
        status => {
            let cfg = lnai_data::storage::s3_config_from_env().map_err(|e| anyhow!("{e}"))?;
            let client = lnai_data::s3::S3Client::new(cfg);
            let objects = client.list_objects(prefix).map_err(|e| anyhow!("{e}"))?;
            let bytes: u64 = objects.iter().map(|(_, s)| s).sum();
            println!(
                "{status:?}\n  objects under `{prefix}`: {} (total {bytes} bytes)",
                objects.len()
            );
            for (key, size) in objects.iter().take(20) {
                println!("  {key} ({size} B)");
            }
            Ok(())
        }
    }
}
