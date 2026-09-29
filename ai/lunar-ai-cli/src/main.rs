use anyhow::Result;
use clap::Parser;
use std::path::Path;

mod cli;
mod commands;
mod util;

use cli::{Cli, Commands};
use commands::*;
use util::sha256_file;

fn main() -> Result<()> {
    load_dotenv_if_present();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch(args) => {
            fetch_stellar_data(&FetchOptions {
                output_path: &args.output,
                username: args.username.as_deref(),
                password: args.password.as_deref(),
                max_rows: args.max_rows,
                ra_min: args.ra_min,
                ra_max: args.ra_max,
                max_ruwe: args.max_ruwe,
                poll_initial_secs: args.poll_initial_secs,
                poll_max_secs: args.poll_max_secs,
                include_velocities: false,
            })?;
        }
        Commands::FetchGnn(args) => {
            fetch_stellar_data(&FetchOptions {
                output_path: &args.output,
                username: args.username.as_deref(),
                password: args.password.as_deref(),
                max_rows: args.max_rows,
                ra_min: args.ra_min,
                ra_max: args.ra_max,
                max_ruwe: args.max_ruwe,
                poll_initial_secs: args.poll_initial_secs,
                poll_max_secs: args.poll_max_secs,
                include_velocities: true,
            })?;
        }
        Commands::Clean(args) => {
            clean_and_transform(&args.input, &args.output, args.print_sha256)?;
        }
        Commands::CleanGnn(args) => {
            clean_and_transform_gnn(&args.input, &args.output, args.print_sha256)?;
        }
        Commands::Sha256(args) => {
            let h = sha256_file(&args.input)?;
            println!("{}  {}", h, args.input);
        }
        Commands::Combine(args) => {
            combine_datasets(&args.inputs, &args.output)?;
        }
        Commands::CollectData(args) => {
            run_collect_data(args)?;
        }
        Commands::BuildDataset(args) => {
            run_build_dataset(
                Path::new(&args.out_dir),
                args.quality_report,
                args.batch_shards,
                args.keep_parts,
            )?;
        }
        Commands::AuthStatus => {
            run_auth_status();
        }
        Commands::EnrichStellar(args) => {
            run_enrich_stellar(
                Path::new(&args.data),
                Path::new(&args.out_dir),
                args.ids_per_query,
                args.concurrency,
                args.max_ids,
                args.join_only,
                args.join_output,
            )?;
        }
        Commands::EnrichFixtures(args) => {
            run_enrich_fixtures(&args.out_dir)?;
        }
        Commands::EnrichReport(args) => {
            run_enrich_report(Path::new(&args.out_dir), args.gaia_parquet.as_deref())?;
        }
        Commands::SpikeMast(args) => {
            run_spike_mast(args.ra, args.dec, args.radius)?;
        }
        Commands::SpikeIrsa(args) => {
            run_spike_irsa(
                args.ra_min,
                args.ra_max,
                args.dec_min,
                args.dec_max,
                args.top,
            )?;
        }
        Commands::JplScenes(args) => {
            run_jpl_scenes(
                &args.body,
                &args.start_time,
                &args.stop_time,
                &args.step_size,
                &args.center,
            )?;
        }
        Commands::StorageUpload(args) => {
            run_storage_upload(Path::new(&args.dataset_dir), &args.prefix)?;
        }
        Commands::StoragePull(args) => {
            run_storage_pull(Path::new(&args.dest_dir), &args.prefix)?;
        }
        Commands::StorageList(args) => {
            run_storage_list(&args.prefix)?;
        }
        Commands::Train(args) => {
            run_train(&TrainOptions {
                model: args.model,
                data: args.data.as_deref(),
                resume: args.resume.as_deref(),
                norm: args.norm.as_deref(),
                output_dir: &args.output_dir,
                epochs: args.epochs,
                batch_size: args.batch_size,
                lr: args.lr,
                physics_weight: args.physics_weight,
                val_frac: args.val_frac,
                gpu_index: args.gpu_index,
                holdout: args.holdout.as_deref(),
                lnai_bin: args.lnai_bin.as_deref(),
                model_file: &args.model_file,
                norm_file: &args.norm_file,
                knn_k: args.knn_k,
                hidden_dim: args.hidden_dim,
                max_group_size: args.max_group_size,
                radius_pc: args.radius_pc,
                texture_size: args.texture_size,
                max_stars: args.max_stars,
                seed: args.seed,
                dataset_manifest_hash: &args.dataset_manifest_hash,
                max_rows: args.max_rows,
                tiles: args.tiles,
                agent_every: args.agent_every,
                agent_model: &args.agent_model,
                agent_fallback_model: &args.agent_fallback_model,
                agent_timeout_secs: args.agent_timeout_secs,
                agent_log_lines: args.agent_log_lines,
                agent_dry_run: args.agent_dry_run,
            })?;
        }
        Commands::AgentFix(args) => {
            run_agent_fix(&args.dir, &args.model, args.message, args.auto_approve)?;
        }
        Commands::Evaluate(args) => {
            run_evaluate_cmd(
                args.model,
                &args.output_dir,
                args.data.as_deref(),
                args.holdout.as_deref(),
                args.batch_size,
                args.seed,
                &args.model_file,
                &args.norm_file,
            )?;
        }
        Commands::Benchmark(args) => {
            run_benchmark_cmd(
                args.model,
                &args.output_dir,
                args.iters,
                args.warmup,
                args.batch_size,
                args.seed,
            )?;
        }
    }

    Ok(())
}
