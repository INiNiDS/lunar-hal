use anyhow::Result;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use crate::cli::args_data::CollectDataArgs;

pub fn run_collect_data(args: CollectDataArgs) -> Result<()> {
    use lnai_data::collector::{CollectConfig, CollectOptions, run_collection};
    use lnai_data::tap::TapFetcher;

    let gaia_user = env::var("GAIA_USERNAME").ok();
    let gaia_pass = env::var("GAIA_PASSWORD").ok();
    let auth_mode = if gaia_user.is_some() && gaia_pass.is_some() {
        "authenticated (env credentials)"
    } else {
        "anonymous"
    };

    let cfg = CollectConfig {
        out_dir: PathBuf::from(&args.out_dir),
        ra_start_deg: args.ra_min,
        ra_end_deg: args.ra_max,
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
        args.ra_min,
        args.ra_max,
        args.mag_limit_g,
        args.out_dir,
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
