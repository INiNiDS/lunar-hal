use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};

pub fn run_enrich_stellar(
    data: &Path,
    out_dir: &Path,
    ids_per_query: usize,
    concurrency: usize,
    max_ids: u64,
    join_only: bool,
    join_output: Option<String>,
) -> Result<()> {
    use lnai_data::stellar_params::{ApEnrichConfig, run_enrich_ap};

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

pub fn run_build_dataset(
    out_dir: &Path,
    write_quality_report: bool,
    batch_shards: usize,
    keep_parts: bool,
) -> Result<()> {
    use lnai_data::assemble::{StreamingAssembleOptions, assemble_dataset_streaming};
    use lnai_data::clean::CleanPolicy;

    for (key, value) in [
        ("POLARS_MAX_THREADS", "4"),
        ("POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS", "16384"),
        ("POLARS_INFLIGHT_SINK_MORSEL_LIMIT", "4"),
    ] {
        if env::var_os(key).is_none() {
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
