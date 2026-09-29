use anyhow::{Result, anyhow};
use std::path::Path;

pub fn require_writable_sink() -> Result<()> {
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

pub fn run_storage_upload(dataset_dir: &Path, prefix: &str) -> Result<()> {
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

pub fn run_storage_pull(dest_dir: &Path, prefix: &str) -> Result<()> {
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

pub fn run_storage_list(prefix: &str) -> Result<()> {
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
