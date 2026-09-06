//! Dataset storage sinks (new, outside the original 4A scope):
//! every artifact the pipeline produces (raw shards, canonical parquet, views,
//! manifest, provenance) can live on any S3-compatible object store.
//!
//! * Local runs and CI default to MinIO via `install/data-minio.compose.yml`
//!   (anonymous read policy + dedicated bucket).
//! * Cloud deployments point the same variables at DigitalOcean Spaces / AWS /
//!   R2 — nothing else changes.
//!
//! Environment contract (non-secret values may be committed to `.env.example`):
//!   `S3_ENDPOINT` | legacy alias `SPACES_ENDPOINT` — scheme://host[:port]
//!   `S3_BUCKET`   | `SPACES_BUCKET`
//!   `S3_REGION`   | `SPACES_REGION` (default us-east-1)
//!   `S3_ACCESS_KEY_ID`     | `SPACES_KEY`      — REQUIRED for writes
//!   `S3_SECRET_ACCESS_KEY` | `SPACES_SECRET`  — REQUIRED for writes
//!   `S3_PATH_STYLE` (default `true`, MinIO-style addressing)
//! Secrets stay in `.env`; they are read by backend/CLI code only.

use crate::auth::{SecretBox, redact, resolve_env_secret};
use crate::s3::{DEFAULT_REGION, S3Client, S3Config};

pub const MANIFEST_OBJECT_SUFFIX: &str = "manifest.json";

#[derive(Debug)]
pub enum StorageError {
    NotConfigured(String),
    Auth(String),
    S3(crate::s3::S3Error),
    Io(std::io::Error),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConfigured(e) => write!(f, "storage not configured: {e}"),
            Self::S3(e) => write!(f, "{e}"),
            Self::Auth(e) => write!(f, "s3 auth config error: {e}"),
            Self::Io(e) => write!(f, "storage io error: {e}"),
        }
    }
}

impl From<crate::s3::S3Error> for StorageError {
    fn from(e: crate::s3::S3Error) -> Self {
        Self::S3(e)
    }
}

// Serializes tests that mutate process-global environment variables.
#[cfg(test)]
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Builds a sink description from environment; returns detailed errors that
/// never echo secret values.
pub fn s3_config_from_env() -> Result<S3Config> {
    let get =
        |names: &[&str]| -> Option<String> { names.iter().find_map(|n| resolve_env_secret(n)) };
    let endpoint = get(&["S3_ENDPOINT", "SPACES_ENDPOINT"]).ok_or_else(|| {
        StorageError::NotConfigured("set S3_ENDPOINT (e.g. http://127.0.0.1:9000)".into())
    })?;
    let bucket = get(&["S3_BUCKET", "SPACES_BUCKET"])
        .ok_or_else(|| StorageError::NotConfigured("set S3_BUCKET".into()))?;
    let region = get(&["S3_REGION", "SPACES_REGION"]).unwrap_or_else(|| DEFAULT_REGION.to_string());
    // Empty-but-set vars already filtered out by resolve_env_secret.
    let path_style = std::env::var("S3_PATH_STYLE")
        .ok()
        .map(|v| !matches!(v.trim(), "0" | "false" | "no"))
        .unwrap_or(true);

    let access_key_id = get(&["S3_ACCESS_KEY_ID", "SPACES_KEY"]);
    let secret_access_key = get(&["S3_SECRET_ACCESS_KEY", "SPACES_SECRET"]);
    if (access_key_id.is_some()) != (secret_access_key.is_some()) {
        return Err(StorageError::Auth(
            "both S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set together".to_string(),
        ));
    }

    Ok(S3Config {
        endpoint,
        region,
        bucket,
        access_key_id: access_key_id.map(SecretBox::new),
        secret_access_key: secret_access_key.map(SecretBox::new),
        path_style,
    })
}

impl From<std::io::Error> for StorageError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

type Result<T> = std::result::Result<T, StorageError>;

/// What kind of sink is active right now — used by diagnostics and Testbench
/// status endpoints. Contains NO credential material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkStatus {
    AnonymousRead {
        endpoint: String,
        bucket: String,
    },
    Authenticated {
        endpoint: String,
        bucket: String,
        /// Redacted head of the access key id (4 chars) for UI diagnostics.
        key_head: String,
    },
    Unconfigured,
}

pub fn sink_status() -> SinkStatus {
    match s3_config_from_env() {
        Err(_) => SinkStatus::Unconfigured,
        Ok(cfg) if cfg.authenticated() => SinkStatus::Authenticated {
            endpoint: cfg.endpoint.clone(),
            bucket: cfg.bucket.clone(),
            key_head: cfg
                .access_key_id
                .as_ref()
                .map(|k| redact(k.expose()))
                .unwrap_or_default(),
        },
        Ok(cfg) => SinkStatus::AnonymousRead {
            endpoint: cfg.endpoint.clone(),
            bucket: cfg.bucket.clone(),
        },
    }
}

/// Pushes every file of an assembled dataset directory into the bucket under
/// `<prefix>`, preserving relative layout:
/// `{prefix}/raw/ra-000.parquet`, `{prefix}/views/pinn.parquet`,
/// `{prefix}/manifest.json`, ... Returns uploaded keys with byte sizes.
pub fn upload_dataset_dir(dir: &std::path::Path, prefix: &str) -> Result<Vec<(String, u64)>> {
    let cfg = s3_config_from_env()?;
    if !cfg.authenticated() {
        return Err(StorageError::Auth(
            "upload requires S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY (read-only anonymous mode)"
                .to_string(),
        ));
    }
    let client = S3Client::new(cfg);
    let mut uploaded = Vec::new();

    for entry in walk_files(dir)? {
        let rel = entry.strip_prefix(dir).unwrap_or(entry.as_path());
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let key = format!("{}/{rel_str}", prefix.trim_end_matches('/'));
        let body = std::fs::read(&entry)?;
        let ct = content_type_of(&entry);
        client.put_object(&key, &body, ct.as_deref())?;
        uploaded.push((key, body.len() as u64));
    }
    Ok(uploaded)
}

/// Pulls remote dataset artifacts into a local directory (resume-friendly:
/// files whose local size matches the remote one are skipped).
pub fn download_dataset_dir(dir: &std::path::Path, prefix: &str) -> Result<Vec<(String, u64)>> {
    let cfg = s3_config_from_env()?;
    let client = S3Client::new(cfg);
    let clean_prefix = prefix.trim_end_matches('/');
    let objects = client.list_objects(clean_prefix)?;
    let mut pulled = Vec::new();
    for (key, size) in &objects {
        let rel = key
            .strip_prefix(prefix)
            .unwrap_or(key)
            .trim_start_matches('/');
        let dest = dir.join(rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if dest.exists() && std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0) == *size {
            continue;
        }
        let data = client.get_object(key)?;
        std::fs::write(&dest, &data)?;
        pulled.push((key.clone(), data.len() as u64));
    }
    Ok(pulled)
}

// -- internals --------------------------------------------------------------

fn walk_files(root: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        if p.is_dir() {
            for child in std::fs::read_dir(&p)?.collect::<std::result::Result<Vec<_>, _>>()? {
                stack.push(child.path());
            }
        } else {
            out.push(p);
        }
    }
    // Final ordering by full path keeps deterministic lexicographic layout.
    out.sort();
    Ok(out)
}

fn content_type_of(path: &std::path::Path) -> Option<String> {
    match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
        "json" => Some("application/json".to_string()),
        "parquet" => Some("application/vnd.apache.parquet".to_string()),
        "csv" => Some("text/csv".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unconfigured_env_yields_secret_free_error() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Clear every storage-related variable for this test.
        for name in [
            "S3_ENDPOINT",
            "S3_BUCKET",
            "S3_REGION",
            "S3_ACCESS_KEY_ID",
            "S3_SECRET_ACCESS_KEY",
            "SPACES_ENDPOINT",
            "SPACES_KEY",
            "SPACES_SECRET",
        ] {
            unsafe { std::env::remove_var(name) };
        }
        match s3_config_from_env() {
            Err(StorageError::NotConfigured(msg)) => {
                assert!(msg.contains("S3_ENDPOINT"), "{msg}");
                assert!(!msg.to_lowercase().contains("secret"));
            }
            other => panic!("expected NotConfigured, got {other:?}"),
        }
        match sink_status() {
            SinkStatus::Unconfigured => {}
            other => panic!("expected Unconfigured, got {other:?}"),
        }
    }

    #[test]
    fn config_resolution_prefers_s3_names_and_falls_back_to_spaces() {
        let _guard = ENV_LOCK.lock().unwrap();
        let probe_ep = "http://lnai-test-endpoint.invalid:9100";
        let probe_bucket = "lnai-test-bucket";
        unsafe { std::env::set_var("S3_ENDPOINT", probe_ep) };
        unsafe { std::env::set_var("S3_BUCKET", probe_bucket) };
        let cfg = s3_config_from_env().expect("configured now");
        assert_eq!(cfg.endpoint, probe_ep);
        assert_eq!(cfg.bucket, probe_bucket);
        assert_eq!(cfg.region, DEFAULT_REGION);
        assert!(!cfg.authenticated());

        unsafe { std::env::remove_var("S3_ENDPOINT") };
        unsafe { std::env::remove_var("S3_BUCKET") };
    }

    #[test]
    fn spaces_alias_is_used_when_s3_names_absent() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("SPACES_ENDPOINT", "http://legacy-alias.test:9200") };
        unsafe { std::env::set_var("SPACES_BUCKET", "legacy-bucket") };
        let cfg = s3_config_from_env().expect("aliased config");
        assert_eq!(cfg.endpoint, "http://legacy-alias.test:9200");
        assert_eq!(cfg.bucket, "legacy-bucket");
        unsafe { std::env::remove_var("SPACES_ENDPOINT") };
        unsafe { std::env::remove_var("SPACES_BUCKET") };
    }

    #[test]
    fn half_configured_keys_are_rejected_not_downgraded() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("S3_ENDPOINT", "http://minio.test:9000") };
        unsafe { std::env::set_var("S3_BUCKET", "b") };
        unsafe { std::env::set_var("S3_ACCESS_KEY_ID", "AKIA-test-key-value-1234567890") };
        let err = s3_config_from_env().expect_err("must reject partial credentials");
        assert!(matches!(err, StorageError::Auth(_)), "{err}");
        unsafe { std::env::remove_var("S3_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("S3_BUCKET") };
        unsafe { std::env::remove_var("S3_ENDPOINT") };
    }

    #[test]
    fn sink_status_never_contains_secret_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("S3_ENDPOINT", "http://minio.test:9000") };
        unsafe { std::env::set_var("S3_BUCKET", "st") };
        unsafe { std::env::set_var("S3_ACCESS_KEY_ID", "AKIA-demo-demo-demo-9876543210") };
        unsafe { std::env::set_var("S3_SECRET_ACCESS_KEY", "SUPERSECRETVALUE123") };
        match sink_status() {
            SinkStatus::Authenticated {
                endpoint,
                bucket,
                key_head,
            } => {
                assert_eq!(endpoint, "http://minio.test:9000");
                assert_eq!(bucket, "st");
                assert!(!key_head.contains("SUPERSECRETVALUE123"));
            }
            other => panic!("expected authenticated status, got {other:?}"),
        }
        unsafe { std::env::remove_var("S3_SECRET_ACCESS_KEY") };
        unsafe { std::env::remove_var("S3_ACCESS_KEY_ID") };
        unsafe { std::env::remove_var("S3_BUCKET") };
        unsafe { std::env::remove_var("S3_ENDPOINT") };
    }

    #[test]
    fn walk_files_enumerates_nested_layout_deterministically() {
        let td = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(td.path().join("views")).unwrap();
        std::fs::write(td.path().join("manifest.json"), b"{}").unwrap();
        std::fs::write(td.path().join("views").join("pinn.parquet"), b"P").unwrap();
        let files = walk_files(td.path()).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| {
                p.strip_prefix(td.path())
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert_eq!(names, vec!["manifest.json", "views/pinn.parquet"]);
    }
}
