//! Stage 4 / item 10: data collection jobs + canonical dataset coverage
//! exposed to the Testbench frontend.
//!
//! Endpoints:
//! * `GET  /data/status?dir=...` — manifest coverage summary for one dataset dir
//! * `POST /data/collect`        — spawn `lnaicli collect-data` as a job
//! * `POST /data/verify`         — spawn checksum re-verification as a job
//! * `POST /data/build`          — spawn canonical parquet/view assembly

use std::path::{Path, PathBuf};
use tokio::process::Command;

use axum::{
    Json,
    extract::{Query, State},
};
use lunar_structures_testbench::{Job, JobKind};
use serde::{Deserialize, Serialize};

use crate::AppState;

#[derive(Deserialize)]
pub struct DataDirQuery {
    pub dir: Option<String>,
}

/// Per-status shard rollup used by the datasets page coverage card.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct DataCoverage {
    pub dir: String,
    pub manifest_found: bool,
    pub schema_version: Option<String>,
    pub source_release: Option<String>,
    pub total_rows: u64,
    pub shards_total: usize,
    pub shards_verified: usize,
    pub shards_failed: usize,
    pub shards_pending: usize,
    pub shards_active: usize,
    pub subdivided_shards: usize,
    /// 0..=100; share of *planned* top-level strips fully verified is not
    /// derivable post-subdivision, so this reports verified/total instead and
    /// `coverage_note` explains gaps per manifest.
    pub verified_percent: f64,
    pub gap_explanations: Vec<String>,
    pub last_updated_ms: Option<u64>,
}

impl DataCoverage {
    fn empty(dir: &str) -> Self {
        Self {
            dir: dir.to_string(),
            manifest_found: false,
            schema_version: None,
            source_release: None,
            total_rows: 0,
            shards_total: 0,
            shards_verified: 0,
            shards_failed: 0,
            shards_pending: 0,
            shards_active: 0,
            subdivided_shards: 0,
            verified_percent: 0.0,
            gap_explanations: vec![],
            last_updated_ms: None,
        }
    }
}

pub async fn get_status(Query(q): Query<DataDirQuery>) -> Json<DataCoverage> {
    let dir = q.dir.unwrap_or_else(|| "data/canonical-v1".to_string());
    Json(read_coverage(&dir))
}

pub fn read_coverage(dir: &str) -> DataCoverage {
    let path = PathBuf::from(dir);
    let manifest_path = path.join("manifest.json");
    let Ok(raw) = std::fs::read_to_string(&manifest_path) else {
        return DataCoverage::empty(dir);
    };
    let Ok(m): Result<lnai_data::manifest::DatasetManifestV1, _> = serde_json::from_str(&raw)
    else {
        return {
            let mut c = DataCoverage::empty(dir);
            c.manifest_found = true; // exists but unreadable — surface via note
            c.gap_explanations
                .push("manifest.json exists but failed to parse".into());
            c
        };
    };

    let mut c = DataCoverage::empty(dir);
    c.manifest_found = true;
    c.schema_version = Some(m.version.clone());
    c.source_release = Some(m.source_release.clone());
    c.total_rows = m.total_rows;
    c.last_updated_ms = Some(m.updated_ms);

    let mut gaps = Vec::new();
    for s in &m.shards {
        use lnai_data::manifest::ShardStatus::*;
        match s.status {
            Verified => {
                if s.row_limit_hit {
                    c.subdivided_shards += 1;
                    gaps.push(format!(
                        "{} subdivided into {:?}",
                        s.shard_id, s.subdivided_into
                    ));
                } else {
                    c.shards_verified += 1;
                }
            }
            Failed => {
                c.shards_failed += 1;
                gaps.push(format!(
                    "{} failed after {} attempt(s)",
                    s.shard_id, s.retries
                ));
            }
            Pending => c.shards_pending += 1,
            Downloading | Downloaded | Verifying => c.shards_active += 1,
        }
    }
    let total = m.shards.len().max(1) as f64;
    c.shards_total = m.shards.len();
    // Subdivided parents carry no rows themselves; still count progress by them.
    let done = (c.shards_verified + c.subdivided_shards) as f64;
    c.verified_percent = ((done / total) * 100.0 * 10.0).round() / 10.0;
    c.gap_explanations = gaps;
    c
}

// ------------------------------- job spawning -------------------------------

const DATA_COLLECT_KIND: &str = "data_collect";
const DATA_VERIFY_KIND: &str = "data_verify";
const DATA_BUILD_KIND: &str = "data_build";

fn lnaicli_path() -> PathBuf {
    workspace_root()
        .join("target")
        .join("release")
        .join("lnaicli")
}

fn workspace_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("Cargo.toml").exists() && cwd.join("testbench").exists() {
        return cwd;
    }
    cwd.parent().map(Path::to_path_buf).unwrap_or(cwd)
}

#[derive(Deserialize)]
pub struct DataCollectSpec {
    #[serde(default = "default_out_dir")]
    pub out_dir: String,
    #[serde(default)]
    pub ra_min: f64,
    #[serde(default = "default_ra_max")]
    pub ra_max: f64,
    #[serde(default = "default_mag")]
    pub mag_limit_g: f64,
    #[serde(default = "default_target_rows")]
    pub target_rows_per_shard: usize,
    #[serde(default = "default_workers")]
    pub concurrency: usize,
    #[serde(default)]
    pub only: Option<String>,
    #[serde(default)]
    pub retry_failed: bool,
}

fn default_out_dir() -> String {
    "data/canonical-v1".into()
}
fn default_ra_max() -> f64 {
    360.0
}
fn default_mag() -> f64 {
    16.0
}
fn default_target_rows() -> usize {
    200_000
}
fn default_workers() -> usize {
    4
}

fn collect_command(spec: &DataCollectSpec) -> Command {
    let mut cmd = Command::new(lnaicli_path());
    cmd.arg("collect-data").arg("--out-dir").arg(&spec.out_dir);
    cmd.arg("--ra-min").arg(spec.ra_min.to_string());
    cmd.arg("--ra-max").arg(spec.ra_max.to_string());
    cmd.arg("--mag-limit-g").arg(spec.mag_limit_g.to_string());
    cmd.arg("--target-rows-per-shard")
        .arg(spec.target_rows_per_shard.to_string());
    cmd.arg("--concurrency").arg(spec.concurrency.to_string());
    if spec.retry_failed {
        cmd.arg("--retry-failed");
    }
    if let Some(only) = &spec.only {
        cmd.arg("--only").arg(only);
    }
    cmd
}

fn verify_command(out_dir: &str) -> Command {
    let mut cmd = Command::new(lnaicli_path());
    cmd.arg("collect-data").arg("--out-dir").arg(out_dir);
    cmd.arg("--verify");
    cmd
}

fn build_command(out_dir: &str) -> Command {
    let mut cmd = Command::new(lnaicli_path());
    cmd.arg("build-dataset").arg("--out-dir").arg(out_dir);
    cmd
}

async fn spawn_data_job(
    state: AppState,
    kind_tag: &'static str,
    title: String,
    cmd: Command,
) -> Result<Json<Job>, String> {
    let job = Job::new(JobKind::Custom(kind_tag.to_string()), title, 0);
    let id = state.registry.spawn(job, cmd).map_err(|e| e.to_string())?;
    state
        .registry
        .get(&id)
        .map(Json)
        .ok_or_else(|| "job not found after spawn".to_string())
}

pub async fn start_collect(
    State(state): State<AppState>,
    Json(spec): Json<DataCollectSpec>,
) -> Result<Json<Job>, String> {
    let title = format!(
        "Data collect · RA[{:.0},{:.0}) · {}",
        spec.ra_min, spec.ra_max, spec.out_dir
    );
    spawn_data_job(state, DATA_COLLECT_KIND, title, collect_command(&spec)).await
}

#[derive(Deserialize)]
pub struct DataDirPayload {
    #[serde(default = "default_out_dir")]
    pub out_dir: String,
}

pub async fn start_verify(
    State(state): State<AppState>,
    Json(p): Json<DataDirPayload>,
) -> Result<Json<Job>, String> {
    let title = format!("Data verify · {}", p.out_dir);
    spawn_data_job(state, DATA_VERIFY_KIND, title, verify_command(&p.out_dir)).await
}

pub async fn start_build(
    State(state): State<AppState>,
    Json(p): Json<DataDirPayload>,
) -> Result<Json<Job>, String> {
    let title = format!("Dataset build · {}", p.out_dir);
    spawn_data_job(state, DATA_BUILD_KIND, title, build_command(&p.out_dir)).await
}
