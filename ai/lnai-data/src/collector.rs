use crate::integrity::{count_tap_csv_rows, sha256_file};
use crate::manifest::{DatasetManifestV1, ShardState, ShardStatus};
use sha2::Digest;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Default per-shard row budget requested with ADQL `TOP`. When the source
/// returns more rows than this, the scheduler subdivides the shard.
pub const DEFAULT_TARGET_ROWS_PER_SHARD: usize = 200_000;

/// RA band width of a top-level shard in degrees. The canonical 360-degree
/// collection is planned as adjacent `[degree, degree + 1)` strips.
pub const TOP_LEVEL_RA_WIDTH_DEG: f64 = 1.0;

/// Depth cap for adaptive subdivision (safety against pathological skies).
pub const MAX_SUBDIVIDE_DEPTH: u32 = 6;

/// Upper bound of recorded lifetime attempts per shard; --retry-failed stops
/// accepting shards past this even across many separate runs.
pub const MAX_LIFETIME_RETRIES_PER_SHARD: u32 = 8;

const MIN_TARGET_ROWS_PER_SHARD: usize = 10;

fn current_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Clone)]
pub struct CollectConfig {
    /// Directory holding `manifest.json` and downloaded shard files.
    pub out_dir: PathBuf,
    /// Inclusive-exclusive RA range collected, in degrees.
    pub ra_start_deg: f64,
    pub ra_end_deg: f64,
    /// Brightness cut applied server-side; part of the manifest query hash.
    pub mag_limit_g: f64,
    /// RUWE quality cut mirrored from the legacy pipeline.
    pub max_ruwe: f64,
    /// Row budget that triggers adaptive subdivision when exceeded.
    pub target_rows_per_shard: usize,
    /// Bounded number of concurrent downloads (worker threads).
    pub concurrency: usize,
    /// Base delay for exponential backoff between retries, in ms.
    pub retry_backoff_ms_base: u64,
    /// Retry attempts per shard before it is left in `Failed`.
    pub retry_max_attempts: u32,
}

impl Default for CollectConfig {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("data/canonical-v1"),
            ra_start_deg: 0.0,
            ra_end_deg: 360.0,
            mag_limit_g: 16.0,
            max_ruwe: 1.4,
            target_rows_per_shard: DEFAULT_TARGET_ROWS_PER_SHARD,
            concurrency: 4,
            retry_backoff_ms_base: 250,
            retry_max_attempts: 3,
        }
    }
}

impl CollectConfig {
    /// Stable string hashed into the manifest as `query_hash`: any change to
    /// cuts or planning constants invalidates previous collections.
    pub fn query_hash(&self) -> String {
        let params = format!(
            "base_v1|ra={:.3}..{:.3}|mag_g<={:.2}|ruwe<={:.3}|top={}",
            self.ra_start_deg,
            self.ra_end_deg,
            self.mag_limit_g,
            self.max_ruwe,
            self.target_rows_per_shard
        );
        let mut h = sha2::Sha256::new();
        h.update(params.as_bytes());
        hex_digest(h.finalize().as_slice())
    }
}

/// Errors surfaced by [`ShardFetcher`] implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchError {
    /// Transport/HTTP failure; retriable.
    Http(String),
    /// Server answered more rows than the shard budget; schedulers must split.
    TooManyRows { limit: usize },
    /// Unexpected payload shape; not retried automatically.
    Protocol(String),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Http(e) => write!(f, "http error: {e}"),
            FetchError::TooManyRows { limit } => {
                write!(f, "row limit exceeded ({limit}); shard must be subdivided")
            }
            FetchError::Protocol(e) => write!(f, "protocol violation: {e}"),
        }
    }
}

/// Abstraction over the actual network download so the whole collector state
/// machine is testable offline with replayed fixtures.
pub trait ShardFetcher: Send + Sync {
    /// Streams the rows produced by `query` into `dest` (a temporary path).
    /// Returns the number of data rows written on success.
    fn fetch(&self, query: &str, dest: &Path) -> Result<u64, FetchError>;
}

/// Generates the ADQL query for one shard. Public so tests can assert the
/// exact contract sent to the archive (part of `query_hash` stability).
pub fn adql_query_for_shard(shard: &PlannedShard, cfg: &CollectConfig) -> String {
    format!(
        "SELECT TOP {} \
         CAST(gs.source_id AS varchar) AS source_id, \
         gs.ra AS ra_deg, \
         gs.dec AS dec_deg, \
         gs.parallax AS parallax_mas, \
         gs.pmra AS pm_ra_mas_yr, \
         gs.pmdec AS pm_dec_mas_yr, \
         gs.radial_velocity AS radial_velocity_kms, \
         gs.phot_g_mean_mag AS mag_g, \
         gs.phot_bp_mean_mag AS mag_bp, \
         gs.phot_rp_mean_mag AS mag_rp, \
         gs.ruwe AS ruwe, \
         gs.astrometric_excess_noise AS astrometric_excess_noise \
         FROM gaiadr3.gaia_source AS gs \
         WHERE gs.ra >= {:.6} AND gs.ra < {:.6} \
           AND gs.dec >= {:.6} AND gs.dec < {:.6} \
           AND gs.phot_g_mean_mag IS NOT NULL AND gs.phot_g_mean_mag < {:.3} \
           AND gs.ruwe IS NOT NULL AND gs.ruwe < {:.3} \
         ORDER BY gs.source_id",
        cfg.target_rows_per_shard + 1,
        shard.ra_range.0,
        shard.ra_range.1,
        shard.dec_range.0,
        shard.dec_range.1,
        cfg.mag_limit_g,
        cfg.max_ruwe,
    )
}

/// A concrete spatial chunk to download: RA range `[start, end)` crossed with a
/// declination band. Top-level shards span the full Dec column; dense regions
/// get subdivided adaptively.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedShard {
    pub id: String,
    pub ra_range: (f64, f64),
    pub dec_range: (f64, f64),
}

impl PlannedShard {
    pub fn file_name(&self) -> String {
        format!("{}.csv", self.id)
    }
}

fn fmt_id(ra: (f64, f64), dec: (f64, f64)) -> String {
    format!("ra_{:.0}_{:.0}_dec{:+.1}_{:+.1}", ra.0, ra.1, dec.0, dec.1)
}

/// Plans all top-level shards covering `[ra_start, ra_end)` degrees: one
/// shard per RA degree, spanning Dec `[-90, 90)`.
pub fn plan_top_level_shards(cfg: &CollectConfig) -> Vec<PlannedShard> {
    let mut shards = Vec::new();
    let mut ra = cfg.ra_start_deg;
    while ra < cfg.ra_end_deg - f64::EPSILON {
        let next = (ra + TOP_LEVEL_RA_WIDTH_DEG).min(cfg.ra_end_deg);
        shards.push(PlannedShard {
            id: fmt_id((ra, next), (-90.0, 90.0)),
            ra_range: (ra, next),
            dec_range: (-90.0, 90.0),
        });
        ra += TOP_LEVEL_RA_WIDTH_DEG;
    }
    shards
}

/// Subdivides a shard into smaller children, deterministically: first by
/// declination bands (no wider than 10 degrees), then, once Dec is narrow, by
/// halving the RA extent ("adaptive split по Dec / меньшему RA").
pub fn subdivide_shard(parent: &PlannedShard, depth: u32) -> Vec<PlannedShard> {
    const SUBDIVIDE_DEC_BAND: f64 = 10.0;
    let dec_span = parent.dec_range.1 - parent.dec_range.0;
    let children: Vec<PlannedShard> = if dec_span > SUBDIVIDE_DEC_BAND {
        let bands = (dec_span / SUBDIVIDE_DEC_BAND).ceil() as usize;
        let step = dec_span / bands as f64;
        (0..bands)
            .map(|i| {
                let lo = parent.dec_range.0 + step * i as f64;
                let hi = if i == bands - 1 {
                    parent.dec_range.1
                } else {
                    lo + step
                };
                PlannedShard {
                    id: fmt_id(parent.ra_range, (lo, hi)),
                    ra_range: parent.ra_range,
                    dec_range: (lo, hi),
                }
            })
            .collect()
    } else {
        let mid = (parent.ra_range.0 + parent.ra_range.1) / 2.0;
        vec![
            PlannedShard {
                id: fmt_id((parent.ra_range.0, mid), parent.dec_range),
                ra_range: (parent.ra_range.0, mid),
                dec_range: parent.dec_range,
            },
            PlannedShard {
                id: fmt_id((mid, parent.ra_range.1), parent.dec_range),
                ra_range: (mid, parent.ra_range.1),
                dec_range: parent.dec_range,
            },
        ]
    };
    children
        .into_iter()
        .map(|mut s| {
            s.id = format!("{}.d{}", s.id, depth);
            s
        })
        .collect()
}

/// Outcome of one collection run, used by callers/tests and CLI reports.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CollectReport {
    pub fetched: u32,
    pub subdivided: u32,
    pub skipped_verified: u32,
    pub failed: u32,
}

/// High level behavior switches for [`run_collection`].
#[derive(Debug, Clone, Default)]
pub struct CollectOptions {
    /// Re-attempt shards currently in `Failed` (default they are skipped).
    pub retry_failed: bool,
    /// Recompute checksums/row counts of `Verified` shards instead of skipping.
    pub verify: bool,
    /// Restrict work to shards whose id contains this substring.
    pub only: Option<String>,
    /// Test hook: simulate a process interruption after N successful shards.
    pub test_interrupt_after_n_shards: u32,
}

/// Runs/resumes the whole collection against `fetcher`.
///
/// Guarantees (tested):
/// * already-`Verified` shards are never re-downloaded unless `verify`;
/// * `Failed` shards are skipped without `retry_failed`;
/// * `TooManyRows` shards are replaced by deterministic children recorded in
///   the manifest (`subdivided_into`, `row_limit_hit` — пункт 12);
/// * shard files are written to unique temp paths and atomically renamed.
pub fn run_collection(
    cfg: CollectConfig,
    fetcher: Arc<dyn ShardFetcher>,
    opts: CollectOptions,
) -> Result<CollectReport, String> {
    fs::create_dir_all(&cfg.out_dir).map_err(|e| e.to_string())?;
    if cfg.target_rows_per_shard < MIN_TARGET_ROWS_PER_SHARD {
        return Err(format!(
            "target_rows_per_shard={} below minimum {}",
            cfg.target_rows_per_shard, MIN_TARGET_ROWS_PER_SHARD
        ));
    }
    if cfg.concurrency == 0 {
        return Err("concurrency must be >= 1".into());
    }

    let manifest_path = cfg.out_dir.join("manifest.json");
    let mut manifest = load_or_init_manifest(&cfg, &manifest_path)?;
    ensure_planned_shards_registered(&mut manifest, &cfg, opts.only.as_deref())?;
    persist_manifest(&manifest, &manifest_path)?;

    let manifest = Arc::new(Mutex::new(manifest));
    // After the simulated-crash budget is spent, every further download of
    // this run fails with a transport error (test hook; 0 disables it).
    let interrupt_active = opts.test_interrupt_after_n_shards > 0;
    let interrupt_left = Arc::new(AtomicU32::new(opts.test_interrupt_after_n_shards));
    let success_counter = Arc::new(AtomicU64::new(0));
    let mut report = CollectReport::default();
    loop {
        let batch = select_batch(&manifest, &cfg, &opts, &manifest_path)?;
        if batch.is_empty() {
            break;
        }
        let progress = process_batch(
            &manifest,
            &cfg,
            fetcher.clone(),
            &batch,
            &interrupt_left,
            &success_counter,
            interrupt_active,
            &manifest_path,
        )?;
        if !progress {
            break; // no forward movement (all remaining failing) — avoid spin
        }
    }

    // Final snapshot + honest statistics derived from the manifest itself.
    let manifest = Arc::try_unwrap(manifest)
        .map_err(|_| "manifest lock leaked".to_string())?
        .into_inner()
        .map_err(|_| "manifest poisoned".to_string())?;
    report.fetched = success_counter.load(Ordering::SeqCst) as u32;
    report.subdivided = manifest.shards.iter().filter(|s| s.row_limit_hit).count() as u32;
    report.failed = manifest
        .shards
        .iter()
        .filter(|s| s.status == ShardStatus::Failed)
        .count() as u32;
    persist_manifest(&manifest, &manifest_path)?;
    Ok(report)
}

fn load_or_init_manifest(cfg: &CollectConfig, path: &Path) -> Result<DatasetManifestV1, String> {
    if path.exists() {
        let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let parsed: DatasetManifestV1 = serde_json::from_str(&raw)
            .map_err(|e| format!("manifest {} unreadable: {e}", path.display()))?;
        if parsed.query_hash != cfg.query_hash() {
            return Err(format!(
                "manifest query_hash {} does not match config {}; delete the directory or align the config",
                parsed.query_hash,
                cfg.query_hash()
            ));
        }
        return Ok(parsed);
    }
    Ok(DatasetManifestV1::new(
        "gaia_dr3",
        &cfg.query_hash(),
        &crate::integrity::schema_hash(),
    ))
}

fn ensure_planned_shards_registered(
    manifest: &mut DatasetManifestV1,
    cfg: &CollectConfig,
    only: Option<&str>,
) -> Result<(), String> {
    let mut existing: std::collections::HashSet<String> =
        manifest.shards.iter().map(|s| s.shard_id.clone()).collect();
    for planned in plan_top_level_shards(cfg) {
        if let Some(pattern) = only {
            if !planned.id.contains(pattern) {
                continue;
            }
        }
        if existing.remove(&planned.id) {
            continue;
        }
        manifest.shards.push(ShardState::new(
            planned.id.clone(),
            (planned.ra_range.0 as f32, planned.ra_range.1 as f32),
            (planned.dec_range.0 as f32, planned.dec_range.1 as f32),
        ));
    }
    Ok(())
}

/// Returns shards ready to work on this pass; also applies `verify` re-checks
/// and `retry_failed` resets. Empty vec = done.
fn select_batch(
    manifest: &Arc<Mutex<DatasetManifestV1>>,
    cfg: &CollectConfig,
    opts: &CollectOptions,
    manifest_path: &Path,
) -> Result<Vec<String>, String> {
    let mut work: Vec<String> = Vec::new();
    let mut dirty = false;

    {
        let mut m = manifest.lock().map_err(|_| "manifest poisoned")?;
        for idx in 0..m.shards.len() {
            let shard = &mut m.shards[idx];
            if let Some(pattern) = &opts.only {
                if !shard.shard_id.contains(pattern.as_str()) {
                    continue;
                }
            }
            match shard.status {
                ShardStatus::Verified => {
                    if opts.verify {
                        let file = cfg.out_dir.join(format!("{}.csv", shard.shard_id));
                        match fs::File::open(&file) {
                            Err(_) => {
                                shard.transition_to(ShardStatus::Failed).ok();
                                dirty = true;
                            }
                            Ok(f) => {
                                let digest = sha256_stream(f);
                                shard.transition_to(ShardStatus::Verifying).ok();
                                if digest == shard.checksum {
                                    shard.transition_to(ShardStatus::Verified).ok();
                                } else {
                                    shard.transition_to(ShardStatus::Failed).ok();
                                }
                                dirty = true;
                            }
                        }
                    }
                }
                ShardStatus::Failed => {
                    if opts.retry_failed && shard.retries < MAX_LIFETIME_RETRIES_PER_SHARD {
                        shard.transition_to(ShardStatus::Pending).ok();
                        work.push(shard.shard_id.clone());
                        dirty = true;
                    }
                }
                ShardStatus::Pending
                | ShardStatus::Downloading
                | ShardStatus::Downloaded
                | ShardStatus::Verifying => {
                    // Downloaded/Downloading/Verifying can only appear after an
                    // external crash; all recover by starting the work again.
                    if shard.status != ShardStatus::Pending {
                        shard.status = ShardStatus::Pending;
                    }
                    work.push(shard.shard_id.clone());
                    dirty = true;
                }
            }
        }
    }

    if dirty {
        persist_locked(manifest, manifest_path)?;
    }

    // Verify-mode re-checks above are pure file operations: no network runs
    // during --verify.
    Ok(work)
}

fn sha256_stream(file: fs::File) -> String {
    let mut h = sha2::Sha256::new();
    let mut reader = std::io::BufReader::new(file);
    let mut buf = [0u8; 64 * 1024];
    use std::io::Read;
    loop {
        let n = reader.read(&mut buf).unwrap_or(0);
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    hex_digest(h.finalize().as_slice())
}

/// Processes `batch` with bounded concurrency. Returns whether any forward
/// progress happened (download completed or subdivision registered).
#[allow(clippy::too_many_arguments)]
fn process_batch(
    manifest: &Arc<Mutex<DatasetManifestV1>>,
    cfg: &CollectConfig,
    fetcher: Arc<dyn ShardFetcher>,
    batch: &[String],
    interrupt_left: &Arc<AtomicU32>,
    success_counter: &Arc<AtomicU64>,
    interrupt_active: bool,
    manifest_path: &Path,
) -> Result<bool, String> {
    let queue: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(batch.to_vec()));
    let progress = Arc::new(AtomicU64::new(0));
    let workers = cfg.concurrency.min(batch.len()).max(1);
    let mut handles = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let queue = Arc::clone(&queue);
        let manifest = Arc::clone(manifest);
        let fetcher = Arc::clone(&fetcher);
        let cfg = cfg.clone();
        let interrupt_left = Arc::clone(interrupt_left);
        let success_counter = Arc::clone(success_counter);
        let progress = Arc::clone(&progress);
        let manifest_path = manifest_path.to_path_buf();
        handles.push(thread::spawn(move || {
            loop {
                let shard_id = {
                    let mut q = queue.lock().unwrap();
                    q.pop()
                };
                let Some(shard_id) = shard_id else { break };
                let outcome = download_with_retries(
                    &shard_id,
                    &manifest,
                    &cfg,
                    &fetcher,
                    &interrupt_left,
                    &success_counter,
                    interrupt_active,
                );
                let moved = !matches!(outcome, ShardOutcome::Exhausted | ShardOutcome::Abort);
                if moved {
                    progress.fetch_add(1, Ordering::SeqCst);
                }
                let _ = worker_id;
                if persist_locked(&manifest, &manifest_path).is_err() {
                    // Keep working; the orchestrator persists again at the end
                    // of the pass and errors surface through poisoning checks.
                }
            }
        }));
    }
    for h in handles {
        h.join().map_err(|_| "worker panicked")?;
    }
    let moved = progress.load(Ordering::SeqCst) > 0;
    // Snapshot persistence already handled by workers; make one final pass.
    {
        let m = manifest.lock().map_err(|_| "manifest poisoned")?;
        persist_manifest(&m, manifest_path)?;
    }
    Ok(moved)
}

enum ShardOutcome {
    Ok,
    Subdivided,
    Exhausted,
    /// Simulated interruption fired: leave everything untouched.
    Abort,
}

#[allow(clippy::too_many_arguments)]
fn download_with_retries(
    shard_id: &str,
    manifest: &Arc<Mutex<DatasetManifestV1>>,
    cfg: &CollectConfig,
    fetcher: &Arc<dyn ShardFetcher>,
    interrupt_left: &Arc<AtomicU32>,
    _success_counter: &Arc<AtomicU64>,
    interrupt_active: bool,
) -> ShardOutcome {
    let mut attempt: u32 = 0;
    loop {
        // Mark active. A previous attempt inside this very call may have left
        // the shard in `Failed`; recover through `Pending` first.
        {
            let mut m = manifest.lock().unwrap();
            if let Some(s) = m.shards.iter_mut().find(|s| s.shard_id == shard_id) {
                match s.status {
                    ShardStatus::Failed => s.transition_to(ShardStatus::Pending).ok(),
                    _ => Some(()),
                };
                s.transition_to(ShardStatus::Downloading).ok();
            }
        }

        let (planned, file_path, tmp_path) = {
            let m = manifest.lock().unwrap();
            let Some(s) = m.shards.iter().find(|s| s.shard_id == shard_id) else {
                return ShardOutcome::Exhausted;
            };
            let planned = PlannedShard {
                id: s.shard_id.clone(),
                ra_range: (s.ra_range.0 as f64, s.ra_range.1 as f64),
                dec_range: (s.dec_range.0 as f64, s.dec_range.1 as f64),
            };
            let file_path = cfg.out_dir.join(planned.file_name());
            let tmp_path = cfg
                .out_dir
                .join(format!(".{}.csv.tmp.{}.part", planned.id, attempt));
            (planned, file_path, tmp_path)
        };

        // Simulated interruption (test hook): while the injected budget lasts
        // downloads proceed normally; afterwards every download fails with a
        // transport error for the remainder of this run_collection call.
        // Simulated interruption behaves like an abrupt process kill: no
        // status changes, no retry accounting, abort the whole pass.
        if interrupt_active && interrupt_left.load(Ordering::SeqCst) == 0 {
            return ShardOutcome::Abort;
        }
        let fetch_result = {
            interrupt_left
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                    if n > 0 { Some(n - 1) } else { Some(n) }
                })
                .ok();
            fetch_shard(
                fetcher.as_ref(),
                &adql_query_for_shard(&planned, cfg),
                &tmp_path,
                cfg.target_rows_per_shard,
            )
        };

        match fetch_result {
            Ok(rows) => {
                let digest = match sha256_file(&tmp_path) {
                    Ok(d) => d,
                    Err(e) => {
                        let _ = fs::remove_file(&tmp_path);
                        record_failure(manifest, shard_id, e);
                        return ShardOutcome::Exhausted;
                    }
                };
                if fs::rename(&tmp_path, &file_path).is_err() {
                    dbg!("rename fail");
                    record_failure(manifest, shard_id, "atomic rename failed".into());
                    return ShardOutcome::Exhausted;
                }
                let mut m = manifest.lock().unwrap();
                if let Some(s) = m.shards.iter_mut().find(|s| s.shard_id == shard_id) {
                    s.row_count = rows;
                    s.checksum = digest;
                    s.bytes_downloaded = fs::metadata(&file_path).map(|m| m.len()).unwrap_or(0);
                    s.last_attempt_ms = Some(current_ms());
                    // Contracted chain: Downloading -> Downloaded -> Verifying -> Verified.
                    s.transition_to(ShardStatus::Downloaded).ok();
                    s.transition_to(ShardStatus::Verifying).ok();
                    s.transition_to(ShardStatus::Verified).ok();
                }
                drop(m);
                _success_counter.fetch_add(1, Ordering::SeqCst);
                return ShardOutcome::Ok;
            }
            Err(FetchError::TooManyRows { .. }) => {
                let _ = fs::remove_file(&tmp_path);
                let subdivided = subdivide_under_lock(manifest, shard_id, cfg);
                return if subdivided {
                    ShardOutcome::Subdivided
                } else {
                    // Depth cap reached: leave explicit Failed state.
                    record_failure(
                        manifest,
                        shard_id,
                        "row limit hit at max subdivision depth".into(),
                    );
                    ShardOutcome::Exhausted
                };
            }
            Err(FetchError::Http(msg)) if msg.contains("simulated interruption") => {
                let _ = fs::remove_file(&tmp_path);
                // Roll the shard back to Pending exactly like a crash would.
                let mut m = manifest.lock().unwrap();
                if let Some(s) = m.shards.iter_mut().find(|s| s.shard_id == shard_id) {
                    if s.status == ShardStatus::Downloading {
                        s.transition_to(ShardStatus::Pending).ok();
                    }
                }
                drop(m);
                return ShardOutcome::Abort;
            }
            Err(other) => {
                let _ = fs::remove_file(&tmp_path);
                record_failure(manifest, shard_id, other.to_string());
                if attempt >= cfg.retry_max_attempts {
                    return ShardOutcome::Exhausted;
                }
                let backoff = cfg
                    .retry_backoff_ms_base
                    .saturating_mul(1 << attempt.min(10));
                thread::sleep(Duration::from_millis(backoff));
                attempt += 1;
            }
        }
    }
}

fn record_failure(manifest: &Arc<Mutex<DatasetManifestV1>>, shard_id: &str, _msg: String) {
    let mut m = manifest.lock().unwrap();
    if let Some(s) = m.shards.iter_mut().find(|s| s.shard_id == shard_id) {
        if matches!(s.status, ShardStatus::Downloading | ShardStatus::Verifying) {
            s.transition_to(ShardStatus::Failed).ok();
        } else if s.status == ShardStatus::Pending {
            s.transition_to(ShardStatus::Downloading).ok();
            s.transition_to(ShardStatus::Failed).ok();
        }
    }
}

/// Registers deterministic children for `shard_id`, marks the parent with
/// `row_limit_hit` + `subdivided_into`, and returns success.
fn subdivide_under_lock(
    manifest: &Arc<Mutex<DatasetManifestV1>>,
    shard_id: &str,
    cfg: &CollectConfig,
) -> bool {
    let mut m = manifest.lock().unwrap();
    let Some(pos) = m.shards.iter().position(|s| s.shard_id == shard_id) else {
        return false;
    };
    let (depth, planned) = {
        let s = &m.shards[pos];
        let depth = s
            .shard_id
            .rsplit_once(".d")
            .and_then(|(_, d)| d.parse::<u32>().ok())
            .unwrap_or(0);
        let planned = PlannedShard {
            id: s.shard_id.clone(),
            ra_range: (s.ra_range.0 as f64, s.ra_range.1 as f64),
            dec_range: (s.dec_range.0 as f64, s.dec_range.1 as f64),
        };
        (depth + 1, planned)
    };
    if depth > MAX_SUBDIVIDE_DEPTH {
        return false;
    }
    let children = subdivide_shard(&planned, depth);
    let child_ids: Vec<String> = children.iter().map(|c| c.id.clone()).collect();
    let existing: std::collections::HashSet<String> =
        m.shards.iter().map(|s| s.shard_id.clone()).collect();
    let mut added = 0u32;
    for child in children {
        if existing.contains(&child.id) {
            continue;
        }
        m.shards.push(ShardState::new(
            child.id.clone(),
            (child.ra_range.0 as f32, child.ra_range.1 as f32),
            (child.dec_range.0 as f32, child.dec_range.1 as f32),
        ));
        added += 1;
    }
    if added == 0 {
        return false;
    }
    let parent = &mut m.shards[pos];
    parent.row_limit_hit = true;
    parent.subdivided_into = child_ids;
    if parent.status == ShardStatus::Downloading {
        parent.transition_to(ShardStatus::Failed).ok();
    }
    drop(m);
    let _ = cfg;
    true
}

fn fetch_shard(
    fetcher: &dyn ShardFetcher,
    query: &str,
    dest: &Path,
    row_budget: usize,
) -> Result<u64, FetchError> {
    let rows = fetcher.fetch(query, dest)?;
    if rows > row_budget as u64 {
        return Err(FetchError::TooManyRows { limit: row_budget });
    }
    // Sanity: the file must parse as a CSV whose row count matches what the
    // fetcher reported (guards against truncated transports).
    let on_disk = count_tap_csv_rows(dest).map_err(FetchError::Protocol)?;
    if on_disk != rows {
        return Err(FetchError::Protocol(format!(
            "row count drift: reported {rows}, on disk {on_disk}"
        )));
    }
    Ok(rows)
}

fn persist_locked(manifest: &Arc<Mutex<DatasetManifestV1>>, path: &Path) -> Result<(), String> {
    let m = manifest.lock().map_err(|_| "manifest poisoned")?;
    persist_manifest(&m, path)
}

/// Atomic JSON write (tmp + rename) so crashes never corrupt the manifest.
fn persist_manifest(manifest: &DatasetManifestV1, path: &Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json).map_err(|e| e.to_string())?;
    fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeFetcher {
        rows_per_shard: usize,
        calls: Mutex<Vec<String>>,
    }

    impl ShardFetcher for FakeFetcher {
        fn fetch(&self, query: &str, dest: &Path) -> Result<u64, FetchError> {
            self.calls.lock().unwrap().push(query.to_string());
            let mut f = std::fs::File::create(dest).unwrap();
            let rows = self.rows_per_shard;
            use std::io::Write;
            writeln!(f, "source_id,ra_deg,dec_deg").unwrap();
            for i in 0..rows {
                writeln!(f, "{}{i},0.5,0.5", fake_source_prefix(query)).unwrap();
            }
            f.flush().unwrap();
            Ok(rows as u64)
        }
    }

    /// Derives a stable per-shard id prefix from the RA bounds embedded in the
    /// query so different shards produce distinct source_ids.
    fn fake_source_prefix(query: &str) -> String {
        // "WHERE gs.ra >= X AND gs.ra < Y" — take the leading digits of each.
        let ra = query.split("gs.ra >= ").nth(1).unwrap_or("0");
        format!(
            "{}_",
            ra.chars().take(5).collect::<String>().replace('.', "")
        )
    }

    fn cfg(dir: &Path) -> CollectConfig {
        CollectConfig {
            out_dir: dir.to_path_buf(),
            ra_start_deg: 0.0,
            ra_end_deg: 2.0,
            target_rows_per_shard: 50,
            concurrency: 2,
            retry_backoff_ms_base: 1,
            retry_max_attempts: 2,
            ..Default::default()
        }
    }

    #[test]
    fn plan_covers_ra_range_with_degree_strips() {
        let c = cfg(Path::new("/tmp/unused"));
        let shards = plan_top_level_shards(&c);
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].id, "ra_0_1_dec-90.0_+90.0");
        assert_eq!(shards[1].id, "ra_1_2_dec-90.0_+90.0");
    }

    #[test]
    fn subdivide_first_by_dec_then_by_ra() {
        let wide = PlannedShard {
            id: "p".into(),
            ra_range: (0.0, 1.0),
            dec_range: (-90.0, 90.0),
        };
        let by_dec = subdivide_shard(&wide, 1);
        assert_eq!(by_dec.len(), 18); // 180 span / 10-degree bands
        assert!(by_dec.iter().all(|s| s.ra_range == (0.0, 1.0)));

        let narrow_dec = PlannedShard {
            id: "p".into(),
            ra_range: (0.0, 1.0),
            dec_range: (0.0, 10.0),
        };
        let by_ra = subdivide_shard(&narrow_dec, 2);
        assert_eq!(by_ra.len(), 2);
        assert_eq!(by_ra[0].ra_range, (0.0, 0.5));
        assert_eq!(by_ra[1].ra_range, (0.5, 1.0));
        assert!(by_ra.iter().all(|s| s.id.ends_with(".d2")));
    }

    #[test]
    fn adql_embeds_bounds_cuts_and_row_budget() {
        let c = cfg(Path::new("/tmp/unused"));
        let shard = PlannedShard {
            id: "x".into(),
            ra_range: (10.0, 11.0),
            dec_range: (-20.0, -10.0),
        };
        let q = adql_query_for_shard(&shard, &c);
        assert!(q.contains("SELECT TOP 51"));
        assert!(q.contains("gs.ra >= 10.000000") && q.contains("gs.ra < 11.000000"));
        assert!(q.contains("gs.dec >= -20.000000") && q.contains("gs.dec < -10.000000"));
        assert!(q.contains("mag < 16.000"));
        assert!(q.contains("ruwe < 1.400"));
        assert!(q.contains("ORDER BY gs.source_id"));
    }

    #[test]
    fn query_hash_changes_when_cuts_change() {
        let d = tempfile::tempdir().unwrap();
        let mut c1 = cfg(d.path());
        c1.mag_limit_g = 16.0;
        let mut c2 = cfg(d.path());
        c2.mag_limit_g = 17.0;
        assert_ne!(c1.query_hash(), c2.query_hash());
    }
}
