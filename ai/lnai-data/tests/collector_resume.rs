//! Integration: resume after an artificial interruption, retry-failed gating,
//! verify re-checking and adaptive subdivision — all against a fake archive.

use lnai_data::collector::{
    CollectConfig, CollectOptions, FetchError, ShardFetcher, run_collection,
};
use lnai_data::manifest::{DatasetManifestV1, ShardStatus};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Deterministic fake archive server: writes header + N rows CSV to the
/// requested temp path, keyed by shard RA so every shard has unique content.
struct FakeArchive {
    rows_per_shard: usize,
    fail_queries_containing: Vec<&'static str>,
    /// Counts every transport attempt including planned outages.
    total_calls: AtomicU64,
}

impl FakeArchive {
    fn new(rows_per_shard: usize) -> Self {
        Self {
            rows_per_shard,
            fail_queries_containing: Vec::new(),
            total_calls: AtomicU64::new(0),
        }
    }

    fn with_always_failing(mut self, marker: &'static str) -> Self {
        self.fail_queries_containing.push(marker);
        self
    }

    fn calls(&self) -> u64 {
        self.total_calls.load(Ordering::SeqCst)
    }
}

impl ShardFetcher for FakeArchive {
    fn fetch(&self, query: &str, dest: &Path) -> Result<u64, FetchError> {
        use std::io::Write;
        self.total_calls.fetch_add(1, Ordering::SeqCst);
        for marker in &self.fail_queries_containing {
            if query.contains(marker) {
                return Err(FetchError::Http(format!("planned outage: {marker}")));
            }
        }
        let x = query
            .split("gs.ra >= ")
            .nth(1)
            .and_then(|s| s.split(" AND").next())
            .unwrap_or("0")
            .to_string();
        let y = query
            .split("gs.ra < ")
            .nth(1)
            .and_then(|s| s.split(" AND").next())
            .unwrap_or("1")
            .to_string();
        let mut f = fs::File::create(dest).map_err(|e| FetchError::Protocol(e.to_string()))?;
        writeln!(f, "source_id,ra_deg,dec_deg").map_err(|e| FetchError::Protocol(e.to_string()))?;
        for i in 0..self.rows_per_shard {
            writeln!(
                f,
                "{}{}{i},{}.5,10.25",
                x.replace('.', ""),
                y.replace('.', ""),
                x
            )
            .map_err(|e| FetchError::Protocol(e.to_string()))?;
        }
        Ok(self.rows_per_shard as u64)
    }
}

fn three_shard_cfg(out_dir: PathBuf) -> CollectConfig {
    CollectConfig {
        out_dir,
        ra_start_deg: 0.0,
        ra_end_deg: 3.0, // three top-level degree strips
        target_rows_per_shard: 100,
        concurrency: 2,
        retry_backoff_ms_base: 1,
        retry_max_attempts: 3,
        ..Default::default()
    }
}

fn read_manifest(dir: &Path) -> DatasetManifestV1 {
    let raw = fs::read_to_string(dir.join("manifest.json")).expect("manifest exists");
    serde_json::from_str(&raw).expect("valid manifest json")
}

#[test]
fn interrupted_run_resumes_without_redownloading_verified_shards() {
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("collect");
    let cfg = three_shard_cfg(out_dir.clone());

    let archive = Arc::new(FakeArchive::new(50));
    // Simulate a crash right after the first successful upload: with the
    // budget spent, every further download silently stops this pass, leaving
    // unstarted shards exactly as an OS kill would (still pending).
    let report1 = run_collection(
        cfg.clone(),
        archive.clone() as Arc<dyn ShardFetcher>,
        CollectOptions {
            test_interrupt_after_n_shards: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(report1.fetched, 1);

    let m1 = read_manifest(&out_dir);
    let verified_after_crash = m1
        .shards
        .iter()
        .filter(|s| s.status == ShardStatus::Verified)
        .count();
    assert_eq!(verified_after_crash, 1);

    let calls_at_crash = archive.calls();

    // Resume (plain defaults): completes everything, never re-touches the
    // already-verified shard.
    let report2 = run_collection(
        cfg.clone(),
        archive.clone() as Arc<dyn ShardFetcher>,
        CollectOptions::default(),
    )
    .unwrap();
    assert_eq!(
        report2.fetched,
        m1.shards.len() as u32 - 1,
        "resume downloads exactly the missing shards"
    );
    assert_eq!(archive.calls() - calls_at_crash, 2);

    let m2 = read_manifest(&out_dir);
    assert!(m2.shards.iter().all(|s| s.status == ShardStatus::Verified));
    for shard in &m2.shards {
        let p = out_dir.join(format!("{}.csv", shard.shard_id));
        assert_eq!(
            lnai_data::integrity::sha256_file(&p).unwrap(),
            shard.checksum
        );
    }

    // One more resume pass is a perfect network no-op.
    let before_noop = archive.calls();
    let calls_at_end = {
        run_collection(cfg, archive.clone(), CollectOptions::default()).unwrap();
        archive.calls()
    };
    assert_eq!(calls_at_end, before_noop);
}

#[test]
fn failed_shards_are_skipped_until_retry_failed_is_requested() {
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("collect");
    let cfg = three_shard_cfg(out_dir.clone());
    let archive = Arc::new(
        FakeArchive::new(20).with_always_failing("gs.ra >= 2"), // shard #3 always dies
    );

    let r1 = run_collection(cfg.clone(), archive.clone(), CollectOptions::default()).unwrap();
    assert_eq!(r1.failed, 1, "one permanently failing shard");

    // Default resume skips it without spending any network call on it.
    let calls_before_skip = archive.calls();
    let _ = run_collection(cfg.clone(), archive.clone(), CollectOptions::default());
    assert_eq!(archive.calls(), calls_before_skip);

    // --retry-failed attempts it again and keeps the failure recorded.
    let _ = run_collection(
        cfg,
        archive.clone(),
        CollectOptions {
            retry_failed: true,
            ..Default::default()
        },
    )
    .unwrap();

    let m = read_manifest(&out_dir);
    let failing = m
        .shards
        .iter()
        .find(|s| s.shard_id.starts_with("ra_2"))
        .expect("shard #3 registered");
    assert_eq!(failing.status, ShardStatus::Failed);
    assert!(failing.retries >= 1, "retries persisted across runs");
    assert_eq!(
        m.shards
            .iter()
            .filter(|s| s.status == ShardStatus::Verified)
            .count(),
        2
    );
}

#[test]
fn row_limit_hits_subdivide_deterministically_and_record_children() {
    // Fake archive always answers 50 rows; budget 10 forces subdivision until
    // children fit under the limit (depth > MAX_SUBDIVIDE_DEPTH leaves
    // explicit Failed entries).
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("collect");
    let mut cfg = three_shard_cfg(out_dir.clone());
    cfg.target_rows_per_shard = 10;

    let archive = Arc::new(FakeArchive::new(50));
    let report = run_collection(
        cfg,
        archive as Arc<dyn ShardFetcher>,
        CollectOptions::default(),
    )
    .unwrap();
    assert!(report.subdivided >= 1, "row limit forced subdivision");

    let m = read_manifest(&out_dir);
    let parents: Vec<_> = m.shards.iter().filter(|s| s.row_limit_hit).collect();
    assert!(!parents.is_empty());
    for parent in parents {
        assert!(
            !parent.subdivided_into.is_empty(),
            "children ids recorded under subdivided_into"
        );
        assert!(
            m.shards
                .iter()
                .any(|c| parent.subdivided_into.contains(&c.shard_id))
        );
    }

    // No temp/partial files may survive anywhere.
    let leftovers: Vec<_> = fs::read_dir(&out_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains(".part"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn verify_mode_rechecks_checksums_and_flags_corruption_then_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("collect");
    let cfg = three_shard_cfg(out_dir.clone());

    let archive = Arc::new(FakeArchive::new(30));
    run_collection(cfg.clone(), archive.clone(), CollectOptions::default()).unwrap();

    // Clean --verify round: everything matches.
    let rv = run_collection(
        cfg.clone(),
        archive.clone(),
        CollectOptions {
            verify: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rv.failed, 0);
    let clean_calls = archive.calls();

    // Corrupt one shard on disk, then verify again: exactly one mismatch.
    let victim = fs::read_dir(&out_dir)
        .unwrap()
        .find_map(|e| {
            let e = e.ok()?;
            e.file_name()
                .to_string_lossy()
                .ends_with(".csv")
                .then(|| e.path())
        })
        .unwrap();
    fs::write(&victim, "source_id,ra_deg,dec_deg\nTAMPERED,1,1\n").unwrap();
    let rc = run_collection(
        cfg.clone(),
        archive.clone(),
        CollectOptions {
            verify: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rc.failed, 1, "corruption detected");
    // Verification itself performs zero downloads.
    assert_eq!(archive.calls(), clean_calls);

    // Recovery: the demoted shard re-downloads with retry_failed...
    let _ = run_collection(
        cfg.clone(),
        archive.clone(),
        CollectOptions {
            retry_failed: true,
            ..Default::default()
        },
    )
    .unwrap();

    // ...after which a fresh --verify round is perfectly clean.
    let rv2 = run_collection(
        cfg,
        archive,
        CollectOptions {
            verify: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rv2.failed, 0);
}
