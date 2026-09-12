//! Stage 4B: Gaia `astrophysical_parameters` enrichment (PINN targets).
//!
//! The canonical-v1 backbone from `collect-data` only carries `gaia_source`
//! columns, so PINN targets (Teff/R/M/L) are missing. This module fetches them
//! from `gaiadr3.astrophysical_parameters` keyed by `source_id`:
//!
//! * IDs are scanned from the assembled canonical parquet (`source_id`).
//! * IDs are fetched in `--ids-per-query` chunks via TAP sync CSV, reusing the
//!   same [`ShardFetcher`] abstraction as the backbone collector.
//! * Progress is a chunk manifest (`ap_manifest.json`); reruns skip finished
//!   chunks, so a ~35k-query full-sky pass survives restarts.
//! * The join is a streaming LEFT JOIN of fetched parts onto the canonical
//!   parquet, so peak RAM stays bounded; `ap_coverage.json` reports honest
//!   per-column coverage (AP rows exist only for a subset of sources).
//!
//! NOTE: a server-side `TAP_UPLOAD` + join would need ~360 queries instead of
//! tens of thousands, but the fetcher is GET-only today; chunked IN-lists
//! reuse proven infra (retry/backoff/atomic writes).

use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use crate::tap::DEFAULT_GAIA_TAP_SYNC_URL;

pub const AP_MANIFEST_VERSION: u32 = 1;
pub const AP_MANIFEST_FILE: &str = "ap_manifest.json";
pub const AP_PARTS_DIR: &str = "ap_parts";
pub const AP_COVERAGE_FILE: &str = "ap_coverage.json";
pub const AP_ENRICHED_FILE: &str = "enriched.parquet";

/// Astrophysical-parameter output columns (canonical names, all nullable).
pub const AP_OUTPUT_COLUMNS: [&str; 4] = [
    "teff_gspphot",
    "radius_gspphot",
    "mass_flame",
    "lum_flame",
];

/// ADQL for one ID chunk. `source_id` is returned as varchar so the join key
/// matches the canonical parquet byte-for-byte (no int/string juggling).
pub fn ap_query_for_ids(ids: &[&str]) -> String {
    let list = ids.join(",");
    format!(
        "SELECT CAST(ap.source_id AS varchar) AS source_id, \
         ap.teff_gspphot AS teff_gspphot, \
         ap.radius_gspphot AS radius_gspphot, \
         ap.mass_flame AS mass_flame, \
         ap.lum_flame AS lum_flame \
         FROM gaiadr3.astrophysical_parameters AS ap \
         WHERE ap.source_id IN ({list})"
    )
}

/// Resume state for the chunked AP fetch. `completed_chunks` is kept sorted.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ApChunkManifest {
    pub version: u32,
    pub data_file: String,
    pub total_ids: u64,
    pub ids_per_query: usize,
    pub max_ids: u64,
    pub total_chunks: u64,
    pub completed_chunks: Vec<u64>,
    pub failed_chunks: Vec<u64>,
}

impl ApChunkManifest {
    pub fn fresh(
        data_file: &str,
        total_ids: u64,
        ids_per_query: usize,
        max_ids: u64,
        total_chunks: u64,
    ) -> Self {
        Self {
            version: AP_MANIFEST_VERSION,
            data_file: data_file.to_string(),
            total_ids,
            ids_per_query,
            max_ids,
            total_chunks,
            completed_chunks: Vec::new(),
            failed_chunks: Vec::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| format!("read manifest {}: {e}", path.display()))?;
        serde_json::from_str(&raw).map_err(|e| format!("parse manifest {}: {e}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let raw =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize manifest: {e}"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, raw).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))
    }

    pub fn is_complete(&self, idx: u64) -> bool {
        self.completed_chunks.binary_search(&idx).is_ok()
    }

    pub fn mark_complete(&mut self, idx: u64) {
        if let Err(pos) = self.completed_chunks.binary_search(&idx) {
            self.completed_chunks.insert(pos, idx);
        }
        self.failed_chunks.retain(|&f| f != idx);
    }

    pub fn mark_failed(&mut self, idx: u64) {
        if !self.failed_chunks.contains(&idx) {
            self.failed_chunks.push(idx);
        }
    }
}

/// One parsed AP row; every value column is optional (coverage is measured,
/// not assumed).
#[derive(Debug, Clone, PartialEq)]
pub struct ApRow {
    pub source_id: String,
    pub teff_gspphot: Option<f64>,
    pub radius_gspphot: Option<f64>,
    pub mass_flame: Option<f64>,
    pub lum_flame: Option<f64>,
}

fn parse_opt_f64(s: &str) -> Result<Option<f64>, String> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") || s.eq_ignore_ascii_case("nan") {
        return Ok(None);
    }
    s.parse::<f64>()
        .map(Some)
        .map_err(|e| format!("bad float {s:?}: {e}"))
}

/// Parses one TAP sync CSV payload. The header is validated so an HTML error
/// page or VOTable can never silently become rows.
pub fn parse_ap_csv(text: &str) -> Result<Vec<ApRow>, String> {
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| "empty TAP response".to_string())?;
    let got: Vec<&str> = header
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .collect();
    let expected = [
        "source_id",
        "teff_gspphot",
        "radius_gspphot",
        "mass_flame",
        "lum_flame",
    ];
    if got != expected {
        return Err(format!("unexpected AP header: {header:?}"));
    }
    let mut rows = Vec::new();
    for (lineno, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 5 {
            return Err(format!(
                "AP csv line {}: expected 5 fields, got {}",
                lineno + 2,
                f.len()
            ));
        }
        rows.push(ApRow {
            source_id: f[0].trim().trim_matches('"').to_string(),
            teff_gspphot: parse_opt_f64(f[1])?,
            radius_gspphot: parse_opt_f64(f[2])?,
            mass_flame: parse_opt_f64(f[3])?,
            lum_flame: parse_opt_f64(f[4])?,
        });
    }
    Ok(rows)
}

fn part_path(out_dir: &Path, idx: u64) -> PathBuf {
    out_dir
        .join(AP_PARTS_DIR)
        .join(format!("ap_part_{idx:06}.parquet"))
}

fn write_ap_part(path: &Path, rows: &[ApRow]) -> Result<(), String> {
    let mut df = df![
        "source_id" => rows.iter().map(|r| r.source_id.clone()).collect::<Vec<_>>(),
        "teff_gspphot" => rows.iter().map(|r| r.teff_gspphot).collect::<Vec<Option<f64>>>(),
        "radius_gspphot" => rows.iter().map(|r| r.radius_gspphot).collect::<Vec<Option<f64>>>(),
        "mass_flame" => rows.iter().map(|r| r.mass_flame).collect::<Vec<Option<f64>>>(),
        "lum_flame" => rows.iter().map(|r| r.lum_flame).collect::<Vec<Option<f64>>>(),
    ]
    .map_err(|e| format!("build AP part frame: {e}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let file =
        std::fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    ParquetWriter::new(file)
        .finish(&mut df)
        .map_err(|e| format!("write AP part {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))
}

/// Reads just the `source_id` column (streaming scan, no full decode).
fn read_source_ids(data: &Path, max_ids: u64) -> Result<Vec<String>, String> {
    let lf = scan_one_parquet(data)?;
    let df = lf
        .select([col("source_id")])
        .collect()
        .map_err(|e| format!("scan source_id from {}: {e}", data.display()))?;
    let ca = df
        .column("source_id")
        .map_err(|e| format!("canonical has no source_id column: {e}"))?
        .str()
        .map_err(|e| format!("source_id is not a string column: {e}"))?;
    let mut ids: Vec<String> = Vec::with_capacity(ca.len().min(1_000_000));
    for opt in ca.iter() {
        match opt {
            Some(s) => ids.push(s.to_string()),
            None => return Err("canonical source_id contains null".to_string()),
        }
        if max_ids > 0 && ids.len() as u64 >= max_ids {
            break;
        }
    }
    if ids.is_empty() {
        return Err("no source_ids found".to_string());
    }
    Ok(ids)
}

fn scan_one_parquet(path: &Path) -> Result<LazyFrame, String> {
    let s = path
        .to_str()
        .ok_or_else(|| format!("non-utf8 path {}", path.display()))?;
    let builder = DslBuilder::scan_parquet(
        ScanSources::Paths(polars_buffer::Buffer::from_iter([PlRefPath::from(s)])),
        ParquetOptions::default(),
        UnifiedScanArgs::default(),
    )
    .map_err(|e| format!("parquet scan plan: {e}"))?;
    Ok(LazyFrame::from(builder.0))
}

/// Streaming sink helper (same bounded-RAM contract as `assemble`).
fn sink_parquet_streaming(lf: LazyFrame, target: &Path) -> Result<(), String> {
    let target_ref = PlRefPath::from(
        target
            .to_str()
            .ok_or_else(|| format!("non-utf8 path {}", target.display()))?,
    );
    let write_options = ParquetWriteOptions {
        row_group_size: Some(64 * 1024),
        data_page_size: Some(1024 * 1024),
        ..Default::default()
    };
    lf.sink(
        SinkDestination::File {
            target: SinkTarget::Path(target_ref),
        },
        FileWriteFormat::Parquet(Arc::new(write_options)),
        UnifiedSinkArgs::default(),
    )
    .map_err(|e| format!("parquet sink plan for {}: {e}", target.display()))?
    .collect_with_engine(Engine::Streaming)
    .map(|_| ())
    .map_err(|e| format!("streaming sink {}: {e}", target.display()))
}

fn fetch_chunk_with_retry(
    client: &reqwest::blocking::Client,
    sync_url: &str,
    user: Option<&str>,
    pass: Option<&str>,
    ids: &[String],
    parts_dir: &Path,
    idx: u64,
    attempts: u32,
) -> Result<u64, String> {
    let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
    let query = ap_query_for_ids(&refs);
    let csv_path = parts_dir.join(format!("csv_tmp_{idx:06}.csv"));
    let mut last_err = String::new();
    for attempt in 1..=attempts {
        match fetch_ap_post(client, sync_url, user, pass, &query, &csv_path) {
            Ok(()) => {
                let text = std::fs::read_to_string(&csv_path)
                    .map_err(|e| format!("read TAP csv: {e}"))?;
                match parse_ap_csv(&text) {
                    Ok(rows) => {
                        if rows.len() as u64 > ids.len() as u64 {
                            last_err = format!(
                                "chunk {idx}: server returned more rows than requested"
                            );
                        } else {
                            let n = rows.len() as u64;
                            write_ap_part(&part_path_by_dir(parts_dir, idx), &rows)?;
                            let _ = std::fs::remove_file(&csv_path);
                            return Ok(n);
                        }
                    }
                    Err(e) => last_err = format!("chunk {idx} attempt {attempt}: {e}"),
                }
            }
            Err(e) => {
                last_err = format!("chunk {idx} attempt {attempt}: {e}");
            }
        }
        let _ = std::fs::remove_file(&csv_path);
        std::thread::sleep(Duration::from_secs(1 << attempt.min(4)));
    }
    Err(last_err)
}

/// TAP sync fetch via HTTP POST (form-encoded). POST is required here because
/// a GET request-target carrying thousands of IDs exceeds URL-length limits
/// on the way to the archive; the shared GET-only fetcher cannot carry them.
/// Credentials (if any) come from the caller (env-only upstream) and are
/// never logged; error text is truncated so multi-KB queries never land in logs.
fn fetch_ap_post(
    client: &reqwest::blocking::Client,
    sync_url: &str,
    user: Option<&str>,
    pass: Option<&str>,
    query: &str,
    dest_csv: &Path,
) -> Result<(), String> {
    let mut req = client.post(sync_url).form(&[
        ("REQUEST", "doQuery"),
        ("LANG", "ADQL"),
        ("FORMAT", "csv"),
        ("QUERY", query),
    ]);
    if let (Some(u), Some(p)) = (user, pass) {
        req = req.basic_auth(u, Some(p));
    }
    let mut response = req.send().map_err(|e| short_err(&e))?;
    let status = response.status();
    if !status.is_success() {
        let code = status.as_u16();
        return Err(match code {
            502 | 503 | 504 => format!("server busy: HTTP {code}"),
            _ => format!("TAP returned HTTP {code}"),
        });
    }
    let tmp = dest_csv.with_extension("csvtmp");
    let mut file =
        std::fs::File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    std::io::copy(&mut response, &mut file)
        .map_err(|e| format!("stream TAP csv: {}", short_err(&e)))?;
    std::fs::rename(&tmp, dest_csv).map_err(|e| format!("rename {}: {e}", dest_csv.display()))?;
    Ok(())
}

/// Truncates transport errors: the embedded request URL carries the whole
/// ID list and must never flood logs.
fn short_err(e: &dyn std::fmt::Display) -> String {
    let s = e.to_string();
    if s.len() > 300 {
        format!(
            "{}...[truncated {} chars]",
            &s[..300],
            s.len() - 300
        )
    } else {
        s
    }
}

fn part_path_by_dir(parts_dir: &Path, idx: u64) -> PathBuf {
    parts_dir.join(format!("ap_part_{idx:06}.parquet"))
}

#[derive(Debug, Clone)]
pub struct ApEnrichConfig {
    pub data_path: PathBuf,
    pub out_dir: PathBuf,
    /// Source IDs per TAP query (POST body; 2000 ≈ 40KB form).
    pub ids_per_query: usize,
    pub concurrency: usize,
    /// Cap on scanned IDs (0 = all). Pilot runs use this.
    pub max_ids: u64,
    /// Skip fetching entirely; join whatever parts exist on disk.
    pub join_only: bool,
    /// Where the enriched parquet lands (default: <out_dir>/enriched.parquet).
    pub join_output: Option<PathBuf>,
    pub retry_attempts: u32,
    /// TAP endpoint override (default: [`DEFAULT_GAIA_TAP_SYNC_URL`]).
    pub sync_url: Option<String>,
    /// Optional basic-auth pair (env-only upstream, never logged).
    pub gaia_user: Option<String>,
    pub gaia_pass: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ApCoverageReport {
    pub canonical_rows: u64,
    pub ap_part_files: u64,
    pub ap_rows: u64,
    pub matched_rows: u64,
    pub non_null_teff_gspphot: u64,
    pub non_null_radius_gspphot: u64,
    pub non_null_mass_flame: u64,
    pub non_null_lum_flame: u64,
    pub output: String,
}

impl ApCoverageReport {
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let raw =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize coverage: {e}"))?;
        std::fs::write(path, raw).map_err(|e| format!("write {}: {e}", path.display()))
    }
}

pub fn run_enrich_ap(cfg: &ApEnrichConfig) -> Result<ApCoverageReport, String> {
    if cfg.ids_per_query == 0 {
        return Err("--ids-per-query must be >= 1".to_string());
    }
    let concurrency = cfg.concurrency.max(1);
    let parts_dir = cfg.out_dir.join(AP_PARTS_DIR);
    std::fs::create_dir_all(&parts_dir)
        .map_err(|e| format!("mkdir {}: {e}", parts_dir.display()))?;
    let manifest_path = cfg.out_dir.join(AP_MANIFEST_FILE);

    let ids = read_source_ids(&cfg.data_path, cfg.max_ids)?;
    let total_chunks = ids.len().div_ceil(cfg.ids_per_query) as u64;
    println!(
        "AP enrich: {} ids in {} chunks ({} ids/query, {} workers) from {}",
        ids.len(),
        total_chunks,
        cfg.ids_per_query,
        concurrency,
        cfg.data_path.display()
    );

    // Load or init resume manifest; reconcile completed entries with files.
    let mut manifest = if manifest_path.exists() {
        let m = ApChunkManifest::load(&manifest_path)?;
        if m.data_file != cfg.data_path.to_string_lossy()
            || m.ids_per_query != cfg.ids_per_query
            || m.max_ids != cfg.max_ids
            || m.total_ids != ids.len() as u64
        {
            return Err(format!(
                "out_dir state mismatch (different --data/--ids-per-query/--max-ids); \
                 use a fresh --out-dir or finish the previous run first"
            ));
        }
        m
    } else {
        ApChunkManifest::fresh(
            &cfg.data_path.to_string_lossy(),
            ids.len() as u64,
            cfg.ids_per_query,
            cfg.max_ids,
            total_chunks,
        )
    };
    manifest.completed_chunks.retain(|&i| {
        i < total_chunks && part_path(&cfg.out_dir, i).exists()
    });
    manifest.failed_chunks.retain(|&i| i < total_chunks);
    manifest.save(&manifest_path)?;

    if !cfg.join_only {
        let pending: Vec<u64> = (0..total_chunks)
            .filter(|i| !manifest.is_complete(*i))
            .collect();
        println!("AP enrich: {} chunks pending, {} already done", pending.len(), manifest.completed_chunks.len());
        if !pending.is_empty() {
            let sync_url = cfg
                .sync_url
                .clone()
                .unwrap_or_else(|| DEFAULT_GAIA_TAP_SYNC_URL.to_string());
            // One shared HTTP client (connection reuse, cheap to reference).
            let client = reqwest::blocking::Client::builder()
                .timeout(Some(Duration::from_secs(3600)))
                .connect_timeout(Some(Duration::from_secs(60)))
                .build()
                .map_err(|e| format!("http client: {}", short_err(&e)))?;
            let cursor = AtomicU64::new(0);
            let done = AtomicU64::new(0);
            let state = Mutex::new(manifest);
            let ids_ref = &ids;
            let pending_ref = &pending;
            let parts_ref = &parts_dir;
            let manifest_ref = &manifest_path;
            let client_ref = &client;
            std::thread::scope(|scope| {
                for _ in 0..concurrency {
                    let cursor = &cursor;
                    let done = &done;
                    let state = &state;
                    let cfg = cfg;
                    let sync_url = &sync_url;
                    scope.spawn(move || {
                        loop {
                            let pos = cursor.fetch_add(1, Ordering::SeqCst) as usize;
                            if pos >= pending_ref.len() {
                                break;
                            }
                            let idx = pending_ref[pos];
                            let start = idx as usize * cfg.ids_per_query;
                            let end = (start + cfg.ids_per_query).min(ids_ref.len());
                            match fetch_chunk_with_retry(
                                client_ref,
                                sync_url,
                                cfg.gaia_user.as_deref(),
                                cfg.gaia_pass.as_deref(),
                                &ids_ref[start..end],
                                parts_ref,
                                idx,
                                cfg.retry_attempts,
                            ) {
                                Ok(n) => {
                                    let mut st = state.lock().expect("manifest lock");
                                    st.mark_complete(idx);
                                    let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                                    if d % 25 == 0 {
                                        let _ = st.save(manifest_ref);
                                    }
                                    println!("  chunk {idx}/{total_chunks}: {n} AP rows");
                                }
                                Err(e) => {
                                    let mut st = state.lock().expect("manifest lock");
                                    st.mark_failed(idx);
                                    eprintln!("  chunk {idx} FAILED: {e}");
                                }
                            }
                        }
                    });
                }
            });
            let st = state.into_inner().expect("manifest lock");
            st.save(&manifest_path)?;
            manifest = st;
        }
        if !manifest.failed_chunks.is_empty() {
            return Err(format!(
                "{} chunk(s) failed ({:?}); re-run to retry before joining",
                manifest.failed_chunks.len(),
                &manifest.failed_chunks[..manifest.failed_chunks.len().min(10)]
            ));
        }
        println!(
            "AP enrich: fetch complete, {}/{} chunks verified",
            manifest.completed_chunks.len(),
            total_chunks
        );
    }

    // ---- Join phase: streaming LEFT JOIN of parts onto canonical. ----
    let mut part_files: Vec<PathBuf> = std::fs::read_dir(&parts_dir)
        .map_err(|e| format!("read {}: {e}", parts_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|x| x.to_str()) == Some("parquet")
                && p.file_name()
                    .and_then(|x| x.to_str())
                    .is_some_and(|n| n.starts_with("ap_part_"))
        })
        .collect();
    part_files.sort();
    if part_files.is_empty() {
        return Err("no AP parts on disk; run the fetch phase first".to_string());
    }
    println!("AP enrich: joining {} part files...", part_files.len());

    let mut ap_refs: Vec<PlRefPath> = Vec::with_capacity(part_files.len());
    for p in &part_files {
        let s = p
            .to_str()
            .ok_or_else(|| format!("non-utf8 path {}", p.display()))?;
        ap_refs.push(PlRefPath::from(s));
    }
    let ap_lf = LazyFrame::from(
        DslBuilder::scan_parquet(
            ScanSources::Paths(polars_buffer::Buffer::from_iter(ap_refs)),
            ParquetOptions::default(),
            UnifiedScanArgs::default(),
        )
        .map_err(|e| format!("AP parts scan plan: {e}"))?
        .0,
    )
    .with_column(col("source_id").cast(DataType::String));

    let canon_lf = scan_one_parquet(&cfg.data_path)?
        .with_column(col("source_id").cast(DataType::String));

    let join_output = cfg
        .join_output
        .clone()
        .unwrap_or_else(|| cfg.out_dir.join(AP_ENRICHED_FILE));
    let joined = canon_lf.join(
        ap_lf,
        vec![col("source_id")],
        vec![col("source_id")],
        JoinArgs::new(JoinType::Left),
    );
    sink_parquet_streaming(joined, &join_output)?;

    // ---- Coverage pass (streaming aggregation, O(1) memory). ----
    let cov = scan_one_parquet(&join_output)?
        .select([
            len().cast(DataType::UInt64).alias("n"),
            col("teff_gspphot")
                .is_not_null()
                .sum()
                .cast(DataType::UInt64)
                .alias("teff"),
            col("radius_gspphot")
                .is_not_null()
                .sum()
                .cast(DataType::UInt64)
                .alias("radius"),
            col("mass_flame")
                .is_not_null()
                .sum()
                .cast(DataType::UInt64)
                .alias("mass"),
            col("lum_flame")
                .is_not_null()
                .sum()
                .cast(DataType::UInt64)
                .alias("lum"),
        ])
        .collect_with_engine(Engine::Streaming)
        .map_err(|e| format!("coverage pass: {e}"))?
        .unwrap_single();
    let get = |name: &str| -> Result<u64, String> {
        cov.column(name)
            .map_err(|e| format!("coverage column {name}: {e}"))?
            .u64()
            .map_err(|e| format!("coverage dtype {name}: {e}"))?
            .get(0)
            .ok_or_else(|| format!("coverage empty {name}"))
    };
    let report = ApCoverageReport {
        canonical_rows: get("n")?,
        ap_part_files: part_files.len() as u64,
        ap_rows: {
            let mut n = 0u64;
            for p in &part_files {
                let lf = scan_one_parquet(p)?;
                let c = lf
                    .select([len().cast(DataType::UInt64)])
                    .collect_with_engine(Engine::Streaming)
                    .map_err(|e| format!("count part {}: {e}", p.display()))?
                    .unwrap_single();
                n += c
                    .column("len")
                    .map_err(|e| e.to_string())?
                    .u64()
                    .map_err(|e| e.to_string())?
                    .get(0)
                    .unwrap_or(0);
            }
            n
        },
        matched_rows: get("teff")?,
        non_null_teff_gspphot: get("teff")?,
        non_null_radius_gspphot: get("radius")?,
        non_null_mass_flame: get("mass")?,
        non_null_lum_flame: get("lum")?,
        output: join_output.to_string_lossy().to_string(),
    };
    let cov_path = cfg.out_dir.join(AP_COVERAGE_FILE);
    report.save(&cov_path)?;
    println!(
        "AP enrich: {}/{} rows matched AP (teff {:.1}%, radius {:.1}%, mass {:.1}%, lum {:.1}%) -> {}",
        report.matched_rows,
        report.canonical_rows,
        100.0 * report.non_null_teff_gspphot as f64 / report.canonical_rows.max(1) as f64,
        100.0 * report.non_null_radius_gspphot as f64 / report.canonical_rows.max(1) as f64,
        100.0 * report.non_null_mass_flame as f64 / report.canonical_rows.max(1) as f64,
        100.0 * report.non_null_lum_flame as f64 / report.canonical_rows.max(1) as f64,
        join_output.display()
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ap_query_lists_ids_and_casts_key() {
        let q = ap_query_for_ids(&["123", "456"]);
        assert!(q.contains("gaiadr3.astrophysical_parameters"), "{q}");
        assert!(q.contains("ap.source_id IN (123,456)"), "{q}");
        assert!(q.contains("CAST(ap.source_id AS varchar)"), "{q}");
        for col in AP_OUTPUT_COLUMNS {
            assert!(q.contains(col), "{q}");
        }
    }

    #[test]
    fn manifest_roundtrip_and_resume_skip() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let path = dir.path().join(AP_MANIFEST_FILE);
        let mut m = ApChunkManifest::fresh("canon.parquet", 10, 5, 0, 2);
        assert!(!m.is_complete(0));
        m.mark_complete(1);
        m.mark_complete(0);
        assert!(m.is_complete(0) && m.is_complete(1));
        m.mark_failed(1);
        m.save(&path).expect("save");
        let back = ApChunkManifest::load(&path).expect("load");
        assert_eq!(back.completed_chunks, vec![0, 1]);
        assert_eq!(back.failed_chunks, vec![1]);
        // Re-marking a failed chunk as complete clears the failure.
        let mut back = back;
        back.mark_complete(1);
        assert!(back.failed_chunks.is_empty());
    }

    #[test]
    fn parse_ap_csv_accepts_nulls_and_rejects_bad_header() {
        let text = "source_id,teff_gspphot,radius_gspphot,mass_flame,lum_flame\n\
                    111,5778,,1.0,1.0\n\
                    222,,,,\n";
        let rows = parse_ap_csv(text).expect("parse");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].source_id, "111");
        assert_eq!(rows[0].teff_gspphot, Some(5778.0));
        assert_eq!(rows[0].radius_gspphot, None);
        assert!(rows[1].teff_gspphot.is_none());
        assert!(parse_ap_csv("<html>error</html>").is_err());
        assert!(parse_ap_csv("").is_err());
    }

    #[test]
    fn parse_ap_csv_rejects_garbage_floats() {
        let text = "source_id,teff_gspphot,radius_gspphot,mass_flame,lum_flame\n111,hot,,,\n";
        assert!(parse_ap_csv(text).is_err());
    }

    #[test]
    fn left_join_keeps_unmatched_canonical_rows() {
        let canon = df![
            "source_id" => ["a", "b", "c"],
            "mag_g" => [10.0f32, 11.0, 12.0],
        ]
        .expect("canon");
        let ap = df![
            "source_id" => ["a", "c"],
            "teff_gspphot" => [Some(5778.0), None],
        ]
        .expect("ap");
        let out = canon
            .lazy()
            .join(
                ap.lazy(),
                vec![col("source_id")],
                vec![col("source_id")],
                JoinArgs::new(JoinType::Left),
            )
            .collect()
            .expect("join");
        assert_eq!(out.height(), 3);
        let teff = out.column("teff_gspphot").expect("col");
        assert_eq!(teff.null_count(), 2);
    }
}
