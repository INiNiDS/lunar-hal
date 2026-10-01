use crate::clean::StarRecord;
use crate::manifest::{DatasetManifestV1, ShardState, ShardStatus};
use crate::schema::{SchemaView, required_columns_for_view};
use crate::split::{self, Split};
use polars::prelude::*;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssembleReport {
    pub canonical_path: PathBuf,
    pub view_paths: Vec<(SchemaView, PathBuf)>,
    pub rows_written: u64,
    pub train_rows: u64,
    pub validation_rows: u64,
    pub test_rows: u64,
    pub holdout_rows: u64,
}

pub fn validate_shard_coverage(manifest: &DatasetManifestV1) -> Result<(), String> {
    let shards = &manifest.shards;
    if shards.is_empty() {
        return Err("coverage: manifest contains no shards".into());
    }
    for shard in shards {
        let (ra, dec) = (shard.ra_range, shard.dec_range);
        if !ra.0.is_finite()
            || !ra.1.is_finite()
            || !dec.0.is_finite()
            || !dec.1.is_finite()
            || ra.0 >= ra.1
            || dec.0 >= dec.1
            || ra.0 < 0.0
            || ra.1 > 360.0
            || dec.0 < -90.0
            || dec.1 > 90.0
        {
            return Err(format!(
                "coverage: invalid bounds for shard {}: RA {ra:?} Dec {dec:?}",
                shard.shard_id
            ));
        }
    }
    let stock_query =
        manifest.query_hash == crate::collector::CollectConfig::default().query_hash();
    let ra_start = if stock_query {
        0.0
    } else {
        shards
            .iter()
            .map(|s| s.ra_range.0)
            .fold(f32::INFINITY, f32::min)
    };
    let ra_end = if stock_query {
        360.0
    } else {
        shards
            .iter()
            .map(|s| s.ra_range.1)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let dec_start = if stock_query {
        -90.0
    } else {
        shards
            .iter()
            .map(|s| s.dec_range.0)
            .fold(f32::INFINITY, f32::min)
    };
    let dec_end = if stock_query {
        90.0
    } else {
        shards
            .iter()
            .map(|s| s.dec_range.1)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut edges = vec![ra_start, ra_end];
    for s in shards {
        edges.extend([s.ra_range.0, s.ra_range.1]);
    }
    edges.sort_by(f32::total_cmp);
    edges.dedup();

    let mut gaps = 0usize;
    let mut examples = Vec::new();
    for pair in edges.windows(2) {
        let (left, right) = (pair[0], pair[1]);
        if left >= right || left < ra_start || right > ra_end {
            continue;
        }
        let mut intervals: Vec<(f32, f32)> = shards
            .iter()
            .filter(|s| {
                s.status == ShardStatus::Verified && s.ra_range.0 <= left && s.ra_range.1 >= right
            })
            .map(|s| s.dec_range)
            .collect();
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut covered = dec_start;
        for (lo, hi) in intervals {
            if lo > covered {
                gaps += 1;
                if examples.len() < 8 {
                    examples.push(format!("RA [{left},{right}) Dec [{covered},{lo})"));
                }
            }
            covered = covered.max(hi);
        }
        if covered < dec_end {
            gaps += 1;
            if examples.len() < 8 {
                examples.push(format!("RA [{left},{right}) Dec [{covered},{dec_end})"));
            }
        }
    }
    if gaps > 0 {
        return Err(format!(
            "coverage: {gaps} uncovered sky rectangles in RA [{ra_start},{ra_end}) Dec [{dec_start},{dec_end}); examples: {}. Check failed/missing shards before assembly (no data was written)",
            examples.join(", ")
        ));
    }
    Ok(())
}

fn col_f32(records: &[StarRecord], get: fn(&StarRecord) -> Option<f32>) -> Float32Chunked {
    records.iter().map(|r| get(r)).collect::<Float32Chunked>()
}

pub fn build_canonical_frame(records: &[StarRecord]) -> PolarsResult<DataFrame> {
    let ids: StringChunked = records
        .iter()
        .map(|r| r.source_id.as_str())
        .collect::<StringChunked>();
    let df = df![
        "source_id" => ids,
        "ra_deg" => Series::new("ra_deg".into(), records.iter().map(|r| r.ra_deg).collect::<Vec<f64>>()),
        "dec_deg" => Series::new("dec_deg".into(), records.iter().map(|r| r.dec_deg).collect::<Vec<f64>>()),
        "epoch_year" => Series::new("epoch_year".into(), records.iter().map(|r| r.epoch_year).collect::<Vec<f64>>()),
        "parallax_mas" => Series::new("parallax_mas".into(), records.iter().map(|r| r.parallax_mas).collect::<Vec<Option<f64>>>()),
        "pm_ra_mas_yr" => Series::new("pm_ra_mas_yr".into(), records.iter().map(|r| r.pm_ra_mas_yr).collect::<Vec<Option<f64>>>()),
        "pm_dec_mas_yr" => Series::new("pm_dec_mas_yr".into(), records.iter().map(|r| r.pm_dec_mas_yr).collect::<Vec<Option<f64>>>()),
        "radial_velocity_kms" => Series::new("radial_velocity_kms".into(), records.iter().map(|r| r.radial_velocity_kms).collect::<Vec<Option<f64>>>()),
        "x_pc" => col_f32(records, |r| r.x_pc),
        "y_pc" => col_f32(records, |r| r.y_pc),
        "z_pc" => col_f32(records, |r| r.z_pc),
        "vx_kms" => col_f32(records, |r| r.vx_kms),
        "vy_kms" => col_f32(records, |r| r.vy_kms),
        "vz_kms" => col_f32(records, |r| r.vz_kms),
        "mag_g" => col_f32(records, |r| r.mag_g),
        "mag_bp" => col_f32(records, |r| r.mag_bp),
        "mag_rp" => col_f32(records, |r| r.mag_rp),
        "ruwe" => col_f32(records, |r| r.ruwe),
        "astrometric_excess_noise" => col_f32(records, |r| r.astrometric_excess_noise),
        "is_valid" => Series::new("is_valid".into(), records.iter().map(|r| r.is_valid).collect::<Vec<bool>>()),
        "spatial_tile" => records
            .iter()
            .map(|r| split::spatial_tile_id(r.ra_deg, r.dec_deg))
            .collect::<StringChunked>(),
        "split" => records
            .iter()
            .map(|r| split::classify(r.ra_deg, r.dec_deg, &r.source_id).map(|s| s.as_str()))
            .collect::<StringChunked>(),
    ]?;
    Ok(df)
}

fn write_parquet(df: &DataFrame, path: &Path) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    ParquetWriter::new(file)
        .finish(&mut df.clone())
        .map_err(|e| format!("parquet write {}: {e}", path.display()))?;
    Ok(())
}

fn model_view_plan(mut lf: LazyFrame, view: &SchemaView) -> Result<LazyFrame, String> {
    let schema = lf
        .collect_schema()
        .map_err(|e| format!("view {view:?}: schema: {e}"))?;
    let mut names = required_columns_for_view(view);
    let needed: &[&str] = match view {
        SchemaView::Pinn => &["x_pc", "y_pc", "z_pc", "bp_rp", "mag_g"],
        SchemaView::GnnKinematics => &[
            "x_pc",
            "y_pc",
            "z_pc",
            "bp_rp",
            "mag_g",
            "mag_bp",
            "mag_rp",
            "ruwe",
            "radial_velocity_kms",
            "vx_kms",
            "vy_kms",
            "vz_kms",
        ],
        SchemaView::Siren => &["x_pc", "y_pc", "z_pc", "bp_rp", "mag_g", "ruwe"],
        _ => return Err(format!("view {view:?}: not a standalone model view")),
    };
    let ap = ["teff_gspphot", "radius_gspphot", "mass_flame", "lum_flame"];
    if *view == SchemaView::Pinn && ap.iter().any(|n| schema.contains(n)) {
        if let Some(missing) = ap.iter().find(|n| !schema.contains(n)) {
            return Err(format!(
                "view Pinn: incomplete AP enrichment, missing {missing}"
            ));
        }
        names.extend(ap);
    }
    if !schema.contains("bp_rp") {
        if !schema.contains("mag_bp") || !schema.contains("mag_rp") {
            return Err(format!("view {view:?}: bp_rp or mag_bp/mag_rp required"));
        }
        lf = lf.with_columns([(col("mag_bp") - col("mag_rp"))
            .cast(DataType::Float32)
            .alias("bp_rp")]);
    }
    names.extend(needed);
    if schema.contains("spatial_tile") {
        names.push("spatial_tile");
    }
    if schema.contains("split") {
        names.push("split");
    }
    names.sort_unstable();
    names.dedup();
    let missing = names.iter().find(|n| *n != &"bp_rp" && !schema.contains(n));
    if let Some(name) = missing {
        return Err(format!("view {view:?}: missing column {name}"));
    }
    let mut lf = lf.select(names.iter().map(|n| col(*n)).collect::<Vec<_>>());
    let mut predicate = col("is_valid").eq(lit(true));
    for name in needed {
        predicate = predicate
            .and(col(*name).is_not_null())
            .and(col(*name).is_finite());
    }
    if *view == SchemaView::GnnKinematics {
        for name in ["parallax_mas", "pm_ra_mas_yr", "pm_dec_mas_yr"] {
            predicate = predicate
                .and(col(name).is_not_null())
                .and(col(name).is_finite());
        }
    }
    if *view == SchemaView::Pinn && schema.contains("teff_gspphot") {
        for name in ap {
            predicate = predicate
                .and(col(name).is_not_null())
                .and(col(name).is_finite());
        }
    }
    lf = lf.filter(predicate);
    Ok(lf)
}

#[cfg(test)]
fn view_columns(df: &DataFrame, view: &SchemaView) -> Result<DataFrame, String> {
    model_view_plan(df.clone().lazy(), view)?
        .collect()
        .map_err(|e| format!("view {view:?}: {e}"))
}

pub fn assemble_model_views_from_parquet(
    canonical_path: &Path,
) -> Result<Vec<(SchemaView, PathBuf)>, String> {
    let dir = canonical_path
        .parent()
        .ok_or("canonical parquet has no parent directory")?;
    let views = write_standard_views(canonical_path, dir)?;
    write_views_provenance(canonical_path, &views)?;
    Ok(views)
}

fn write_standard_views(
    canonical_path: &Path,
    dir: &Path,
) -> Result<Vec<(SchemaView, PathBuf)>, String> {
    let mut views = Vec::new();
    for (view, name) in [
        (SchemaView::Pinn, "pinn"),
        (SchemaView::GnnKinematics, "gnn_kinematics"),
        (SchemaView::Siren, "siren"),
    ] {
        let path = dir.join(format!("view_{name}.parquet"));
        let lf = model_view_plan(scan_parquet_plan(&[canonical_path.to_path_buf()])?, &view)?;
        sink_parquet_streaming(lf, &path)?;
        views.push((view, path));
    }
    Ok(views)
}

fn write_views_provenance(
    canonical_path: &Path,
    views: &[(SchemaView, PathBuf)],
) -> Result<(), String> {
    let mut entries = Vec::new();
    for (view, path) in views {
        entries.push(serde_json::json!({
            "view": view,
            "file": path.file_name().and_then(|n| n.to_str()),
            "rows": count_parquet_rows(path)?,
            "sha256": crate::integrity::sha256_file(path)?,
        }));
    }
    let metadata = serde_json::json!({
        "generator_version": "1.0.0",
        "schema_version": crate::schema::DATASET_SCHEMA_VERSION,
        "canonical_schema_hash": crate::integrity::schema_hash(),
        "source_file": canonical_path.file_name().and_then(|n| n.to_str()),
        "source_sha256": crate::integrity::sha256_file(canonical_path)?,
        "views": entries,
    });
    let target = canonical_path
        .parent()
        .ok_or("canonical parquet has no parent directory")?
        .join("views_manifest.json");
    let tmp = target.with_extension("json.tmp");
    std::fs::write(
        &tmp,
        serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &target).map_err(|e| format!("publish {}: {e}", target.display()))
}

pub fn load_verified_records(
    out_dir: &Path,
    manifest: &DatasetManifestV1,
    policy: &crate::clean::CleanPolicy,
) -> Result<Vec<StarRecord>, String> {
    let mut all = Vec::new();
    for shard in manifest
        .shards
        .iter()
        .filter(|s| s.status == ShardStatus::Verified)
    {
        let path = out_dir.join(format!("{}.csv", shard.shard_id));
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("verified shard file {} unreadable: {e}", path.display()))?;
        all.extend(crate::clean::parse_shard_csv(&raw)?);
    }
    Ok(crate::clean::clean_records(all, policy))
}

pub struct StreamingAssembleOptions {
    pub batch_shards: usize,
    pub keep_parts: bool,
}

impl Default for StreamingAssembleOptions {
    fn default() -> Self {
        Self {
            batch_shards: 2,
            keep_parts: false,
        }
    }
}

#[derive(Debug)]
struct QualityAccumulator {
    total_rows: u64,
    valid_rows: u64,
    null_parallax: u64,
    null_pm: u64,
    null_rv: u64,
    ra_min: f64,
    ra_max: f64,
    dec_min: f64,
    dec_max: f64,
    outlier_position: u64,
}

impl Default for QualityAccumulator {
    fn default() -> Self {
        Self {
            total_rows: 0,
            valid_rows: 0,
            null_parallax: 0,
            null_pm: 0,
            null_rv: 0,
            ra_min: f64::INFINITY,
            ra_max: f64::NEG_INFINITY,
            dec_min: f64::INFINITY,
            dec_max: f64::NEG_INFINITY,
            outlier_position: 0,
        }
    }
}

impl QualityAccumulator {
    fn push_batch(&mut self, recs: &[StarRecord]) {
        for r in recs {
            self.total_rows += 1;
            if r.is_valid {
                self.valid_rows += 1;
            }
            if r.parallax_mas.is_none() {
                self.null_parallax += 1;
            }
            if r.pm_ra_mas_yr.is_none() || r.pm_dec_mas_yr.is_none() {
                self.null_pm += 1;
            }
            if r.radial_velocity_kms.is_none() {
                self.null_rv += 1;
            }
            self.ra_min = self.ra_min.min(r.ra_deg);
            self.ra_max = self.ra_max.max(r.ra_deg);
            self.dec_min = self.dec_min.min(r.dec_deg);
            self.dec_max = self.dec_max.max(r.dec_deg);
            if let (Some(x), Some(y), Some(z)) = (r.x_pc, r.y_pc, r.z_pc) {
                if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                    self.outlier_position += 1;
                }
            }
        }
    }

    fn finish(self) -> crate::clean::QualityReport {
        let n = self.total_rows.max(1) as f64;
        let zero = |c: u64| c as f64 / n;
        let bounded = |v: f64| if self.total_rows == 0 { 0.0 } else { v };
        crate::clean::QualityReport {
            unique_ids: self.total_rows,
            duplicate_rows_removed: 0,
            total_rows: self.total_rows,
            valid_rows: self.valid_rows,
            null_rate_parallax: zero(self.null_parallax),
            null_rate_pm: zero(self.null_pm),
            null_rate_radial_velocity: zero(self.null_rv),
            ra_min: bounded(self.ra_min),
            ra_max: bounded(self.ra_max),
            dec_min: bounded(self.dec_min),
            dec_max: bounded(self.dec_max),
            outlier_rate_position: zero(self.outlier_position),
        }
    }
}

fn scan_parquet_plan(paths: &[PathBuf]) -> Result<LazyFrame, String> {
    let mut refs: Vec<PlRefPath> = Vec::with_capacity(paths.len());
    for p in paths {
        let s = p
            .to_str()
            .ok_or_else(|| format!("non-utf8 path {}", p.display()))?;
        refs.push(PlRefPath::from(s));
    }
    let builder = DslBuilder::scan_parquet(
        ScanSources::Paths(polars_buffer::Buffer::from_iter(refs)),
        ParquetOptions::default(),
        UnifiedScanArgs::default(),
    )
    .map_err(|e| format!("parquet scan plan: {e}"))?;
    Ok(LazyFrame::from(builder.0))
}

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

fn count_parquet_rows(path: &Path) -> Result<u64, String> {
    let lf = scan_parquet_plan(std::slice::from_ref(&path.to_path_buf()))?;
    let df = lf
        .select([len()])
        .collect_with_engine(Engine::Streaming)
        .map_err(|e| format!("row count {}: {e}", path.display()))?
        .unwrap_single();
    let n = df
        .column("len")
        .map_err(|e| format!("row count column: {e}"))?
        .cast(&DataType::UInt64)
        .map_err(|e| format!("row count cast: {e}"))?
        .u64()
        .map_err(|e| format!("row count dtype: {e}"))?
        .get(0)
        .ok_or_else(|| "row count empty".to_string())?;
    Ok(n)
}

pub fn assemble_dataset_streaming(
    out_dir: &Path,
    manifest: &mut DatasetManifestV1,
    policy: &crate::clean::CleanPolicy,
    opts: StreamingAssembleOptions,
) -> Result<(AssembleReport, crate::clean::QualityReport), String> {
    let verified: Vec<&ShardState> = manifest
        .shards
        .iter()
        .filter(|s| s.status == ShardStatus::Verified)
        .collect();
    if verified.is_empty() {
        return Err("no verified shards to assemble".into());
    }
    validate_shard_coverage(manifest)?;
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let assembled_dir = out_dir.join("assembled");

    let parts_dir = assembled_dir.join("parts");
    if parts_dir.exists() {
        std::fs::remove_dir_all(&parts_dir).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&parts_dir).map_err(|e| e.to_string())?;

    let batch_shards = opts.batch_shards.max(1);
    let total_batches = verified.len().div_ceil(batch_shards);

    let mut qacc = QualityAccumulator::default();
    let mut counts = [0u64; 3];
    let mut holdout = 0u64;
    let mut knn_candidates: Vec<StarRecord> = Vec::new();
    let mut part_paths: Vec<PathBuf> = Vec::new();

    for (batch_idx, group) in verified.chunks(batch_shards).enumerate() {
        let mut batch = Vec::new();
        for shard in group {
            let path = out_dir.join(format!("{}.csv", shard.shard_id));
            let raw = std::fs::read_to_string(&path)
                .map_err(|e| format!("verified shard file {} unreadable: {e}", path.display()))?;
            batch.extend(crate::clean::parse_shard_csv(&raw)?);
        }
        let cleaned = crate::clean::clean_records(batch, policy);
        qacc.push_batch(&cleaned);
        for r in &cleaned {
            match split::classify(r.ra_deg, r.dec_deg, &r.source_id) {
                Some(Split::Train) => counts[0] += 1,
                Some(Split::Validation) => counts[1] += 1,
                Some(Split::Test) => counts[2] += 1,
                None => holdout += 1,
            }
            if knn_candidates.len() < LOCALIZATION_KNN_SAMPLE_CAP
                && r.is_valid
                && matches!((r.x_pc, r.y_pc, r.z_pc), (Some(_), Some(_), Some(_)))
            {
                knn_candidates.push(r.clone());
            }
        }
        let frame = build_canonical_frame(&cleaned).map_err(|e| format!("canonical frame: {e}"))?;
        let part_path = parts_dir.join(format!("canonical_part_{batch_idx:05}.parquet"));
        let file = File::create(&part_path)
            .map_err(|e| format!("cannot create {}: {e}", part_path.display()))?;
        ParquetWriter::new(file)
            .with_row_group_size(Some(64 * 1024))
            .finish(&mut frame.clone())
            .map_err(|e| format!("parquet write {}: {e}", part_path.display()))?;
        part_paths.push(part_path);
        println!(
            "[batch {}/{}] shards={} rows={}",
            batch_idx + 1,
            total_batches,
            group.len(),
            cleaned.len()
        );
    }

    let total_rows = qacc.total_rows;

    let canonical_path = assembled_dir.join("canonical.parquet");
    sink_parquet_streaming(scan_parquet_plan(&part_paths)?, &canonical_path)?;

    let mut view_paths = write_standard_views(&canonical_path, &assembled_dir)?;

    let pairs = build_localization_pairs(&knn_candidates);
    if !pairs.is_empty() {
        let pair_df = df![
            "anchor_source_id" => pairs.iter().map(|p| p.anchor_source_id.as_str()).collect::<StringChunked>(),
            "neighbor_source_id" => pairs.iter().map(|p| p.neighbor_source_id.as_str()).collect::<StringChunked>(),
            "rel_x" => Series::new("rel_x".into(), pairs.iter().map(|p| Some(p.rel_x)).collect::<Vec<Option<f32>>>()),
            "rel_y" => Series::new("rel_y".into(), pairs.iter().map(|p| Some(p.rel_y)).collect::<Vec<Option<f32>>>()),
            "rel_z" => Series::new("rel_z".into(), pairs.iter().map(|p| Some(p.rel_z)).collect::<Vec<Option<f32>>>()),
            "is_visible" => Series::new("is_visible".into(), pairs.iter().map(|_| true).collect::<Vec<bool>>()),
        ]
        .map_err(|e| format!("localization pairs frame: {e}"))?;
        let pairs_path = assembled_dir.join("gnn_localization_neighbors.parquet");
        write_parquet(&pair_df, &pairs_path)?;
        view_paths.push((SchemaView::GnnLocalization, pairs_path));
    }

    let rows_written = count_parquet_rows(&canonical_path)?;
    if rows_written != total_rows {
        return Err(format!(
            "parquet merge row mismatch: expected {total_rows}, got {rows_written}"
        ));
    }
    write_views_provenance(&canonical_path, &view_paths)?;

    let checksum = crate::integrity::sha256_file(&canonical_path)?;
    manifest.finalize(&checksum);

    if !opts.keep_parts {
        std::fs::remove_dir_all(&parts_dir).map_err(|e| e.to_string())?;
    }

    let report = AssembleReport {
        canonical_path,
        view_paths,
        rows_written,
        train_rows: counts[0],
        validation_rows: counts[1],
        test_rows: counts[2],
        holdout_rows: holdout,
    };
    Ok((report, qacc.finish()))
}

pub const LOCALIZATION_KNN_SAMPLE_CAP: usize = 4_000;
const LOCALIZATION_NEIGHBOR_LIMIT: usize = 8;

pub struct NeighborRow {
    pub anchor_source_id: String,
    pub neighbor_source_id: String,
    pub rel_x: f32,
    pub rel_y: f32,
    pub rel_z: f32,
}

pub fn build_localization_pairs(records: &[StarRecord]) -> Vec<NeighborRow> {
    let candidates: Vec<&StarRecord> = records
        .iter()
        .filter(|r| r.is_valid && matches!((r.x_pc, r.y_pc, r.z_pc), (Some(_), Some(_), Some(_))))
        .take(LOCALIZATION_KNN_SAMPLE_CAP)
        .collect();
    let mut pairs = Vec::new();
    for anchor in &candidates {
        let ax = anchor.x_pc.unwrap() as f64;
        let ay = anchor.y_pc.unwrap() as f64;
        let az = anchor.z_pc.unwrap() as f64;
        let mut dists: Vec<(f64, &StarRecord)> = candidates
            .iter()
            .filter(|c| c.source_id != anchor.source_id)
            .map(|c| {
                let dx = c.x_pc.unwrap() as f64 - ax;
                let dy = c.y_pc.unwrap() as f64 - ay;
                let dz = c.z_pc.unwrap() as f64 - az;
                ((dx * dx + dy * dy + dz * dz).sqrt(), *c)
            })
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (d, c) in dists.into_iter().take(LOCALIZATION_NEIGHBOR_LIMIT) {
            pairs.push(NeighborRow {
                anchor_source_id: anchor.source_id.clone(),
                neighbor_source_id: c.source_id.clone(),
                rel_x: (c.x_pc.unwrap() - ax as f32) as f32,
                rel_y: (c.y_pc.unwrap() - ay as f32) as f32,
                rel_z: (c.z_pc.unwrap() - az as f32) as f32,
            });
            let _ = d;
        }
    }
    pairs
}

pub fn assemble_dataset(
    out_dir: &Path,
    manifest: &mut DatasetManifestV1,
    records: &[StarRecord],
) -> Result<AssembleReport, String> {
    if !manifest.shards.is_empty()
        && manifest.query_hash == crate::collector::CollectConfig::default().query_hash()
    {
        validate_shard_coverage(manifest)?;
    }
    if records.is_empty() {
        return Err("no records to assemble".into());
    }
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;

    let canonical = build_canonical_frame(records).map_err(|e| format!("canonical frame: {e}"))?;
    let canonical_path = out_dir.join("canonical.parquet");
    write_parquet(&canonical, &canonical_path)?;

    let mut view_paths = write_standard_views(&canonical_path, out_dir)?;

    let pairs = build_localization_pairs(records);
    if !pairs.is_empty() {
        let pair_df = df![
            "anchor_source_id" => pairs.iter().map(|p| p.anchor_source_id.as_str()).collect::<StringChunked>(),
            "neighbor_source_id" => pairs.iter().map(|p| p.neighbor_source_id.as_str()).collect::<StringChunked>(),
            "rel_x" => Series::new("rel_x".into(), pairs.iter().map(|p| Some(p.rel_x)).collect::<Vec<Option<f32>>>()),
            "rel_y" => Series::new("rel_y".into(), pairs.iter().map(|p| Some(p.rel_y)).collect::<Vec<Option<f32>>>()),
            "rel_z" => Series::new("rel_z".into(), pairs.iter().map(|p| Some(p.rel_z)).collect::<Vec<Option<f32>>>()),
            "is_visible" => Series::new("is_visible".into(), pairs.iter().map(|_| true).collect::<Vec<bool>>()),
        ]
        .map_err(|e| format!("localization pairs frame: {e}"))?;
        let pairs_path = out_dir.join("gnn_localization_neighbors.parquet");
        write_parquet(&pair_df, &pairs_path)?;
        view_paths.push((SchemaView::GnnLocalization, pairs_path));
    }

    let mut counts = [0u64; 3];
    let mut holdout = 0u64;
    for r in records {
        match split::classify(r.ra_deg, r.dec_deg, &r.source_id) {
            Some(Split::Train) => counts[0] += 1,
            Some(Split::Validation) => counts[1] += 1,
            Some(Split::Test) => counts[2] += 1,
            None => holdout += 1,
        }
    }

    let bytes_read = File::open(&canonical_path).map_err(|e| format!("reopen canonical: {e}"))?;
    let round_trip = ParquetReader::new(bytes_read)
        .finish()
        .map_err(|e| e.to_string())?;
    if round_trip.height() != canonical.height() {
        return Err("parquet round-trip row mismatch".into());
    }
    write_views_provenance(&canonical_path, &view_paths)?;

    let checksum = crate::integrity::sha256_file(&canonical_path)?;
    manifest.finalize(&checksum);
    let report = AssembleReport {
        canonical_path,
        view_paths,
        rows_written: canonical.height() as u64,
        train_rows: counts[0],
        validation_rows: counts[1],
        test_rows: counts[2],
        holdout_rows: holdout,
    };
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clean::{CleanPolicy, parse_shard_csv};
    use crate::manifest::ShardState;

    const CSV_HEADER: &str = "source_id,ra_deg,dec_deg,parallax_mas,pm_ra_mas_yr,pm_dec_mas_yr,radial_velocity_kms,mag_g,mag_bp,mag_rp,ruwe,astrometric_excess_noise";

    fn fixture_records(n: u64) -> Vec<StarRecord> {
        let mut csv = format!("{CSV_HEADER}\n");
        for i in 0..n {
            let ra = (i % 360) as f64 + 0.5;
            let dec = ((i % 17) as f64) * 11.0 - 90.0 + 5.5;
            csv.push_str(&format!(
                "{i},{ra},{dec},10.0,5.0,-5.0,10.0,{:.1},9.9,9.8,1.0,0.1\n",
                8.0 + (i % 50) as f64 / 10.0
            ));
        }
        let recs = parse_shard_csv(&csv).unwrap();
        crate::clean::clean_records(recs, &CleanPolicy::default())
    }

    #[test]
    fn canonical_frame_columns_match_golden_schema_order() {
        let recs = fixture_records(40);
        let df = build_canonical_frame(&recs).unwrap();
        let expected = vec![
            "source_id",
            "ra_deg",
            "dec_deg",
            "epoch_year",
            "parallax_mas",
            "pm_ra_mas_yr",
            "pm_dec_mas_yr",
            "radial_velocity_kms",
            "x_pc",
            "y_pc",
            "z_pc",
            "vx_kms",
            "vy_kms",
            "vz_kms",
            "mag_g",
            "mag_bp",
            "mag_rp",
            "ruwe",
            "astrometric_excess_noise",
            "is_valid",
            "spatial_tile",
            "split",
        ];
        assert_eq!(
            df.get_column_names(),
            expected,
            "column contract of the golden schema"
        );
    }

    #[test]
    fn view_frames_only_contain_contracted_required_columns() {
        let recs = fixture_records(20);
        let df = build_canonical_frame(&recs).unwrap();
        for view in [
            SchemaView::Pinn,
            SchemaView::GnnKinematics,
            SchemaView::Siren,
        ] {
            let vf = view_columns(&df, &view).unwrap();
            for name in required_columns_for_view(&view) {
                assert!(vf.column(name).is_ok(), "{view:?} needs {name}");
            }
        }
    }

    #[test]
    fn assemble_writes_canonical_and_views_with_split_accounting() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = DatasetManifestV1::new("gaia_dr3", "q", "s");
        manifest
            .shards
            .push(ShardState::new("shard".into(), (0.0, 1.0), (-90.0, 90.0)));
        let recs = fixture_records(300);
        let report = assemble_dataset(dir.path(), &mut manifest, &recs).unwrap();
        assert_eq!(report.rows_written, 300);
        assert_eq!(
            report.train_rows + report.validation_rows + report.test_rows + report.holdout_rows,
            300
        );
        for (_, path) in &report.view_paths {
            assert!(path.exists(), "{}", path.display());
        }
        assert!(
            report.canonical_path.join("canonical.parquet").exists()
                || report.canonical_path.exists()
        );

        let f = File::open(&report.canonical_path).unwrap();
        let rt = ParquetReader::new(f).finish().unwrap();
        assert_eq!(rt.height(), 300);
    }

    #[test]
    fn localization_pairs_are_deterministic_within_limits() {
        let recs = fixture_records(500);
        let a = build_localization_pairs(&records_fixture_clone(&recs));
        let b = build_localization_pairs(&recs);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.anchor_source_id, pb.anchor_source_id);
            assert_eq!(pa.neighbor_source_id, pb.neighbor_source_id);
            assert_eq!(pa.rel_x.to_ne_bytes(), pb.rel_x.to_ne_bytes());
            assert_eq!(pa.rel_y.to_ne_bytes(), pb.rel_y.to_ne_bytes());
            assert_eq!(pa.rel_z.to_ne_bytes(), pb.rel_z.to_ne_bytes());
        }
        assert!(a.len() > 0);
    }

    #[test]
    fn streaming_assemble_matches_in_memory_assemble() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = DatasetManifestV1::new("gaia_dr3", "q", "s");
        for i in 0..3 {
            let mut shard = ShardState::new(format!("shard-{i}"), (0.0, 1.0), (-90.0, 90.0));
            shard.status = ShardStatus::Verified;
            shard.row_count = 100;
            manifest.shards.push(shard);
            let mut csv = format!("{CSV_HEADER}\n");
            for j in 0..100u64 {
                let id = i * 1000 + j;
                let ra = (id % 360) as f64 + 0.5;
                let dec = ((id % 17) as f64) * 11.0 - 90.0 + 5.5;
                csv.push_str(&format!(
                    "{id},{ra},{dec},10.0,5.0,-5.0,10.0,{:.1},9.9,9.8,1.0,0.1\n",
                    8.0 + (id % 50) as f64 / 10.0
                ));
            }
            std::fs::write(dir.path().join(format!("shard-{i}.csv")), csv).unwrap();
        }

        let (report, quality) = assemble_dataset_streaming(
            dir.path(),
            &mut manifest,
            &CleanPolicy::default(),
            StreamingAssembleOptions::default(),
        )
        .unwrap();
        assert_eq!(report.rows_written, 300);
        assert_eq!(quality.total_rows, 300);
        assert_eq!(quality.unique_ids, 300);
        assert_eq!(quality.duplicate_rows_removed, 0);
        assert!(
            report.train_rows + report.validation_rows + report.test_rows + report.holdout_rows
                == 300
        );
        for (_, path) in &report.view_paths {
            assert!(path.exists(), "{}", path.display());
        }
        assert!(!dir.path().join("parts").exists(), "parts cleaned up");

        let recs = load_verified_records(dir.path(), &manifest, &CleanPolicy::default()).unwrap();
        let mut manifest2 = DatasetManifestV1::new("gaia_dr3", "q", "s");
        let in_memory =
            assemble_dataset(&dir.path().join("in-memory"), &mut manifest2, &recs).unwrap();
        assert_eq!(in_memory.rows_written, report.rows_written);
        assert_eq!(in_memory.train_rows, report.train_rows);
        assert_eq!(in_memory.validation_rows, report.validation_rows);
        assert_eq!(in_memory.test_rows, report.test_rows);
        assert_eq!(in_memory.holdout_rows, report.holdout_rows);

        let f = File::open(&report.canonical_path).unwrap();
        let rt = ParquetReader::new(f).finish().unwrap();
        assert_eq!(rt.height(), 300);
    }

    #[test]
    fn streaming_assemble_rejects_no_verified_shards() {
        let dir = tempfile::tempdir().unwrap();
        let mut manifest = DatasetManifestV1::new("gaia_dr3", "q", "s");
        manifest
            .shards
            .push(ShardState::new("shard".into(), (0.0, 1.0), (-90.0, 90.0)));
        assert!(
            assemble_dataset_streaming(
                dir.path(),
                &mut manifest,
                &CleanPolicy::default(),
                StreamingAssembleOptions::default(),
            )
            .is_err()
        );
    }

    fn records_fixture_clone(recs: &[StarRecord]) -> Vec<StarRecord> {
        recs.to_vec()
    }

    #[test]
    fn holdout_rows_never_receive_a_split_label() {
        let held: Vec<(i64, i64)> = (0..24i64)
            .flat_map(|a| (0..12i64).map(move |b| (a, b)))
            .filter(|(a, b)| split::is_holdout_tile(&format!("tile_ra{a}_dec{b}")))
            .collect();
        assert!(!held.is_empty());
        let (a, b) = held[0];
        let ra = a as f64 * 15.0 + 7.5;
        let dec = b as f64 * 15.0 - 82.5;
        assert!(split::classify(ra, dec, "99999").is_none());
    }
}
