//! Integration: deterministic hash split + spatial-tile holdout must never
//! leak holdout geography into train/validation/test artifacts.

use lnai_data::clean::{CleanPolicy, clean_records, parse_shard_csv};
use lnai_data::manifest::{DatasetManifestV1, ShardState};
use lnai_data::schema::{SchemaView, required_columns_for_view};
use lnai_data::split;
use std::collections::HashSet;

const CSV_HEADER: &str = "source_id,ra_deg,dec_deg,parallax_mas,pm_ra_mas_yr,pm_dec_mas_yr,radial_velocity_kms,mag_g,mag_bp,mag_rp,ruwe,astrometric_excess_noise";

/// Synthetic sky covering all RA strips and Dec bands so every spatial tile
/// gets populated.
fn fixture_csv(rows_per_band: u64) -> String {
    let mut csv = format!("{CSV_HEADER}\n");
    let mut id = 0u64;
    for band in 0..12i64 {
        for k in 0..rows_per_band {
            let ra = (k % 24) as f64 * 15.0 + (band % 15) as f64 + 3.5;
            let dec = band as f64 * 15.0 - 82.5;
            csv.push_str(&format!(
                "{id},{ra},{dec},10.0,5.0,-5.0,10.0,{:.1},9.9,9.8,1.0,0.1\n",
                8.0 + id as f64 % 40.0 / 8.0
            ));
            id += 1;
        }
    }
    csv
}

#[test]
fn assembled_views_respect_hash_split_and_holdout_exclusion() {
    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("dataset");

    let raw = parse_shard_csv(&fixture_csv(64)).unwrap();
    let records = clean_records(raw, &CleanPolicy::default());
    assert!(records.len() >= 280);

    let mut manifest =
        DatasetManifestV1::new("gaia_dr3", "q", &lnai_data::integrity::schema_hash());
    manifest.shards.push(ShardState::new(
        "synthetic".into(),
        (0.0, 24.0),
        (-90.0, 90.0),
    ));

    let report = lnai_data::assemble::assemble_dataset(&out_dir, &mut manifest, &records).unwrap();
    assert_eq!(report.rows_written, records.len() as u64);

    // Accounting closes exactly and no double assignment happened.
    assert_eq!(
        report.train_rows + report.validation_rows + report.test_rows + report.holdout_rows,
        report.rows_written
    );
    assert!(report.holdout_rows > 0, "fixture spans whole sky");

    // Read canonical parquet back and validate split columns against pure logic.
    let file = std::fs::File::open(&report.canonical_path).unwrap();
    let df = polars_io_parquet(&file);
    let ids: Vec<String> = col_strings(&df, "source_id");
    let splits: Vec<Option<String>> = col_opt_strings(&df, "split");

    assert_eq!(ids.len(), records.len());
    let mut seen = HashSet::new();
    for (i, id) in ids.iter().enumerate() {
        assert!(seen.insert(id.clone()), "source ids must be unique");
        let rec = records.iter().find(|r| r.source_id == *id).unwrap();
        let expected = split::classify(rec.ra_deg, rec.dec_deg, id);
        match (&splits[i], expected) {
            (Some(s), Some(e)) => {
                assert_eq!(s.as_str(), e.as_str());
                assert!(!split::is_spatial_holdout(rec.ra_deg, rec.dec_deg));
            }
            (None, None) => {
                assert!(split::is_spatial_holdout(rec.ra_deg, rec.dec_deg));
            }
            other => panic!("split mismatch row {i}: {other:?}"),
        }
    }

    // View files contain only contracted columns (no leakage of non-required).
    for (view, path) in &report.view_paths {
        assert!(path.exists(), "{}", path.display());
        if matches!(
            view,
            SchemaView::Pinn | SchemaView::GnnKinematics | SchemaView::Siren
        ) {
            let f = std::fs::File::open(path).unwrap();
            let vdf = polars_io_parquet(&f);
            let got: HashSet<String> = vdf
                .get_column_names()
                .into_iter()
                .map(|s| s.as_str().to_string())
                .collect();
            for name in required_columns_for_view(view) {
                assert!(got.contains(name), "{view:?} missing {name}");
            }
        }
    }

    // Localization neighbors reference only real anchors from the dataset.
    let pairs_path = out_dir.join("gnn_localization_neighbors.parquet");
    assert!(pairs_path.exists());
    let pf = std::fs::File::open(pairs_path).unwrap();
    let pdf = polars_io_parquet(&pf);
    let anchors: Vec<String> = col_strings(&pdf, "anchor_source_id");
    let neighbors: Vec<String> = col_strings(&pdf, "neighbor_source_id");
    assert!(!anchors.is_empty());
    let known: HashSet<&str> = ids.iter().map(String::as_str).collect();
    for pair in anchors.iter().chain(neighbors.iter()) {
        assert!(known.contains(pair.as_str()), "unknown id {pair}");
    }
}

// --- small polars helpers kept local so the integration test stays standalone

use polars::prelude::*;

fn polars_io_parquet(file: &std::fs::File) -> DataFrame {
    ParquetReader::new(file.try_clone().expect("file clone"))
        .finish()
        .expect("parquet decode")
}

fn col_strings(df: &DataFrame, name: &str) -> Vec<String> {
    df.column(name)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .flatten()
        .map(|s| s.to_string())
        .collect()
}

fn col_opt_strings(df: &DataFrame, name: &str) -> Vec<Option<String>> {
    df.column(name)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|o| o.map(|s| s.to_string()))
        .collect()
}
