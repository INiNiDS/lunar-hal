//! Stage 3, tasks 1-2: frozen fixture integrity and train/eval leakage proof.

use lnai_training::e2e::{FixtureManifestV1, SpatialTile, assert_no_leakage, fixture_dir};
use polars::prelude::*;

fn load_fixture() -> FixtureManifestV1 {
    FixtureManifestV1::load(&fixture_dir()).expect("fixture manifest must parse")
}

#[test]
fn committed_parquet_matches_manifest_checksum() {
    let manifest = load_fixture();
    assert_eq!(manifest.version, "1.0.0");
    assert_eq!(manifest.checksum_algorithm, "sha256");
    assert_eq!(
        manifest.row_count as usize,
        manifest.source_ids.len(),
        "manifest row_count must match source id list"
    );
    manifest
        .verify_checksum(&fixture_dir())
        .expect("committed parquet must match manifest checksum");
}

#[test]
fn parquet_source_ids_and_tiles_match_manifest() {
    let manifest = load_fixture();

    let file = std::fs::File::open(fixture_dir().join("fixture.parquet")).unwrap();
    let df = ParquetReader::new(file).finish().unwrap();

    let ids = df
        .column("source_id")
        .unwrap()
        .str()
        .unwrap()
        .into_no_null_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), manifest.row_count as usize);
    assert_eq!(
        ids, manifest.source_ids,
        "parquet id order must match manifest"
    );

    let ra = df.column("ra_deg").unwrap().f64().unwrap();
    let dec = df.column("dec_deg").unwrap().f64().unwrap();

    // Every row must land in exactly one declared tile.
    for i in 0..ids.len() {
        let matches: Vec<&SpatialTile> = manifest
            .spatial_tiles
            .iter()
            .filter(|t| t.contains(ra.get(i).unwrap() as f32, dec.get(i).unwrap() as f32))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "row {i} must belong to exactly one spatial tile"
        );
    }
}

#[test]
fn leakage_guard_flags_train_tiles_inside_fixture_coverage() {
    let fixture = load_fixture();

    // stellar-e2e-v1 intentionally spans the full sphere, so any real
    // training catalog MUST declare disjoint tiles; the guard has to flag a
    // naive train manifest that reuses fixture sky regions.
    let train = FixtureManifestV1 {
        version: "1.0.0".into(),
        schema_hash: fixture.schema_hash.clone(),
        source_release: "gaia_dr3-train".into(),
        row_count: 2,
        checksum: String::new(),
        checksum_algorithm: "sha256".into(),
        seed: 1,
        spatial_tiles: vec![
            SpatialTile {
                id: "train_ra_000_090".into(),
                ra_range: (0.0, 90.0),
                dec_range: (-90.0, 90.0),
            },
            SpatialTile {
                id: "train_ra_180_270".into(),
                ra_range: (180.0, 270.0),
                dec_range: (-90.0, 90.0),
            },
        ],
        source_ids: vec!["gaia-0001".into(), "gaia-0002".into()],
    };

    let err =
        assert_no_leakage(&fixture, &train).expect_err("overlapping sky coverage must be reported");
    assert!(err.contains("spatial tile leakage"), "{err}");
}
