//! Regenerates the frozen `stellar-e2e-v1` fixture byte-identically:
//! `cargo run -p lnai-training --example gen-fixture`.
//!
//! Writes `ai/fixtures/stellar-e2e-v1/{fixture.parquet,manifest.json}`.
//! The manifest records the SHA-256 of the parquet bytes, the spatial tiles
//! and every source ID so CI can prove fixture integrity and train/eval
//! disjointness (Stage 3, tasks 1-2).

use polars::prelude::*;
use std::fs::File;
use std::path::Path;

use lnai_training::e2e::{
    FIXTURE_SCHEMA_VERSION, FixtureManifestV1, SpatialTile, fixture_dir, sha256_hex,
};

const ROWS_PER_TILE: usize = 64;
pub const FIXTURE_SEED: u64 = 0x05E2E101;

/// Deterministic LCG; no external rand dependency needed for reproducibility.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f32 / (1u64 << 53) as f32
    }
}

fn schema_hash() -> String {
    // Deterministic digest of the canonical schema definition (names, types,
    // nullability, units). Any structural change alters the fixture manifest.
    let mut buf: Vec<u8> = Vec::new();
    for column in lnai_data::schema::canonical_columns() {
        buf.extend_from_slice(column.name.as_bytes());
        buf.extend_from_slice(format!("{:?}", column.data_type).as_bytes());
        buf.push(u8::from(column.nullable));
        if let Some(units) = column.units {
            buf.extend_from_slice(units.as_bytes());
        }
        buf.push(0xFF);
    }
    sha256_hex(&buf)
}

fn tiles() -> Vec<SpatialTile> {
    [0.0_f32, 90.0, 180.0, 270.0]
        .into_iter()
        .map(|start| SpatialTile {
            id: format!("ra_{start:03}_{:03}", start + 90.0),
            ra_range: (start, start + 90.0),
            dec_range: (-90.0, 90.0),
        })
        .collect()
}

struct StarRow {
    source_id: String,
    ra_deg: f64,
    dec_deg: f64,
    parallax_mas: f64,
    pm_ra_mas_yr: f64,
    pm_dec_mas_yr: f64,
    radial_velocity_kms: f64,
    vx_kms: f32,
    vy_kms: f32,
    vz_kms: f32,
    mag_g: f32,
    mag_bp: f32,
    mag_rp: f32,
    ruwe: f32,
    astrometric_excess_noise: f32,
}

fn generate_rows() -> (Vec<StarRow>, Vec<SpatialTile>) {
    let tile_defs = tiles();
    let mut rng = Lcg(FIXTURE_SEED);
    let mut rows = Vec::with_capacity(tile_defs.len() * ROWS_PER_TILE);
    let mut index = 0_u64;

    for tile in &tile_defs {
        let ra_span = tile.ra_range.1 - tile.ra_range.0;
        for _ in 0..ROWS_PER_TILE {
            let ra = f64::from(tile.ra_range.0 + rng.next_f32() * ra_span);
            let dec = f64::from(-90.0 + rng.next_f32() * 180.0);
            let vx = 40.0 * rng.next_f32() - 20.0;
            let vy = 40.0 * rng.next_f32() - 20.0;
            let vz = 40.0 * rng.next_f32() - 20.0;
            rows.push(StarRow {
                source_id: format!("e2e-v1-{index:06}"),
                ra_deg: ra,
                dec_deg: dec,
                parallax_mas: 0.1 + f64::from(rng.next_f32()) * 9.9,
                pm_ra_mas_yr: f64::from(rng.next_f32() * 20.0 - 10.0),
                pm_dec_mas_yr: f64::from(rng.next_f32() * 20.0 - 10.0),
                radial_velocity_kms: f64::from(rng.next_f32() * 120.0 - 60.0),
                vx_kms: vx,
                vy_kms: vy,
                vz_kms: vz,
                mag_g: 6.0 + rng.next_f32() * 12.0,
                mag_bp: 7.0 + rng.next_f32() * 12.0,
                mag_rp: 6.5 + rng.next_f32() * 11.0,
                ruwe: rng.next_f32() * 1.4,
                astrometric_excess_noise: rng.next_f32(),
            });
            index += 1;
        }
    }
    (rows, tile_defs)
}

fn write_parquet(rows: &[StarRow], path: &Path) -> anyhow_result::Result<()> {
    let n = rows.len();
    let df = DataFrame::new(
        n,
        vec![
            Column::new(
                "source_id".into(),
                rows.iter().map(|r| r.source_id.clone()).collect::<Vec<_>>(),
            ),
            Column::new(
                "ra_deg".into(),
                rows.iter().map(|r| r.ra_deg).collect::<Vec<_>>(),
            ),
            Column::new(
                "dec_deg".into(),
                rows.iter().map(|r| r.dec_deg).collect::<Vec<_>>(),
            ),
            Column::new("epoch_year".into(), vec![2000.0_f64; n]),
            Column::new(
                "parallax_mas".into(),
                rows.iter().map(|r| r.parallax_mas).collect::<Vec<_>>(),
            ),
            Column::new(
                "pm_ra_mas_yr".into(),
                rows.iter().map(|r| r.pm_ra_mas_yr).collect::<Vec<_>>(),
            ),
            Column::new(
                "pm_dec_mas_yr".into(),
                rows.iter().map(|r| r.pm_dec_mas_yr).collect::<Vec<_>>(),
            ),
            Column::new(
                "radial_velocity_kms".into(),
                rows.iter()
                    .map(|r| r.radial_velocity_kms)
                    .collect::<Vec<_>>(),
            ),
            Column::new(
                "vx_kms".into(),
                rows.iter().map(|r| r.vx_kms).collect::<Vec<_>>(),
            ),
            Column::new(
                "vy_kms".into(),
                rows.iter().map(|r| r.vy_kms).collect::<Vec<_>>(),
            ),
            Column::new(
                "vz_kms".into(),
                rows.iter().map(|r| r.vz_kms).collect::<Vec<_>>(),
            ),
            Column::new(
                "mag_g".into(),
                rows.iter().map(|r| r.mag_g).collect::<Vec<_>>(),
            ),
            Column::new(
                "mag_bp".into(),
                rows.iter().map(|r| r.mag_bp).collect::<Vec<_>>(),
            ),
            Column::new(
                "mag_rp".into(),
                rows.iter().map(|r| r.mag_rp).collect::<Vec<_>>(),
            ),
            Column::new(
                "ruwe".into(),
                rows.iter().map(|r| r.ruwe).collect::<Vec<_>>(),
            ),
            Column::new(
                "astrometric_excess_noise".into(),
                rows.iter()
                    .map(|r| r.astrometric_excess_noise)
                    .collect::<Vec<_>>(),
            ),
            Column::new("is_valid".into(), vec![true; n]),
        ],
    )?;

    let file = File::create(path)?;
    ParquetWriter::new(file).finish(&mut df.clone())?;
    drop(df);
    Ok(())
}

mod anyhow_result {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}

fn main() -> anyhow_result::Result<()> {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir)?;

    let (rows, tile_defs) = generate_rows();
    let parquet_path = dir.join("fixture.parquet");
    write_parquet(&rows, &parquet_path)?;

    let checksum = sha256_hex(&std::fs::read(&parquet_path)?);
    let manifest = FixtureManifestV1 {
        version: FIXTURE_SCHEMA_VERSION.into(),
        schema_hash: schema_hash(),
        source_release: "synthetic-stellar-e2e-v1".into(),
        row_count: rows.len() as u64,
        checksum,
        checksum_algorithm: "sha256".into(),
        seed: FIXTURE_SEED,
        spatial_tiles: tile_defs,
        source_ids: rows.iter().map(|r| r.source_id.clone()).collect(),
    };
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;

    println!(
        "fixture written: {} rows, checksum {}",
        manifest.row_count,
        &manifest.checksum[..16]
    );
    Ok(())
}
