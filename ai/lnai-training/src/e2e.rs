//! E2E foundation: the frozen `stellar-e2e-v1` fixture contract.
//!
//! The fixture is a tiny synthetic stellar catalog committed under
//! `ai/fixtures/stellar-e2e-v1/` together with `manifest.json`. It is fully
//! deterministic (`examples/gen-fixture.rs` regenerates byte-identical
//! artifacts), so CI can verify integrity by recomputing the SHA-256 of the
//! parquet file and comparing source IDs / spatial tiles against the manifest.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Fixture schema version.
pub const FIXTURE_SCHEMA_VERSION: &str = "1.0.0";

/// Directory containing the committed fixture artifacts.
pub const FIXTURE_DIR: &str = "ai/fixtures/stellar-e2e-v1";

/// Spatial tile descriptor: a closed-open RA/Dec box in degrees.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SpatialTile {
    pub id: String,
    /// `[start, end)` in degrees.
    pub ra_range: (f32, f32),
    /// `[start, end)` in degrees.
    pub dec_range: (f32, f32),
}

impl SpatialTile {
    /// Returns true when the coordinates belong to this tile.
    pub fn contains(&self, ra_deg: f32, dec_deg: f32) -> bool {
        ra_deg >= self.ra_range.0
            && ra_deg < self.ra_range.1
            && dec_deg >= self.dec_range.0
            && dec_deg < self.dec_range.1
    }
}

/// Frozen manifest of `stellar-e2e-v1`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FixtureManifestV1 {
    pub version: String,
    /// Hash of the canonical dataset schema this fixture conforms to.
    pub schema_hash: String,
    pub source_release: String,
    pub row_count: u64,
    /// SHA-256 of `fixture.parquet` bytes.
    pub checksum: String,
    pub checksum_algorithm: String,
    /// Deterministic seed used to synthesize the rows.
    pub seed: u64,
    pub spatial_tiles: Vec<SpatialTile>,
    /// All source IDs in fixture order; enables exact leakage proofs.
    pub source_ids: Vec<String>,
}

impl FixtureManifestV1 {
    /// Loads and parses `manifest.json` from the fixture directory.
    pub fn load(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join("manifest.json");
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Recomputes the SHA-256 of `fixture.parquet` and checks it against the
    /// recorded checksum.
    pub fn verify_checksum(&self, dir: &Path) -> Result<(), String> {
        let bytes = std::fs::read(dir.join("fixture.parquet"))
            .map_err(|e| format!("failed to read fixture parquet: {e}"))?;
        let actual = sha256_hex(&bytes);
        if actual == self.checksum {
            Ok(())
        } else {
            Err(format!(
                "fixture checksum mismatch: manifest {}, actual {actual}",
                self.checksum
            ))
        }
    }
}

/// Workspace-rooted path of the fixture directory.
///
/// Resolved against the crate manifest so tests/benches/examples find the
/// artifacts regardless of the invocation working directory.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Workspace-rooted path of the fixture directory.
pub fn fixture_dir() -> PathBuf {
    workspace_root().join(FIXTURE_DIR)
}

/// SHA-256 as lowercase hex (single shared implementation for reports/fixtures).
pub fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Proof that an evaluation fixture shares nothing with training data:
/// no common source ID **and** no overlapping spatial tile.
pub fn assert_no_leakage(
    fixture: &FixtureManifestV1,
    train: &FixtureManifestV1,
) -> Result<(), String> {
    let train_ids: std::collections::HashSet<&String> = train.source_ids.iter().collect();
    let shared_ids: Vec<&String> = fixture
        .source_ids
        .iter()
        .filter(|id| train_ids.contains(id))
        .collect();
    if !shared_ids.is_empty() {
        return Err(format!(
            "source ID leakage detected: {} shared ids, e.g. {:?}",
            shared_ids.len(),
            &shared_ids[..shared_ids.len().min(3)]
        ));
    }

    for ftile in &fixture.spatial_tiles {
        for ttile in &train.spatial_tiles {
            let ra_overlap =
                ftile.ra_range.0 < ttile.ra_range.1 && ttile.ra_range.0 < ftile.ra_range.1;
            let dec_overlap =
                ftile.dec_range.0 < ttile.dec_range.1 && ttile.dec_range.0 < ftile.dec_range.1;
            if ra_overlap && dec_overlap {
                return Err(format!(
                    "spatial tile leakage: fixture tile '{}' overlaps train tile '{}'",
                    ftile.id, ttile.id
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(id_prefix: &str, tiles: Vec<(f32, f32)>) -> FixtureManifestV1 {
        FixtureManifestV1 {
            version: FIXTURE_SCHEMA_VERSION.into(),
            schema_hash: "schema".into(),
            source_release: "synthetic".into(),
            row_count: 2,
            checksum: "abc".into(),
            checksum_algorithm: "sha256".into(),
            seed: 7,
            spatial_tiles: tiles
                .into_iter()
                .enumerate()
                .map(|(i, (a, b))| SpatialTile {
                    id: format!("{id_prefix}-tile-{i}"),
                    ra_range: (a, b),
                    dec_range: (-90.0, 90.0),
                })
                .collect(),
            source_ids: vec![format!("{id_prefix}-1"), format!("{id_prefix}-2")],
        }
    }

    #[test]
    fn leakage_proof_accepts_disjoint_manifests() {
        let fixture = manifest("e2e", vec![(350.0, 360.0), (340.0, 350.0)]);
        let train = manifest("train", vec![(0.0, 10.0), (10.0, 20.0)]);
        assert!(assert_no_leakage(&fixture, &train).is_ok());
    }

    #[test]
    fn leakage_proof_detects_shared_source_id() {
        let mut fixture = manifest("e2e", vec![(0.0, 10.0)]);
        let train = manifest("train", vec![(180.0, 190.0)]);
        fixture.source_ids.push("train-1".into());
        let err = assert_no_leakage(&fixture, &train).unwrap_err();
        assert!(err.contains("source ID leakage"), "{err}");
    }

    #[test]
    fn leakage_proof_detects_overlapping_tiles() {
        let fixture = manifest("e2e", vec![(5.0, 15.0)]);
        let train = manifest("train", vec![(14.0, 25.0)]);
        let err = assert_no_leakage(&fixture, &train).unwrap_err();
        assert!(err.contains("spatial tile leakage"), "{err}");
    }

    #[test]
    fn tile_contains_respects_half_open_ranges() {
        let tile = SpatialTile {
            id: "t".into(),
            ra_range: (10.0, 20.0),
            dec_range: (-30.0, 30.0),
        };
        assert!(tile.contains(10.0, 0.0));
        assert!(!tile.contains(20.0, 0.0));
        assert!(!tile.contains(9.9, 0.0));
        assert!(!tile.contains(15.0, 30.0));
    }

    #[test]
    fn sha256_hex_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
