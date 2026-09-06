use serde::{Deserialize, Serialize};
use sha2::Digest;

/// Split salt: bump to re-shuffle all splits contractually (e.g. after adding
/// new data), while remaining deterministic for a given dataset version.
pub const SPLIT_SALT: &str = "lnai-split-v1";

/// Model/data split buckets. Deterministic hash assignment means the same
/// source always lands in the same bucket on every machine and run.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        }
    }
}

/// Assigns `source_id` into train/validation/test via
/// SHA256(salt|source_id) mod 1000 with the 80/10/10 policy.
pub fn split_for_source(source_id: &str) -> Split {
    let mut h = sha2::Sha256::new();
    h.update(format!("{SPLIT_SALT}|{source_id}").as_bytes());
    let digest = h.finalize();
    // Use the first 8 bytes for a stable bucket.
    let mut bucket = 0u64;
    for b in digest.iter().take(8) {
        bucket = (bucket << 8) | *b as u64;
    }
    match bucket % 1000 {
        0..=799 => Split::Train,
        800..=899 => Split::Validation,
        _ => Split::Test,
    }
}

/// Spatial tiling used by the holdout: RA bands of 15 degrees and Dec bands of
/// 15 degrees cover the whole sky with 24*12 = 288 tiles.
pub const TILE_RA_STEP_DEG: f64 = 15.0;
pub const TILE_DEC_STEP_DEG: f64 = 15.0;

/// Stable tile identifier covering `(ra, dec)`.
pub fn spatial_tile_id(ra_deg: f64, dec_deg: f64) -> String {
    let ra_idx = ((ra_deg.rem_euclid(360.0) / TILE_RA_STEP_DEG).floor() as i64).clamp(0, 23);
    let dec_idx = ((dec_deg + 90.0) / TILE_DEC_STEP_DEG)
        .floor()
        .clamp(0.0, 11.999) as i64;
    format!("tile_ra{ra_idx}_dec{dec_idx}")
}

const HOLDOUT_TILE_SALT: &str = "lnai-holdout-v1";
const HOLDOUT_BUCKET_OF_20: u64 = 3; // ~5% of sky tiles held out entirely

/// Whether a spatial tile is part of the holdout set. Deterministic choice
/// driven by tile identity, not row content — so models never see *any* row
/// from these regions during training/evaluation.
pub fn is_holdout_tile(tile_id: &str) -> bool {
    let mut h = sha2::Sha256::new();
    h.update(format!("{HOLDOUT_TILE_SALT}|{tile_id}").as_bytes());
    let digest = h.finalize();
    (digest[0] as u64) % 20 == HOLDOUT_BUCKET_OF_20
}

/// Combined check: does this coordinate fall inside a holdout tile?
pub fn is_spatial_holdout(ra_deg: f64, dec_deg: f64) -> bool {
    is_holdout_tile(&spatial_tile_id(ra_deg, dec_deg))
}

/// Every non-holdout row gets one of the three splits; holdout rows report
/// neither split nor presence in train/val/test files.
pub fn classify(ra_deg: f64, dec_deg: f64, source_id: &str) -> Option<Split> {
    if is_spatial_holdout(ra_deg, dec_deg) {
        return None;
    }
    Some(split_for_source(source_id))
}

/// Aggregate composition of a classified dataset; surfaced in reports/UI.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct SplitSummary {
    pub train: u64,
    pub validation: u64,
    pub test: u64,
    pub holdout_rows: u64,
    pub holdout_tiles: Vec<String>,
}

pub fn summarize(records: &[(f64, f64, String)]) -> SplitSummary {
    let mut s = SplitSummary::default();
    for (ra, dec, id) in records {
        match classify(*ra, *dec, id) {
            Some(Split::Train) => s.train += 1,
            Some(Split::Validation) => s.validation += 1,
            Some(Split::Test) => s.test += 1,
            None => s.holdout_rows += 1,
        }
    }
    s.holdout_tiles = collect_holdout_tile_ids(records);
    s
}

fn collect_holdout_tile_ids(records: &[(f64, f64, String)]) -> Vec<String> {
    let mut tiles: Vec<String> = records
        .iter()
        .map(|(ra, dec, _)| spatial_tile_id(*ra, *dec))
        .filter(|t| is_holdout_tile(t))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    tiles.sort_unstable(); // already sorted by BTreeSet but keep intent clear
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_deterministic_and_balanced() {
        assert_eq!(split_for_source("42"), split_for_source("42"));
        let mut counts = [0u32; 3];
        for id in 0..50_000u64 {
            counts[match split_for_source(&id.to_string()) {
                Split::Train => 0,
                Split::Validation => 1,
                Split::Test => 2,
            }] += 1;
        }
        let n = 50_000;
        assert!(
            (counts[0] as f64 - 0.8 * n as f64).abs() < 0.02 * n as f64,
            "train {counts:?}"
        );
        assert!(counts[1] > n / 40 && counts[1] < n / 5, "{counts:?}");
        assert!(counts[2] > n / 40 && counts[2] < n / 5, "{counts:?}");
    }

    #[test]
    fn spatial_tiling_is_deterministic_and_covers_grid() {
        // Deterministic mapping.
        assert_eq!(spatial_tile_id(359.9, 89.9), spatial_tile_id(-0.01, 89.9));
        assert_eq!(spatial_tile_id(14.9, 14.9), "tile_ra0_dec6");
        // Whole sky lands inside the 24x12 grid.
        let probe: Vec<String> = (0..24i64)
            .flat_map(|a| (0..12i64).map(move |b| (a, b)))
            .map(|(a, b)| spatial_tile_id(a as f64 * 15.0 + 7.5, b as f64 * 15.0 - 82.5))
            .collect();
        let unique: std::collections::BTreeSet<&String> = probe.iter().collect();
        assert_eq!(unique.len(), 288, "every tile center hits a unique tile");
    }

    #[test]
    fn holdout_selection_is_stable_and_bounded() {
        let mut held = Vec::new();
        for ra_i in 0..24i64 {
            for dec_i in 0..12i64 {
                let tile = format!("tile_ra{ra_i}_dec{dec_i}");
                if is_holdout_tile(&tile) {
                    held.push(tile);
                }
            }
        }
        // Deterministic and small (~5% of 288): repeated call agrees.
        let again: Vec<String> = (0..24i64)
            .flat_map(|ra_i| (0..12i64).map(move |dec_i| (ra_i, dec_i)))
            .filter(|(a, b)| is_holdout_tile(&format!("tile_ra{a}_dec{b}")))
            .map(|(a, b)| format!("tile_ra{a}_dec{b}"))
            .collect();
        assert_eq!(held, again);
        assert!(!held.is_empty(), "at least one tile must be held out");
        assert!(held.len() <= 288 / 4, "holdout must stay modest");
    }

    #[test]
    fn summarize_excludes_holdout_from_splits() {
        // Build synthetic rows; then verify that every classified split avoids
        // rows inside holdout tiles (the leakage invariant).
        let mut rows = Vec::new();
        for i in 0..2000u64 {
            let ra = (i % 24) as f64 * 15.0 + 7.5;
            let dec = ((i / 24) % 12) as f64 * 15.0 - 82.5;
            rows.push((ra, dec, format!("{i}")));
        }
        let summary = summarize(&rows);
        assert_eq!(
            summary.train + summary.validation + summary.test + summary.holdout_rows,
            rows.len() as u64
        );
        for (ra, dec, _) in &rows {
            if is_spatial_holdout(*ra, *dec) {
                continue;
            }
            // Non-holdout rows must always receive a split.
            assert!(classify(*ra, *dec, "").is_some());
        }
        assert!(summary.holdout_rows > 0, "fixture spans whole sky");
        assert_eq!(
            summary.holdout_tiles.len(),
            summary
                .holdout_tiles
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        );
    }

    #[test]
    fn split_json_round_trip() {
        for s in [Split::Train, Split::Validation, Split::Test] {
            let j = serde_json::to_string(&s).unwrap();
            let back: Split = serde_json::from_str(&j).unwrap();
            assert_eq!(back, s);
        }
        assert_eq!(serde_json::to_string(&Split::Train).unwrap(), "\"train\"");
    }
}
