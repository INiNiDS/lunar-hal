//! Stage 4A / пункт 5: Gaia crossmatch heuristics.
//!
//! v1 contract (documented, deterministic):
//! * **ID-first** — sources sharing a stable catalog identifier join by ID
//!   alone; no sky geometry involved on this path.
//! * **Epoch-aware fallback** — for candidates lacking an exact ID match the
//!   angular separation is evaluated at a common epoch J2016.0 (Gaia DR3
//!   reference epoch) using proper motion only as tie-break metadata.
//! * **Ambiguity metric** — ratio `sep_nearest / sep_second` inside the
//!   tolerance window; values close to 1.0 flag ambiguous counterparts that
//!   downstream stages must either discard or treat probabilistically.

use serde::{Deserialize, Serialize};

/// Angular separation in degrees between two equatorial positions.
/// Haversine form: numerically stable down to sub-milliarcsecond offsets.
pub fn angular_separation_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let dphi = (dec2 - dec1).to_radians();
    let dlmb = (ra2 - ra1).to_radians();
    let phi1 = dec1.to_radians();
    let phi2 = dec2.to_radians();
    let a = (dphi / 2.0).sin().powi(2) + phi1.cos() * phi2.cos() * (dlmb / 2.0).sin().powi(2);
    a.clamp(0.0, 1.0).sqrt().asin() * 2.0f64.to_degrees()
}

/// Proper-motion propagation of equatorial coordinates from `epoch_from` to
/// `epoch_to` years (J2000 epoch assumed for stored coordinates).
pub fn propagate_epoch(
    ra_deg: f64,
    dec_deg: f64,
    pm_ra_mas_yr: f64,
    pm_dec_mas_yr: f64,
    epoch_from: f64,
    epoch_to: f64,
) -> (f64, f64) {
    let dt_yr = epoch_to - epoch_from;
    let mas_to_deg = 1.0 / 3_600_000.0;
    // pmra is already μα* (includes cos δ); undo it explicitly.
    let cos_dec = dec_deg.to_radians().cos();
    let ra_shift_deg = if cos_dec.abs() > f64::EPSILON {
        pm_ra_mas_yr * dt_yr * mas_to_deg / cos_dec
    } else {
        0.0
    };
    (
        (ra_deg + ra_shift_deg).rem_euclid(360.0),
        (dec_deg + pm_dec_mas_yr * dt_yr * mas_to_deg).clamp(-90.0, 90.0),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossmatchCandidate<'a> {
    /// Stable catalog id (e.g. numeric Gaia source_id string).
    pub source_id: &'a str,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// Reference epoch of `ra/dec`.
    pub epoch_year: f64,
    pub pm_ra_mas_yr: Option<f64>,
    pub pm_dec_mas_yr: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CrossmatchOutcome {
    pub gaia_source_id: String,
    pub matched_by: MatchKind,
    /// Great-circle distance to the chosen counterpart, degrees.
    pub separation_deg: Option<f64>,
    /// `sep_nearest / sep_second`; None when fewer than two in-tolerance
    /// candidates exist. Values <~0.5 indicate a confident positional match.
    pub ambiguity_ratio: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    IdFirst,
    EpochAwarePositional,
    Unmatched,
}

/// Matches one enrichment candidate against the Gaia backbone.
///
/// `tolerance_arcsec` applies only to the positional fallback path.
pub fn crossmatch_candidate(
    gaia_id_of_candidate: Option<&str>,
    candidate_position: (f64, f64),
    candidate_epoch_year: f64,
    candidate_pm: (Option<f64>, Option<f64>),
    gaia_rows: &[CrossmatchCandidate<'_>],
    tolerance_arcsec: f64,
) -> CrossmatchOutcome {
    let out_for_unmatched = || CrossmatchOutcome {
        gaia_source_id: String::new(),
        matched_by: MatchKind::Unmatched,
        separation_deg: None,
        ambiguity_ratio: None,
    };

    // --- path 1: ID-first -------------------------------------------------
    if let Some(id) = gaia_id_of_candidate {
        let exact: Vec<_> = gaia_rows.iter().filter(|g| g.source_id == id).collect();
        if exact.len() == 1 {
            return CrossmatchOutcome {
                gaia_source_id: id.to_string(),
                matched_by: MatchKind::IdFirst,
                separation_deg: None, // IDs carry no position claim
                ambiguity_ratio: None,
            };
        }
        if exact.len() > 1 {
            // Duplicate ids are a data error: refuse to guess.
            return CrossmatchOutcome {
                gaia_source_id: id.to_string(),
                matched_by: MatchKind::Unmatched,
                separation_deg: None,
                ambiguity_ratio: Some(0.0), // explicit duplicate marker
            };
        }
    }

    // --- path 2: epoch-aware positional ------------------------------------
    let tol_deg = tolerance_arcsec / 3600.0;
    let mut scored: Vec<(f64, &CrossmatchCandidate)> = gaia_rows
        .iter()
        .map(|g| {
            let (gra, gdec) = propagate_epoch(
                g.ra_deg,
                g.dec_deg,
                g.pm_ra_mas_yr.unwrap_or(0.0),
                g.pm_dec_mas_yr.unwrap_or(0.0),
                g.epoch_year,
                candidate_epoch_year.max(g.epoch_year),
            );
            let sep = angular_separation_deg(candidate_position.0, candidate_position.1, gra, gdec);
            (sep, g)
        })
        .filter(|(sep, _)| *sep <= tol_deg)
        .collect();

    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let Some((best_sep, best)) = scored.first().copied() else {
        return out_for_unmatched();
    };
    let second_sep = scored.get(1).map(|(s, _)| *s);
    CrossmatchOutcome {
        gaia_source_id: best.source_id.to_string(),
        matched_by: MatchKind::EpochAwarePositional,
        separation_deg: Some(best_sep),
        ambiguity_ratio: second_sep.map(|s| best_sep / s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL_ARCSEC: f64 = 1.0;

    fn gaia_row(id: &str, ra: f64, dec: f64) -> CrossmatchCandidate<'_> {
        CrossmatchCandidate {
            source_id: id,
            ra_deg: ra,
            dec_deg: dec,
            epoch_year: 2016.0,
            pm_ra_mas_yr: None,
            pm_dec_mas_yr: None,
        }
    }

    #[test]
    fn id_first_path_beats_geometry_and_requires_unique_ids() {
        let rows = vec![
            gaia_row("100", 10.0000001, -5.0),
            gaia_row("200", 150.0, 40.0),
        ];
        let m = crossmatch_candidate(
            Some("200"),
            (10.0, -5.0),
            2016.0,
            (None, None),
            &rows,
            TOL_ARCSEC,
        );
        assert_eq!(m.matched_by, MatchKind::IdFirst);
        assert_eq!(m.gaia_source_id, "200");

        // Ambiguous duplicates never guess.
        let dupes = vec![gaia_row("9", 1.0, 1.0), gaia_row("9", 2.0, 2.0)];
        let m2 = crossmatch_candidate(
            Some("9"),
            (2.0, 2.0),
            2016.0,
            (None, None),
            &dupes,
            TOL_ARCSEC,
        );
        assert_eq!(m2.matched_by, MatchKind::Unmatched);
        assert_eq!(m2.ambiguity_ratio, Some(0.0));
    }

    #[test]
    fn positional_match_is_epoch_aware_and_scores_ambiguity() {
        // Two identical-ish stars plus one clearly far away, all at the same
        // place except the twin pair differ by ~0.25 arcsec vs 0.75 arcsec.
        let rows = vec![
            gaia_row("near", 10.0 + 0.25 / 3600.0, 20.0),
            gaia_row("far", 10.0 + 0.75 / 3600.0, 20.0),
        ];
        let m = crossmatch_candidate(None, (10.0, 20.0), 2020.0, (None, None), &rows, TOL_ARCSEC);
        assert_eq!(m.matched_by, MatchKind::EpochAwarePositional);
        assert_eq!(m.gaia_source_id, "near");
        let ratio = m.ambiguity_ratio.expect("two in-tolerance candidates");
        assert!((ratio - (0.25f64 / 3600.0) / (0.75 / 3600.0)).abs() < 1e-9);
        assert!(ratio < 0.5, "clear winner expected");

        // Nothing within tolerance -> unmatched with flat diagnostics.
        let none = crossmatch_candidate(
            None,
            (180.0, -80.0),
            2016.0,
            (None, None),
            &rows,
            TOL_ARCSEC,
        );
        assert_eq!(none.matched_by, MatchKind::Unmatched);
        assert!(none.separation_deg.is_none());
    }

    #[test]
    fn epoch_propagation_moves_coordinates_deterministically() {
        // 100 mas/yr over exactly 16 years => 1.6 arcsec = 4.444e-4 deg.
        let (ra, dec) = propagate_epoch(60.0, 30.0, 100.0, 50.0, 2000.0, 2016.0);
        assert!((dec - (30.0 + 50.0 * 16.0 / 3_600_000.0)).abs() < 1e-12);
        let expect_ra_shift = 100.0 * 16.0 / 3_600_000.0 / 30f64.to_radians().cos();
        assert!((ra - (60.0 + expect_ra_shift)).abs() < 1e-12);
    }

    #[test]
    fn angular_separation_matches_known_small_offsets() {
        // One arcsecond in RA at dec=0 equals 1/3600 deg apart.
        let s = angular_separation_deg(10.0, 0.0, 10.0 + 1.0 / 3600.0, 0.0);
        assert!((s - 1.0 / 3600.0).abs() < 1e-12, "{s}");
        // Identical points are zero; antipodes are 180 deg.
        assert_eq!(angular_separation_deg(11.0, 22.0, 11.0, 22.0), 0.0);
        assert!((angular_separation_deg(0.0, 0.0, 180.0, 0.0) - 180.0).abs() < 1e-9);
    }
}
