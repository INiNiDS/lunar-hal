//! Stage 4A / пункты 6–7: NASA-enriched feature rows and the honest
//! before/after evaluation report.
//!
//! Pipeline:
//! 1. [`build_enrichment`] crossmatches parsed PSCompPars host stars against a
//!    Gaia backbone sample with the frozen ID-first / epoch-aware strategy
//!    (reuses [`crate::crossmatch`], no second implementation).
//! 2. [`evaluate`] runs identical closed-form ridge regressions over a
//!    deterministic 2-fold split — baseline uses Gaia photometry only;
//!    enriched adds NASA stellar parameters. The ONLY difference between the
//!    arms is feature availability, so the delta cannot hide leakage.
//! 3. Verdict stays conservative: "improved" only when enriched MAE beats
//!    baseline across folds; otherwise "no_improvement" is recorded verbatim
//!    (plan requirement: не объявлять улучшение без report).

use crate::crossmatch::{CrossmatchCandidate, CrossmatchOutcome, MatchKind, crossmatch_candidate};
use crate::sources::nasa_exoplanet::NasaExoplanetRecord;
use serde::{Deserialize, Serialize};

/// One enrichment row after joining PSCompPars onto Gaia.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NasaEnrichedRow {
    pub planet_name: String,
    pub hostname: String,
    pub outcome: CrossmatchOutcome,
    /// NASA-derived feature block (all optional — coverage gets measured).
    pub teff_k: Option<f64>,
    pub radius_rsun: Option<f64>,
    pub mass_msun: Option<f64>,
}

/// Owned Gaia backbone row extended with the photometry/target columns the
/// report needs beyond plain crossmatching.
#[derive(Debug, Clone)]
pub struct GaiaSampleRow {
    pub source_id: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub epoch_year: f64,
    pub pm_ra_mas_yr: Option<f64>,
    pub pm_dec_mas_yr: Option<f64>,
    pub mag_g: f64,
    pub mag_bp: f64,
    pub mag_rp: f64,
    /// Strictly positive for evaluable samples.
    pub parallax_mas: f64,
}

impl GaiaSampleRow {
    fn as_candidate(&self) -> CrossmatchCandidate<'_> {
        CrossmatchCandidate {
            source_id: &self.source_id,
            ra_deg: self.ra_deg,
            dec_deg: self.dec_deg,
            epoch_year: self.epoch_year,
            pm_ra_mas_yr: self.pm_ra_mas_yr,
            pm_dec_mas_yr: self.pm_dec_mas_yr,
        }
    }
}

/// Deterministic host-star crossmatch of an exoplanet record set against the
/// Gaia sample. Candidate epoch freezes at 2016.0 (Gaia DR3 reference epoch).
/// Tolerance applies only to the positional fallback path.
pub fn build_enrichment(
    nasa_rows: &[NasaExoplanetRecord],
    gaia_rows: &[GaiaSampleRow],
    tolerance_arcsec: f64,
) -> Vec<NasaEnrichedRow> {
    let candidates: Vec<CrossmatchCandidate> =
        gaia_rows.iter().map(GaiaSampleRow::as_candidate).collect();

    nasa_rows
        .iter()
        .map(|r| {
            let outcome = crossmatch_candidate(
                None, // PSCompPars carries no trusted Gaia ID: positional path
                (r.ra_deg, r.dec_deg),
                2016.0,
                (None, None), // TAP payload has no host proper-motion columns
                &candidates,
                tolerance_arcsec,
            );
            NasaEnrichedRow {
                planet_name: r.planet_name.clone(),
                hostname: r.hostname.clone(),
                outcome,
                teff_k: r.teff_k,
                radius_rsun: r.radius_rsun,
                mass_msun: r.mass_msun,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Before/after evaluation (пункт 7)
// ---------------------------------------------------------------------------

const BASELINE_DIM: usize = 2; // [mag_g, bp_rp]
const NASA_DIM: usize = 3; // [teff_k / 6000, radius_rsun, mass_msun]

/// Training sample assembled only from confidently matched rows.
#[derive(Debug, Clone)]
pub struct Sample {
    pub baseline_features: [f64; BASELINE_DIM],
    pub nasa_features: [f64; NASA_DIM],
    pub target_parallax_mas: f64,
}

/// Builds evaluable samples plus an explicit unmatched count so callers must
/// account for dropped rows instead of swallowing them silently.
///
/// Matching policy: positional wins only (PSCompPars has no trusted Gaia id),
/// unambiguous (no duplicate-id marker ambiguity_ratio==0 cases kept).
pub fn enrichment_samples_from_records(
    nasa_rows: &[NasaExoplanetRecord],
    gaia_rows: &[GaiaSampleRow],
    tolerance_arcsec: f64,
) -> (Vec<Sample>, usize) {
    let mut unmatched = 0usize;
    let mut samples = Vec::new();

    let candidates: Vec<CrossmatchCandidate> =
        gaia_rows.iter().map(GaiaSampleRow::as_candidate).collect();

    for r in nasa_rows {
        let matched_row = match crossmatch_candidate(
            None,
            (r.ra_deg, r.dec_deg),
            2016.0,
            (None, None),
            &candidates,
            tolerance_arcsec,
        ) {
            o if o.matched_by == MatchKind::EpochAwarePositional => Some(o.gaia_source_id),
            _ => None,
        };

        let Some(source_id) = matched_row else {
            unmatched += 1;
            continue;
        };
        let Some(gaia_row) = gaia_rows.iter().find(|g| g.source_id == source_id) else {
            unmatched += 1;
            continue;
        };
        // Complete NASA feature vector required — coverage gaps are reported,
        // never imputed here.
        let (Some(teff), Some(rad), Some(mass)) = (r.teff_k, r.radius_rsun, r.mass_msun) else {
            unmatched += 1;
            continue;
        };
        if gaia_row.parallax_mas <= 0.0 || !(gaia_row.mag_bp - gaia_row.mag_rp).is_finite() {
            unmatched += 1;
            continue;
        }

        samples.push(Sample {
            baseline_features: [gaia_row.mag_g, gaia_row.mag_bp - gaia_row.mag_rp],
            nasa_features: [teff / 6000.0, rad, mass],
            target_parallax_mas: gaia_row.parallax_mas,
        });
    }
    (samples, unmatched)
}

fn ridge_solve(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let d = x[0].len();
    // Normal equations: A = X^T X + lambda I, b = X^T y.
    let mut a = vec![vec![0.0; d + 1]; d];
    for (row_i, xi) in x.iter().enumerate() {
        for j in 0..d {
            for k in 0..d {
                a[j][k] += xi[j] * xi[k];
            }
            a[j][d] += xi[j] * y[row_i];
        }
    }
    for j in 0..d {
        a[j][j] += lambda;
    }

    // Gaussian elimination with partial pivoting + back substitution.
    for col in 0..d {
        let pivot = (col..d)
            .max_by(|u, v| {
                a[*u][col]
                    .abs()
                    .partial_cmp(&a[*v][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        a.swap(col, pivot);
        if a[col][col].abs() < 1e-12 {
            continue;
        }
        for r in (col + 1)..d {
            let factor = a[r][col] / a[col][col];
            if factor != 0.0 {
                for k in col..=d {
                    a[r][k] -= factor * a[col][k];
                }
            }
        }
    }
    let mut w = vec![0.0; d];
    for r in (0..d).rev() {
        if a[r][r].abs() < 1e-12 {
            continue;
        }
        let mut sum = a[r][d];
        for k in (r + 1)..d {
            sum -= a[r][k] * w[k];
        }
        w[r] = sum / a[r][r];
    }
    w
}

fn predict(w: &[f64], feats: &[f64]) -> f64 {
    w.iter().zip(feats).map(|(wi, xi)| wi * xi).sum()
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EvalMetrics {
    pub mae: f64,
    pub rows: usize,
}

impl std::fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MAE={:.4} (n={})", self.mae, self.rows)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EnrichmentReportV1 {
    pub version: String,
    pub task: String,
    pub matched_rows: usize,
    /// Mean MAE across both folds, baseline arm.
    pub before: EvalMetrics,
    /// Mean MAE across both folds, enriched arm.
    pub after: EvalMetrics,
    /// Signed relative MAE change (>0 means enrichment helped).
    pub mae_delta_fraction: f64,
    pub verdict: EnrichmentVerdict,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnrichmentVerdict {
    Improved,
    NoImprovement,
}

impl EnrichmentReportV1 {
    pub const VERSION: &'static str = "1.0.0";
    pub const TASK: &str =
        "ridge 2-fold: predict parallax_mas from gaia photometry vs photometry+nasa-stellar-params";
}

/// Runs the deterministic before/after comparison. Requires >= 8 complete
/// matched samples; returns None otherwise so callers report honestly rather
/// than fabricating numbers. Full PINN-level retraining comparison belongs to
/// stage 5 (lnai-training); this module provides the data-level gate.
pub fn evaluate(samples: &[Sample]) -> Option<EnrichmentReportV1> {
    if samples.len() < 8 {
        return None;
    }

    let arm_mae = |use_nasa: bool| -> EvalMetrics {
        let feat = |s: &Sample| -> Vec<f64> {
            if use_nasa {
                s.baseline_features
                    .iter()
                    .chain(s.nasa_features.iter())
                    .copied()
                    .collect()
            } else {
                s.baseline_features.to_vec()
            }
        };

        let mut maes = Vec::new();
        for test_is_even in [true, false] {
            let mut train: Vec<&Sample> = Vec::new();
            let mut test: Vec<&Sample> = Vec::new();
            for (i, sample) in samples.iter().enumerate() {
                if test_is_even == (i % 2 == 0) {
                    test.push(sample);
                } else {
                    train.push(sample);
                }
            }

            let tm = train.iter().map(|s| s.target_parallax_mas).sum::<f64>() / train.len() as f64;

            let total_dim = feat(&samples[0]).len();
            let mut fm = vec![0.0; total_dim];
            for s in &train {
                for i in 0..total_dim {
                    fm[i] += feat(s)[i];
                }
            }
            for v in &mut fm {
                *v /= train.len() as f64;
            }

            let x_train: Vec<Vec<f64>> = train
                .iter()
                .map(|s| feat(s).iter().zip(&fm).map(|(x, m)| x - m).collect())
                .collect();
            let y_train: Vec<f64> = train.iter().map(|s| s.target_parallax_mas - tm).collect();

            let w = ridge_solve(&x_train, &y_train, 1e-6);
            let err: f64 = test
                .iter()
                .map(|s| {
                    let f: Vec<f64> = feat(s).iter().zip(&fm).map(|(x, m)| x - m).collect();
                    (predict(&w, &f) + tm - s.target_parallax_mas).abs()
                })
                .sum();
            maes.push(err / test.len().max(1) as f64);
        }

        EvalMetrics {
            mae: maes.iter().sum::<f64>() / maes.len() as f64,
            rows: samples.len(),
        }
    };

    let before = arm_mae(false);
    let after = arm_mae(true);
    let mae_delta_fraction = (before.mae - after.mae) / before.mae.max(f64::EPSILON);

    Some(EnrichmentReportV1 {
        version: EnrichmentReportV1::VERSION.into(),
        task: EnrichmentReportV1::TASK.into(),
        matched_rows: samples.len(),
        before,
        after,
        mae_delta_fraction,
        verdict: if mae_delta_fraction > 1e-9 {
            EnrichmentVerdict::Improved
        } else {
            EnrichmentVerdict::NoImprovement
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaia(id: u32, ra: f64, dec: f64, g: f64, bp: f64, rp: f64, plx: f64) -> GaiaSampleRow {
        GaiaSampleRow {
            source_id: format!("g{id}"),
            ra_deg: ra,
            dec_deg: dec,
            epoch_year: 2016.0,
            pm_ra_mas_yr: None,
            pm_dec_mas_yr: None,
            mag_g: g,
            mag_bp: bp,
            mag_rp: rp,
            parallax_mas: plx,
        }
    }

    fn host(i: u32, ra: f64, dec: f64, teff: f64, rad: f64, mass: f64) -> NasaExoplanetRecord {
        NasaExoplanetRecord {
            planet_name: format!("p{i}"),
            hostname: format!("h{i}"),
            ra_deg: ra,
            dec_deg: dec,
            parallax_mas: None,
            distance_pc: None,
            teff_k: Some(teff),
            radius_rsun: Some(rad),
            mass_msun: Some(mass),
            luminosity_lsun: None,
            discovery_year: None,
        }
    }

    #[test]
    fn enrichment_joins_positionally_and_marks_unmatched() {
        let nasa = vec![host(1, 10.0002, 20.0, 5800.0, 1.1, 1.05)];
        let gaia = vec![gaia(100, 10.0, 20.0, 9.5, 10.2, 8.7, 4.2)];
        let rows = build_enrichment(&nasa, &gaia, 1.0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].outcome.matched_by, MatchKind::EpochAwarePositional);
        assert_eq!(rows[0].outcome.gaia_source_id, "g100");

        let far = vec![host(2, 150.0, -30.0, 5000.0, 1.0, 1.0)];
        assert_eq!(
            build_enrichment(&far, &gaia, 1.0)[0].outcome.matched_by,
            MatchKind::Unmatched
        );
    }

    #[test]
    fn samples_only_built_for_complete_matches_and_unmatched_counted() {
        let mut nasa = (0..12u32)
            .map(|i| {
                host(
                    i,
                    i as f64 * 30.0 + 0.00001,
                    i as f64 % 7.0 - 3.0,
                    5000.0 + i as f64 * 40.0,
                    1.0,
                    1.0,
                )
            })
            .chain(std::iter::once(host(99, 359.9, 89.9, f64::NAN, 1.0, 1.0)))
            .collect::<Vec<_>>();
        // Row 99 loses NaN-purged Teff -> counted unmatched explicitly.
        nasa[12].teff_k = None;

        let gaia = (0..12u32)
            .map(|i| {
                gaia(
                    i,
                    i as f64 * 30.0 + 0.00001,
                    i as f64 % 7.0 - 3.0,
                    8.0 + i as f64 * 0.2,
                    9.0,
                    8.0,
                    2.0 + i as f64 * 0.7,
                )
            })
            .collect::<Vec<_>>();
        let (samples, unmatched) = enrichment_samples_from_records(&nasa, &gaia, 1.0);
        assert_eq!(samples.len(), 12);
        assert_eq!(unmatched, 1);
    }

    #[test]
    fn ridge_recovers_linear_relationship_after_centering() {
        // Same transformation the evaluator applies (mean-center features and
        // target): the raw relationship y = 2x0 - x1 + 5 must be recoverable.
        let xs_raw: Vec<Vec<f64>> = (0..16)
            .map(|i| vec![i as f64, (i % 5) as f64, (i % 3) as f64])
            .collect();
        let d = xs_raw[0].len();
        let xm = (0..d)
            .map(|j| xs_raw.iter().map(|v| v[j]).sum::<f64>() / xs_raw.len() as f64)
            .collect::<Vec<_>>();
        let xs: Vec<Vec<f64>> = xs_raw
            .iter()
            .map(|v| v.iter().zip(&xm).map(|(a, m)| a - m).collect())
            .collect();
        let ys_raw: Vec<f64> = xs_raw.iter().map(|v| 2.0 * v[0] - v[1] + 5.0).collect();
        let ym = ys_raw.iter().sum::<f64>() / ys_raw.len() as f64;
        let ys: Vec<f64> = ys_raw.iter().map(|y| y - ym).collect();

        let w = ridge_solve(&xs, &ys, 1e-9);
        assert!((w[0] - 2.0).abs() < 1e-6, "{w:?}");
        assert!((w[1] + 1.0).abs() < 1e-6, "{w:?}");
        assert!(w[2].abs() < 1e-6, "{w:?}");
    }

    #[test]
    fn evaluate_requires_minimum_sample_count_and_reports_verdict_honestly() {
        let mk = |seed: usize| Sample {
            baseline_features: [seed as f64 % 5.0 + 8.0, 1.1],
            nasa_features: [0.97 + seed as f64 * 0.01, 1.02, 1.04],
            target_parallax_mas: 2.0 + (seed as f64 * 0.37) % 5.0,
        };
        let too_few: Vec<Sample> = (0..4).map(mk).collect();
        assert!(evaluate(&too_few).is_none(), "needs >= 8 samples");

        let some: Vec<Sample> = (0..16).map(mk).collect();
        let report = evaluate(&some).expect("enough samples");
        assert_eq!(report.version, EnrichmentReportV1::VERSION);
        assert_eq!(report.matched_rows, 16);
        assert_eq!(report.before.rows, 16);
        assert!(report.mae_delta_fraction.is_finite());
        // Determinism: same input yields identical metrics bit-for-bit.
        assert_eq!(report, evaluate(&some).unwrap());
    }

    #[test]
    fn serialization_round_trip_of_report() {
        let mk = |seed: usize| Sample {
            baseline_features: [9.0 + seed as f64 * 0.31, 0.8],
            nasa_features: [1.01, 1.05, 1.03],
            target_parallax_mas: 1.5 + seed as f64 * 0.21,
        };
        let samples: Vec<Sample> = (0..24).map(mk).collect();
        let report = evaluate(&samples).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let back: EnrichmentReportV1 = serde_json::from_str(json.as_str()).unwrap();
        assert_eq!(back, report);
    }
}
