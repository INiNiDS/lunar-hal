//! PINN correctness metrics: per-target errors and the Stefan–Boltzmann
//! physics residual.
//!
//! The PINN head emits four normalized log10 targets:
//! `(log10_teff, log10_rad, log10_mass, log10_lum)`. In solar units the
//! Stefan–Boltzmann law is `L = R^2 * T^4`, i.e. in log10 space
//! `log10_lum = 2*log10_rad + 4*log10_teff`.

/// Contracted target order of the PINN output head.
pub const PINN_TARGETS: [&str; 4] = ["log10_teff", "log10_rad", "log10_mass", "log10_lum"];

/// Exponents of the Stefan–Boltzmann law in log10 solar units.
pub const SB_RAD_EXPONENT: f32 = 2.0;
pub const SB_TEFF_EXPONENT: f32 = 4.0;

/// Error statistics for one output target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerTargetMetrics {
    pub target: &'static str,
    pub mse: f64,
    pub mae: f64,
    pub max_abs_err: f32,
}

/// Computes per-target MSE/MAE/max-error over `[N, 4]` prediction/truth rows.
///
/// # Panics
/// Panics when the slices differ in length.
pub fn per_target_metrics(pred: &[[f32; 4]], truth: &[[f32; 4]]) -> Vec<PerTargetMetrics> {
    assert_eq!(pred.len(), truth.len(), "pred/truth length mismatch");
    let n = pred.len() as f64;

    let mut sums = [(0.0f64, 0.0f64, 0.0f32); 4];
    for (p, t) in pred.iter().zip(truth.iter()) {
        for k in 0..4 {
            let err = p[k] - t[k];
            sums[k].0 += (err * err) as f64;
            sums[k].1 += (err.abs()) as f64;
            sums[k].2 = sums[k].2.max(err.abs());
        }
    }

    sums.into_iter()
        .zip(PINN_TARGETS)
        .map(|((sq, abs, max), target)| PerTargetMetrics {
            target,
            mse: sq / n,
            mae: abs / n,
            max_abs_err: max,
        })
        .collect()
}

/// Stefan–Boltzmann residual in log10 solar units:
/// `log10_lum - (4*log10_teff + 2*log10_rad)`.
/// Zero for physically consistent predictions.
pub fn stefan_boltzmann_residual(row: &[f32; 4]) -> f32 {
    let [log10_teff, log10_rad, _log10_mass, log10_lum] = *row;
    log10_lum - (SB_TEFF_EXPONENT * log10_teff + SB_RAD_EXPONENT * log10_rad)
}

/// Mean absolute Stefan–Boltzmann residual over `[N, 4]` rows.
pub fn mean_abs_sb_residual(rows: &[[f32; 4]]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter()
        .map(stefan_boltzmann_residual)
        .map(f32::abs)
        .sum::<f32>() as f64
        / rows.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_target_metrics_match_hand_computed_values() {
        let truth = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]];
        let pred = [[2.0, 2.0, 3.0, 6.0], [1.0, 4.0, 5.0, 4.0]];
        let metrics = per_target_metrics(&pred, &truth);
        let by_name = |name: &str| metrics.iter().find(|m| m.target == name).unwrap();

        let teff = by_name("log10_teff");
        assert_eq!(teff.mse, 0.5);
        assert_eq!(teff.mae, 0.5);
        assert_eq!(teff.max_abs_err, 1.0);

        let rad = by_name("log10_rad");
        assert_eq!(rad.mse, 2.0);
        assert_eq!(rad.max_abs_err, 2.0);

        assert_eq!(by_name("log10_mass").mse, 2.0);
        assert_eq!(by_name("log10_lum").mse, 2.0);
    }

    #[test]
    fn sb_residual_is_zero_for_physical_rows_and_detects_violations() {
        // L = R^2 * T^4 in log10, row layout [teff, rad, mass, lum]:
        // lum must equal 4*0.0 + 2*1.0 = 2.0.
        let physical = [0.0_f32, 1.0, 0.0, 2.0];
        assert!(stefan_boltzmann_residual(&physical).abs() < 1e-6);

        let violating = [0.0_f32, 1.0, 0.0, 3.0];
        assert!((stefan_boltzmann_residual(&violating) - 1.0).abs() < 1e-6);
        assert!(mean_abs_sb_residual(&[physical, violating]) > 0.0);
    }

    #[test]
    fn target_order_is_frozen() {
        assert_eq!(
            PINN_TARGETS,
            ["log10_teff", "log10_rad", "log10_mass", "log10_lum"]
        );
    }
}
