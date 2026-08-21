//! Stage 3, task 4: PINN per-target metrics + Stefan–Boltzmann residual,
//! emitting a correctness report (report-only gates).

use lnai_training::metrics::pinn::{
    PINN_TARGETS, mean_abs_sb_residual, per_target_metrics, stefan_boltzmann_residual,
};
use lnai_training::report::{ReportKind, ReportV1, identity_from_env};

const SEED: u64 = 42;

fn synthetic_batch(n: usize, noise: f32) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut lcg = SEED;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) % 10_000) as f32 / 10_000.0 - 0.5
    };

    let mut truth = Vec::with_capacity(n);
    let mut pred = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / n as f32;
        // Physically consistent truth: lum = 4*teff + 2*rad.
        let row = [
            3.6 + t,
            0.9 - t * 0.3,
            0.0 + t * 0.2,
            4.0 * (3.6 + t) + 2.0 * (0.9 - t * 0.3),
        ];
        let noisy = [
            row[0] + noise * next(),
            row[1] + noise * next(),
            row[2] + noise * next(),
            row[3] + noise * next(),
        ];
        truth.push(row);
        pred.push(noisy);
    }
    (pred, truth)
}

#[test]
fn pinn_per_target_metrics_and_sb_residual_report() {
    let (pred, truth) = synthetic_batch(2048, 0.02);

    // Contract: exactly four named targets, all finite.
    let metrics = per_target_metrics(&pred, &truth);
    assert_eq!(metrics.len(), PINN_TARGETS.len());
    for m in &metrics {
        assert!(m.mse.is_finite() && m.mae.is_finite());
    }

    // Truth rows are SB-exact by construction; predictions deviate ~noise.
    let truth_residual = mean_abs_sb_residual(&truth);
    let pred_residual = mean_abs_sb_residual(&pred);
    assert!(
        truth_residual < 1e-5,
        "truth must satisfy SB law, got {truth_residual}"
    );
    assert!(
        pred_residual > truth_residual,
        "noisy predictions must show a larger physics residual"
    );

    // Deterministic evaluation: same inputs -> identical metrics.
    let again = per_target_metrics(&pred, &truth);
    assert_eq!(again, metrics);

    let mut report = ReportV1::new(
        ReportKind::Correctness,
        "pinn_accuracy",
        identity_from_env(SEED),
    );
    for m in &metrics {
        report.add_metric(&format!("mse_{}", m.target), m.mse);
        report.add_metric(&format!("mae_{}", m.target), m.mae);
    }
    report.add_metric("mean_abs_sb_residual_truth", truth_residual);
    report.add_metric("mean_abs_sb_residual_pred", pred_residual);
    report.add_note("report-only baseline: thresholds are frozen in Stage 7 after model fixes");

    // Provisional sanity gate (loose): noise 0.02 must not explode into MSE > 1e-2.
    let worst_mse = metrics.iter().map(|m| m.mse).fold(0.0, f64::max);
    report.passed = worst_mse < 1e-2;
    assert!(
        report.metric("mse_log10_lum").unwrap() >= 0.0,
        "MSE cannot be negative"
    );
    assert_eq!(stefan_boltzmann_residual(&truth[0]), 0.0);

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write pinn report");
    println!("pinn report: {}", path.display());
}
