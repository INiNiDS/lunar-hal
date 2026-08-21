//! Stage 3, task 6: position rollout as a kinematic-consistency test.

use lnai_training::metrics::rollout::{mean_squared_error3, position_rollout};
use lnai_training::report::{ReportKind, ReportV1, identity_from_env};

const SEED: u64 = 11;
const N: usize = 512;
const DT: f32 = 0.5;
const STEPS: usize = 12;

#[test]
fn rollout_error_grows_linearly_for_constant_bias() {
    let p0 = vec![[0.0_f32; 3]; N];
    let v_truth = vec![[1.0_f32; 3]; N];
    // Constant per-axis bias of 0.05 => error after k steps is exactly linear.
    let v_pred = vec![[1.05_f32; 3]; N];

    let result = position_rollout(&p0, &v_pred, &v_truth, DT, STEPS);
    assert_eq!(result.errors.len(), STEPS);

    let expected_step = |k: usize| (k as f64 * DT as f64) * ((0.05_f32 * 3.0_f32.sqrt()) as f64);
    for (idx, err) in result.errors.iter().enumerate() {
        assert!(
            (err - expected_step(idx + 1)).abs() < 1e-4,
            "step {}: {err} vs {}",
            idx + 1,
            expected_step(idx + 1)
        );
    }

    // Linearity check: error(k)/k must stay constant (f32 rounding tolerated).
    let ratios: Vec<f64> = result
        .errors
        .iter()
        .enumerate()
        .map(|(i, e)| e / (i + 1) as f64)
        .collect();
    for pair in ratios.windows(2) {
        assert!(
            (pair[0] - pair[1]).abs() / pair[0] < 1e-4,
            "growth must be strictly linear"
        );
    }
}

#[test]
fn rollout_report_with_noisy_model_velocities() {
    let mut lcg = SEED;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) % 10_000) as f32 / 10_000.0 - 0.5
    };

    let p0: Vec<[f32; 3]> = vec![[0.0; 3]; N];
    let v_truth: Vec<[f32; 3]> = (0..N).map(|_| [next() * 2.0; 3]).collect();
    let v_pred: Vec<[f32; 3]> = v_truth
        .iter()
        .map(|v| {
            [
                v[0] + 0.04 * next(),
                v[1] + 0.04 * next(),
                v[2] + 0.04 * next(),
            ]
        })
        .collect();

    let result = position_rollout(&p0, &v_pred, &v_truth, DT, STEPS);

    // Zero-velocity-error reference must stay flat.
    let exact = position_rollout(&p0, &v_truth, &v_truth, DT, STEPS);
    assert!(exact.final_error() < 1e-5);

    let growth = result.final_error() / result.errors[0];
    assert!(
        (growth - STEPS as f64).abs() < 1e-6,
        "linear regime must scale ~steps, got {growth}"
    );

    let mse = mean_squared_error3(&v_pred, &v_truth);

    let mut report = ReportV1::new(
        ReportKind::Correctness,
        "position_rollout",
        identity_from_env(SEED),
    );
    report.add_metric("final_position_error", result.final_error());
    report.add_metric("first_step_error", result.errors[0]);
    report.add_metric("velocity_mse", mse);
    report.add_metric("linear_growth_factor", growth);
    report.add_note("report-only consistency harness for future trained GNNs");
    report.passed = true;

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write rollout report");
    println!("rollout report: {}", path.display());
}
