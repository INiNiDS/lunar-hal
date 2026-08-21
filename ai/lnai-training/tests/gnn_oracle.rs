//! Stage 3, task 5: GNN-Kinematics oracle/chained/baseline metrics.

use lnai_training::metrics::gnn::KinematicsMetrics;
use lnai_training::report::{ReportKind, ReportV1, identity_from_env};

const SEED: u64 = 7;
const N: usize = 1024;

fn synthetic_velocities() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 3]>) {
    // truth = smooth field, chained = truth + noise (a "model"),
    // baseline = batch-mean velocity (the trivial predictor).
    let mut lcg = SEED;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((lcg >> 11) % 10_000) as f32 / 10_000.0 - 0.5
    };

    let truth: Vec<[f32; 3]> = (0..N)
        .map(|i| {
            let t = i as f32 / N as f32;
            [t * 2.0 - 1.0, 0.5 * (t * std::f32::consts::PI).sin(), -t]
        })
        .collect();

    let chained: Vec<[f32; 3]> = truth
        .iter()
        .map(|v| {
            [
                v[0] + 0.05 * next(),
                v[1] + 0.05 * next(),
                v[2] + 0.05 * next(),
            ]
        })
        .collect();

    let mean = [
        truth.iter().map(|v| v[0]).sum::<f32>() / N as f32,
        truth.iter().map(|v| v[1]).sum::<f32>() / N as f32,
        truth.iter().map(|v| v[2]).sum::<f32>() / N as f32,
    ];
    let baseline = vec![mean; N];

    (chained, baseline, truth)
}

#[test]
fn gnn_oracle_chained_baseline_ordering_and_report() {
    let (chained, baseline, truth) = synthetic_velocities();

    let metrics = KinematicsMetrics::evaluate(&chained, &truth, &baseline);

    // Oracle (= truth itself) must be the zero upper bound.
    assert_eq!(metrics.oracle_mse, 0.0);
    assert!(
        metrics.ordering_is_sane(),
        "oracle {oracle} must not lose to chained {chained}",
        oracle = metrics.oracle_mse,
        chained = metrics.chained_mse
    );
    assert!(
        metrics.chained_mse < metrics.baseline_mse,
        "the seeded model noise is small, so it should beat the mean predictor"
    );
    assert!(
        (0.0..=1.0).contains(&metrics.chained_skill),
        "skill must be in [0, 1], got {}",
        metrics.chained_skill
    );

    let mut report = ReportV1::new(
        ReportKind::Correctness,
        "gnn_oracle",
        identity_from_env(SEED),
    );
    report.add_metric("oracle_mse", metrics.oracle_mse);
    report.add_metric("chained_mse", metrics.chained_mse);
    report.add_metric("baseline_mse", metrics.baseline_mse);
    report.add_metric("chained_skill", metrics.chained_skill);
    report.add_note("report-only: real weights arrive with Stage 7 model fixes");
    report.passed = true;

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write gnn report");
    println!("gnn report: {}", path.display());
}
