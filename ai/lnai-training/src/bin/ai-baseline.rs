//! Report-only CPU baseline runner (Stage 3, task 8).
//!
//! Executes an identical deterministic workload `--runs N` times on the same
//! machine, measures wall time per run and reports natural noise:
//!
//! ```bash
//! cargo run -p lnai-training --bin ai-baseline -- --runs 3
//! ```
//!
//! The output JSON lands in `$LUNAR_AI_REPORT_DIR` or `target/ai-reports`.

use lnai_training::metrics::pinn::{per_target_metrics, stefan_boltzmann_residual};
use lnai_training::metrics::rollout::position_rollout;
use lnai_training::report::{BaselineStats, ReportKind, ReportV1, RunIdentity, identity_from_env};
use std::time::Instant;

const ROWS: usize = 4096;
const WORKLOAD_ITERATIONS: usize = 25;

/// Deterministic LCG so every run performs identical work.
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

fn synthetic_rows(seed: u64) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let mut rng = Lcg(seed);
    let mut truth = Vec::with_capacity(ROWS);
    let mut pred = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let t = i as f32 / ROWS as f32;
        let row = [
            t,
            0.5 - t * 0.2,
            t * t * 0.1,
            4.0 * t + 2.0 * (0.5 - t * 0.2),
        ];
        let noisy = [
            row[0],
            row[1],
            row[2],
            row[3] + (rng.next_f32() - 0.5) * 0.01,
        ];
        truth.push(row);
        pred.push(noisy);
    }
    (pred, truth)
}

fn run_workload(seed: u64) -> f64 {
    let (pred, truth) = synthetic_rows(seed);
    let start = Instant::now();
    let mut sink = 0.0_f64;

    for _ in 0..WORKLOAD_ITERATIONS {
        let metrics = per_target_metrics(&pred, &truth);
        sink += metrics.iter().map(|m| m.mse).sum::<f64>();
        sink += pred
            .iter()
            .map(stefan_boltzmann_residual)
            .map(f32::abs)
            .sum::<f32>() as f64;

        let p0 = vec![[0.0_f32; 3]; ROWS];
        let vp = vec![[1.01_f32; 3]; ROWS];
        let vt = vec![[1.0_f32; 3]; ROWS];
        sink += position_rollout(&p0, &vp, &vt, 0.5, 16).final_error();
    }

    if sink.is_nan() {
        panic!("workload produced NaN");
    }
    start.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let runs: usize = std::env::args()
        .find_map(|a| a.strip_prefix("--runs=").map(|v| v.to_string()))
        .or_else(|| {
            let mut args = std::env::args().skip(1);
            while let Some(a) = args.next() {
                if a == "--runs" {
                    return args.next();
                }
            }
            None
        })
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(3);

    println!("ai-baseline: executing {runs} report-only runs...");
    let durations: Vec<f64> = (0..runs)
        .map(|i| {
            let ms = run_workload(1000 + i as u64);
            println!("  run {}: {ms:.1} ms", i + 1);
            ms
        })
        .collect();

    let stats = BaselineStats::from_runs(durations.clone()).expect("non-empty runs");
    let mut report = ReportV1::new(
        ReportKind::Performance,
        "baseline_cpu",
        identity_from_env(42),
    );
    report.add_metric("mean_ms", stats.mean_ms);
    report.add_metric("stddev_ms", stats.stddev_ms);
    report.add_metric("rel_spread_pct", stats.rel_spread_pct);
    report.add_metric(
        "stable_enough",
        f64::from(u8::from(stats.is_stable_enough())),
    );
    for (i, d) in durations.iter().enumerate() {
        report.add_metric(&format!("run_{}_ms", i + 1), *d);
    }

    // Identity carries no artifact/dataset for a pure synthetic workload.
    report.identity = RunIdentity {
        dataset_manifest_hash: String::new(),
        artifact_hash: None,
        device: report.identity.device.clone(),
        seed: report.identity.seed,
        git_revision: report.identity.git_revision.clone(),
        created_ms: report.identity.created_ms,
    };

    match stats.is_stable_enough() {
        true => report.add_note(format!(
            "natural noise {:.1}% is within the provisional 25% budget",
            stats.rel_spread_pct
        )),
        false => report.add_note(format!(
            "natural noise {:.1}% exceeds the provisional 25% budget; investigate runner stability",
            stats.rel_spread_pct
        )),
    };
    report.passed = stats.is_stable_enough();

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write report");
    println!(
        "baseline complete: mean {:.1} ms, spread {:.1}% -> {}",
        stats.mean_ms,
        stats.rel_spread_pct,
        path.display()
    );
}
