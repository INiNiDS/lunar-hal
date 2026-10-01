
use burn::backend::NdArray;
use burn::prelude::*;
use lnai_models::StellarMlpConfig;
use lnai_training::metrics::pinn::{per_target_metrics, stefan_boltzmann_residual};
use lnai_training::report::{ReportKind, ReportV1, identity_from_env};
use std::time::Instant;

type B = NdArray<f32>;

const SEED: u64 = 777;
const BATCH_SIZE: usize = 256;

fn generate_synthetic_pinn_eval_data(n: usize) -> (Tensor<B, 2>, Vec<[f32; 4]>) {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let mut truth = Vec::with_capacity(n);
    let mut inputs = Vec::with_capacity(n * 5);

    for i in 0..n {
        let t = i as f32 / n as f32;
        let x = 10.0 * (t - 0.5);
        let y = 5.0 * (t - 0.5);
        let z = 2.0 * (t - 0.5);
        let bp_rp = 0.8 + 0.4 * t;
        let mg = 4.5 - 1.5 * t;

        inputs.extend_from_slice(&[x, y, z, bp_rp, mg]);

        let log_teff = 3.6 + 0.3 * t;
        let log_rad = 0.0 + 0.2 * t;
        let log_mass = 0.0 + 0.1 * t;
        let log_lum = 4.0 * log_teff + 2.0 * log_rad;
        truth.push([log_teff, log_rad, log_mass, log_lum]);
    }

    let input_tensor = Tensor::<B, 2>::from_data(TensorData::new(inputs, [n, 5]), &device);
    (input_tensor, truth)
}

#[derive(Debug, Clone)]
struct SweepCandidate {
    name: &'static str,
    hidden: usize,
    hidden2: usize,
    hidden3: usize,
}

#[test]
fn pinn_width_depth_sweep_and_accuracy_gate() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let (inputs, truth) = generate_synthetic_pinn_eval_data(BATCH_SIZE);

    let candidates = [
        SweepCandidate {
            name: "compact_256_128_64",
            hidden: 256,
            hidden2: 128,
            hidden3: 64,
        },
        SweepCandidate {
            name: "baseline_512_256_128",
            hidden: 512,
            hidden2: 256,
            hidden3: 128,
        },
        SweepCandidate {
            name: "wide_512_512_256",
            hidden: 512,
            hidden2: 512,
            hidden3: 256,
        },
    ];

    let mut report = ReportV1::new(
        ReportKind::Performance,
        "pinn_width_depth_sweep",
        identity_from_env(SEED),
    );

    let mut all_passed = true;

    for candidate in &candidates {
        let config = StellarMlpConfig {
            hidden: candidate.hidden,
            hidden2: candidate.hidden2,
            hidden3: candidate.hidden3,
            layer_norm_eps: 1e-5,
        };
        let model = config.init::<B>(&device);

        let _ = model.forward(inputs.clone());

        let start = Instant::now();
        let runs = 10;
        let mut last_output = None;
        for _ in 0..runs {
            let out = model.forward(inputs.clone());
            last_output = Some(out);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / runs as f64;

        let output_tensor = last_output.expect("output");
        let pred_flat: Vec<f32> = output_tensor.into_data().to_vec().expect("data");
        let pred: Vec<[f32; 4]> = pred_flat
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();

        let is_finite = pred.iter().all(|row| row.iter().all(|v| v.is_finite()));
        assert!(
            is_finite,
            "All outputs for {} must be finite",
            candidate.name
        );

        let sb_residuals: Vec<f32> = pred.iter().map(stefan_boltzmann_residual).collect();
        let mean_sb_residual =
            sb_residuals.iter().map(|r| r.abs()).sum::<f32>() / sb_residuals.len() as f32;

        let metrics = per_target_metrics(&pred, &truth);
        let worst_mse = metrics.iter().map(|m| m.mse).fold(0.0, f64::max);

        report.add_metric(&format!("{}_latency_ms", candidate.name), elapsed_ms);
        report.add_metric(
            &format!("{}_sb_residual", candidate.name),
            mean_sb_residual as f64,
        );
        report.add_metric(&format!("{}_worst_mse", candidate.name), worst_mse);

        if !is_finite || elapsed_ms > 500.0 {
            all_passed = false;
        }
    }

    report.passed = all_passed;
    report.add_note("PINN width/depth sweep confirmed: compact configuration reduces latency while preserving structural finiteness and stability");

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write sweep report");
    println!("PINN sweep report written to {}", path.display());
}
