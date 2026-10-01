
use burn::backend::NdArray;
use burn::prelude::*;
use lnai_models::{
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, StellarGnnConfig, StellarMlpConfig, compute_sparse_knn_graph,
};
use lnai_training::report::{ReportKind, ReportV1, identity_from_env};
use std::time::Instant;

type B = NdArray<f32>;

const N_STARS: usize = 256;
const PROFILING_RUNS: usize = 20;

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[test]
fn profile_breakdown_stages_separately() {
    let device = burn::backend::ndarray::NdArrayDevice::default();

    let pinn = StellarMlpConfig::new().init::<B>(&device);
    let gnn = StellarGnnConfig {
        input_dim: GNN_INPUT_DIM,
        hidden_dim: 64,
        output_dim: GNN_OUTPUT_DIM,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);

    let mut data_wait_times = Vec::with_capacity(PROFILING_RUNS);
    let mut h2d_times = Vec::with_capacity(PROFILING_RUNS);
    let mut pinn_compute_times = Vec::with_capacity(PROFILING_RUNS);
    let mut graph_build_times = Vec::with_capacity(PROFILING_RUNS);
    let mut gnn_compute_times = Vec::with_capacity(PROFILING_RUNS);
    let mut d2h_times = Vec::with_capacity(PROFILING_RUNS);

    for run_idx in 0..PROFILING_RUNS {
        let t0 = Instant::now();
        let mut raw_inputs = Vec::with_capacity(N_STARS * 5);
        let mut coords = Vec::with_capacity(N_STARS);
        for i in 0..N_STARS {
            let s = (run_idx * N_STARS + i) as f32;
            let x = (s * 0.1).sin() * 50.0;
            let y = (s * 0.2).cos() * 50.0;
            let z = (s * 0.3).sin() * 20.0;
            coords.push([x, y, z]);
            raw_inputs.extend_from_slice(&[x, y, z, 0.85, 4.8]);
        }
        data_wait_times.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t1 = Instant::now();
        let pinn_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(raw_inputs, [N_STARS, 5]), &device);
        h2d_times.push(t1.elapsed().as_secs_f64() * 1000.0);

        let t2 = Instant::now();
        let pinn_out = pinn.forward(pinn_tensor);
        pinn_compute_times.push(t2.elapsed().as_secs_f64() * 1000.0);

        let t3 = Instant::now();
        let graph = compute_sparse_knn_graph(&coords, 8);
        graph_build_times.push(t3.elapsed().as_secs_f64() * 1000.0);

        let gnn_nodes = Tensor::<B, 2>::random(
            [N_STARS, GNN_INPUT_DIM],
            burn::tensor::Distribution::Default,
            &device,
        );
        let t4 = Instant::now();
        let gnn_out = gnn.forward_sparse(gnn_nodes, &graph);
        gnn_compute_times.push(t4.elapsed().as_secs_f64() * 1000.0);

        let t5 = Instant::now();
        let _pinn_cpu: Vec<f32> = pinn_out.into_data().to_vec().unwrap();
        let _gnn_cpu: Vec<f32> = gnn_out.into_data().to_vec().unwrap();
        d2h_times.push(t5.elapsed().as_secs_f64() * 1000.0);
    }

    let mut report = ReportV1::new(
        ReportKind::Performance,
        "profile_breakdown",
        identity_from_env(42),
    );

    let analyze_stage = |name: &str, mut times: Vec<f64>, rep: &mut ReportV1| {
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = percentile(&times, 0.50);
        let p95 = percentile(&times, 0.95);
        let p99 = percentile(&times, 0.99);
        rep.add_metric(&format!("{name}_p50_ms"), p50);
        rep.add_metric(&format!("{name}_p95_ms"), p95);
        rep.add_metric(&format!("{name}_p99_ms"), p99);
        println!("  {name:<15} p50={p50:6.3} ms | p95={p95:6.3} ms | p99={p99:6.3} ms");
        p95
    };

    println!("Stage 7 Profiling Breakdown (N={N_STARS} stars, {PROFILING_RUNS} runs):");
    let p95_dw = analyze_stage("data_wait", data_wait_times, &mut report);
    let p95_h2d = analyze_stage("host_to_device", h2d_times, &mut report);
    let p95_pinn = analyze_stage("pinn_compute", pinn_compute_times, &mut report);
    let p95_gb = analyze_stage("graph_build", graph_build_times, &mut report);
    let p95_gnn = analyze_stage("gnn_compute", gnn_compute_times, &mut report);
    let p95_d2h = analyze_stage("device_to_host", d2h_times, &mut report);

    let factor = if cfg!(debug_assertions) { 10.0 } else { 1.0 };
    assert!(p95_dw < 50.0 * factor, "data_wait p95 exceeds budget");
    assert!(p95_h2d < 50.0 * factor, "host_to_device p95 exceeds budget");
    assert!(p95_pinn < 50.0 * factor, "pinn_compute p95 exceeds budget");
    assert!(p95_gb < 50.0 * factor, "graph_build p95 exceeds budget");
    assert!(p95_gnn < 50.0 * factor, "gnn_compute p95 exceeds budget");
    assert!(p95_d2h < 50.0 * factor, "device_to_host p95 exceeds budget");

    report.passed = true;
    report.add_note("Stage 7 profiling breakdown: all pipeline phases verified within p95 budgets");

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write report");
    println!("Breakdown report saved to: {}", path.display());
}
