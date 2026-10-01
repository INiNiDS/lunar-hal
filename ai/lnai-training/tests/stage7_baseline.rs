
use burn::backend::NdArray;
use burn::prelude::*;
use lnai_models::{
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, StellarGnnConfig, StellarSirenConfig, fourier_encode,
    fourier_encode_cached,
};
use lnai_training::gnn::GraphCache;
use serde_json::Value;
use std::sync::Arc;

type B = NdArray<f32>;

fn stage7_baseline() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fixtures/stage7-approved-baseline.json"
    );
    let raw = std::fs::read_to_string(path).expect("stage 7 baseline JSON must exist");
    serde_json::from_str(&raw).expect("baseline JSON must parse")
}

#[test]
fn baseline_record_covers_all_stage7_suites() {
    let base = stage7_baseline();
    assert_eq!(base["stage"], 7);
    let suites = base["suites"].as_array().expect("suites array");
    let files: Vec<&str> = suites
        .iter()
        .map(|s| s["file"].as_str().expect("suite file"))
        .collect();

    for expected in [
        "ai/lnai-training/tests/fourier_cache_parity.rs",
        "ai/lnai-training/tests/gnn_sparse_parity.rs",
        "ai/lnai-training/tests/pinn_sweep.rs",
        "ai/lnai-training/tests/profile_breakdown.rs",
        "ai/lnai-training/tests/stage7_baseline.rs",
    ] {
        assert!(
            files.contains(&expected),
            "stage 7 baseline must cover {expected}: {files:?}"
        );
    }

    for suite in suites {
        assert_eq!(
            suite["status"], "passed",
            "suite must be marked passed: {}",
            suite["file"]
        );
    }

    assert_eq!(base["release_qualification"], "not_qualified");
    assert_eq!(base["evidence_scope"]["trained_checkpoint_quality"], false);
    assert!(
        base["exit_gate"]["performance_improved"]
            .as_str()
            .is_some_and(|value| value.starts_with("not_verified"))
    );
    assert!(
        base["exit_gate"]["accuracy_calibration"]
            .as_str()
            .is_some_and(|value| value.starts_with("not_verified"))
    );
}

#[test]
fn fourier_cache_parity_gate() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    B::seed(&device, 101);
    let xyz = Tensor::<B, 2>::random(
        [12, 3],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );

    let legacy: Vec<f32> = fourier_encode(xyz.clone(), 8).into_data().to_vec().unwrap();
    let cached: Vec<f32> = fourier_encode_cached(xyz, 8).into_data().to_vec().unwrap();

    assert_eq!(legacy.len(), cached.len());
    assert_eq!(
        legacy, cached,
        "Fourier cache must be bitwise identical to legacy"
    );
}

#[test]
fn gnn_sparse_parity_and_cache_gate() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = StellarGnnConfig {
        input_dim: GNN_INPUT_DIM,
        hidden_dim: 32,
        output_dim: GNN_OUTPUT_DIM,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);

    let n = 20;
    let coords: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let fi = i as f32;
            [fi.sin() * 10.0, fi.cos() * 10.0, (fi * 0.5).sin() * 5.0]
        })
        .collect();

    let g1 = GraphCache::get_or_build(&coords, 4);
    let g2 = GraphCache::get_or_build(&coords, 4);
    assert!(
        Arc::ptr_eq(&g1, &g2),
        "GraphCache must reuse identical Arc for immutable graph"
    );

    let nodes = Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.5f32; n * GNN_INPUT_DIM], [n, GNN_INPUT_DIM]),
        &device,
    );

    let adj_dense = g1.to_dense_adjacency::<B>(&device);
    let out_dense: Vec<f32> = model
        .forward(nodes.clone(), adj_dense)
        .into_data()
        .to_vec()
        .unwrap();
    let out_sparse: Vec<f32> = model
        .forward_sparse(nodes, &g1)
        .into_data()
        .to_vec()
        .unwrap();

    for (d, s) in out_dense.iter().zip(out_sparse.iter()) {
        assert!(
            (d - s).abs() < 1e-4,
            "GNN sparse parity tolerance violation: {d} vs {s}"
        );
    }
}

#[test]
fn siren_chunked_forward_parity_gate() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = StellarSirenConfig {
        hidden: 32,
        w0: 30.0,
    }
    .init::<B>(&device);

    let rows = 64;
    let inputs =
        Tensor::<B, 2>::from_data(TensorData::new(vec![0.25f32; rows * 5], [rows, 5]), &device);

    let full_out: Vec<f32> = model.forward(inputs.clone()).into_data().to_vec().unwrap();
    let chunked_out: Vec<f32> = model
        .forward_chunked(inputs, 16)
        .into_data()
        .to_vec()
        .unwrap();

    assert_eq!(full_out.len(), chunked_out.len());
    for (f, c) in full_out.iter().zip(chunked_out.iter()) {
        assert_eq!(
            f, c,
            "SIREN chunked inference must be bitwise identical to unchunked"
        );
    }
}
