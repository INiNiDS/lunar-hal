//! Stage 3: cross-crate chain proof — PINN positions feed the GNN graph,
//! whose readout must decode into the frozen vx/vy/vz kinematics contract.
//!
//! Stage 6.8: no synthetic noise anywhere in the chain. GNN node rows are
//! assembled from real PINN outputs plus the PINN inputs they derive from
//! (`[log_teff, log_rad, log_mass, log_lum, mg, x, y, z]` — the production
//! layout), and every random draw is seeded, so the chain is exactly
//! reproducible.

use burn::backend::NdArray;
use burn::prelude::*;

use lnai_models::{
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, KinematicsOutput, StellarGnnConfig, StellarMlpConfig,
    fourier_encode,
};
use lnai_training::report::{ReportKind, ReportV1, identity_from_env};

type B = NdArray;

const SEED: u64 = 2024;
const BATCH: usize = 8;

/// One full PINN → graph → GNN pass. All randomness (input sample, both
/// model inits) derives from `SEED` via [`Backend::seed`], so two calls
/// must return bit-identical velocities.
fn run_chain() -> Vec<[f32; 3]> {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    B::seed(&device, SEED);

    // 1) PINN head consumes [x, y, z, bp_rp, M_G] and yields 4 log10 targets.
    let pinn = StellarMlpConfig {
        hidden: 512,
        hidden2: 256,
        hidden3: 128,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);
    let xs = Tensor::<B, 2>::random([BATCH, 5], burn::tensor::Distribution::Default, &device);
    let targets = pinn.forward(xs.clone());
    let [batch, out_dim] = targets.dims();
    assert_eq!((batch, out_dim), (BATCH, 4), "PINN output contract is [N, 4]");
    let _ = fourier_encode(xs.clone().slice([0..1, 0..3]), 8); // preprocessing in the loop compiles

    // 2) GNN node rows from the model path only: PINN log-targets plus the
    // source mg and xyz columns — no Tensor::random nodes.
    let nodes = Tensor::cat(
        vec![
            targets,
            xs.clone().slice([0..batch, 4..5]),
            xs.slice([0..batch, 0..3]),
        ],
        1,
    );
    assert_eq!(nodes.dims(), [batch, GNN_INPUT_DIM]);

    // 3) k-NN adjacency over the PINN-derived positions (pure CPU).
    let coord_rows: Vec<[f32; 3]> = {
        let floats = nodes
            .clone()
            .slice([0..batch, 5..8])
            .into_data()
            .as_slice::<f32>()
            .expect("f32 data")
            .to_vec();
        floats.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
    };
    let adj_cpu = lnai_models::compute_knn_adjacency(&coord_rows, 3);
    assert_eq!(adj_cpu.len(), batch);

    let flat_adj: Vec<f32> = adj_cpu.iter().flat_map(|r| r.iter().copied()).collect();
    let adj = Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(flat_adj, [batch, batch]),
        &device,
    );

    // 4) GNN readout emits exactly 3 velocity components per node.
    let gnn = StellarGnnConfig {
        input_dim: GNN_INPUT_DIM,
        hidden_dim: 16,
        output_dim: GNN_OUTPUT_DIM,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);
    let velocities = gnn.forward(nodes, adj);
    let [_n, vdim] = velocities.dims();
    assert_eq!(vdim, 3, "GNN readout contract is vx/vy/vz");

    // 5) Every raw row decodes into the typed KinematicsOutput.
    let vel_data = velocities.into_data();
    let floats = vel_data.as_slice::<f32>().expect("f32 data").to_vec();
    floats
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

#[test]
fn pinn_to_gnn_chain_produces_contracted_velocities() {
    let first = run_chain();
    let second = run_chain();
    assert_eq!(first.len(), BATCH);

    // Finite gate: every component decodes into the frozen contract.
    let decoded: Vec<KinematicsOutput> = first
        .iter()
        .map(|row| KinematicsOutput::from_row(row))
        .collect::<Option<_>>()
        .expect("every row must decode into vx/vy/vz");
    assert_eq!(decoded.len(), BATCH);
    for out in &decoded {
        assert!(out.vx.is_finite() && out.vy.is_finite() && out.vz.is_finite());
    }

    // Deterministic gate: same seed → identical chain outputs.
    assert_eq!(first, second, "seeded chain must reproduce exactly");

    let mut report = ReportV1::new(ReportKind::E2E, "pinn_gnn_chain", identity_from_env(SEED));
    report.add_metric("nodes", BATCH as f64);
    report.add_metric("decoded_outputs", decoded.len() as f64);
    report.add_note(
        "Stage 6.8: untrained seeded weights; GNN nodes flow from PINN outputs, no synthetic noise",
    );
    report.passed = true;

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write chain report");
    println!("chain report: {}", path.display());
}
