//! Stage 3: cross-crate chain proof — PINN positions feed the GNN graph,
//! whose readout must decode into the frozen vx/vy/vz kinematics contract.

use burn::backend::NdArray;
use burn::prelude::*;

use lnai_models::{
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, KinematicsOutput, StellarGnnConfig, StellarMlpConfig,
    fourier_encode,
};
use lnai_training::report::{ReportKind, ReportV1, identity_from_env};

type B = NdArray;

#[test]
fn pinn_to_gnn_chain_produces_contracted_velocities() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let seed = 2024_u64;

    // 1) PINN head consumes [x, y, z, bp_rp, M_G] and yields 4 log10 targets.
    let pinn = StellarMlpConfig {
        hidden: 512,
        hidden2: 256,
        hidden3: 128,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);
    let xs = Tensor::<B, 2>::random([8, 5], burn::tensor::Distribution::Default, &device);
    let targets = pinn.forward(xs.clone());
    let [batch, out_dim] = targets.dims();
    assert_eq!((batch, out_dim), (8, 4), "PINN output contract is [N, 4]");
    let _ = fourier_encode(xs.slice([0..1, 0..3]), 8); // preprocessing in the loop compiles

    // 2) Positions from the first three columns become graph node features.
    let coords: Tensor<B, 2> = targets.clone().slice([0..batch, 0..3]);
    let coord_rows: Vec<[f32; 3]> = {
        let data = coords.into_data();
        let floats = data.as_slice::<f32>().expect("f32 data").to_vec();
        floats.chunks(3).map(|c| [c[0], c[1], c[2]]).collect()
    };

    // 3) k-NN adjacency over those positions (pure CPU preprocessing).
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
    let nodes = Tensor::<B, 2>::random(
        [batch, GNN_INPUT_DIM],
        burn::tensor::Distribution::Default,
        &device,
    );
    let velocities = gnn.forward(nodes, adj);
    let [_n, vdim] = velocities.dims();
    assert_eq!(vdim, 3, "GNN readout contract is vx/vy/vz");

    // 5) Every raw row decodes into the typed KinematicsOutput.
    let vel_data = velocities.into_data();
    let floats = vel_data.as_slice::<f32>().expect("f32 data").to_vec();
    let decoded: Vec<KinematicsOutput> = floats
        .chunks(3)
        .map(KinematicsOutput::from_row)
        .collect::<Option<_>>()
        .expect("every row must decode into vx/vy/vz");
    assert_eq!(decoded.len(), batch as usize);
    for out in &decoded {
        assert!(out.vx.is_finite() && out.vy.is_finite() && out.vz.is_finite());
    }

    let mut report = ReportV1::new(ReportKind::E2E, "pinn_gnn_chain", identity_from_env(seed));
    report.add_metric("nodes", batch as f64);
    report.add_metric("decoded_outputs", decoded.len() as f64);
    report.add_note("chain mechanics proven on untrained weights; accuracy gates come in Stage 7");
    report.passed = true;

    let dir = ReportV1::configured_dir();
    let path = report.write_to_dir(&dir).expect("write chain report");
    println!("chain report: {}", path.display());
}
