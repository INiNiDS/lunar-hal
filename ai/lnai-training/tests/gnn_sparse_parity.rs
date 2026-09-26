//! Stage 7: GNN sparse edge representation and parity test.
//!
//! Asserts that:
//! 1. `GraphBatch::to_dense_adjacency` produces the exact same adjacency matrix
//!    as `compute_knn_adjacency`.
//! 2. `StellarGnn::forward_sparse` produces numerically equivalent outputs to
//!    the dense `forward` pass within floating-point tolerance (< 1e-4).
//! 3. Output shapes, determinism, and finiteness gates hold on the sparse path.
//! 4. Deterministic head under `sample_stellar_dynamics` returns mean velocities
//!    without applying stochastic noise.

use burn::backend::NdArray;
use burn::prelude::*;
use lnai_models::{
    GNN_INPUT_DIM, GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM, GraphBatch, StellarGnnConfig,
    compute_knn_adjacency, compute_sparse_knn_graph, sample_stellar_dynamics,
};

type B = NdArray<f32>;

fn seeded_coords(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            [(); 3].map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) % 4000) as f32 / 100.0 - 20.0
            })
        })
        .collect()
}

fn seeded_nodes(
    n: usize,
    seed: u64,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> Tensor<B, 2> {
    B::seed(device, seed);
    Tensor::<B, 2>::random(
        [n, GNN_INPUT_DIM],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    )
}

#[test]
fn sparse_graph_adjacency_matches_dense_adjacency_matrix() {
    for n in [5, 16, 42] {
        let coords = seeded_coords(n, 1234 + n as u64);
        let k = 4;
        let dense_matrix = compute_knn_adjacency(&coords, k);
        let sparse_graph = compute_sparse_knn_graph(&coords, k);

        assert_eq!(sparse_graph.num_nodes, n);
        assert!(sparse_graph.num_edges > 0);

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let reconstructed = sparse_graph.to_dense_adjacency::<B>(&device);
        let rec_data: Vec<f32> = reconstructed.into_data().to_vec().expect("data");

        for i in 0..n {
            for j in 0..n {
                let expected = dense_matrix[i][j];
                let actual = rec_data[i * n + j];
                assert!(
                    (expected - actual).abs() < 1e-6,
                    "Mismatch at ({i}, {j}): expected {expected}, got {actual}"
                );
            }
        }
    }
}

#[test]
fn sparse_gnn_forward_matches_dense_forward_pass() {
    let device = burn::backend::ndarray::NdArrayDevice::default();

    for output_dim in [GNN_OUTPUT_DIM, GNN_VARIATIONAL_DIM] {
        let model = StellarGnnConfig {
            input_dim: GNN_INPUT_DIM,
            hidden_dim: 32,
            output_dim,
            layer_norm_eps: 1e-5,
        }
        .init::<B>(&device);

        for n in [8, 25] {
            let coords = seeded_coords(n, 777 + n as u64);
            let nodes = seeded_nodes(n, 888 + n as u64, &device);
            let k = 5;

            let graph = compute_sparse_knn_graph(&coords, k);
            let adj_dense = graph.to_dense_adjacency::<B>(&device);

            let dense_out: Vec<f32> = model
                .forward(nodes.clone(), adj_dense)
                .into_data()
                .to_vec()
                .expect("dense forward data");

            let sparse_out: Vec<f32> = model
                .forward_sparse(nodes, &graph)
                .into_data()
                .to_vec()
                .expect("sparse forward data");

            assert_eq!(dense_out.len(), sparse_out.len());
            assert_eq!(dense_out.len(), n * output_dim);

            let mut max_diff: f32 = 0.0;
            for (i, (&d, &s)) in dense_out.iter().zip(sparse_out.iter()).enumerate() {
                assert!(d.is_finite(), "dense out finite at {i}: {d}");
                assert!(s.is_finite(), "sparse out finite at {i}: {s}");
                let diff = (d - s).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                assert!(
                    diff < 1e-4,
                    "Parity violation at {i}: dense={d}, sparse={s}, diff={diff}"
                );
            }
            assert!(
                max_diff < 1e-4,
                "Sparse vs dense max diff must be under 1e-4, got {max_diff}"
            );
        }
    }
}

#[test]
fn graph_batch_csr_invariants() {
    let empty = GraphBatch::empty();
    assert_eq!(empty.num_nodes, 0);
    assert_eq!(empty.num_edges, 0);
    assert_eq!(empty.row_ptr, vec![0]);

    let manual = GraphBatch::new(2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);
    assert_eq!(manual.num_nodes, 2);
    assert_eq!(manual.num_edges, 2);
}

#[test]
fn deterministic_head_sampling_returns_mean_without_noise() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let n = 10;
    let deterministic_output = Tensor::<B, 2>::from_data(
        TensorData::new(vec![1.5f32; n * GNN_OUTPUT_DIM], [n, GNN_OUTPUT_DIM]),
        &device,
    );

    // With temperature > 0.0, deterministic head must still return exact mean (no random noise).
    let sampled = sample_stellar_dynamics(deterministic_output.clone(), 1.0, &device);
    let orig_vals: Vec<f32> = deterministic_output.into_data().to_vec().unwrap();
    let sampled_vals: Vec<f32> = sampled.into_data().to_vec().unwrap();

    assert_eq!(
        orig_vals, sampled_vals,
        "Deterministic head must not have variational sampling applied"
    );
}
