//! GNN graph-construction microbenches: preprocessing k-NN weights and the
//! 3-layer GCN forward pass (Stage 3, task 7 "preprocessing/k-NN").

use burn::backend::NdArray;
use burn::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use lnai_models::{GNN_INPUT_DIM, GNN_OUTPUT_DIM, StellarGnnConfig, compute_knn_adjacency};

type B = NdArray;

fn random_coords(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut lcg = seed;
    (0..n)
        .map(|_| {
            [(); 3].map(|_| {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((lcg >> 11) % 4000) as f32 / 100.0 - 20.0
            })
        })
        .collect()
}

fn dense_to_tensor(
    adj: &[Vec<f32>],
    device: &burn::backend::ndarray::NdArrayDevice,
) -> Tensor<B, 2> {
    let n = adj.len();
    let flat: Vec<f32> = adj.iter().flat_map(|row| row.iter().copied()).collect();
    Tensor::<B, 2>::from_data(burn::tensor::TensorData::new(flat, [n, n]), device)
}

fn bench_gnn_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("gnn_preprocessing");
    for &n in &[256_usize, 512] {
        group.bench_function(format!("knn_k8_n{n}"), |b| {
            let coords = random_coords(n, 42);
            b.iter(|| black_box(compute_knn_adjacency(black_box(&coords), 8)))
        });
    }
    group.finish();

    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = StellarGnnConfig {
        input_dim: GNN_INPUT_DIM,
        hidden_dim: 64,
        output_dim: GNN_OUTPUT_DIM,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);

    let n = 256;
    let nodes = Tensor::<B, 2>::random(
        [n, GNN_INPUT_DIM],
        burn::tensor::Distribution::Default,
        &device,
    );
    let adj = dense_to_tensor(&compute_knn_adjacency(&random_coords(n, 42), 8), &device);

    c.bench_function("gnn_forward_n256", |b| {
        b.iter(|| black_box(model.forward(nodes.clone(), adj.clone())))
    });
}

criterion_group!(benches, bench_gnn_graph);
criterion_main!(benches);
