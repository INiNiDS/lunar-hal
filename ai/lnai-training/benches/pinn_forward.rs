//! PINN forward-pass microbench: Fourier features + 6-block MLP on CPU.
//!
//! Run: `cargo bench -p lnai-training --bench pinn_forward`

use burn::backend::NdArray;
use burn::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use lnai_models::{StellarMlpConfig, fourier_encode};

type B = NdArray;

fn random_input(batch: usize, dim: usize, seed: u64) -> Tensor<B, 2> {
    let mut lcg = seed;
    let data: Vec<f32> = (0..batch * dim)
        .map(|_| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg >> 11) % 2000) as f32 / 1000.0 - 1.0
        })
        .collect();
    Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(data, [batch * dim]),
        &burn::backend::ndarray::NdArrayDevice::default(),
    )
    .reshape([batch, dim])
}

fn bench_pinn_forward(c: &mut Criterion) {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = StellarMlpConfig {
        hidden: 512,
        hidden2: 256,
        hidden3: 128,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);

    c.bench_function("pinn_fourier_encode_1024", |b| {
        let xs = random_input(1024, 3, 7);
        b.iter(|| black_box(fourier_encode(xs.clone(), 8)))
    });

    c.bench_function("pinn_forward_1024", |b| {
        let xs = random_input(1024, 5, 11);
        b.iter(|| black_box(model.forward(xs.clone())))
    });
}

criterion_group!(benches, bench_pinn_forward);
criterion_main!(benches);
