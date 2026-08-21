//! SIREN texture microbench: forward pass over a 64x64 texture grid.

use burn::backend::NdArray;
use burn::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use lnai_models::{SIREN_INPUT_DIM, StellarSirenConfig};

type B = NdArray;

fn texture_grid(side: usize, device: &burn::backend::ndarray::NdArrayDevice) -> Tensor<B, 2> {
    // (u, v, bp_rp, g_mag, m_abs) per pixel — matches SIREN input contract.
    let mut data = Vec::with_capacity(side * side * SIREN_INPUT_DIM);
    for y in 0..side {
        for x in 0..side {
            let u = x as f32 / side as f32;
            let v = y as f32 / side as f32;
            data.extend_from_slice(&[u, v, 0.85, 4.83, 0.5]);
        }
    }
    Tensor::<B, 2>::from_data(
        burn::tensor::TensorData::new(data, [side * side, SIREN_INPUT_DIM]),
        device,
    )
}

fn bench_siren_texture(c: &mut Criterion) {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let model = StellarSirenConfig { hidden: 64 }.init::<B>(&device);
    let grid = texture_grid(64, &device);

    c.bench_function("siren_forward_64x64", |b| {
        b.iter(|| black_box(model.forward(grid.clone())))
    });
}

criterion_group!(benches, bench_siren_texture);
criterion_main!(benches);
