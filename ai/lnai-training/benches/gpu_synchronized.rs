
#![cfg(feature = "gpu-harness")]

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use lnai_models::{StellarMlpConfig, StellarSirenConfig};

type B = Cuda;

fn synchronized_forward(
    xs: Tensor<B, 2>,
    run: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    iterations: u32,
) -> std::time::Duration {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let out = run(xs.clone());
        black_box(out.slice([0..1, 0..1]).into_scalar());
    }
    start.elapsed()
}

fn bench_gpu_synchronized(c: &mut Criterion) {
    let device = CudaDevice::default();
    let mut group = c.benchmark_group("gpu_synchronized");

    let pinn = StellarMlpConfig {
        hidden: 512,
        hidden2: 256,
        hidden3: 128,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);
    let pinn_input =
        Tensor::<B, 2>::random([1024, 5], burn::tensor::Distribution::Default, &device);
    group.bench_function("pinn_forward_1024_sync", |b| {
        b.iter_custom(|iters| {
            synchronized_forward(
                black_box(pinn_input.clone()),
                |x| pinn.forward(x),
                iters as u32,
            )
        })
    });

    let siren = StellarSirenConfig { hidden: 64 }.init::<B>(&device);
    let siren_input =
        Tensor::<B, 2>::random([4096, 5], burn::tensor::Distribution::Default, &device);
    group.bench_function("siren_forward_4096_sync", |b| {
        b.iter_custom(|iters| {
            synchronized_forward(
                black_box(siren_input.clone()),
                |x| siren.forward(x),
                iters as u32,
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_gpu_synchronized);
criterion_main!(benches);
