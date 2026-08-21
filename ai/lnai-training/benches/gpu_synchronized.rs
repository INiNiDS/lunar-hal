//! Synchronized GPU harness (report-only, Stage 3 task 7).
//!
//! Unlike the CPU benches, every iteration here ends with an explicit
//! device-to-host readback so timings include the full GPU pipeline and are
//! not hidden by async execution. Compiled **only** with
//! `--features gpu-harness` and intended for the nightly/manual GPU runner:
//!
//! ```bash
//! cargo bench -p lnai-training --features gpu-harness --bench gpu_synchronized
//! ```

#![cfg(feature = "gpu-harness")]

use burn::backend::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::prelude::*;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use lnai_models::{StellarMlpConfig, StellarSirenConfig};

type B = Autodiff<Cuda>;

/// Runs `iterations` forwards and forces a synchronization each iteration by
/// reading one element back to the host.
fn synchronized_forward_ms(
    xs: Tensor<B, 2>,
    run: impl Fn(Tensor<B, 2>) -> Tensor<B, 2>,
    iterations: u32,
) -> f64 {
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let out = run(xs.clone());
        // Explicit readback = device synchronization.
        black_box(out.slice([0..1, 0..1]).into_scalar());
    }
    start.elapsed().as_secs_f64() * 1000.0
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
            synchronized_forward_ms(
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
            synchronized_forward_ms(
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
