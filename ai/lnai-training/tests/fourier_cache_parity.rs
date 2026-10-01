
use burn::backend::NdArray;
use burn::prelude::*;
use lnai_models::{COND_DIM, StellarMlpConfig, fourier_encode, fourier_encode_cached};

type B = NdArray<f32>;

fn seeded_tensor(rows: usize, cols: usize, seed: u64) -> Tensor<B, 2> {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    B::seed(&device, seed);
    Tensor::<B, 2>::random(
        [rows, cols],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    )
}

#[test]
fn cached_fourier_matches_legacy_encode_exactly() {
    for levels in [1, 8] {
        let xyz = seeded_tensor(17, 3, 42 + levels as u64);
        let legacy: Vec<f32> = fourier_encode(xyz.clone(), levels)
            .into_data()
            .to_vec()
            .expect("legacy encode data");
        let cached: Vec<f32> = fourier_encode_cached(xyz, levels)
            .into_data()
            .to_vec()
            .expect("cached encode data");
        assert_eq!(legacy.len(), cached.len());
        assert!(
            legacy == cached,
            "cached encode must be bitwise identical to legacy (levels={levels})"
        );
    }
}

#[test]
fn forward_on_cached_path_is_deterministic_finite_and_shaped() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    B::seed(&device, 7);
    let model = StellarMlpConfig {
        hidden: 32,
        hidden2: 16,
        hidden3: 8,
        layer_norm_eps: 1e-5,
    }
    .init::<B>(&device);
    let xyz = seeded_tensor(8, 3, 99);
    let cond = Tensor::<B, 2>::zeros([8, COND_DIM], &device);
    let input = Tensor::cat(vec![xyz, cond], 1);
    let first: Vec<f32> = model
        .forward(input.clone())
        .into_data()
        .to_vec()
        .expect("forward data");
    let second: Vec<f32> = model
        .forward(input)
        .into_data()
        .to_vec()
        .expect("forward data");
    assert_eq!(first.len(), 8 * 4);
    assert_eq!(first, second, "forward must be deterministic");
    assert!(
        first.iter().all(|v| v.is_finite()),
        "finite gate on cached forward path"
    );
}
