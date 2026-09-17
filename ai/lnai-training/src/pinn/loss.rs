use burn::prelude::*;

use crate::spec::PinnLossKind;

use super::dataset::NormParams;

const LOG_T_SUN: f64 = 3.5617974672827754;

/// Huber elementwise penalty: quadratic inside `|err| <= delta`, linear
/// past the knee (in whatever units `err` carries).
fn huber_penalty<B: Backend>(err: Tensor<B, 2>, delta: f32) -> Tensor<B, 2> {
    let clipped = err.clone().clamp(-delta, delta);
    let quad = clipped.clone().square().mul_scalar(0.5);
    let linear = err
        .abs()
        .sub(clipped.abs())
        .mul_scalar(delta);
    quad + linear
}

/// Weighted mean of a per-element penalty over `[B, 4]` errors with
/// per-target weights in PINN target order: `sum(w_i * mean_i) / sum(w)`.
fn weighted_target_mean<B: Backend>(penalty: Tensor<B, 2>, weights: &[f32; 4]) -> Tensor<B, 1> {
    let w_sum: f32 = weights.iter().sum();
    let w = Tensor::<B, 1>::from_floats(*weights, &penalty.device()).unsqueeze::<2>();
    let per_target = penalty.mean_dim(0);
    (per_target * w).sum().div_scalar(w_sum.max(f32::EPSILON))
}

/// Stefan–Boltzmann residual in dex, **normalized** by the luminosity
/// target scale so the physics term is O(1) like the normalized data
/// loss (Stage 6: previously raw dex², incomparable with data loss and
/// implicitly re-scaling `physics_weight` per dataset).
fn normalized_sb_residual<B: Backend>(
    predictions: Tensor<B, 2>,
    norm: &NormParams,
) -> Tensor<B, 2> {
    let [batch, _] = predictions.dims();
    let log_t_pred = predictions.clone().slice([0..batch, 0..1]);
    let log_r_pred = predictions.clone().slice([0..batch, 1..2]);
    let log_l_pred = predictions.slice([0..batch, 3..4]);

    let log_t = denorm(log_t_pred, norm.log_teff_mean, norm.log_teff_std);
    let log_r = denorm(log_r_pred, norm.log_rad_mean, norm.log_rad_std);
    let log_l = denorm(log_l_pred, norm.log_lum_mean, norm.log_lum_std);

    let sb_lhs = log_l;
    let sb_rhs = log_r * 2.0 + (log_t - LOG_T_SUN as f32) * 4.0;

    (sb_lhs - sb_rhs).div_scalar(norm.log_lum_std.max(f32::EPSILON))
}

fn denorm<B: Backend>(tensor: Tensor<B, 2>, mean: f32, std: f32) -> Tensor<B, 2> {
    let scaled = tensor * std;
    scaled + mean
}

pub fn compute_pinn_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
    norm: &NormParams,
    loss_kind: PinnLossKind,
    huber_delta: f32,
    target_weights: &[f32; 4],
) -> Tensor<B, 1> {
    let err = predictions.clone() - targets;
    // Legacy fast path: plain MSE keeps bit-identical numerics for every
    // pre-Stage-6 spec (uniform weights + MSE).
    let data_loss = if loss_kind == PinnLossKind::Mse
        && *target_weights == [1.0, 1.0, 1.0, 1.0]
    {
        err.square().mean()
    } else {
        let penalty = match loss_kind {
            PinnLossKind::Mse => err.square(),
            PinnLossKind::Huber => huber_penalty(err, huber_delta.max(f32::EPSILON)),
        };
        weighted_target_mean(penalty, target_weights)
    };

    let residual = normalized_sb_residual(predictions, norm);
    let physics_loss = residual.square().mean();
    data_loss + physics_loss * physics_weight as f32
}

pub fn compute_data_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (predictions - targets).square().mean()
}

pub fn compute_physics_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    norm: &NormParams,
) -> Tensor<B, 1> {
    normalized_sb_residual(predictions, norm).square().mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn unit_norm() -> NormParams {
        NormParams {
            x_mean: 0.0,
            x_std: 1.0,
            y_mean: 0.0,
            y_std: 1.0,
            z_mean: 0.0,
            z_std: 1.0,
            bp_rp_mean: 0.0,
            bp_rp_std: 1.0,
            mg_mean: 0.0,
            mg_std: 1.0,
            log_teff_mean: 0.0,
            log_teff_std: 1.0,
            log_rad_mean: 0.0,
            log_rad_std: 1.0,
            log_mass_mean: 0.0,
            log_mass_std: 1.0,
            log_lum_mean: 0.0,
            log_lum_std: 1.0,
        }
    }

    fn scalar(t: Tensor<TestBackend, 1>) -> f32 {
        t.into_scalar()
    }

    #[test]
    fn legacy_mse_path_matches_plain_mean_of_squares() {
        let device = Default::default();
        let pred: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]], &device);
        let truth: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.5, 2.0, 2.0, 5.0], [0.5, -0.5, 2.0, 1.0]], &device);
        let norm = unit_norm();
        let full = scalar(compute_pinn_loss(
            pred.clone(),
            truth.clone(),
            0.0,
            &norm,
            PinnLossKind::Mse,
            1.0,
            &[1.0, 1.0, 1.0, 1.0],
        ));
        let plain = scalar(compute_data_loss(pred, truth));
        assert!((full - plain).abs() < 1e-6, "{full} vs {plain}");
    }

    #[test]
    fn huber_matches_half_mse_for_small_errors_and_linear_for_large() {
        let device = Default::default();
        let norm = unit_norm();
        // All errors 0.1 << delta 10: Huber == 0.5 * MSE.
        let pred: Tensor<TestBackend, 2> = Tensor::from_floats([[0.1, -0.1, 0.1, -0.1]], &device);
        let truth: Tensor<TestBackend, 2> = Tensor::from_floats([[0.0, 0.0, 0.0, 0.0]], &device);
        let huber = scalar(compute_pinn_loss(
            pred.clone(),
            truth.clone(),
            0.0,
            &norm,
            PinnLossKind::Huber,
            10.0,
            &[1.0, 1.0, 1.0, 1.0],
        ));
        let mse = scalar(compute_data_loss(pred, truth));
        assert!((huber - 0.5 * mse).abs() < 1e-6, "{huber} vs {mse}");
        // Single error 4.0 >> delta 1.0: penalty == 1*(4 - 0.5) == 3.5.
        let pred: Tensor<TestBackend, 2> = Tensor::from_floats([[4.0, 0.0, 0.0, 0.0]], &device);
        let truth: Tensor<TestBackend, 2> = Tensor::from_floats([[0.0, 0.0, 0.0, 0.0]], &device);
        let huber = scalar(compute_pinn_loss(
            pred,
            truth,
            0.0,
            &norm,
            PinnLossKind::Huber,
            1.0,
            &[1.0, 0.0, 0.0, 0.0],
        ));
        assert!((huber - 3.5).abs() < 1e-5, "got {huber}");
    }

    #[test]
    fn target_weights_select_columns() {
        let device = Default::default();
        let norm = unit_norm();
        let pred: Tensor<TestBackend, 2> = Tensor::from_floats([[2.0, 0.0, 0.0, 0.0]], &device);
        let truth: Tensor<TestBackend, 2> = Tensor::from_floats([[0.0, 0.0, 0.0, 0.0]], &device);
        // Only teff weighted: mean over batch of 2^2.
        let weighted = scalar(compute_pinn_loss(
            pred.clone(),
            truth.clone(),
            0.0,
            &norm,
            PinnLossKind::Mse,
            1.0,
            &[1.0, 0.0, 0.0, 0.0],
        ));
        assert!((weighted - 4.0).abs() < 1e-6, "got {weighted}");
        // Uniform explicit weights == plain MSE up to fp summation order.
        let uniform = scalar(compute_pinn_loss(
            pred,
            truth,
            0.0,
            &norm,
            PinnLossKind::Mse,
            1.0,
            &[1.0, 1.0, 1.0, 1.0],
        ));
        assert!((uniform - 1.0).abs() < 1e-5, "got {uniform}");
    }

    #[test]
    fn physics_residual_is_zero_for_consistent_rows_and_scaled_by_lum_std() {
        let device = Default::default();
        let norm = unit_norm();
        // SB-consistent in denormalized space: lum = 2*rad + 4*(teff - T_sun).
        let teff = LOG_T_SUN as f32;
        let pred: Tensor<TestBackend, 2> =
            Tensor::from_floats([[teff, 0.5, 0.0, 1.0]], &device);
        let phys = scalar(compute_physics_loss(pred, &norm));
        assert!(phys.abs() < 1e-6, "got {phys}");
        // Scale behaviour: the same physical row under a different norm
        // yields the same physical residual, divided by that norm's lum
        // scale — loss scales with 1/std².
        let teff_p = 3.7f32;
        let rad_p = 0.5f32;
        // SB-consistent luminosity, plus a +1 dex violation.
        let lum_p = 2.0 * rad_p + 4.0 * (teff_p - LOG_T_SUN as f32) + 1.0;
        let unit = unit_norm();
        let row_unit: Tensor<TestBackend, 2> =
            Tensor::from_floats([[teff_p, rad_p, 0.0, lum_p]], &device);
        let loss_unit = scalar(compute_physics_loss(row_unit, &unit));
        assert!((loss_unit - 1.0).abs() < 1e-5, "got {loss_unit}");
        let scaled_norm = NormParams {
            log_teff_mean: 0.5,
            log_teff_std: 2.0,
            log_rad_mean: 0.1,
            log_rad_std: 2.0,
            log_mass_mean: 0.0,
            log_mass_std: 2.0,
            log_lum_mean: 0.2,
            log_lum_std: 4.0,
            ..unit_norm()
        };
        let row_scaled: Tensor<TestBackend, 2> = Tensor::from_floats(
            [[
                (teff_p - 0.5) / 2.0,
                (rad_p - 0.1) / 2.0,
                0.0,
                (lum_p - 0.2) / 4.0,
            ]],
            &device,
        );
        let loss_scaled = scalar(compute_physics_loss(row_scaled, &scaled_norm));
        assert!((loss_scaled - 1.0 / 16.0).abs() < 1e-5, "got {loss_scaled}");
    }
}
