use burn::prelude::*;

const LOG_T_SUN: f64 = 3.5617974672827754;

fn denorm<B: Backend>(tensor: Tensor<B, 2>, mean: f32, std: f32) -> Tensor<B, 2> {
    let scaled = tensor * std;
    scaled + mean
}

pub fn compute_pinn_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
    log_teff_mean: f32,
    log_teff_std: f32,
    log_rad_mean: f32,
    log_rad_std: f32,
    log_lum_mean: f32,
    log_lum_std: f32,
) -> Tensor<B, 1> {
    let [batch, _] = predictions.dims();
    let data_loss = (predictions.clone() - targets).square().mean();

    let log_t_pred = predictions.clone().slice([0..batch, 0..1]);
    let log_r_pred = predictions.clone().slice([0..batch, 1..2]);
    let log_l_pred = predictions.slice([0..batch, 3..4]);

    let log_t = denorm(log_t_pred, log_teff_mean, log_teff_std);
    let log_r = denorm(log_r_pred, log_rad_mean, log_rad_std);
    let log_l = denorm(log_l_pred, log_lum_mean, log_lum_std);

    let sb_lhs = log_l;
    let sb_rhs = log_r * 2.0 + (log_t - LOG_T_SUN as f32) * 4.0;

    let physics_loss = (sb_lhs - sb_rhs).square().mean();
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
    log_teff_mean: f32,
    log_teff_std: f32,
    log_rad_mean: f32,
    log_rad_std: f32,
    log_lum_mean: f32,
    log_lum_std: f32,
) -> Tensor<B, 1> {
    let [batch, _] = predictions.dims();

    let log_t_pred = predictions.clone().slice([0..batch, 0..1]);
    let log_r_pred = predictions.clone().slice([0..batch, 1..2]);
    let log_l_pred = predictions.slice([0..batch, 3..4]);

    let log_t = denorm(log_t_pred, log_teff_mean, log_teff_std);
    let log_r = denorm(log_r_pred, log_rad_mean, log_rad_std);
    let log_l = denorm(log_l_pred, log_lum_mean, log_lum_std);

    let sb_lhs = log_l;
    let sb_rhs = log_r * 2.0 + (log_t - LOG_T_SUN as f32) * 4.0;

    (sb_lhs - sb_rhs).square().mean()
}