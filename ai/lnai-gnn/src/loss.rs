use burn::prelude::*;

pub fn compute_gnn_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (predictions - targets).square().mean()
}

pub fn compute_gnn_physics_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
) -> Tensor<B, 1> {
    let data_loss = compute_gnn_loss(predictions.clone(), targets);

    let [n, _] = predictions.dims();
    let pred_vx = predictions.clone().slice([0..n, 0..1]);
    let pred_vy = predictions.clone().slice([0..n, 1..2]);
    let pred_vz = predictions.slice([0..n, 2..3]);

    let mean_vx = pred_vx.clone().mean_dim(0);
    let mean_vy = pred_vy.clone().mean_dim(0);
    let mean_vz = pred_vz.clone().mean_dim(0);

    let momentum_loss = mean_vx.clone().square() + mean_vy.clone().square() + mean_vz.clone().square();

    let vx_var = (pred_vx - mean_vx).square().mean_dim(0);
    let vy_var = (pred_vy - mean_vy).square().mean_dim(0);
    let vz_var = (pred_vz - mean_vz).square().mean_dim(0);

    let kinetic_reg = vx_var + vy_var + vz_var;

    let phys_scalar = momentum_loss + kinetic_reg.mul_scalar(0.01);

    let phys_scalar_1d: Tensor<B, 1> = phys_scalar.reshape([1]);

    data_loss + phys_scalar_1d.mul_scalar(physics_weight)
}

#[allow(dead_code)]
pub fn compute_velocity_conservation_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    adj: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [n, _] = predictions.dims();

    let pred_vx = predictions.clone().slice([0..n, 0..1]);
    let pred_vy = predictions.clone().slice([0..n, 1..2]);
    let pred_vz = predictions.slice([0..n, 2..3]);

    let neighbor_vx = adj.clone().matmul(pred_vx.clone());
    let neighbor_vy = adj.clone().matmul(pred_vy.clone());
    let neighbor_vz = adj.matmul(pred_vz.clone());

    let smooth_loss = (neighbor_vx - pred_vx).square().mean()
        + (neighbor_vy - pred_vy).square().mean()
        + (neighbor_vz - pred_vz).square().mean();

    smooth_loss
}