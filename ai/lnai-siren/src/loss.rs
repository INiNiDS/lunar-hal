use burn::prelude::*;

pub fn compute_siren_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let mse = (predictions.clone() - targets).square().mean();

    let [batch, _dims] = predictions.dims();

    let pred_r = predictions.clone().slice([0..batch, 0..1]);
    let pred_g = predictions.clone().slice([0..batch, 1..2]);
    let pred_b = predictions.clone().slice([0..batch, 2..3]);

    let luma = pred_r.mul_scalar(0.299) + pred_g.mul_scalar(0.587) + pred_b.mul_scalar(0.114);
    let luma_mean = luma.clone().mean();

    let luma_mean_expanded = luma_mean.clone().reshape([1, 1]).repeat_dim(0, batch);
    let contrast_loss = (luma - luma_mean_expanded).square().mean();

    mse + contrast_loss.mul_scalar(0.001)
}

pub fn compute_data_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (predictions - targets).square().mean()
}