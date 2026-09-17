use burn::prelude::*;

/// Contrast-term base weight (legacy value, kept for the uniform path).
const CONTRAST_WEIGHT: f32 = 0.001;

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

    mse + contrast_loss.mul_scalar(CONTRAST_WEIGHT)
}

/// Stage 6 target-aware image loss: same MSE + luma-contrast structure,
/// but the contrast term is weighted per row by the star's activity class
/// read from `conditioning` (`[B, 3]` normalized `(bp_rp, mg, ruwe)` —
/// columns 2..5 of the SIREN inputs).
///
/// Redder (later-type, more active) stars carry stronger spot/granulation
/// texture, so their contrast deviations count up to 3×; blue quiet stars
/// stay near the legacy weight. Uniform (zero) conditioning reproduces the
/// legacy loss up to float summation order.
pub fn compute_siren_loss_conditioned<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    conditioning: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let mse = (predictions.clone() - targets).square().mean();

    let [batch, _] = predictions.dims();

    let pred_r = predictions.clone().slice([0..batch, 0..1]);
    let pred_g = predictions.clone().slice([0..batch, 1..2]);
    let pred_b = predictions.clone().slice([0..batch, 2..3]);

    let luma = pred_r.mul_scalar(0.299) + pred_g.mul_scalar(0.587) + pred_b.mul_scalar(0.114);
    let luma_mean = luma.clone().mean();
    let dev_sq = (luma - luma_mean.reshape([1, 1]).repeat_dim(0, batch)).square();

    // Activity weight from normalized bp_rp (column 0): redder → busier.
    let bp = conditioning.slice([0..batch, 0..1]);
    let activity = bp.clamp(0.0, 2.0);
    let weights = activity.add_scalar(1.0);
    let contrast_loss = (dev_sq * weights).mean();

    mse + contrast_loss.mul_scalar(CONTRAST_WEIGHT)
}

pub fn compute_data_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (predictions - targets).square().mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn scalar(t: Tensor<TestBackend, 1>) -> f32 {
        t.into_scalar()
    }

    #[test]
    fn uniform_conditioning_matches_legacy_loss() {
        let device = Default::default();
        let pred: Tensor<TestBackend, 2> =
            Tensor::from_floats([[0.8, 0.2, 0.4], [0.3, 0.9, 0.1]], &device);
        let truth: Tensor<TestBackend, 2> =
            Tensor::from_floats([[0.7, 0.3, 0.5], [0.4, 0.8, 0.2]], &device);
        let cond: Tensor<TestBackend, 2> =
            Tensor::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);
        let legacy = scalar(compute_siren_loss(pred.clone(), truth.clone()));
        let conditioned = scalar(compute_siren_loss_conditioned(pred, truth, cond));
        assert!(
            (legacy - conditioned).abs() < 1e-6,
            "{legacy} vs {conditioned}"
        );
    }

    #[test]
    fn red_conditioning_upweights_contrast_vs_blue() {
        let device = Default::default();
        // Textured predictions: non-flat luma so the contrast term matters.
        let pred: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], &device);
        let truth: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], &device);
        // MSE is zero here; only the contrast term differs.
        let red: Tensor<TestBackend, 2> =
            Tensor::from_floats([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], &device);
        let blue: Tensor<TestBackend, 2> =
            Tensor::from_floats([[-2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], &device);
        let loss_red = scalar(compute_siren_loss_conditioned(
            pred.clone(),
            truth.clone(),
            red,
        ));
        let loss_blue = scalar(compute_siren_loss_conditioned(pred, truth, blue));
        assert!(loss_red > loss_blue, "{loss_red} vs {loss_blue}");
        // Red weight 3× vs blue weight 1× on identical deviations.
        assert!(
            (loss_red - 3.0 * loss_blue).abs() < 1e-5,
            "{loss_red} vs {loss_blue}"
        );
    }
}
