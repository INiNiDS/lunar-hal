//! Stage 6: unified GNN-Kinematics loss contract.
//!
//! One definition of the loss is shared by the train step, checkpoint
//! selection, evaluation and reporting:
//! `total = data(mean, targets) + physics_weight * physics(mean, targets)
//!        + kl_weight * kl(mean, logvar)` (KL only for the variational head).
//!
//! The physics term is **target-centered**: it pulls the predicted batch
//! mean and variance toward the *observed* batch mean and variance instead
//! of toward zero. The pre-Stage-6 term (`mean² + 0.01·var`, no targets)
//! provably suppressed velocity dispersion; see `docs` history.

use burn::prelude::*;
use lnai_models::{GnnHeadKind, split_mean_logvar, variational_kl};

/// Pure data loss on `[N, 3]` mean velocities (legacy metric, unchanged).
pub fn compute_gnn_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (predictions - targets).square().mean()
}

/// Target-centered physics constraint on `[N, 3]` mean velocities, in
/// normalized units: squared error of batch mean plus (softer) squared
/// error of batch variance, both measured against the batch *targets*.
///
/// Unlike the legacy zero-seeking term this cannot suppress a real
/// velocity dispersion — a high-dispersion group has high target variance
/// and the predictions are pulled toward matching it.
pub fn target_centered_physics<B: Backend>(
    mean_pred: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let mean_p = mean_pred.clone().mean_dim(0);
    let mean_t = targets.clone().mean_dim(0);
    let mean_loss: Tensor<B, 1> = (mean_p.clone() - mean_t.clone()).square().sum().reshape([1]);

    let var_p = (mean_pred - mean_p).square().mean_dim(0);
    let var_t = (targets - mean_t).square().mean_dim(0);
    let var_loss: Tensor<B, 1> = (var_p - var_t).square().sum().reshape([1]);

    mean_loss + var_loss.mul_scalar(0.1)
}

/// Full readout `[N, 3]` (deterministic) or `[N, 6]` (variational) plus
/// `[N, 3]` targets → optimized total. Panics on unsupported widths so a
/// misconfigured head can never train silently.
pub fn compute_gnn_total_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
    kl_weight: f64,
) -> Tensor<B, 1> {
    let [_, width] = predictions.dims();
    let head = GnnHeadKind::from_output_dim(width).expect(
        "gnn readout width must be 3 (deterministic) or 6 (variational), check output_dim",
    );
    let (mean, logvar) = split_mean_logvar(predictions, head);
    let data = compute_gnn_loss(mean.clone(), targets.clone());
    let physics = target_centered_physics(mean.clone(), targets);
    let kl = match (head, logvar) {
        (GnnHeadKind::Variational, Some(logvar)) => variational_kl(mean, logvar),
        _ => Tensor::<B, 1>::zeros([1], &data.device()),
    };
    data + physics.mul_scalar(physics_weight as f32) + kl.mul_scalar(kl_weight as f32)
}

/// Legacy-named entry point, now defined as the unified total (Stage 6).
/// Kept under this name so train, checkpoint selection and evaluation
/// cannot drift apart again.
pub fn compute_gnn_physics_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
) -> Tensor<B, 1> {
    compute_gnn_total_loss(predictions, targets, physics_weight, 0.0)
}

/// Scalar breakdown of one forward pass for logging/reporting:
/// `(total, physics_part)`. The physics part excludes data and KL terms;
/// both values derive from the same `predictions`/`targets` pair.
pub fn gnn_loss_scalars<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
    kl_weight: f64,
) -> (f32, f32) {
    let total: f32 = compute_gnn_total_loss(
        predictions.clone(),
        targets.clone(),
        physics_weight,
        kl_weight,
    )
    .into_scalar()
    .elem();
    let [_, width] = predictions.dims();
    let head = GnnHeadKind::from_output_dim(width).expect(
        "gnn readout width must be 3 (deterministic) or 6 (variational), check output_dim",
    );
    let (mean, _) = split_mean_logvar(predictions, head);
    let physics: f32 = target_centered_physics(mean, targets)
        .mul_scalar(physics_weight as f32)
        .into_scalar()
        .elem();
    (total, physics)
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
    fn target_centered_physics_matches_group_stats_instead_of_zero() {
        let device = Default::default();
        // High-dispersion group: vx in {-100, +100} (normalized units).
        let mean: Tensor<TestBackend, 2> =
            Tensor::from_floats([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]], &device);
        let targets: Tensor<TestBackend, 2> =
            Tensor::from_floats([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]], &device);
        // Perfect match → zero constraint despite huge dispersion: the old
        // zero-seeking term would have returned a large penalty here.
        let phys = scalar(target_centered_physics(mean.clone(), targets.clone()));
        assert!(phys.abs() < 1e-3, "got {phys}");
        // Collapsed predictions (zero dispersion) against dispersed targets
        // must be penalized through the variance term.
        let collapsed: Tensor<TestBackend, 2> =
            Tensor::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);
        let phys = scalar(target_centered_physics(collapsed, targets));
        assert!(phys > 1e6, "collapsed dispersion must be penalized, got {phys}");
    }

    #[test]
    fn total_loss_selects_head_by_width_and_adds_kl() {
        let device = Default::default();
        let targets: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
        // Deterministic: data-only when weights are zero.
        let det: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]], &device);
        let total = scalar(compute_gnn_total_loss(det, targets.clone(), 0.0, 0.0));
        // Squared errors: (0+0+1 + 0+0+0)/6 = 1/6.
        assert!((total - 1.0 / 6.0).abs() < 1e-6, "got {total}");
        // Variational with zero logvar/mean offset: KL > 0 raises total.
        let var: Tensor<TestBackend, 2> = Tensor::from_floats(
            [[1.0, 2.0, 4.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]],
            &device,
        );
        let no_kl = scalar(compute_gnn_total_loss(var.clone(), targets.clone(), 0.0, 0.0));
        let with_kl = scalar(compute_gnn_total_loss(var, targets, 0.0, 1.0));
        assert!(with_kl > no_kl, "{with_kl} vs {no_kl}");
    }

    #[test]
    #[should_panic(expected = "readout width must be 3")]
    fn total_loss_rejects_unknown_widths() {
        let device = Default::default();
        let bad: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let targets: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0]], &device);
        let _ = compute_gnn_total_loss(bad, targets, 0.0, 0.0);
    }
}
