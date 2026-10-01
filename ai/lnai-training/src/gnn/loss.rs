
use burn::prelude::*;
use lnai_models::{GnnHeadKind, split_mean_logvar, variational_kl};

pub fn compute_gnn_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    (predictions - targets).square().mean()
}

pub fn target_centered_physics<B: Backend>(
    mean_pred: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let mean_p = mean_pred.clone().mean_dim(0);
    let mean_t = targets.clone().mean_dim(0);
    let mean_loss: Tensor<B, 1> = (mean_p.clone() - mean_t.clone())
        .square()
        .sum()
        .reshape([1]);

    let var_p = (mean_pred - mean_p).square().mean_dim(0);
    let var_t = (targets - mean_t).square().mean_dim(0);
    let var_loss: Tensor<B, 1> = (var_p - var_t).square().sum().reshape([1]);

    mean_loss + var_loss.mul_scalar(0.1)
}

pub fn compute_gnn_total_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
    kl_weight: f64,
) -> Tensor<B, 1> {
    let [n, width] = predictions.dims();
    assert!(
        n >= 2,
        "gnn loss requires a group of >= 2 nodes, got {n}: single-node GNN is excluded (Stage 6.6)"
    );
    let head = GnnHeadKind::from_output_dim(width)
        .expect("gnn readout width must be 3 (deterministic) or 6 (variational), check output_dim");
    let (mean, logvar) = split_mean_logvar(predictions, head);
    let data = compute_gnn_loss(mean.clone(), targets.clone());
    let physics = target_centered_physics(mean.clone(), targets);
    let kl = match (head, logvar) {
        (GnnHeadKind::Variational, Some(logvar)) => variational_kl(mean, logvar),
        _ => Tensor::<B, 1>::zeros([1], &data.device()),
    };
    data + physics.mul_scalar(physics_weight as f32) + kl.mul_scalar(kl_weight as f32)
}

pub fn compute_gnn_physics_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
) -> Tensor<B, 1> {
    compute_gnn_total_loss(predictions, targets, physics_weight, 0.0)
}

pub fn gnn_loss_scalars<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    physics_weight: f64,
    kl_weight: f64,
) -> (f32, f32) {
    let [n, _] = predictions.dims();
    assert!(
        n >= 2,
        "gnn loss requires a group of >= 2 nodes, got {n}: single-node GNN is excluded (Stage 6.6)"
    );
    let total: f32 = compute_gnn_total_loss(
        predictions.clone(),
        targets.clone(),
        physics_weight,
        kl_weight,
    )
    .into_scalar()
    .elem();
    let [_, width] = predictions.dims();
    let head = GnnHeadKind::from_output_dim(width)
        .expect("gnn readout width must be 3 (deterministic) or 6 (variational), check output_dim");
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
        let mean: Tensor<TestBackend, 2> =
            Tensor::from_floats([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]], &device);
        let targets: Tensor<TestBackend, 2> =
            Tensor::from_floats([[-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]], &device);
        let phys = scalar(target_centered_physics(mean.clone(), targets.clone()));
        assert!(phys.abs() < 1e-3, "got {phys}");
        let collapsed: Tensor<TestBackend, 2> =
            Tensor::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);
        let phys = scalar(target_centered_physics(collapsed, targets));
        assert!(
            phys > 1e6,
            "collapsed dispersion must be penalized, got {phys}"
        );
    }

    #[test]
    fn total_loss_selects_head_by_width_and_adds_kl() {
        let device = Default::default();
        let targets: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
        let det: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]], &device);
        let total = scalar(compute_gnn_total_loss(det, targets.clone(), 0.0, 0.0));
        assert!((total - 1.0 / 6.0).abs() < 1e-6, "got {total}");
        let var: Tensor<TestBackend, 2> = Tensor::from_floats(
            [
                [1.0, 2.0, 4.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            ],
            &device,
        );
        let no_kl = scalar(compute_gnn_total_loss(
            var.clone(),
            targets.clone(),
            0.0,
            0.0,
        ));
        let with_kl = scalar(compute_gnn_total_loss(var, targets, 0.0, 1.0));
        assert!(with_kl > no_kl, "{with_kl} vs {no_kl}");
    }

    #[test]
    #[should_panic(expected = "readout width must be 3")]
    fn total_loss_rejects_unknown_widths() {
        let device = Default::default();
        let bad: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], &device);
        let targets: Tensor<TestBackend, 2> =
            Tensor::from_floats([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], &device);
        let _ = compute_gnn_total_loss(bad, targets, 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = ">= 2 nodes")]
    fn total_loss_rejects_single_node_batches() {
        let device = Default::default();
        let single: Tensor<TestBackend, 2> = Tensor::from_floats([[1.0, 2.0, 3.0]], &device);
        let targets: Tensor<TestBackend, 2> = Tensor::from_floats([[1.0, 2.0, 3.0]], &device);
        let _ = compute_gnn_total_loss(single, targets, 0.0, 0.0);
    }
}
