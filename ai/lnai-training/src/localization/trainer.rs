use anyhow::{Result, bail, ensure};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::ElementConversion;
use burn::tensor::backend::AutodiffBackend;
use lnai_data::split::is_spatial_holdout;
use lnai_models::{
    GNN_LOC_INPUT_DIM, GNN_LOC_SLOT_DIM, GraphBatch, NoGraphMlpBaseline, NoGraphMlpConfig,
    StellarGnnLocalization, StellarGnnLocalizationConfig,
};
use serde::{Deserialize, Serialize};

use crate::localization::dataset::{
    LocalStar, audit_leakage, build_visible_graph_batch, build_visible_node_features,
    generate_synthetic_stars, mask_neighborhood,
};
use crate::localization::evaluation::{
    MaskedEvaluationReport, MaskedSampleEval, NeighborsEvaluationReport, evaluate_masked_set,
    evaluate_neighbors_dataset,
};
use crate::localization::loss::hungarian_match;
use crate::spec::{LocalizationConfig, ModelConfig, TrainingSpec};

/// Normalization data for GNN-Localization model.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LocalizationNorm {
    pub radius_pc: f32,
    pub max_slots: u32,
    pub bp_rp_mean: f32,
    pub bp_rp_std: f32,
    pub g_mag_mean: f32,
    pub g_mag_std: f32,
}

impl Default for LocalizationNorm {
    fn default() -> Self {
        Self {
            radius_pc: 25.0,
            max_slots: 16,
            bp_rp_mean: 1.0,
            bp_rp_std: 0.5,
            g_mag_mean: 15.0,
            g_mag_std: 2.0,
        }
    }
}

struct MaskedExample {
    visible: Vec<LocalStar>,
    graph: GraphBatch,
    target: [f32; 3],
    anchor: [f32; 3],
}

struct NeighborsExample {
    visible: Vec<LocalStar>,
    graph: GraphBatch,
    hidden: Vec<LocalStar>,
    anchor: [f32; 3],
    seed: u64,
}

// The seed domains and catalog IDs are disjoint across train, validation and
// spatial holdout. No held-out catalog is used for checkpoint selection.
fn split_seed(seed: u64, split: u64, index: u64) -> u64 {
    seed.wrapping_add(split.wrapping_mul(1_000_003))
        .wrapping_add(index)
}

fn check_spec(spec: &TrainingSpec, config: &LocalizationConfig) -> Result<()> {
    spec.validate()
        .map_err(|errors| anyhow::anyhow!("invalid localization spec: {}", errors.join("; ")))?;
    ensure!(
        matches!(&spec.config, ModelConfig::GnnLocalization(cfg) if cfg == config),
        "localization config does not match TrainingSpec"
    );
    ensure!(
        spec.data_path.is_none(),
        "localization data_path is not implemented: refusing to substitute synthetic data for a real catalog"
    );
    ensure!(
        spec.resume_from.is_none() && spec.tiles.is_none() && spec.max_rows.is_none(),
        "localization resume_from, tiles and max_rows are not implemented"
    );
    ensure!(
        spec.val_frac > 0.0,
        "localization requires a nonempty validation split"
    );
    ensure!(
        spec.clip_grad_norm.is_finite() && spec.clip_grad_norm > 0.0,
        "localization clip_grad_norm must be positive and finite"
    );
    ensure!(
        config.max_slots <= 16,
        "localization synthetic training supports at most 16 slots"
    );
    ensure!(
        config.loss_weights.existence
            + config.loss_weights.position_nll
            + config.loss_weights.chamfer
            + config.loss_weights.feature
            + config.loss_weights.calibration
            > 0.0,
        "localization training needs a positive differentiable loss weight"
    );
    Ok(())
}

fn masked_example(seed: u64, radius: f32, holdout: bool) -> Result<MaskedExample> {
    let mut stars = generate_synthetic_stars(if holdout { 256 } else { 28 }, seed, radius);
    stars.retain(|s| is_spatial_holdout(s.ra_deg, s.dec_deg) == holdout);
    masked_from_stars(stars, seed, holdout)
}

fn masked_from_stars(mut stars: Vec<LocalStar>, seed: u64, holdout: bool) -> Result<MaskedExample> {
    ensure!(stars.len() >= 3, "not enough stars in masked split {seed}");
    let target = stars.remove(1);
    let anchor = [stars[0].x, stars[0].y, stars[0].z];
    let mut proxy = stars[0].clone();
    proxy.source_id = format!("masked_{seed}");
    proxy.bp_rp = target.bp_rp;
    proxy.g_mag = target.g_mag;
    proxy.ruwe = target.ruwe;
    // The proxy has only the known anchor position. Its edges cannot depend on
    // the masked star's position; neither can any other visible-visible edge.
    stars.insert(0, proxy);
    let graph = build_visible_graph_batch(&stars, 4);
    let mut hidden = target.clone();
    hidden.is_visible = false;
    audit_leakage(&stars, &[hidden], &graph, !holdout)?;
    Ok(MaskedExample {
        visible: stars,
        graph,
        target: [target.x, target.y, target.z],
        anchor,
    })
}

fn neighbors_example(
    seed: u64,
    config: &LocalizationConfig,
    holdout: bool,
    negative: bool,
) -> Result<NeighborsExample> {
    let mut stars = generate_synthetic_stars(if holdout { 160 } else { 16 }, seed, config.radius);
    stars.retain(|s| is_spatial_holdout(s.ra_deg, s.dec_deg) == holdout);
    ensure!(
        stars.len() >= 2,
        "not enough stars in neighbor split {seed}"
    );
    let (visible, hidden) =
        mask_neighborhood(stars, if negative { 0.0 } else { config.mask_ratio }, seed);
    ensure!(
        hidden.len() <= config.max_slots as usize,
        "hidden count exceeds max_slots in {seed}"
    );
    let graph = build_visible_graph_batch(&visible, 4);
    audit_leakage(&visible, &hidden, &graph, !holdout)?;
    let anchor = [visible[0].x, visible[0].y, visible[0].z];
    Ok(NeighborsExample {
        visible,
        graph,
        hidden,
        anchor,
        seed,
    })
}

fn nodes<B: Backend>(
    visible: &[LocalStar],
    anchor: [f32; 3],
    radius: f32,
    device: &Device<B>,
) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(
            build_visible_node_features(anchor, visible, radius),
            [visible.len(), GNN_LOC_INPUT_DIM],
        ),
        device,
    )
}

fn masked_nodes<B: Backend>(ex: &MaskedExample, radius: f32, device: &Device<B>) -> Tensor<B, 2> {
    let mut features = build_visible_node_features(ex.anchor, &ex.visible, radius);
    features[6] = 0.0;
    features[7] = 0.0;
    Tensor::from_data(
        TensorData::new(features, [ex.visible.len(), GNN_LOC_INPUT_DIM]),
        device,
    )
}

fn masked_loss<B: Backend>(
    out: Tensor<B, 2>,
    target: [f32; 3],
    anchor: [f32; 3],
    radius: f32,
    device: &Device<B>,
) -> Tensor<B, 1> {
    let wanted = Tensor::<B, 2>::from_data(
        TensorData::new(
            (0..3)
                .map(|i| (target[i] - anchor[i]) / radius)
                .collect::<Vec<_>>(),
            [1, 3],
        ),
        device,
    );
    let position = (out.clone().slice([0..1, 0..3]) - wanted)
        .powf_scalar(2.0)
        .mean();
    // Fit a nonzero uncertainty head rather than shipping random covariance.
    let log_variance = out.slice([0..1, 3..6]);
    position + (log_variance + 1.0).powf_scalar(2.0).mean() * 0.001
}

fn masked_forward<B: Backend>(
    model: &StellarGnnLocalization<B>,
    ex: &MaskedExample,
    radius: f32,
    device: &Device<B>,
) -> Tensor<B, 2> {
    model.forward_masked(
        masked_nodes(ex, radius, device),
        ex.graph.to_dense_adjacency(device),
    )
}

fn masked_report<B: Backend>(
    model: &StellarGnnLocalization<B>,
    mlp: &NoGraphMlpBaseline<B>,
    examples: &[MaskedExample],
    radius: f32,
    device: &Device<B>,
) -> MaskedEvaluationReport {
    let mut samples = Vec::with_capacity(examples.len());
    for ex in examples {
        let out = masked_forward(model, ex, radius, device).into_data();
        let pred: &[f32] = out.as_slice().expect("masked f32");
        let baseline = mlp
            .forward(masked_nodes(ex, radius, device).slice([0..1, 0..GNN_LOC_INPUT_DIM]))
            .into_data();
        let mlp_pred: &[f32] = baseline.as_slice().expect("mlp f32");
        let shift = |values: &[f32]| -> [f32; 3] {
            std::array::from_fn(|i| ex.anchor[i] + values[i] * radius)
        };
        let vis_positions: Vec<[f32; 3]> = ex
            .visible
            .iter()
            .skip(1)
            .map(|s| [s.x - ex.anchor[0], s.y - ex.anchor[1], s.z - ex.anchor[2]])
            .collect();
        let knn = lnai_models::knn_interpolation_baseline(&vis_positions, 4);
        samples.push(MaskedSampleEval {
            pred_pos: shift(pred),
            true_pos: ex.target,
            variances: std::array::from_fn(|i| (pred[3 + i].exp() * 0.1).clamp(1e-4, 100.0)),
            mlp_pred_pos: shift(mlp_pred),
            knn_pred_pos: std::array::from_fn(|i| ex.anchor[i] + knn[i]),
        });
    }
    evaluate_masked_set(&samples)
}

fn masked_gate(report: &MaskedEvaluationReport) -> Result<()> {
    ensure!(
        report.sample_count > 0
            && report.median_error_pc.is_finite()
            && report.baseline_mlp_median_pc.is_finite()
            && report.baseline_knn_median_pc.is_finite()
            && report.median_error_pc < report.baseline_mlp_median_pc
            && report.median_error_pc < report.baseline_knn_median_pc,
        "masked spatial holdout did not beat trained MLP and k-NN baselines: {report:?}"
    );
    Ok(())
}

fn checkpoint_refusal() -> Result<()> {
    // A single gnn-loc-v1 weight name represents *both* heads in the serving
    // registry. This run trains one head only; publishing would advertise the
    // other random head as trained. Do not write an unservable checkpoint.
    bail!(
        "localization artifact contract has one shared model for masked and neighbor heads; cannot publish a single-head checkpoint without changes to artifacts/spec and serving"
    )
}

/// Optimizes the masked-coordinate head on leakage-safe synthetic catalogs.
/// Returns an error rather than claiming an unpublishable single-head artifact.
pub fn run_train_masked<B: AutodiffBackend>(
    spec: &TrainingSpec,
    config: &LocalizationConfig,
    device: &Device<B>,
) -> Result<MaskedEvaluationReport> {
    check_spec(spec, config)?;
    let samples = |split, count, holdout| -> Result<Vec<_>> {
        (0..count)
            .map(|i| masked_example(split_seed(config.seed, split, i), config.radius, holdout))
            .collect()
    };
    let train = samples(1, 10, false)?;
    let validation = samples(2, 4, false)?;
    let holdout = samples(3, 8, true)?;
    let model_cfg = StellarGnnLocalizationConfig::new()
        .with_hidden_dim(128)
        .with_max_slots(config.max_slots as usize);
    let mut model = model_cfg.init::<B>(device);
    let mut mlp = NoGraphMlpConfig::new().init::<B>(device);
    let optim_config = AdamWConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(
        spec.clip_grad_norm as f32,
    )));
    let mut optimizer = optim_config.init();
    let mut mlp_optimizer = optim_config.init();
    let mut best = model.clone();
    let mut best_mlp = mlp.clone();
    let mut best_val = f32::INFINITY;
    let mut best_mlp_val = f32::INFINITY;
    for _ in 0..spec.epochs {
        for ex in &train {
            let loss = masked_loss(
                masked_forward(&model, ex, config.radius, device),
                ex.target,
                ex.anchor,
                config.radius,
                device,
            );
            ensure!(
                loss.clone().into_scalar().elem::<f32>().is_finite(),
                "non-finite masked train loss"
            );
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(spec.lr, model, grads);
            let out = mlp.forward(
                masked_nodes(ex, config.radius, device).slice([0..1, 0..GNN_LOC_INPUT_DIM]),
            );
            let loss = masked_loss(out, ex.target, ex.anchor, config.radius, device);
            let grads = GradientsParams::from_grads(loss.backward(), &mlp);
            mlp = mlp_optimizer.step(spec.lr, mlp, grads);
        }
        let val: f32 = validation
            .iter()
            .map(|ex| {
                masked_loss(
                    masked_forward(&model, ex, config.radius, device),
                    ex.target,
                    ex.anchor,
                    config.radius,
                    device,
                )
                .into_scalar()
                .elem::<f32>()
            })
            .sum::<f32>()
            / validation.len() as f32;
        let mlp_val: f32 = validation
            .iter()
            .map(|ex| {
                masked_loss(
                    mlp.forward(
                        masked_nodes(ex, config.radius, device).slice([0..1, 0..GNN_LOC_INPUT_DIM]),
                    ),
                    ex.target,
                    ex.anchor,
                    config.radius,
                    device,
                )
                .into_scalar()
                .elem::<f32>()
            })
            .sum::<f32>()
            / validation.len() as f32;
        ensure!(
            val.is_finite() && mlp_val.is_finite(),
            "non-finite masked validation loss"
        );
        if val < best_val {
            best_val = val;
            best = model.clone();
        }
        if mlp_val < best_mlp_val {
            best_mlp_val = mlp_val;
            best_mlp = mlp.clone();
        }
    }
    let report = masked_report(&best, &best_mlp, &holdout, config.radius, device);
    masked_gate(&report)?;
    checkpoint_refusal()?;
    unreachable!()
}

fn slot_targets(ex: &NeighborsExample, radius: f32) -> Vec<[f32; 5]> {
    ex.hidden
        .iter()
        .map(|s| {
            [
                (s.x - ex.anchor[0]) / radius,
                (s.y - ex.anchor[1]) / radius,
                (s.z - ex.anchor[2]) / radius,
                s.bp_rp,
                s.g_mag,
            ]
        })
        .collect()
}

fn neighbor_forward<B: Backend>(
    model: &StellarGnnLocalization<B>,
    ex: &NeighborsExample,
    radius: f32,
    device: &Device<B>,
) -> Tensor<B, 2> {
    model.forward_slots(
        nodes(&ex.visible, ex.anchor, radius, device),
        ex.graph.to_dense_adjacency(device),
        0,
    )
}

fn neighbor_loss<B: Backend>(
    slots: Tensor<B, 2>,
    targets: &[[f32; 5]],
    config: &LocalizationConfig,
    device: &Device<B>,
) -> Tensor<B, 1> {
    let [k, d] = slots.dims();
    assert_eq!(d, GNN_LOC_SLOT_DIM);
    let raw = slots.clone().into_data();
    let values: &[f32] = raw.as_slice().expect("slot f32");
    let costs: Vec<Vec<f32>> = (0..k)
        .map(|i| {
            targets
                .iter()
                .map(|target| {
                    (0..3)
                        .map(|c| (values[i * d + c + 1] - target[c]).abs())
                        .sum::<f32>()
                        - values[i * d]
                })
                .collect()
        })
        .collect();
    let matches = hungarian_match(&costs);
    let mut loss = Tensor::<B, 1>::zeros([1], device);
    for i in 0..k {
        let label = if matches.iter().any(|(s, _)| *s == i) {
            1.0
        } else {
            0.0
        };
        let logit = slots.clone().slice([i..i + 1, 0..1]);
        let probability = burn::tensor::activation::sigmoid(logit).clamp(1e-5, 1.0 - 1e-5);
        let bce = if label > 0.0 {
            probability.log().mul_scalar(-1.0)
        } else {
            probability
                .mul_scalar(-1.0)
                .add_scalar(1.0)
                .log()
                .mul_scalar(-1.0)
        };
        loss = loss
            + bce
                .reshape([1])
                .mul_scalar(config.loss_weights.existence / k as f32);
    }
    for (i, j) in &matches {
        let target =
            Tensor::<B, 2>::from_data(TensorData::new(targets[*j][..3].to_vec(), [1, 3]), device);
        let position = slots.clone().slice([*i..*i + 1, 1..4]);
        let squared = (position - target).powf_scalar(2.0);
        let scale = 1.0 / matches.len() as f32;
        loss = loss
            + squared.clone().mean().mul_scalar(
                (config.loss_weights.position_nll + config.loss_weights.chamfer) * scale,
            );
        let features =
            Tensor::<B, 2>::from_data(TensorData::new(targets[*j][3..5].to_vec(), [1, 2]), device);
        loss = loss
            + (slots.clone().slice([*i..*i + 1, 10..12]) - features)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(config.loss_weights.feature * 0.01 * scale);
        let log_variance = slots.clone().slice([*i..*i + 1, 4..7]);
        loss = loss
            + (log_variance + 1.0)
                .powf_scalar(2.0)
                .mean()
                .mul_scalar(config.loss_weights.calibration * 0.01 * scale);
    }
    loss
}

fn neighbor_reports<B: Backend>(
    model: &StellarGnnLocalization<B>,
    examples: &[NeighborsExample],
    config: &LocalizationConfig,
    device: &Device<B>,
) -> (NeighborsEvaluationReport, NeighborsEvaluationReport) {
    let mut predictions = Vec::new();
    let mut poisson = Vec::new();
    for ex in examples {
        let slots = neighbor_forward(model, ex, config.radius, device);
        let output = model.decode_candidates(&slots, config.radius, 0.5);
        let truth: Vec<[f32; 3]> = ex
            .hidden
            .iter()
            .map(|s| [s.x - ex.anchor[0], s.y - ex.anchor[1], s.z - ex.anchor[2]])
            .collect();
        let volume = (4.0 / 3.0) * std::f32::consts::PI * config.radius.powi(3);
        let density = ex.visible.len() as f32 * config.mask_ratio
            / (1.0 - config.mask_ratio).max(0.01)
            / volume;
        let baseline = lnai_models::density_poisson_baseline(config.radius, density, ex.seed);
        predictions.push((output, truth.clone()));
        poisson.push((baseline, truth));
    }
    let threshold = config.radius * 0.2;
    (
        evaluate_neighbors_dataset(&predictions, threshold, &poisson),
        evaluate_neighbors_dataset(&poisson, threshold, &poisson),
    )
}

fn neighbors_gate(
    report: &NeighborsEvaluationReport,
    baseline: &NeighborsEvaluationReport,
) -> Result<()> {
    ensure!(
        report.f1.is_finite()
            && report.chamfer_distance.is_finite()
            && baseline.f1.is_finite()
            && baseline.chamfer_distance.is_finite()
            && report.f1 > baseline.f1
            && report.chamfer_distance < baseline.chamfer_distance,
        "neighbor spatial holdout did not beat Poisson F1 and Chamfer baselines: model={report:?}, baseline={baseline:?}"
    );
    Ok(())
}

/// Optimizes the neighbor head with differentiable matched-slot losses.
/// Returns an error rather than claiming an unpublishable single-head artifact.
pub fn run_train_neighbors<B: AutodiffBackend>(
    spec: &TrainingSpec,
    config: &LocalizationConfig,
    device: &Device<B>,
) -> Result<NeighborsEvaluationReport> {
    check_spec(spec, config)?;
    ensure!(
        config.mask_ratio > 0.0 && config.mask_ratio < 1.0,
        "neighbor training needs a positive mask ratio below one and negative examples are added separately"
    );
    let samples = |split, count, holdout| -> Result<Vec<_>> {
        (0..count)
            .map(|i| {
                neighbors_example(
                    split_seed(config.seed, split, i),
                    config,
                    holdout,
                    i % 4 == 0,
                )
            })
            .collect()
    };
    let train = samples(4, 10, false)?;
    let validation = samples(5, 4, false)?;
    let holdout = samples(6, 8, true)?;
    ensure!(
        holdout.iter().any(|ex| !ex.hidden.is_empty())
            && holdout.iter().any(|ex| ex.hidden.is_empty()),
        "neighbor holdout needs both positive and negative neighborhoods"
    );
    let cfg = StellarGnnLocalizationConfig::new()
        .with_hidden_dim(128)
        .with_max_slots(config.max_slots as usize);
    let mut model = cfg.init::<B>(device);
    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(
            spec.clip_grad_norm as f32,
        )))
        .init();
    let mut best = model.clone();
    let mut best_val = f32::INFINITY;
    for _ in 0..spec.epochs {
        for ex in &train {
            let loss = neighbor_loss(
                neighbor_forward(&model, ex, config.radius, device),
                &slot_targets(ex, config.radius),
                config,
                device,
            );
            ensure!(
                loss.clone().into_scalar().elem::<f32>().is_finite(),
                "non-finite neighbor train loss"
            );
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(spec.lr, model, grads);
        }
        let val: f32 = validation
            .iter()
            .map(|ex| {
                neighbor_loss(
                    neighbor_forward(&model, ex, config.radius, device),
                    &slot_targets(ex, config.radius),
                    config,
                    device,
                )
                .into_scalar()
                .elem::<f32>()
            })
            .sum::<f32>()
            / validation.len() as f32;
        ensure!(val.is_finite(), "non-finite neighbor validation loss");
        if val < best_val {
            best_val = val;
            best = model.clone();
        }
    }
    let (report, baseline) = neighbor_reports(&best, &holdout, config, device);
    neighbors_gate(&report, &baseline)?;
    checkpoint_refusal()?;
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    type Train = burn::backend::Autodiff<burn::backend::NdArray>;

    fn config() -> LocalizationConfig {
        LocalizationConfig {
            radius: 25.0,
            max_slots: 8,
            mask_ratio: 0.25,
            loss_weights: Default::default(),
            seed: 42,
        }
    }

    #[test]
    fn masked_target_coordinates_never_change_inputs_or_edges() {
        let device = Default::default();
        let mut stars = generate_synthetic_stars(20, 31, 25.0);
        stars.retain(|s| !is_spatial_holdout(s.ra_deg, s.dec_deg));
        let first = masked_from_stars(stars.clone(), 31, false).unwrap();
        stars[1].x += 20.0;
        stars[1].y -= 10.0;
        let second = masked_from_stars(stars, 31, false).unwrap();
        assert_ne!(first.target, second.target);
        assert_eq!(first.graph, second.graph);
        assert_eq!(
            masked_nodes::<Train>(&first, 25.0, &device).into_data(),
            masked_nodes::<Train>(&second, 25.0, &device).into_data(),
        );
        assert_eq!(first.visible[0].x, first.anchor[0]);
    }

    #[test]
    fn masked_dense_autodiff_reduces_loss() {
        let device = Default::default();
        Train::seed(&device, 42);
        let ex = masked_example(101, 25.0, false).unwrap();
        let mut model = StellarGnnLocalizationConfig::new()
            .with_hidden_dim(16)
            .with_max_slots(4)
            .init::<Train>(&device);
        let mut optimizer = AdamWConfig::new().init();
        let evaluate = |m: &StellarGnnLocalization<Train>| {
            masked_loss(
                masked_forward(m, &ex, 25.0, &device),
                ex.target,
                ex.anchor,
                25.0,
                &device,
            )
        };
        let before = evaluate(&model).into_scalar().elem::<f32>();
        for _ in 0..25 {
            let grads = GradientsParams::from_grads(evaluate(&model).backward(), &model);
            model = optimizer.step(1e-2, model, grads);
        }
        let after = evaluate(&model).into_scalar().elem::<f32>();
        assert!(
            after.is_finite() && after < before * 0.8,
            "masked loss did not fall: {before} -> {after}"
        );
    }

    #[test]
    fn neighbor_dense_autodiff_reduces_positive_and_negative_loss() {
        let device = Default::default();
        Train::seed(&device, 19);
        let cfg = config();
        let positive = neighbors_example(105, &cfg, false, false).unwrap();
        assert!(!positive.hidden.is_empty());
        let negative = neighbors_example(106, &cfg, false, true).unwrap();
        assert!(negative.hidden.is_empty());
        for ex in [&positive, &negative] {
            let mut model = StellarGnnLocalizationConfig::new()
                .with_hidden_dim(16)
                .with_max_slots(8)
                .init::<Train>(&device);
            let mut optimizer = AdamWConfig::new().init();
            let evaluate = |m: &StellarGnnLocalization<Train>| {
                neighbor_loss(
                    neighbor_forward(m, ex, cfg.radius, &device),
                    &slot_targets(ex, cfg.radius),
                    &cfg,
                    &device,
                )
            };
            let before = evaluate(&model).into_scalar().elem::<f32>();
            for _ in 0..30 {
                let grads = GradientsParams::from_grads(evaluate(&model).backward(), &model);
                model = optimizer.step(2e-3, model, grads);
            }
            let after = evaluate(&model).into_scalar().elem::<f32>();
            assert!(
                after.is_finite() && after < before * 0.8,
                "neighbor loss ({} hidden) did not fall: {before} -> {after}",
                ex.hidden.len()
            );
        }
    }

    #[test]
    fn holdout_gates_reject_equal_or_worse_baselines() {
        let mut masked = evaluate_masked_set(&[]);
        masked.sample_count = 1;
        masked.median_error_pc = 2.0;
        masked.baseline_mlp_median_pc = 2.0;
        masked.baseline_knn_median_pc = 3.0;
        assert!(masked_gate(&masked).is_err());
        masked.baseline_mlp_median_pc = 3.0;
        masked.baseline_knn_median_pc = 2.0;
        assert!(masked_gate(&masked).is_err());
        masked.baseline_knn_median_pc = 3.0;
        assert!(masked_gate(&masked).is_ok());

        let mut neighbor = evaluate_neighbors_dataset(&[], 5.0, &[]);
        let mut poisson = neighbor.clone();
        neighbor.f1 = 0.5;
        poisson.f1 = 0.5;
        neighbor.chamfer_distance = 2.0;
        poisson.chamfer_distance = 3.0;
        assert!(neighbors_gate(&neighbor, &poisson).is_err());
        poisson.f1 = 0.4;
        poisson.chamfer_distance = 2.0;
        assert!(neighbors_gate(&neighbor, &poisson).is_err());
        poisson.chamfer_distance = 3.0;
        assert!(neighbors_gate(&neighbor, &poisson).is_ok());
    }
}
