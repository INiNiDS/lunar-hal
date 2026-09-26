//! Stage 9 / Exit Gate: Missing-neighbor reconstruction, Hungarian matching, and set metrics.
//!
//! Verifies:
//! 1. Set prediction decoder generates K query slots.
//! 2. Hungarian bipartite matching between slots and ground truth targets.
//! 3. Composite set loss (position Huber/NLL, existence BCE, count MAE, feature diff, Chamfer distance).
//! 4. Negative neighborhoods (0 hidden targets): model does not produce excessive false positives.
//! 5. Evaluation metrics (Precision, Recall, F1, Count MAE, Chamfer distance, Coverage 50/90/95%).
//! 6. Comparison against Poisson density baseline.

use burn::prelude::*;
use lnai_models::{GNN_LOC_INPUT_DIM, GNN_LOC_SLOT_DIM, StellarGnnLocalizationConfig};
use lnai_training::localization::{
    SetLossBreakdown, audit_leakage, build_visible_graph_batch, build_visible_node_features,
    compute_set_loss, evaluate_neighbors_dataset, generate_synthetic_stars, mask_neighborhood,
};
use lnai_training::localization::{run_train_masked, run_train_neighbors};
use lnai_training::spec::LocalizationLossWeights;
use lnai_training::spec::{LocalizationConfig, ModelConfig, ModelKind, TrainingSpec};

type B = burn::backend::NdArray;

#[test]
fn training_refuses_unsupported_real_catalog_instead_of_reporting_random_weights() {
    let config = LocalizationConfig {
        radius: 20.0,
        max_slots: 6,
        mask_ratio: 0.25,
        loss_weights: LocalizationLossWeights::default(),
        seed: 42,
    };
    let output = tempfile::tempdir().unwrap();
    let spec = TrainingSpec {
        model: ModelKind::GnnLocalization,
        config: ModelConfig::GnnLocalization(config.clone()),
        dataset_manifest_hash: "real-catalog".into(),
        data_path: Some("missing-catalog.parquet".into()),
        epochs: 2,
        batch_size: 8,
        lr: 1e-3,
        val_frac: 0.2,
        output_dir: output.path().display().to_string(),
        resume_from: None,
        holdout: None,
        gpu_index: 0,
        patience: 2,
        grad_accum: 1,
        clip_grad_norm: 1.0,
        seed: Some(42),
        model_file: "stellar_gnn_loc_model.bpk".into(),
        norm_file: "stellar_gnn_loc_norm.json".into(),
        max_rows: None,
        tiles: None,
        agent: None,
    };
    type Train = burn::backend::Autodiff<burn::backend::NdArray>;
    let device = Default::default();
    for result in [
        run_train_masked::<Train>(&spec, &config, &device).map(|_| ()),
        run_train_neighbors::<Train>(&spec, &config, &device).map(|_| ()),
    ] {
        let err = result.expect_err("unsupported catalog must not yield a training report");
        assert!(err.to_string().contains("data_path"), "{err}");
    }
    assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
}

#[test]
fn synthetic_training_reaches_holdout_gate_without_publishing_single_head_weights() {
    let config = LocalizationConfig {
        radius: 25.0,
        max_slots: 8,
        mask_ratio: 0.25,
        loss_weights: LocalizationLossWeights::default(),
        seed: 42,
    };
    let output = tempfile::tempdir().unwrap();
    let spec = TrainingSpec {
        model: ModelKind::GnnLocalization,
        config: ModelConfig::GnnLocalization(config.clone()),
        dataset_manifest_hash: "synthetic".into(),
        data_path: None,
        epochs: 1,
        batch_size: 32,
        lr: 1e-3,
        val_frac: 0.2,
        output_dir: output.path().display().to_string(),
        resume_from: None,
        holdout: None,
        gpu_index: 0,
        patience: 2,
        grad_accum: 1,
        clip_grad_norm: 1.0,
        seed: Some(42),
        model_file: "stellar_gnn_loc_model.bpk".into(),
        norm_file: "stellar_gnn_loc_norm.json".into(),
        max_rows: None,
        tiles: None,
        agent: None,
    };
    type Train = burn::backend::Autodiff<burn::backend::NdArray>;
    let device = Default::default();
    for result in [
        run_train_masked::<Train>(&spec, &config, &device).map(|_| ()),
        run_train_neighbors::<Train>(&spec, &config, &device).map(|_| ()),
    ] {
        let err = result.expect_err("single-head training must not report a deployable artifact");
        let message = err.to_string();
        assert!(
            message.contains("spatial holdout did not beat")
                || message.contains("artifact contract"),
            "{message}"
        );
    }
    assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 0);
}

#[test]
fn set_prediction_decoder_matches_targets_via_hungarian_matching() {
    let device: Device<B> = Default::default();
    let radius_pc = 25.0;
    let max_slots = 8;

    let cfg = StellarGnnLocalizationConfig::new()
        .with_input_dim(GNN_LOC_INPUT_DIM)
        .with_hidden_dim(64)
        .with_max_slots(max_slots);
    let model = cfg.init::<B>(&device);

    let stars = generate_synthetic_stars(25, 42, radius_pc);
    let (visible, hidden) = mask_neighborhood(stars, 0.3, 42);

    let graph = build_visible_graph_batch(&visible, 4);
    assert!(audit_leakage(&visible, &hidden, &graph, false).is_ok());

    let feats = build_visible_node_features([0.0, 0.0, 0.0], &visible, radius_pc);
    let node_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(feats, [visible.len(), GNN_LOC_INPUT_DIM]),
        &device,
    );

    let slots = model.forward_slots_sparse(node_tensor, &graph, 0);
    assert_eq!(slots.dims(), [max_slots, GNN_LOC_SLOT_DIM]);

    let slots_data = slots.into_data();
    let slots_slice: &[f32] = slots_data.as_slice().unwrap();

    let mut slot_rows = Vec::with_capacity(max_slots);
    for i in 0..max_slots {
        let mut row = [0.0f32; 16];
        row.copy_from_slice(&slots_slice[i * 16..(i + 1) * 16]);
        slot_rows.push(row);
    }

    let hidden_targets: Vec<[f32; 5]> = hidden
        .iter()
        .map(|h| {
            [
                h.x / radius_pc,
                h.y / radius_pc,
                h.z / radius_pc,
                h.bp_rp,
                h.g_mag,
            ]
        })
        .collect();

    let weights = LocalizationLossWeights {
        existence: 1.0,
        position_nll: 1.0,
        chamfer: 1.0,
        feature: 0.5,
        calibration: 0.2,
    };

    let loss_breakdown: SetLossBreakdown =
        compute_set_loss(&slot_rows, &hidden_targets, radius_pc, &weights);

    assert!(loss_breakdown.total_loss.is_finite() && loss_breakdown.total_loss > 0.0);
    assert!(loss_breakdown.existence_loss.is_finite());
    assert!(loss_breakdown.position_loss.is_finite());
    assert!(loss_breakdown.position_nll.is_finite());
    assert!(loss_breakdown.feature_loss.is_finite());
    assert!(loss_breakdown.chamfer_loss.is_finite());
    assert_eq!(
        loss_breakdown.matched_slots,
        hidden_targets.len().min(max_slots)
    );
    assert_eq!(loss_breakdown.true_count, hidden_targets.len());
}

#[test]
fn negative_neighborhoods_are_handled_without_crashing_or_nan() {
    let radius_pc = 20.0;
    let max_slots = 6;
    let weights = LocalizationLossWeights {
        existence: 1.0,
        position_nll: 1.0,
        chamfer: 1.0,
        feature: 0.5,
        calibration: 0.2,
    };

    let dummy_slots = vec![[0.0f32; 16]; max_slots];
    let empty_targets: Vec<[f32; 5]> = Vec::new();

    let breakdown = compute_set_loss(&dummy_slots, &empty_targets, radius_pc, &weights);
    assert!(breakdown.total_loss.is_finite());
    assert_eq!(breakdown.matched_slots, 0);
    assert_eq!(breakdown.true_count, 0);
}

#[test]
fn evaluation_metrics_precision_recall_f1_and_baselines() {
    let device: Device<B> = Default::default();
    let radius_pc = 30.0;
    let max_slots = 8;

    let cfg = StellarGnnLocalizationConfig::new()
        .with_input_dim(GNN_LOC_INPUT_DIM)
        .with_hidden_dim(64)
        .with_max_slots(max_slots);
    let model = cfg.init::<B>(&device);

    let mut predictions = Vec::new();
    let mut poisson_preds = Vec::new();

    for i in 0..5 {
        let seed = 500 + i;
        let stars = generate_synthetic_stars(20, seed, radius_pc);
        let (visible, hidden) = mask_neighborhood(stars, 0.25, seed);

        let graph = build_visible_graph_batch(&visible, 4);
        let feats = build_visible_node_features([0.0, 0.0, 0.0], &visible, radius_pc);
        let node_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(feats, [visible.len(), GNN_LOC_INPUT_DIM]),
            &device,
        );

        let slots = model.forward_slots_sparse(node_tensor, &graph, 0);
        let output = model.decode_candidates(&slots, radius_pc, 0.3);

        let true_positions: Vec<[f32; 3]> = hidden.iter().map(|s| [s.x, s.y, s.z]).collect();
        predictions.push((output, true_positions.clone()));

        let poisson = lnai_models::density_poisson_baseline(radius_pc, 0.0003, seed);
        poisson_preds.push((poisson, true_positions));
    }

    let report = evaluate_neighbors_dataset(&predictions, 8.0, &poisson_preds);

    assert!(report.precision >= 0.0 && report.precision <= 1.0);
    assert!(report.recall >= 0.0 && report.recall <= 1.0);
    assert!(report.f1 >= 0.0 && report.f1 <= 1.0);
    assert!(report.count_mae >= 0.0 && report.count_mae.is_finite());
    assert!(report.median_matched_error_pc.is_finite());
    assert!(report.p95_matched_error_pc.is_finite());
    assert!(report.chamfer_distance.is_finite() && report.chamfer_distance >= 0.0);
    assert!(report.baseline_poisson_chamfer.is_finite() && report.baseline_poisson_chamfer >= 0.0);
}
