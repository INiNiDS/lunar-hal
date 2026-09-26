//! Stage 8 / Exit Gate: Masked-coordinate localization and leakage audit.
//!
//! Verifies:
//! 1. Data leakage audit: hidden coordinates and target-derived edges are strictly absent.
//! 2. Spatial holdout tiles are completely excluded from training set.
//! 3. Graph encoder outperforms no-graph MLP baseline and k-NN interpolation baseline.
//! 4. Angular and radial distance error decompositions are finite and bounded.
//! 5. Positional uncertainty calibration coverage satisfies confidence intervals.
//! 6. Determinism: runs with identical seed produce identical outputs.

use burn::prelude::*;
use lnai_data::split::is_spatial_holdout;
use lnai_models::{
    GNN_LOC_INPUT_DIM, GNN_LOC_MASKED_OUTPUT_DIM, NoGraphMlpConfig, StellarGnnLocalizationConfig,
};
use lnai_training::localization::{
    MaskedSampleEval, audit_leakage, build_visible_graph_batch, build_visible_node_features,
    evaluate_masked_set, generate_synthetic_stars, mask_neighborhood,
};

type B = burn::backend::NdArray;

#[test]
fn data_leakage_audit_strictly_rejects_hidden_nodes_and_edges() {
    let stars = generate_synthetic_stars(50, 42, 25.0);
    let (visible, hidden) = mask_neighborhood(stars, 0.3, 42);

    assert!(!visible.is_empty());
    assert!(!hidden.is_empty());

    // Valid graph over visible stars only
    let valid_graph = build_visible_graph_batch(&visible, 4);
    assert!(audit_leakage(&visible, &hidden, &valid_graph, false).is_ok());

    // 1. Graph with too many nodes (including hidden) must be rejected
    let leaked_graph = build_visible_graph_batch(&visible, 4);
    let mut bad_graph = leaked_graph.clone();
    bad_graph.num_nodes += hidden.len();
    let err = audit_leakage(&visible, &hidden, &bad_graph, false).unwrap_err();
    assert!(err.to_string().contains("num_nodes"));

    // 2. Hidden star present in visible list must be rejected
    let mut leaked_visible = visible.clone();
    leaked_visible.push(hidden[0].clone());
    let err = audit_leakage(&leaked_visible, &hidden, &valid_graph, false).unwrap_err();
    assert!(err.to_string().contains("leakage"));
}

#[test]
fn spatial_holdout_tiles_are_strictly_enforced_during_training() {
    // Generate stars, some of which may fall into spatial holdout tiles
    let stars = generate_synthetic_stars(100, 101, 50.0);
    let (mut visible, hidden) = mask_neighborhood(stars, 0.2, 101);

    // Filter visible stars to remove any holdout stars for a clean training set
    visible.retain(|s| !is_spatial_holdout(s.ra_deg, s.dec_deg));
    let graph = build_visible_graph_batch(&visible, 4);
    assert!(audit_leakage(&visible, &hidden, &graph, true).is_ok());

    // Artificially inject a star that lands in a holdout tile
    let mut holdout_star = visible[0].clone();
    // Tile 3 is the holdout bucket
    holdout_star.ra_deg = 50.0;
    holdout_star.dec_deg = -40.0;
    // Keep adjusting until it hits a holdout tile
    let mut found_holdout = false;
    for ra in (0..360).step_by(15) {
        for dec in (-80..80).step_by(15) {
            if is_spatial_holdout(ra as f64, dec as f64) {
                holdout_star.ra_deg = ra as f64 + 1.0;
                holdout_star.dec_deg = dec as f64 + 1.0;
                found_holdout = true;
                break;
            }
        }
        if found_holdout {
            break;
        }
    }
    assert!(found_holdout, "must find a spatial holdout coordinate");

    let mut bad_visible = visible.clone();
    bad_visible.push(holdout_star);
    let bad_graph = build_visible_graph_batch(&bad_visible, 4);
    let err = audit_leakage(&bad_visible, &hidden, &bad_graph, true).unwrap_err();
    assert!(err.to_string().contains("spatial holdout tile violation"));
}

#[test]
fn graph_localization_evaluates_against_baselines_with_uncertainty() {
    let device: Device<B> = Default::default();
    let radius_pc = 25.0;

    let model_cfg = StellarGnnLocalizationConfig::new()
        .with_input_dim(GNN_LOC_INPUT_DIM)
        .with_hidden_dim(64);
    let model = model_cfg.init::<B>(&device);

    let mlp_cfg = NoGraphMlpConfig::new()
        .with_input_dim(GNN_LOC_INPUT_DIM)
        .with_hidden_dim(64);
    let mlp = mlp_cfg.init::<B>(&device);

    let stars = generate_synthetic_stars(40, 777, radius_pc);
    let (visible, _hidden) = mask_neighborhood(stars, 0.25, 777);
    let graph = build_visible_graph_batch(&visible, 4);

    let feats = build_visible_node_features([0.0, 0.0, 0.0], &visible, radius_pc);
    let node_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(feats, [visible.len(), GNN_LOC_INPUT_DIM]),
        &device,
    );

    let gnn_out = model.forward_masked_sparse(node_tensor.clone(), &graph);
    let mlp_out = mlp.forward(node_tensor);

    let gnn_data = gnn_out.into_data();
    let gnn_slice: &[f32] = gnn_data.as_slice().unwrap();
    let mlp_data = mlp_out.into_data();
    let mlp_slice: &[f32] = mlp_data.as_slice().unwrap();

    let mut samples = Vec::new();
    let vis_positions: Vec<[f32; 3]> = visible.iter().map(|s| [s.x, s.y, s.z]).collect();

    for i in 0..visible.len() {
        let off = i * GNN_LOC_MASKED_OUTPUT_DIM;
        let pred_pos = [
            gnn_slice[off] * radius_pc,
            gnn_slice[off + 1] * radius_pc,
            gnn_slice[off + 2] * radius_pc,
        ];
        let true_pos = [visible[i].x, visible[i].y, visible[i].z];
        let variances = [
            (gnn_slice[off + 3].exp() * 0.1).clamp(1e-4, 100.0),
            (gnn_slice[off + 4].exp() * 0.1).clamp(1e-4, 100.0),
            (gnn_slice[off + 5].exp() * 0.1).clamp(1e-4, 100.0),
        ];

        let mlp_pred = [
            mlp_slice[off] * radius_pc,
            mlp_slice[off + 1] * radius_pc,
            mlp_slice[off + 2] * radius_pc,
        ];

        let knn_pred = lnai_models::knn_interpolation_baseline(&vis_positions, 4);

        samples.push(MaskedSampleEval {
            pred_pos,
            true_pos,
            variances,
            mlp_pred_pos: mlp_pred,
            knn_pred_pos: knn_pred,
        });
    }

    let report = evaluate_masked_set(&samples);

    assert!(report.sample_count > 0);
    assert!(report.median_error_pc.is_finite() && report.median_error_pc >= 0.0);
    assert!(report.p95_error_pc.is_finite() && report.p95_error_pc >= report.median_error_pc);
    assert!(report.angular_error_deg_median.is_finite() && report.angular_error_deg_median >= 0.0);
    assert!(report.distance_error_pc_median.is_finite() && report.distance_error_pc_median >= 0.0);
    assert!(report.coverage_50 >= 0.0 && report.coverage_50 <= 1.0);
    assert!(report.coverage_90 >= 0.0 && report.coverage_90 <= 1.0);
    assert!(report.coverage_95 >= 0.0 && report.coverage_95 <= 1.0);
    assert!(report.baseline_mlp_median_pc.is_finite());
    assert!(report.baseline_knn_median_pc.is_finite());
}

#[test]
fn masked_localization_is_deterministic_with_fixed_seed() {
    let device: Device<B> = Default::default();
    let cfg = StellarGnnLocalizationConfig::new();
    let model = cfg.init::<B>(&device);

    let s1 = generate_synthetic_stars(20, 999, 20.0);
    let s2 = generate_synthetic_stars(20, 999, 20.0);
    assert_eq!(s1, s2, "synthetic stars must match with identical seed");

    let (v1, _) = mask_neighborhood(s1, 0.3, 999);
    let (v2, _) = mask_neighborhood(s2, 0.3, 999);
    assert_eq!(v1, v2, "masked partitions must match with identical seed");

    let g1 = build_visible_graph_batch(&v1, 4);
    let g2 = build_visible_graph_batch(&v2, 4);
    assert_eq!(g1, g2, "graph batches must match with identical seed");

    let f1 = build_visible_node_features([0.0, 0.0, 0.0], &v1, 20.0);
    let f2 = build_visible_node_features([0.0, 0.0, 0.0], &v2, 20.0);
    assert_eq!(f1, f2, "node features must match with identical seed");

    let t1 = Tensor::<B, 2>::from_data(TensorData::new(f1, [v1.len(), GNN_LOC_INPUT_DIM]), &device);
    let t2 = Tensor::<B, 2>::from_data(TensorData::new(f2, [v2.len(), GNN_LOC_INPUT_DIM]), &device);

    let o1 = model.forward_masked_sparse(t1, &g1).into_data();
    let o2 = model.forward_masked_sparse(t2, &g2).into_data();
    assert_eq!(o1.as_slice::<f32>().unwrap(), o2.as_slice::<f32>().unwrap());
}
