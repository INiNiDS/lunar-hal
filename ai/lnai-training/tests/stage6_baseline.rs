//! Stage 6.8: living approved baseline.
//!
//! `ai/fixtures/stage6-approved-baseline.json` records the frozen
//! correctness suite state at the approved commit. This test pins it:
//! the record must parse and cover all six frozen suites, and the key
//! Stage 6 behaviours (finite metrics, determinism, star-disjoint
//! splits, spec gates) are re-asserted through the real library paths so
//! a regression fails here, not in production.

use burn::backend::NdArray;
use burn::prelude::*;
use serde_json::Value;

type B = NdArray<f32>;

fn baseline() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fixtures/stage6-approved-baseline.json"
    );
    let raw = std::fs::read_to_string(path).expect("stage 6 baseline JSON must exist");
    serde_json::from_str(&raw).expect("baseline JSON must parse")
}

#[test]
fn baseline_record_covers_all_frozen_suites() {
    let base = baseline();
    assert_eq!(base["stage"], 6);
    let suites = base["suites"].as_array().expect("suites array");
    let files: Vec<&str> = suites
        .iter()
        .map(|s| s["file"].as_str().expect("suite file"))
        .collect();
    for expected in [
        "ai/lnai-training/tests/fixture_leakage.rs",
        "ai/lnai-training/tests/gnn_oracle.rs",
        "ai/lnai-training/tests/pinn_accuracy.rs",
        "ai/lnai-training/tests/pinn_gnn_chain.rs",
        "ai/lnai-training/tests/position_rollout.rs",
        "ai/lnai-training/tests/training_parity.rs",
    ] {
        assert!(
            files.contains(&expected),
            "baseline must cover {expected}: {files:?}"
        );
    }
    for suite in suites {
        assert_eq!(
            suite["status"], "passed",
            "frozen suite must be green: {}",
            suite["file"]
        );
        assert!(
            suite["tests"].as_u64().unwrap_or(0) > 0,
            "suite must pin a test count: {}",
            suite["file"]
        );
    }
    assert!(
        base["approved_code_commit"]
            .as_str()
            .is_some_and(|s| s.len() == 40),
        "baseline must reference the approved full commit hash"
    );
}

#[test]
fn baseline_pins_finite_deterministic_pinn_metrics() {
    use lnai_training::metrics::pinn::{per_target_metrics, weighted_mean_mse};
    let truth = [[3.6, 0.9, 0.0, 16.2], [3.7, 0.8, 0.1, 16.4]];
    let pred = [[3.61, 0.89, 0.01, 16.25], [3.69, 0.81, 0.09, 16.38]];
    let first = per_target_metrics(&pred, &truth);
    let second = per_target_metrics(&pred, &truth);
    assert_eq!(first, second, "metrics must be deterministic");
    for m in &first {
        assert!(m.mse.is_finite() && m.mae.is_finite(), "finite gate");
    }
    let agg = weighted_mean_mse(&first, &[1.0, 1.0, 1.0, 1.0]).expect("uniform weights");
    assert!(agg.is_finite() && agg >= 0.0);
}

#[test]
fn baseline_pins_gnn_total_and_siren_conditioned_loss_finite() {
    let device = burn::backend::ndarray::NdArrayDevice::default();
    // GNN: two-node group through the unified total (deterministic head).
    let pred: Tensor<B, 2> = Tensor::from_floats([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], &device);
    let targets: Tensor<B, 2> = Tensor::from_floats([[1.1, 1.9, 3.2], [0.4, -1.2, 2.1]], &device);
    let total: f32 =
        lnai_training::gnn::loss::compute_gnn_total_loss(pred, targets, 0.05, 0.0).into_scalar();
    assert!(total.is_finite() && total >= 0.0, "finite gate: {total}");
    // SIREN: conditioned image loss is finite and target-aware.
    let pred: Tensor<B, 2> = Tensor::from_floats([[0.8, 0.2, 0.4]], &device);
    let truth: Tensor<B, 2> = Tensor::from_floats([[0.7, 0.3, 0.5]], &device);
    let cond: Tensor<B, 2> = Tensor::from_floats([[1.5, 0.0, 0.0]], &device);
    let loss: f32 =
        lnai_training::siren::loss::compute_siren_loss_conditioned(pred, truth, cond).into_scalar();
    assert!(loss.is_finite() && loss >= 0.0, "finite gate: {loss}");
}

#[test]
fn baseline_pins_star_disjoint_splits() {
    use lnai_training::siren::split_star_indices;
    let (train, val) = split_star_indices(50, 0.2, 99);
    assert_eq!(val.len(), 10);
    assert_eq!(train.len(), 40);
    let mut all = train.clone();
    all.extend(val.iter().copied());
    all.sort_unstable();
    assert_eq!(all, (0..50).collect::<Vec<_>>(), "no leakage, full cover");
}

#[test]
fn baseline_pins_stage6_spec_gates() {
    use lnai_training::spec::{
        GnnKinematicsConfig, ModelConfig, ModelKind, PinnConfig, PinnLossKind, TrainingSpec,
    };
    let mut pinn = PinnConfig::default();
    pinn.loss = PinnLossKind::Huber;
    pinn.huber_delta = 0.0;
    let spec = TrainingSpec {
        model: ModelKind::Pinn,
        config: ModelConfig::Pinn(pinn),
        dataset_manifest_hash: "m".into(),
        data_path: None,
        epochs: 1,
        batch_size: 8,
        lr: 1e-4,
        val_frac: 0.1,
        output_dir: "x".into(),
        resume_from: None,
        holdout: None,
        gpu_index: 0,
        patience: 1,
        grad_accum: 1,
        clip_grad_norm: 1.0,
        seed: Some(1),
        model_file: "m.bpk".into(),
        norm_file: "n.json".into(),
        max_rows: None,
        tiles: None,
        agent: None,
    };
    assert!(
        spec.validate().is_err(),
        "non-positive huber_delta must fail validation"
    );
    let bad_gnn = GnnKinematicsConfig {
        knn_k: 8,
        hidden_dim: 64,
        output_dim: 7,
        max_group_size: 16,
        radius_pc: 50.0,
        physics_weight: 0.05,
        kl_weight: 0.0,
    };
    let mut spec2 = spec.clone();
    spec2.model = ModelKind::GnnKinematics;
    spec2.config = ModelConfig::GnnKinematics(bad_gnn);
    assert!(
        spec2.validate().is_err(),
        "output_dim outside {{3, 6}} must fail validation"
    );
}
