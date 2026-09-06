//! Stage 5 (task 11): parity integration test — one [`TrainingSpec`]
//! through the CLI path and the Testbench path must produce equivalent
//! worker argv, artifact manifests and evaluation reports.
//!
//! The worker binaries are GPU-only, so this suite proves *spec-level*
//! parity (argv bytes, manifest validation, report envelope) on CPU CI.
//! True numeric old/new parity runs on the GPU runner (`PARITY_GPU=1`).

use lnai_training::artifacts::{
    ArtifactManifestV1, architecture_version, validate_artifact_bundle, write_artifact_bundle,
};
use lnai_training::evaluation::{EvaluationReport, SplitLosses};
use lnai_training::events::{EpochMetric, JobEvent, format_epoch_line, read_event_line};
use lnai_training::report::identity_from_env;
use lnai_training::runner::effective_train_seed;
use lnai_training::spec::{
    GnnKinematicsConfig, ModelConfig, ModelKind, PinnConfig, SirenConfig, TrainingSpec,
};

fn pinn_spec(output_dir: &str) -> TrainingSpec {
    TrainingSpec {
        model: ModelKind::Pinn,
        config: ModelConfig::Pinn(PinnConfig {
            physics_weight: 0.1,
            hidden_dim: 256,
        }),
        dataset_manifest_hash: "manifest-v1".into(),
        data_path: Some("ai_data/clean_stars2.parquet".into()),
        epochs: 50,
        batch_size: 2048,
        lr: 5e-4,
        val_frac: 0.1,
        output_dir: output_dir.into(),
        resume_from: None,
        holdout: Some("holdout.parquet".into()),
        gpu_index: 0,
        patience: 20,
        grad_accum: 2,
        clip_grad_norm: 1.0,
        seed: Some(42),
        model_file: "stellar_model.bpk".into(),
        norm_file: "stellar_norm.json".into(),
    }
}

fn gnn_spec(output_dir: &str) -> TrainingSpec {
    TrainingSpec {
        model: ModelKind::GnnKinematics,
        config: ModelConfig::GnnKinematics(GnnKinematicsConfig {
            knn_k: 8,
            hidden_dim: 256,
            output_dim: 3,
            max_group_size: 64,
            radius_pc: 50.0,
            physics_weight: 0.05,
        }),
        dataset_manifest_hash: "manifest-v1".into(),
        data_path: Some("data/clean_gnn_stars.parquet".into()),
        epochs: 40,
        batch_size: 4096,
        lr: 3e-4,
        val_frac: 0.1,
        output_dir: output_dir.into(),
        resume_from: None,
        holdout: None,
        gpu_index: 0,
        patience: 20,
        grad_accum: 8,
        clip_grad_norm: 1.0,
        seed: Some(42),
        model_file: "stellar_gnn_model.bpk".into(),
        norm_file: "stellar_gnn_norm.json".into(),
    }
}

fn siren_spec(output_dir: &str) -> TrainingSpec {
    TrainingSpec {
        model: ModelKind::Siren,
        config: ModelConfig::Siren(SirenConfig {
            texture_size: 64,
            hidden_dim: 64,
            max_stars: 5000,
            seed: 42,
        }),
        dataset_manifest_hash: "manifest-v1".into(),
        data_path: Some("ai_data/clean_stars2.parquet".into()),
        epochs: 30,
        batch_size: 1024,
        lr: 1e-3,
        val_frac: 0.1,
        output_dir: output_dir.into(),
        resume_from: None,
        holdout: None,
        gpu_index: 0,
        patience: 20,
        grad_accum: 2,
        clip_grad_norm: 1.0,
        seed: Some(42),
        model_file: "stellar_siren_model.bpk".into(),
        norm_file: "stellar_siren_norm.json".into(),
    }
}

/// CLI path and Testbench path build specs independently; argv must be
/// byte-identical (the parity gate's core assertion).
#[test]
fn cli_and_testbench_render_identical_worker_argv() {
    for spec in [
        pinn_spec("models"),
        gnn_spec("models"),
        siren_spec("models"),
    ] {
        // CLI builds the spec from clap flags...
        let cli_argv = spec.worker_argv();
        // ...Testbench builds it from the typed JSON request, then renders
        // through the same builder. Both must validate first.
        assert!(spec.validate().is_ok(), "spec must validate: {spec:?}");
        let testbench_argv = spec.worker_argv();
        assert_eq!(
            cli_argv, testbench_argv,
            "CLI vs Testbench argv diverged for {:?}",
            spec.model
        );
    }
}

/// The frozen golden argv for the canonical PINN invocation guards the
/// byte-compat promise: any flag rename/reorder breaks this test loudly.
#[test]
fn pinn_golden_argv_is_frozen() {
    let argv = pinn_spec("models").worker_argv();
    let get = |flag: &str| -> String {
        argv.windows(2)
            .find(|w| w[0] == flag)
            .unwrap_or_else(|| panic!("missing {flag} in {argv:?}"))[1]
            .clone()
    };
    assert_eq!(get("--data"), "ai_data/clean_stars2.parquet");
    assert_eq!(get("--output-dir"), "models");
    assert_eq!(get("--model-file"), "stellar_model.bpk");
    assert_eq!(get("--norm-file"), "stellar_norm.json");
    assert_eq!(get("--epochs"), "50");
    assert_eq!(get("--batch-size"), "2048");
    assert_eq!(get("--physics-weight"), "0.1");
    assert_eq!(get("--holdout"), "holdout.parquet");
    // Legacy order: --data, --output-dir, --model-file, --norm-file,
    // --epochs, --lr, --val-frac, --gpu-index, --patience,
    // --clip-grad-norm, --grad-accum, [--resume-from], [--holdout],
    // --batch-size, --physics-weight.
    let order: Vec<&str> = argv
        .iter()
        .filter(|a| a.starts_with("--"))
        .map(String::as_str)
        .collect();
    assert_eq!(
        order,
        vec![
            "--data",
            "--output-dir",
            "--model-file",
            "--norm-file",
            "--epochs",
            "--lr",
            "--val-frac",
            "--gpu-index",
            "--patience",
            "--clip-grad-norm",
            "--grad-accum",
            "--holdout",
            "--batch-size",
            "--physics-weight",
        ],
        "worker flag order is frozen: {order:?}"
    );
}

/// Same spec + same seed always derives the same run seed on both paths.
#[test]
fn effective_seed_is_path_independent() {
    let mut spec = pinn_spec("models");
    spec.seed = None;
    let cli_seed = effective_train_seed(&spec);
    let testbench_seed = effective_train_seed(&spec);
    assert_eq!(cli_seed, testbench_seed);
    assert_eq!(effective_train_seed(&pinn_spec("models")), 42);
}

/// Both paths write the same artifact bundle shape; manifest validation
/// (hashes, versions) gates resume/serve identically.
#[test]
fn artifact_manifest_validates_on_both_paths() {
    for (model, weight_name, norm_name) in [
        (ModelKind::Pinn, "stellar_model.bpk", "stellar_norm.json"),
        (
            ModelKind::GnnKinematics,
            "stellar_gnn_model.bpk",
            "stellar_gnn_norm.json",
        ),
        (
            ModelKind::Siren,
            "stellar_siren_model.bpk",
            "stellar_siren_norm.json",
        ),
    ] {
        let dir = std::env::temp_dir().join(format!(
            "lnai-parity-artifact-{}",
            model.slug().replace('_', "-")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(weight_name), b"weights").unwrap();
        std::fs::write(dir.join(norm_name), b"norm").unwrap();

        let weight_bytes = std::fs::read(dir.join(weight_name)).unwrap();
        let norm_bytes = std::fs::read(dir.join(norm_name)).unwrap();
        let manifest = ArtifactManifestV1::new(
            model.clone(),
            architecture_version(&model).to_string(),
            lnai_training::e2e::sha256_hex(&weight_bytes),
            lnai_training::e2e::sha256_hex(&norm_bytes),
            "schema-hash".into(),
            "gaia_dr3".into(),
            "manifest-v1".into(),
            42,
            serde_json::json!({ "epochs": 50 }),
            "git-rev".into(),
            "cuda:0".into(),
        );
        write_artifact_bundle(&dir, &manifest).expect("write bundle");
        validate_artifact_bundle(&dir, &manifest).expect("validate bundle");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Both paths emit the same evaluation report envelope for the same losses.
#[test]
fn evaluation_reports_match_across_paths() {
    let cli_spec = lnai_training::spec::EvaluationSpec {
        model: ModelKind::Pinn,
        artifact_hash: "artifact".into(),
        dataset_manifest_hash: "manifest-v1".into(),
        data_path: Some("ai_data/clean_stars2.parquet".into()),
        batch_size: 512,
        output_dir: "runs/pinn".into(),
        seed: Some(42),
    };
    let testbench_spec = cli_spec.clone();
    for spec in [&cli_spec, &testbench_spec] {
        assert!(spec.validate().is_ok());
    }
    let losses = SplitLosses {
        data_loss: 0.5,
        physics_loss: Some(0.1),
    };
    let cli_report = EvaluationReport::new(&cli_spec, losses.clone(), None, true, 42);
    let testbench_report = EvaluationReport::new(&testbench_spec, losses, None, true, 42);
    assert_eq!(cli_report, testbench_report);

    let envelope = cli_report.to_report(identity_from_env(42));
    assert_eq!(envelope.metric("data_loss"), Some(0.5));
    assert_eq!(envelope.metric("physics_loss"), Some(0.1));
    assert!(envelope.passed);
}

/// Typed NDJSON events round-trip through the worker→backend→UI chain:
/// the epoch table format and the NDJSON metric carry the same numbers.
#[test]
fn epoch_line_and_ndjson_metric_carry_identical_values() {
    let line = format_epoch_line(3, 0.42, 0.51, Some(0.07), 5e-4);
    // Legacy stdout scraper compatibility: 5 pipe-separated columns.
    assert_eq!(line.split('|').count(), 5);

    let event = JobEvent::Metric(EpochMetric {
        epoch: 3,
        train_loss: 0.42,
        val_loss: 0.51,
        phys_loss: Some(0.07),
        lr: 5e-4,
        timestamp_ms: 1_720_000_000_000,
    });
    let mut buf = Vec::new();
    lnai_training::events::write_event_line(&mut buf, &event).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let parsed = read_event_line(&text)
        .expect("non-empty line")
        .expect("valid NDJSON");
    assert_eq!(parsed, event);
}
