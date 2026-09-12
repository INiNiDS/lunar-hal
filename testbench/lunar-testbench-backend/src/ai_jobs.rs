//! Stage 5 (task 10): typed AI job orchestration.
//!
//! The backend no longer hand-assembles worker CLI args. Typed
//! [`TrainingRequest`] / [`EvaluationRequest`] / [`BenchmarkRequest`]
//! payloads convert 1:1 into `lnai-training` specs; worker argv renders
//! through the shared [`worker_argv`](lnai_training::spec::TrainingSpec::worker_argv)
//! builders, so CLI and Testbench spawn byte-identical commands.
//!
//! Stdout stays a log attachment: the epoch table is still scraped for the
//! live metric stream (legacy UI compatibility), but the typed NDJSON
//! `events.ndjson` the worker writes is the authoritative metric protocol
//! (exit gate: "stdout больше не является протоколом метрик").

use std::path::Path;

use axum::{Json, extract::State};
use lunar_structures_testbench::typed::{
    BenchmarkRequest, EvaluationRequest, JobEventDto, ModelKindDto, TrainingRequest,
};
use lunar_structures_testbench::{Job, JobIdPayload, JobKind};
use tokio::process::Command;

use crate::AppState;

/// Converts a Testbench typed training request into the shared library spec.
/// Model-specific options that do not belong to the model kind are rejected
/// here (capability gate) instead of being silently dropped.
pub fn training_spec_from_request(
    req: &TrainingRequest,
) -> Result<lnai_training::spec::TrainingSpec, String> {
    use lnai_training::spec::{
        GnnKinematicsConfig, ModelConfig, ModelKind, PinnConfig, SirenConfig, TrainingSpec,
    };
    let (model, config) = match req.model {
        ModelKindDto::Pinn => {
            reject_unrelated_options(req, "pinn")?;
            (
                ModelKind::Pinn,
                ModelConfig::Pinn(PinnConfig {
                    physics_weight: req.physics_weight,
                    hidden_dim: 256,
                }),
            )
        }
        ModelKindDto::Gnn => {
            reject_unrelated_options(req, "gnn")?;
            (
                ModelKind::GnnKinematics,
                ModelConfig::GnnKinematics(GnnKinematicsConfig {
                    knn_k: req.knn_k.unwrap_or(8),
                    hidden_dim: req.hidden_dim.unwrap_or(256),
                    output_dim: 3,
                    max_group_size: req.max_group_size.unwrap_or(64),
                    radius_pc: req.radius_pc.unwrap_or(50.0),
                    physics_weight: req.physics_weight,
                }),
            )
        }
        ModelKindDto::Siren => {
            reject_unrelated_options(req, "siren")?;
            (
                ModelKind::Siren,
                ModelConfig::Siren(SirenConfig {
                    texture_size: req.texture_size.unwrap_or(64),
                    hidden_dim: 64,
                    max_stars: req.max_stars.unwrap_or(5000),
                    seed: req.seed.unwrap_or(42),
                }),
            )
        }
    };
    let spec = TrainingSpec {
        model,
        config,
        dataset_manifest_hash: req.dataset_manifest_hash.clone().unwrap_or_default(),
        data_path: Some(req.data_path.clone()),
        epochs: req.epochs,
        batch_size: req.batch_size,
        lr: req.lr,
        val_frac: req.val_frac,
        output_dir: req.output_dir.clone(),
        resume_from: req.resume_from.clone(),
        holdout: req.holdout.clone(),
        gpu_index: req.gpu_index,
        patience: req.patience,
        grad_accum: req.grad_accum,
        clip_grad_norm: req.clip_grad_norm,
        seed: req.seed,
        model_file: lnai_training::artifacts::weight_file_name(&match req.model {
            ModelKindDto::Pinn => lnai_training::spec::ModelKind::Pinn,
            ModelKindDto::Gnn => lnai_training::spec::ModelKind::GnnKinematics,
            ModelKindDto::Siren => lnai_training::spec::ModelKind::Siren,
        })
        .to_string(),
        norm_file: lnai_training::artifacts::norm_file_name(&match req.model {
            ModelKindDto::Pinn => lnai_training::spec::ModelKind::Pinn,
            ModelKindDto::Gnn => lnai_training::spec::ModelKind::GnnKinematics,
            ModelKindDto::Siren => lnai_training::spec::ModelKind::Siren,
        })
        .to_string(),
        // Epoch-watch agent / sampling / tiles are CLI-only for now.
        max_rows: None,
        tiles: None,
        agent: None,
    };
    spec.validate().map_err(|errs| errs.join("; "))?;
    Ok(spec)
}

fn reject_unrelated_options(req: &TrainingRequest, kind: &str) -> Result<(), String> {
    // GNN graph options (knn_k/hidden_dim/max_group_size/radius_pc) and
    // SIREN texture options (texture_size/max_stars) belong to their own
    // kinds; every other kind must not carry them. The check is per-kind so
    // GNN keeps its graph flags and SIREN keeps its texture flags.
    let mut stray = Vec::new();
    if kind != "gnn"
        && (req.knn_k.is_some()
            || req.hidden_dim.is_some()
            || req.max_group_size.is_some()
            || req.radius_pc.is_some())
    {
        stray.push("knn_k/hidden_dim/max_group_size/radius_pc (gnn-only)");
    }
    if kind != "siren" && (req.texture_size.is_some() || req.max_stars.is_some()) {
        stray.push("texture_size/max_stars (siren-only)");
    }
    if stray.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{kind} request carries unrelated options: {}",
            stray.join(", ")
        ))
    }
}

/// Converts a typed evaluation request into the shared library spec.
pub fn evaluation_spec_from_request(
    req: &EvaluationRequest,
) -> Result<lnai_training::spec::EvaluationSpec, String> {
    use lnai_training::spec::{EvaluationSpec, ModelKind};
    let model = match req.model {
        ModelKindDto::Pinn => ModelKind::Pinn,
        ModelKindDto::Gnn => ModelKind::GnnKinematics,
        ModelKindDto::Siren => ModelKind::Siren,
    };
    let spec = EvaluationSpec {
        model,
        artifact_hash: req
            .artifact_hash
            .clone()
            .unwrap_or_else(|| "unspecified".to_string()),
        dataset_manifest_hash: req.dataset_manifest_hash.clone().unwrap_or_default(),
        data_path: Some(req.data_path.clone()),
        batch_size: req.batch_size,
        output_dir: req.output_dir.clone(),
        seed: req.seed,
    };
    spec.validate().map_err(|errs| errs.join("; "))?;
    Ok(spec)
}

/// Converts a typed benchmark request into the shared library spec.
pub fn benchmark_spec_from_request(
    req: &BenchmarkRequest,
) -> Result<lnai_training::spec::BenchmarkSpec, String> {
    use lnai_training::spec::{BenchmarkSpec, ModelKind};
    let model = match req.model {
        ModelKindDto::Pinn => ModelKind::Pinn,
        ModelKindDto::Gnn => ModelKind::GnnKinematics,
        ModelKindDto::Siren => ModelKind::Siren,
    };
    let spec = BenchmarkSpec {
        model,
        artifact_hash: "unspecified".to_string(),
        iterations: req.iterations,
        warmup_iterations: req.warmup_iterations,
        output_dir: req.output_dir.clone(),
        batch_size: req.batch_size,
        seed: req.seed,
    };
    spec.validate().map_err(|errs| errs.join("; "))?;
    Ok(spec)
}

/// Resolves the worker binary for a model kind (release preferred, debug
/// fallback — same order as the pre-Stage-5 hand-rolled spawn code).
pub fn worker_binary_path(model: &lnai_training::spec::ModelKind) -> std::path::PathBuf {
    let ws = crate::jobs::workspace_root();
    for profile in ["release", "debug"] {
        let candidate = ws.join("target").join(profile).join(model.binary_name());
        if candidate.exists() {
            return candidate;
        }
    }
    ws.join("target").join("release").join(model.binary_name())
}

/// Builds the worker [`Command`] from a shared training spec — the single
/// argv source CLI and Testbench share (parity gate).
pub fn train_command_from_spec(spec: &lnai_training::spec::TrainingSpec) -> Command {
    let mut cmd = Command::new(worker_binary_path(&spec.model));
    for arg in spec.worker_argv() {
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }
    cmd
}

/// Builds the worker [`Command`] for read-only evaluation
/// (`--evaluate-only`: no optimizer step, no checkpoint rewrite).
pub fn evaluate_command_from_spec(
    spec: &lnai_training::spec::EvaluationSpec,
    holdout: Option<&str>,
    model_file: &str,
    norm_file: &str,
) -> Command {
    let mut cmd = Command::new(worker_binary_path(&spec.model));
    for arg in spec.worker_argv(holdout) {
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }
    cmd.arg("--model-file").arg(model_file);
    cmd.arg("--norm-file").arg(norm_file);
    cmd
}

/// Builds the worker [`Command`] for the forward-pass benchmark harness.
pub fn benchmark_command_from_spec(spec: &lnai_training::spec::BenchmarkSpec) -> Command {
    let mut cmd = Command::new(worker_binary_path(&spec.model));
    for arg in spec.worker_argv() {
        if arg.is_empty() {
            continue;
        }
        cmd.arg(arg);
    }
    cmd
}

// ------------------------------- HTTP routes -------------------------------

/// `POST /jobs/training` — typed training job (new canonical route).
pub async fn start_training(
    State(state): State<AppState>,
    Json(req): Json<TrainingRequest>,
) -> Result<Json<Job>, String> {
    let spec = training_spec_from_request(&req)?;
    let cmd = train_command_from_spec(&spec);
    let total_epochs = spec.epochs;
    let title = format!(
        "{} train · {}",
        match req.model {
            ModelKindDto::Pinn => "PINN",
            ModelKindDto::Gnn => "GNN",
            ModelKindDto::Siren => "SIREN",
        },
        req.data_path
    );
    // The legacy Job envelope keeps the old UI working; the typed spec is
    // the spawn source of truth.
    let legacy = legacy_train_spec(&req);
    let job = Job::new(JobKind::Train(legacy), title, total_epochs);
    let id = state.registry.spawn(job, cmd).map_err(|e| e.to_string())?;
    state
        .registry
        .get(&id)
        .map(Json)
        .ok_or_else(|| "job not found after spawn".to_string())
}

/// `POST /jobs/evaluation` — typed read-only evaluation job.
pub async fn start_evaluation(
    State(state): State<AppState>,
    Json(req): Json<EvaluationRequest>,
) -> Result<Json<Job>, String> {
    let spec = evaluation_spec_from_request(&req)?;
    let model_file = lnai_training::artifacts::weight_file_name(&spec.model).to_string();
    let norm_file = lnai_training::artifacts::norm_file_name(&spec.model).to_string();
    let holdout = req.holdout.clone();
    let cmd = evaluate_command_from_spec(&spec, holdout.as_deref(), &model_file, &norm_file);
    let title = format!("{} evaluate · {}", spec.model.slug(), req.data_path);
    let job = Job::new(JobKind::Custom("evaluate".to_string()), title, 1);
    let id = state.registry.spawn(job, cmd).map_err(|e| e.to_string())?;
    state
        .registry
        .get(&id)
        .map(Json)
        .ok_or_else(|| "job not found after spawn".to_string())
}

/// `POST /jobs/benchmark` — typed forward-pass benchmark job.
pub async fn start_benchmark(
    State(state): State<AppState>,
    Json(req): Json<BenchmarkRequest>,
) -> Result<Json<Job>, String> {
    let spec = benchmark_spec_from_request(&req)?;
    let cmd = benchmark_command_from_spec(&spec);
    let title = format!("{} benchmark · {}", spec.model.slug(), req.output_dir);
    let job = Job::new(JobKind::Custom("benchmark".to_string()), title, 1);
    let id = state.registry.spawn(job, cmd).map_err(|e| e.to_string())?;
    state
        .registry
        .get(&id)
        .map(Json)
        .ok_or_else(|| "job not found after spawn".to_string())
}

/// `GET /jobs/events/:id` — replays the worker's typed NDJSON `events.ndjson`
/// as JSON (the authoritative metric stream; stdout remains log-only).
pub async fn job_typed_events(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Vec<JobEventDto>>, String> {
    let job = state
        .registry
        .get(&id)
        .ok_or_else(|| "not found".to_string())?;
    // Output dir is recovered from the legacy spec stored on the job.
    let output_dir = match &job.spec {
        JobKind::Train(t) => Some(t.output_dir.clone()),
        JobKind::Validate(v) => Some(v.output_dir.clone()),
        JobKind::Custom(_) => None,
    };
    let Some(output_dir) = output_dir else {
        return Ok(Json(Vec::new()));
    };
    let path = Path::new(&output_dir).join("events.ndjson");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Ok(Json(Vec::new()));
    };
    let mut out = Vec::new();
    for line in raw.lines() {
        let Some(parsed) = lnai_training::events::read_event_line(line) else {
            continue;
        };
        match parsed {
            Ok(event) => out.push(map_job_event(event)),
            Err(_) => continue,
        }
    }
    Ok(Json(out))
}

fn map_job_event(event: lnai_training::events::JobEvent) -> JobEventDto {
    use lnai_training::events::JobEvent;
    match event {
        JobEvent::Queued => JobEventDto::Queued,
        JobEvent::Started => JobEventDto::Started,
        JobEvent::Progress {
            epoch,
            total_epochs,
        } => JobEventDto::Progress {
            epoch,
            total_epochs,
        },
        JobEvent::Metric(m) => JobEventDto::Metric(lunar_structures_testbench::EpochMetric {
            epoch: m.epoch,
            train_loss: m.train_loss,
            val_loss: m.val_loss,
            phys_loss: m.phys_loss,
            lr: m.lr,
            timestamp_ms: m.timestamp_ms,
        }),
        JobEvent::Checkpoint { epoch, path, hash } => JobEventDto::Checkpoint { epoch, path, hash },
        JobEvent::Completed { exit_code } => JobEventDto::Completed { exit_code },
        JobEvent::Failed {
            error_summary,
            exit_code,
        } => JobEventDto::Failed {
            error_summary,
            exit_code,
        },
        JobEvent::Cancelled => JobEventDto::Cancelled,
    }
}

fn legacy_train_spec(req: &TrainingRequest) -> lunar_structures_testbench::TrainSpec {
    use lunar_structures_testbench::ModelKind;
    lunar_structures_testbench::TrainSpec {
        model: match req.model {
            ModelKindDto::Pinn => ModelKind::Pinn,
            ModelKindDto::Gnn => ModelKind::Gnn,
            ModelKindDto::Siren => ModelKind::Siren,
        },
        epochs: req.epochs,
        batch_size: req.batch_size,
        lr: req.lr,
        physics_weight: req.physics_weight,
        val_frac: req.val_frac,
        data_path: req.data_path.clone(),
        output_dir: req.output_dir.clone(),
        resume_from: req.resume_from.clone(),
        holdout: req.holdout.clone(),
        gpu_index: req.gpu_index,
        knn_k: req.knn_k,
        hidden_dim: req.hidden_dim,
        texture_size: req.texture_size,
        max_stars: req.max_stars,
        patience: req.patience,
        grad_accum: req.grad_accum,
        clip_grad_norm: req.clip_grad_norm,
    }
}

#[allow(dead_code)]
fn _payload_type_check(_: &JobIdPayload) {}

#[cfg(test)]
mod tests {
    use super::*;
    use lunar_structures_testbench::typed::TrainingRequest as Req;
    use tokio::process::Command as TokioCommand;

    fn pinn_request() -> Req {
        Req {
            model: ModelKindDto::Pinn,
            epochs: 50,
            batch_size: 2048,
            lr: 5e-4,
            physics_weight: 0.1,
            val_frac: 0.1,
            data_path: "ai_data/clean_stars2.parquet".into(),
            output_dir: "models".into(),
            resume_from: None,
            holdout: Some("holdout.parquet".into()),
            gpu_index: 0,
            knn_k: None,
            hidden_dim: None,
            max_group_size: None,
            radius_pc: None,
            texture_size: None,
            max_stars: None,
            seed: Some(42),
            dataset_manifest_hash: None,
            patience: 20,
            grad_accum: 2,
            clip_grad_norm: 1.0,
        }
    }

    fn command_args(command: &TokioCommand) -> Vec<String> {
        command
            .as_std()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn has_flag_value(args: &[String], flag: &str, value: &str) -> bool {
        args.windows(2)
            .any(|pair| pair[0] == flag && pair[1] == value)
    }

    #[test]
    fn pinn_request_converts_and_renders_legacy_compatible_argv() {
        let spec = training_spec_from_request(&pinn_request()).expect("valid pinn request");
        assert_eq!(spec.model, lnai_training::spec::ModelKind::Pinn);
        assert_eq!(spec.model_file, "stellar_model.bpk");
        let argv = spec.worker_argv();
        assert!(has_flag_value(&argv, "--batch-size", "2048"));
        assert!(has_flag_value(&argv, "--physics-weight", "0.1"));
        assert!(has_flag_value(&argv, "--holdout", "holdout.parquet"));
        assert!(!argv.iter().any(|a| a == "--max-nodes"));
        assert!(!argv.iter().any(|a| a == "--knn-k"));
    }

    #[test]
    fn gnn_request_carries_graph_flags_and_rejects_siren_options() {
        let mut req = pinn_request();
        req.model = ModelKindDto::Gnn;
        req.knn_k = Some(8);
        req.hidden_dim = Some(256);
        req.max_group_size = Some(64);
        req.radius_pc = Some(50.0);
        let spec = training_spec_from_request(&req).expect("valid gnn request");
        let argv = spec.worker_argv();
        assert!(has_flag_value(&argv, "--max-nodes", "2048"));
        assert!(has_flag_value(&argv, "--knn-k", "8"));
        assert!(has_flag_value(&argv, "--hidden-dim", "256"));
        assert!(!argv.iter().any(|a| a == "--batch-size"));

        req.texture_size = Some(64);
        assert!(training_spec_from_request(&req).is_err());
    }

    #[test]
    fn siren_request_carries_texture_flags() {
        let mut req = pinn_request();
        req.model = ModelKindDto::Siren;
        req.texture_size = Some(64);
        req.max_stars = Some(100);
        let spec = training_spec_from_request(&req).expect("valid siren request");
        let argv = spec.worker_argv();
        assert!(has_flag_value(&argv, "--texture-size", "64"));
        assert!(has_flag_value(&argv, "--max-stars", "100"));
        assert!(!argv.iter().any(|a| a == "--physics-weight"));
    }

    #[test]
    fn gnn_output_dim_is_frozen_to_three() {
        let mut req = pinn_request();
        req.model = ModelKindDto::Gnn;
        let spec = training_spec_from_request(&req).expect("valid gnn request");
        match spec.config {
            lnai_training::spec::ModelConfig::GnnKinematics(cfg) => {
                assert_eq!(cfg.output_dim, 3);
            }
            other => panic!("unexpected config {other:?}"),
        }
    }

    #[test]
    fn evaluation_and_benchmark_commands_use_read_only_modes() {
        let eval = evaluation_spec_from_request(&EvaluationRequest {
            model: ModelKindDto::Pinn,
            data_path: "data.parquet".into(),
            output_dir: "runs/pinn".into(),
            holdout: None,
            batch_size: 512,
            seed: Some(7),
            artifact_hash: None,
            dataset_manifest_hash: None,
        })
        .expect("valid eval request");
        let eval_args = command_args(&evaluate_command_from_spec(
            &eval,
            None,
            "stellar_model.bpk",
            "stellar_norm.json",
        ));
        assert!(eval_args.contains(&"--evaluate-only".to_string()));
        assert!(!eval_args.iter().any(|a| a == "--epochs"));

        let bench = benchmark_spec_from_request(&BenchmarkRequest {
            model: ModelKindDto::Siren,
            output_dir: "runs/siren".into(),
            batch_size: 1024,
            iterations: 50,
            warmup_iterations: 5,
            seed: None,
        })
        .expect("valid bench request");
        let bench_args = command_args(&benchmark_command_from_spec(&bench));
        assert!(bench_args.contains(&"--benchmark-iters".to_string()));
        assert!(!bench_args.iter().any(|a| a == "--epochs"));
    }
}
