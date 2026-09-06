use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use axum::{
    Json,
    extract::{Query, State},
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use lunar_structures_testbench::{
    EpochMetric, Job, JobIdPayload, JobKind, JobStatus, LogEntry, LogLineKind, ModelKind,
    TrainSpec, ValidateSpec,
};

use crate::AppState;

struct JobHandle {
    cancel: CancellationToken,
}

pub struct JobRegistry {
    jobs: RwLock<HashMap<String, Arc<RwLock<Job>>>>,
    channels: RwLock<HashMap<String, broadcast::Sender<JobEvent>>>,
    handles: RwLock<HashMap<String, JobHandle>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JobEvent {
    Log {
        line: LogEntry,
    },
    Metric {
        metric: EpochMetric,
    },
    Status {
        status: JobStatus,
        message: Option<String>,
    },
    Snapshot {
        job: Box<Job>,
    },
}

impl JobRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            jobs: RwLock::new(HashMap::new()),
            channels: RwLock::new(HashMap::new()),
            handles: RwLock::new(HashMap::new()),
        })
    }

    pub fn list(&self) -> Vec<Job> {
        self.jobs
            .read()
            .values()
            .map(|j| j.read().clone())
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.jobs.read().get(id).map(|j| j.read().clone())
    }

    pub fn cancel(&self, id: &str) -> Result<()> {
        if let Some(h) = self.handles.read().get(id) {
            h.cancel.cancel();
            Ok(())
        } else {
            Err(anyhow!("no running handle for job {id}"))
        }
    }

    pub fn spawn(self: &Arc<Self>, mut job: Job, mut cmd: Command) -> Result<String> {
        let id = job.id.clone();
        let (tx, _rx) = broadcast::channel::<JobEvent>(512);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.stdin(Stdio::null());

        let cancel = CancellationToken::new();
        let cancel_for_handle = cancel.clone();
        let cancel_for_task = cancel.clone();

        let mut child: Child = cmd
            .spawn()
            .map_err(|e| anyhow!("failed to spawn process: {e}"))?;
        job.started_ms = Some(now_ms());
        job.status = JobStatus::Running;

        let job_arc = Arc::new(RwLock::new(job));
        self.jobs.write().insert(id.clone(), job_arc.clone());
        self.channels.write().insert(id.clone(), tx.clone());
        self.handles.write().insert(
            id.clone(),
            JobHandle {
                cancel: cancel_for_handle,
            },
        );

        let registry = self.clone();
        let tx_clone = tx.clone();
        let id_for_task = id.clone();
        let job_arc_for_task = job_arc.clone();
        let model_kind = job_arc_for_task.read().spec.model_kind();

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let stdout_task = tokio::spawn(read_stream(
            stdout,
            LogLineKind::Raw,
            job_arc_for_task.clone(),
            tx_clone.clone(),
            Some(model_kind.clone()),
        ));
        let stderr_task = tokio::spawn(read_stream(
            stderr,
            LogLineKind::Error,
            job_arc_for_task.clone(),
            tx_clone.clone(),
            None,
        ));

        tokio::spawn(async move {
            let child_exit = tokio::select! {
                result = child.wait() => Some(result),
                _ = cancel_for_task.cancelled() => None,
            };

            let status = match child_exit {
                Some(Ok(exit_status)) => {
                    job_arc_for_task.write().exit_code = exit_status.code();
                    if exit_status.success() {
                        JobStatus::Completed
                    } else {
                        JobStatus::Failed
                    }
                }
                Some(Err(error)) => {
                    job_arc_for_task.write().error_summary = Some(error.to_string());
                    JobStatus::Failed
                }
                None => {
                    let kill_error = child
                        .kill()
                        .await
                        .err()
                        .filter(|error| error.kind() != std::io::ErrorKind::InvalidInput);
                    match child.wait().await {
                        Ok(exit_status) => {
                            let mut job = job_arc_for_task.write();
                            job.exit_code = exit_status.code();
                            if let Some(error) = kill_error {
                                job.error_summary =
                                    Some(format!("failed to terminate cancelled job: {error}"));
                                JobStatus::Failed
                            } else {
                                JobStatus::Cancelled
                            }
                        }
                        Err(error) => {
                            job_arc_for_task.write().error_summary =
                                Some(format!("failed to reap cancelled job: {error}"));
                            JobStatus::Failed
                        }
                    }
                }
            };

            let _ = stdout_task.await;
            let _ = stderr_task.await;

            {
                let mut j = job_arc_for_task.write();
                j.status = status.clone();
                j.finished_ms = Some(now_ms());
            }

            registry.handles.write().remove(&id_for_task);
            let snap = job_arc_for_task.read().clone();
            let _ = tx_clone.send(JobEvent::Status {
                status,
                message: snap.error_summary.clone(),
            });
            let _ = tx_clone.send(JobEvent::Snapshot {
                job: Box::new(snap),
            });
        });

        Ok(id)
    }
}

fn maybe_update_best_loss(job: &mut Job, metric: &EpochMetric) {
    if job.best_val_loss.is_none() || metric.val_loss < job.best_val_loss.unwrap() {
        job.best_val_loss = Some(metric.val_loss);
    }
    job.last_metrics.push(metric.clone());
}

fn append_log_with_cap(job: &mut Job, entry: LogEntry) {
    job.log_tail.push(entry);
    if job.log_tail.len() > 1024 {
        let drop_count = job.log_tail.len() - 1024;
        job.log_tail.drain(0..drop_count);
    }
}

fn determine_line_kind(default_kind: &LogLineKind, raw: &str) -> LogLineKind {
    if matches!(default_kind, LogLineKind::Error) {
        LogLineKind::Error
    } else {
        classify(raw)
    }
}

async fn process_line(
    raw: String,
    default_kind: &LogLineKind,
    job_arc: &Arc<RwLock<Job>>,
    tx: &broadcast::Sender<JobEvent>,
    model_kind: &Option<ModelKind>,
) {
    let kind = determine_line_kind(default_kind, &raw);
    let entry = LogEntry {
        timestamp_ms: now_ms(),
        line: raw.clone(),
        kind: kind.clone(),
    };

    if let Some(mk) = model_kind
        && let Some(metric) = parse_epoch_line(mk, &raw)
    {
        let mut j = job_arc.write();
        maybe_update_best_loss(&mut j, &metric);
        drop(j);
        let _ = tx.send(JobEvent::Metric { metric });
    }

    {
        let mut j = job_arc.write();
        append_log_with_cap(&mut j, entry.clone());
    }
    let _ = tx.send(JobEvent::Log { line: entry });
}

async fn read_stream<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    reader: Option<R>,
    default_kind: LogLineKind,
    job_arc: Arc<RwLock<Job>>,
    tx: broadcast::Sender<JobEvent>,
    model_kind: Option<ModelKind>,
) {
    let Some(mut reader) = reader else {
        return;
    };
    let mut buf_reader = BufReader::new(&mut reader);
    let mut line = String::new();
    loop {
        line.clear();
        match buf_reader.read_line(&mut line).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let raw = line.trim_end_matches(['\r', '\n']).to_string();
        process_line(raw, &default_kind, &job_arc, &tx, &model_kind).await;
    }
}

fn classify(line: &str) -> LogLineKind {
    let t = line.trim_start();
    if t.starts_with("===") || t.starts_with("---") {
        LogLineKind::Header
    } else if t.starts_with("WARNING") || t.starts_with("⚠") {
        LogLineKind::Warning
    } else if t.starts_with("ERROR") || t.starts_with("❌") {
        LogLineKind::Error
    } else if t.starts_with("✅")
        || t.starts_with("✓")
        || t.contains("Checkpoint saved")
        || t.starts_with("Best model saved")
    {
        LogLineKind::Checkpoint
    } else {
        LogLineKind::Raw
    }
}

fn parse_epoch_line(model: &ModelKind, line: &str) -> Option<EpochMetric> {
    let trimmed = line.trim_start();
    if !trimmed
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        return None;
    }
    let mut pipe_count = 0;
    for c in trimmed.chars() {
        if c == '|' {
            pipe_count += 1;
        }
    }
    let expected = match model {
        ModelKind::Pinn | ModelKind::Gnn => 4,
        ModelKind::Siren => 3,
    };
    if pipe_count < expected {
        return None;
    }
    let first = trimmed.split('|').next()?.trim();
    if first.is_empty() {
        return None;
    }
    let epoch: u32 = first.parse().ok()?;
    let parts: Vec<&str> = trimmed.split('|').collect();
    let train_loss: f64 = parts.get(1)?.trim().parse().ok()?;
    let val_loss: f64 = parts.get(2)?.trim().parse().ok()?;
    let third: f64 = parts.get(3)?.trim().parse().ok()?;
    let lr: f64 = if expected == 4 {
        parts.get(4)?.trim().parse().ok()?
    } else {
        third
    };
    let phys_loss = if expected == 4 { Some(third) } else { None };
    Some(EpochMetric {
        epoch,
        train_loss,
        val_loss,
        phys_loss,
        lr,
        timestamp_ms: now_ms(),
    })
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn workspace_root() -> PathBuf {
    if let Ok(p) = std::env::var("LUNAR_WORKSPACE_ROOT") {
        return PathBuf::from(p);
    }
    let exe = std::env::current_exe().ok();
    if let Some(exe) = exe {
        let mut cur = exe.parent();
        while let Some(dir) = cur {
            if dir.join("Cargo.toml").exists() && dir.join("crates").exists() {
                return dir.to_path_buf();
            }
            cur = dir.parent();
        }
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("Cargo.toml").exists() && cwd.join("crates").exists() {
        return cwd;
    }
    cwd
}

fn build_train_command(workspace_root: &Path, spec: &TrainSpec) -> Command {
    // Compatibility alias (Stage 5): the old `TrainSpec` shape renders
    // through the shared `worker_argv` builder — same flags the CLI emits.
    // No separate implementation lives here anymore.
    let _ = workspace_root;
    use lunar_structures_testbench::typed::{ModelKindDto, TrainingRequest};
    let model = match spec.model {
        ModelKind::Pinn => ModelKindDto::Pinn,
        ModelKind::Gnn => ModelKindDto::Gnn,
        ModelKind::Siren => ModelKindDto::Siren,
    };
    let req = TrainingRequest {
        model,
        epochs: spec.epochs,
        batch_size: spec.batch_size,
        lr: spec.lr,
        physics_weight: spec.physics_weight,
        val_frac: spec.val_frac,
        data_path: spec.data_path.clone(),
        output_dir: spec.output_dir.clone(),
        resume_from: spec.resume_from.clone(),
        holdout: spec.holdout.clone(),
        gpu_index: spec.gpu_index,
        knn_k: spec.knn_k,
        hidden_dim: spec.hidden_dim,
        max_group_size: None,
        radius_pc: None,
        texture_size: spec.texture_size,
        max_stars: spec.max_stars,
        seed: None,
        dataset_manifest_hash: None,
        patience: spec.patience,
        grad_accum: spec.grad_accum,
        clip_grad_norm: spec.clip_grad_norm,
    };
    let shared =
        crate::ai_jobs::training_spec_from_request(&req).expect("legacy train spec must convert");
    crate::ai_jobs::train_command_from_spec(&shared)
}

fn build_validate_command(workspace_root: &Path, spec: &ValidateSpec) -> Command {
    // Compatibility alias (Stage 5): the old `ValidateSpec` shape renders
    // through the shared read-only evaluation builder — same binary, plus
    // `--evaluate-only`, never an optimizer step. No separate implementation.
    let _ = workspace_root;
    use lunar_structures_testbench::typed::{EvaluationRequest, ModelKindDto};
    let model = match spec.model {
        ModelKind::Pinn => ModelKindDto::Pinn,
        ModelKind::Gnn => ModelKindDto::Gnn,
        ModelKind::Siren => ModelKindDto::Siren,
    };
    let req = EvaluationRequest {
        model,
        data_path: spec.data_path.clone(),
        output_dir: spec.output_dir.clone(),
        holdout: None,
        batch_size: spec.batch_size,
        seed: None,
        artifact_hash: None,
        dataset_manifest_hash: None,
    };
    let eval = crate::ai_jobs::evaluation_spec_from_request(&req)
        .expect("legacy validate spec must convert");
    let kind = match spec.model {
        ModelKind::Pinn => "stellar_model.bpk",
        ModelKind::Gnn => "stellar_gnn_model.bpk",
        ModelKind::Siren => "stellar_siren_model.bpk",
    };
    let norm = match spec.model {
        ModelKind::Pinn => "stellar_norm.json",
        ModelKind::Gnn => "stellar_gnn_norm.json",
        ModelKind::Siren => "stellar_siren_norm.json",
    };
    crate::ai_jobs::evaluate_command_from_spec(&eval, None, kind, norm)
}

pub async fn list_jobs(State(state): State<AppState>) -> Json<Vec<Job>> {
    Json(state.registry.list())
}

pub async fn get_job(
    State(state): State<AppState>,
    Json(p): Json<JobIdPayload>,
) -> Result<Json<Job>, String> {
    state
        .registry
        .get(&p.id)
        .map(Json)
        .ok_or_else(|| "not found".to_string())
}

pub async fn cancel_job(
    State(state): State<AppState>,
    Json(p): Json<JobIdPayload>,
) -> Result<Json<()>, String> {
    state
        .registry
        .cancel(&p.id)
        .map(|_| Json(()))
        .map_err(|e| e.to_string())
}

#[derive(Deserialize)]
pub struct IdQuery {
    pub id: Option<String>,
}

pub async fn get_job_by_query(
    State(state): State<AppState>,
    Query(q): Query<IdQuery>,
) -> Result<Json<Job>, String> {
    let id = q.id.ok_or_else(|| "missing id".to_string())?;
    state
        .registry
        .get(&id)
        .map(Json)
        .ok_or_else(|| "not found".to_string())
}

pub async fn cancel_job_by_query(
    State(state): State<AppState>,
    Query(q): Query<IdQuery>,
) -> Result<Json<()>, String> {
    let id = q.id.ok_or_else(|| "missing id".to_string())?;
    state
        .registry
        .cancel(&id)
        .map(|_| Json(()))
        .map_err(|e| e.to_string())
}

pub async fn start_train(
    State(state): State<AppState>,
    Json(spec): Json<TrainSpec>,
) -> Result<Json<Job>, String> {
    let ws = workspace_root();
    let cmd = build_train_command(&ws, &spec);
    let total_epochs = spec.epochs;
    let title = format!("{} train · {}", spec.model.label(), spec.data_path);
    let job = Job::new(JobKind::Train(spec), title, total_epochs);
    let id = state.registry.spawn(job, cmd).map_err(|e| e.to_string())?;
    state
        .registry
        .get(&id)
        .map(Json)
        .ok_or_else(|| "job not found after spawn".to_string())
}

pub async fn start_validate(
    State(state): State<AppState>,
    Json(spec): Json<ValidateSpec>,
) -> Result<Json<Job>, String> {
    let ws = workspace_root();
    let cmd = build_validate_command(&ws, &spec);
    let total_epochs = spec.epochs;
    let title = format!("{} validate · {}", spec.model.label(), spec.data_path);
    let job = Job::new(JobKind::Validate(spec), title, total_epochs);
    let id = state.registry.spawn(job, cmd).map_err(|e| e.to_string())?;
    state
        .registry
        .get(&id)
        .map(Json)
        .ok_or_else(|| "job not found after spawn".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;

    fn train_spec(model: ModelKind) -> TrainSpec {
        let (knn_k, hidden_dim, texture_size, max_stars) = match model {
            ModelKind::Pinn => (None, None, None, None),
            ModelKind::Gnn => (Some(7), Some(128), None, None),
            ModelKind::Siren => (None, None, Some(96), Some(500)),
        };
        TrainSpec {
            model,
            epochs: 3,
            batch_size: 64,
            lr: 0.001,
            physics_weight: 0.2,
            val_frac: 0.15,
            data_path: "fixture.parquet".into(),
            output_dir: "out".into(),
            resume_from: Some("resume".into()),
            holdout: Some("holdout.parquet".into()),
            gpu_index: 1,
            knn_k,
            hidden_dim,
            texture_size,
            max_stars,
            patience: 12,
            grad_accum: 4,
            clip_grad_norm: 0.8,
        }
    }

    fn validate_spec(model: ModelKind) -> ValidateSpec {
        let (hidden_dim, knn_k, texture_size, max_stars) = match model {
            ModelKind::Pinn => (None, None, None, None),
            ModelKind::Gnn => (Some(128), Some(7), None, None),
            ModelKind::Siren => (None, None, Some(96), Some(500)),
        };
        ValidateSpec {
            model,
            data_path: "fixture.parquet".into(),
            epochs: 1,
            batch_size: 64,
            val_frac: 0.15,
            output_dir: "out".into(),
            hidden_dim,
            knn_k,
            texture_size,
            max_stars,
        }
    }

    fn command_args(command: &Command) -> Vec<String> {
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
    fn train_commands_only_include_flags_supported_by_each_model() {
        // Compatibility-alias contract (Stage 5): legacy specs render through
        // the shared worker_argv builder. Model-specific flags must appear
        // only for their own kind; unrelated options are a conversion error.
        let root = Path::new("/workspace");

        let pinn_args = command_args(&build_train_command(root, &train_spec(ModelKind::Pinn)));
        assert!(has_flag_value(&pinn_args, "--batch-size", "64"));
        assert!(has_flag_value(&pinn_args, "--physics-weight", "0.2"));
        assert!(!pinn_args.iter().any(|arg| arg == "--max-nodes"));
        assert!(!pinn_args.iter().any(|arg| arg == "--knn-k"));
        assert!(!pinn_args.iter().any(|arg| arg == "--texture-size"));

        let gnn_args = command_args(&build_train_command(root, &train_spec(ModelKind::Gnn)));
        assert!(has_flag_value(&gnn_args, "--max-nodes", "64"));
        assert!(has_flag_value(&gnn_args, "--physics-weight", "0.2"));
        assert!(has_flag_value(&gnn_args, "--knn-k", "7"));
        assert!(has_flag_value(&gnn_args, "--hidden-dim", "128"));
        assert!(!gnn_args.iter().any(|arg| arg == "--batch-size"));
        assert!(!gnn_args.iter().any(|arg| arg == "--texture-size"));

        let siren_args = command_args(&build_train_command(root, &train_spec(ModelKind::Siren)));
        assert!(has_flag_value(&siren_args, "--batch-size", "64"));
        assert!(has_flag_value(&siren_args, "--texture-size", "96"));
        assert!(has_flag_value(&siren_args, "--max-stars", "500"));
        assert!(!siren_args.iter().any(|a| a == "--physics-weight"));
        assert!(!siren_args.iter().any(|a| a == "--knn-k"));
        assert!(!siren_args.iter().any(|a| a == "--hidden-dim"));
    }

    #[test]
    fn validation_alias_uses_read_only_evaluate_mode() {
        // The old `/jobs/validate` route is now an alias for read-only
        // evaluation: same binary, `--evaluate-only`, no training flags.
        let root = Path::new("/workspace");
        for model in [ModelKind::Pinn, ModelKind::Gnn, ModelKind::Siren] {
            let args = command_args(&build_validate_command(root, &validate_spec(model)));
            assert!(
                args.contains(&"--evaluate-only".to_string()),
                "validate alias must be read-only: {args:?}"
            );
            assert!(
                !args.iter().any(|a| a == "--epochs"),
                "evaluation must not train: {args:?}"
            );
        }

        let pinn_args = command_args(&build_validate_command(
            root,
            &validate_spec(ModelKind::Pinn),
        ));
        assert!(has_flag_value(&pinn_args, "--batch-size", "64"));

        let gnn_args = command_args(&build_validate_command(
            root,
            &validate_spec(ModelKind::Gnn),
        ));
        assert!(has_flag_value(&gnn_args, "--batch-size", "64"));

        let siren_args = command_args(&build_validate_command(
            root,
            &validate_spec(ModelKind::Siren),
        ));
        assert!(has_flag_value(&siren_args, "--batch-size", "64"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_terminates_running_child() {
        let registry = JobRegistry::new();
        let mut command = Command::new("sleep");
        command.arg("30");
        let job = Job::new(
            JobKind::Custom("cancellation-test".into()),
            "test".into(),
            1,
        );
        let id = registry.spawn(job, command).expect("spawn job");

        registry.cancel(&id).expect("cancel job");
        let completed = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let job = registry.get(&id).expect("job exists");
                if matches!(
                    job.status,
                    JobStatus::Cancelled | JobStatus::Completed | JobStatus::Failed
                ) {
                    break job;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("cancelled child should terminate promptly");

        assert_eq!(completed.status, JobStatus::Cancelled);
        assert!(completed.finished_ms.is_some());
    }
}
