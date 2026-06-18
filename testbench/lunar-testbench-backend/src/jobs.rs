use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use axum::{
    extract::{Query, State},
    Json,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use lunar_structures_testbench::{
    EpochMetric, Job, JobIdPayload, JobKind,
    JobStatus, LogEntry, LogLineKind, ModelKind,
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
    Log { line: LogEntry },
    Metric { metric: EpochMetric },
    Status { status: JobStatus, message: Option<String> },
    Snapshot { job: Box<Job> },
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

    pub fn spawn(
        self: &Arc<Self>,
        mut job: Job,
        mut cmd: Command,
    ) -> Result<String> {
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
            let _ = stdout_task.await;
            let _ = stderr_task.await;

            let cancelled = cancel_for_task.is_cancelled();
            let status = match child.wait().await {
                Ok(s) => {
                    let code = s.code();
                    job_arc_for_task.write().exit_code = code;
                    if cancelled {
                        JobStatus::Cancelled
                    } else if s.success() {
                        JobStatus::Completed
                    } else {
                        JobStatus::Failed
                    }
                }
                Err(e) => {
                    job_arc_for_task.write().error_summary = Some(e.to_string());
                    JobStatus::Failed
                }
            };

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
            let _ = tx_clone.send(JobEvent::Snapshot { job: Box::new(snap) });
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
    } else if t.starts_with("✅") || t.starts_with("✓") || t.contains("Checkpoint saved") || t.starts_with("Best model saved") {
        LogLineKind::Checkpoint
    } else {
        LogLineKind::Raw
    }
}

fn parse_epoch_line(model: &ModelKind, line: &str) -> Option<EpochMetric> {
    let trimmed = line.trim_start();
    if !trimmed.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
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
    let mut cmd = Command::new(
        workspace_root
            .join("target")
            .join("release")
            .join(spec.model.binary_name()),
    );
    cmd.arg("--data").arg(&spec.data_path);
    cmd.arg("--epochs").arg(spec.epochs.to_string());
    cmd.arg("--batch-size").arg(spec.batch_size.to_string());
    cmd.arg("--lr").arg(format!("{}", spec.lr));
    cmd.arg("--physics-weight").arg(format!("{}", spec.physics_weight));
    cmd.arg("--val-frac").arg(format!("{}", spec.val_frac));
    cmd.arg("--gpu-index").arg(spec.gpu_index.to_string());
    cmd.arg("--patience").arg(spec.patience.to_string());
    cmd.arg("--clip-grad-norm")
        .arg(format!("{}", spec.clip_grad_norm));
    cmd.arg("--grad-accum").arg(spec.grad_accum.to_string());
    cmd.arg("--output-dir").arg(&spec.output_dir);

    if let Some(resume) = &spec.resume_from
        && !resume.is_empty()
    {
        cmd.arg("--resume-from").arg(resume);
    }
    if let Some(holdout) = &spec.holdout
        && !holdout.is_empty()
    {
        cmd.arg("--holdout").arg(holdout);
    }
    if let Some(k) = spec.knn_k {
        cmd.arg("--knn-k").arg(k.to_string());
    }
    if let Some(h) = spec.hidden_dim {
        cmd.arg("--hidden-dim").arg(h.to_string());
    }
    if let Some(t) = spec.texture_size {
        cmd.arg("--texture-size").arg(t.to_string());
    }
    if let Some(m) = spec.max_stars {
        cmd.arg("--max-stars").arg(m.to_string());
    }
    cmd
}

fn build_validate_command(workspace_root: &Path, spec: &ValidateSpec) -> Command {
    let mut cmd = Command::new(
        workspace_root
            .join("target")
            .join("release")
            .join(spec.model.binary_name()),
    );
    cmd.arg("--data").arg(&spec.data_path);
    cmd.arg("--epochs").arg(spec.epochs.to_string());
    cmd.arg("--batch-size").arg(spec.batch_size.to_string());
    cmd.arg("--val-frac").arg(format!("{}", spec.val_frac));
    cmd.arg("--output-dir").arg(&spec.output_dir);
    cmd.arg("--patience").arg("99999");
    if let Some(h) = spec.hidden_dim {
        cmd.arg("--hidden-dim").arg(h.to_string());
    }
    if let Some(k) = spec.knn_k {
        cmd.arg("--knn-k").arg(k.to_string());
    }
    if let Some(t) = spec.texture_size {
        cmd.arg("--texture-size").arg(t.to_string());
    }
    if let Some(m) = spec.max_stars {
        cmd.arg("--max-stars").arg(m.to_string());
    }
    cmd
}

pub async fn list_jobs(State(state): State<AppState>) -> Json<Vec<Job>> {
    Json(state.registry.list())
}

pub async fn get_job(
    State(state): State<AppState>,
    Json(p): Json<JobIdPayload>,
) -> Result<Json<Job>, String> {
    state.registry.get(&p.id).map(Json).ok_or_else(|| "not found".to_string())
}

pub async fn cancel_job(
    State(state): State<AppState>,
    Json(p): Json<JobIdPayload>,
) -> Result<Json<()>, String> {
    state.registry.cancel(&p.id).map(|_| Json(())).map_err(|e| e.to_string())
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
    state.registry.get(&id).map(Json).ok_or_else(|| "not found".to_string())
}

pub async fn cancel_job_by_query(
    State(state): State<AppState>,
    Query(q): Query<IdQuery>,
) -> Result<Json<()>, String> {
    let id = q.id.ok_or_else(|| "missing id".to_string())?;
    state.registry.cancel(&id).map(|_| Json(())).map_err(|e| e.to_string())
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
    state.registry.get(&id).map(Json).ok_or_else(|| "job not found after spawn".to_string())
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
    state.registry.get(&id).map(Json).ok_or_else(|| "job not found after spawn".to_string())
}