use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use axum::{
    extract::Query,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tower_http::cors::{Any, CorsLayer};
use walkdir::WalkDir;
use reqwest::Client;

use lunar_structures::{
    PinnResponse, PipelineRequest, PipelineResponse, SirenTextureResponse,
    StarDescriptionPayload, StarLore, StellarMetadata,
};
use lunar_structures_testbench::{
    BackendStatus, BinaryInfo, DatasetInfo, EpochMetric, HostInfo, Job, JobIdPayload, JobKind,
    JobStatus, LogEntry, LogLineKind, ModelArtifact, ModelKind, NormSnapshot,
    SystemSnapshot, TrainSpec, ValidateSpec,
};

// =============================================================
// Job registry (unchanged)
// =============================================================

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
    Snapshot { job: Job },
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
            let _ = tx_clone.send(JobEvent::Snapshot { job: snap });
        });

        Ok(id)
    }
}

async fn read_stream<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    reader: Option<R>,
    default_kind: LogLineKind,
    job_arc: Arc<RwLock<Job>>,
    tx: tokio::sync::broadcast::Sender<JobEvent>,
    model_kind: Option<ModelKind>,
) {
    let Some(mut reader) = reader else {
        return;
    };
    let mut buf_reader = BufReader::new(&mut reader);
    let mut line = String::new();
    loop {
        line.clear();
        let n = match buf_reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        let _ = n;
        let raw = line.trim_end_matches(['\r', '\n']).to_string();
        let kind = if matches!(default_kind, LogLineKind::Error) {
            LogLineKind::Error
        } else {
            classify(&raw)
        };
        let entry = LogEntry {
            timestamp_ms: now_ms(),
            line: raw.clone(),
            kind: kind.clone(),
        };
        if let Some(mk) = &model_kind {
            if let Some(metric) = parse_epoch_line(mk, &raw) {
                let mut j = job_arc.write();
                if j.best_val_loss.is_none() || metric.val_loss < j.best_val_loss.unwrap() {
                    j.best_val_loss = Some(metric.val_loss);
                }
                j.last_metrics.push(metric.clone());
                drop(j);
                let _ = tx.send(JobEvent::Metric { metric });
            }
        }
        {
            let mut j = job_arc.write();
            j.log_tail.push(entry.clone());
            if j.log_tail.len() > 1024 {
                let drop_count = j.log_tail.len() - 1024;
                j.log_tail.drain(0..drop_count);
            }
        }
        let _ = tx.send(JobEvent::Log { line: entry });
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
    } else if t.starts_with("✅") || t.starts_with("✓") {
        LogLineKind::Checkpoint
    } else if t.contains("Checkpoint saved") || t.starts_with("Best model saved") {
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

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// =============================================================
// Command builders (unchanged)
// =============================================================

fn workspace_root() -> PathBuf {
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

fn build_train_command(workspace_root: &PathBuf, spec: &TrainSpec) -> Command {
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

    if let Some(resume) = &spec.resume_from {
        if !resume.is_empty() {
            cmd.arg("--resume-from").arg(resume);
        }
    }
    if let Some(holdout) = &spec.holdout {
        if !holdout.is_empty() {
            cmd.arg("--holdout").arg(holdout);
        }
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

fn build_validate_command(workspace_root: &PathBuf, spec: &ValidateSpec) -> Command {
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

// =============================================================
// Job HTTP handlers (unchanged)
// =============================================================

#[derive(Clone)]
struct AppState {
    registry: Arc<JobRegistry>,
}

async fn list_jobs(axum::extract::State(state): axum::extract::State<AppState>) -> Json<Vec<Job>> {
    Json(state.registry.list())
}

async fn get_job(
    axum::extract::State(state): axum::extract::State<AppState>,
    Query(q): Query<JobIdPayload>,
) -> Result<Json<Job>, String> {
    state.registry.get(&q.id).map(Json).ok_or_else(|| "not found".to_string())
}

async fn cancel_job(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(p): Json<JobIdPayload>,
) -> Result<Json<()>, String> {
    state.registry.cancel(&p.id).map(|_| Json(())).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
struct IdQuery {
    id: Option<String>,
}

async fn get_job_by_query(
    axum::extract::State(state): axum::extract::State<AppState>,
    Query(q): Query<IdQuery>,
) -> Result<Json<Job>, String> {
    let id = q.id.ok_or_else(|| "missing id".to_string())?;
    state.registry.get(&id).map(Json).ok_or_else(|| "not found".to_string())
}

async fn cancel_job_by_query(
    axum::extract::State(state): axum::extract::State<AppState>,
    Query(q): Query<IdQuery>,
) -> Result<Json<()>, String> {
    let id = q.id.ok_or_else(|| "missing id".to_string())?;
    state.registry.cancel(&id).map(|_| Json(())).map_err(|e| e.to_string())
}

async fn start_train(
    axum::extract::State(state): axum::extract::State<AppState>,
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

async fn start_validate(
    axum::extract::State(state): axum::extract::State<AppState>,
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

// =============================================================
// NEW: HTML pages served directly
// =============================================================

static PLAYGROUND_HTML: &str = include_str!("playground.html");
static SIREN_UI_HTML: &str = include_str!("siren_ui.html");

async fn playground_ui() -> Html<&'static str> {
    Html(PLAYGROUND_HTML)
}

async fn siren_ui_page() -> Html<&'static str> {
    Html(SIREN_UI_HTML)
}

// =============================================================
// NEW: System snapshot (moved from lunar-backend)
// =============================================================

fn scan_models(ws: &Path) -> Vec<ModelArtifact> {
    let models_dir = ws.join("models");
    let mut out = Vec::new();
    if !models_dir.exists() {
        return out;
    }
    for entry in WalkDir::new(&models_dir).max_depth(2).into_iter().flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if !(name.ends_with(".bpk") || name.ends_with(".safetensors") || name.ends_with(".json")) {
            continue;
        }
        let md = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let kind = match name.as_str() {
            n if n.starts_with("stellar_model") => "pinn",
            n if n.starts_with("stellar_gnn") => "gnn",
            n if n.starts_with("stellar_siren") => "siren",
            n if n.starts_with("stellar_lore") => "lore",
            _ => "other",
        };
        let mtime_ms = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        out.push(ModelArtifact {
            name,
            path: path.to_string_lossy().to_string(),
            kind: kind.into(),
            size_bytes: md.len(),
            mtime_ms,
            exists: true,
        });
    }
    out.sort_by(|a, b| a.kind.cmp(&b.kind).then(a.name.cmp(&b.name)));
    out
}

fn scan_datasets(ws: &Path) -> Vec<DatasetInfo> {
    let dirs = ["ai_data", "data/chunks", "data"];
    let mut out = Vec::new();
    for d in dirs {
        let path = ws.join(d);
        if !path.exists() {
            continue;
        }
        for entry in WalkDir::new(&path).max_depth(2).into_iter().flatten() {
            let p = entry.path();
            if !p.is_file() {
                continue;
            }
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            if !(name.ends_with(".parquet") || name.ends_with(".csv")) {
                continue;
            }
            let md = match std::fs::metadata(p) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let mtime_ms = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            out.push(DatasetInfo {
                name: name.clone(),
                path: p.to_string_lossy().to_string(),
                size_bytes: md.len(),
                mtime_ms,
                kind: detect_dataset_kind(&name),
            });
        }
    }
    out
}

fn detect_dataset_kind(name: &str) -> String {
    if name.contains("gnn") {
        "gnn".to_string()
    } else if name.contains("combined") || name.contains("chunk") {
        "chunks".into()
    } else if name.contains("holdout") {
        "holdout".into()
    } else if name.contains("raw") {
        "raw".into()
    } else {
        "clean".into()
    }
}

fn binary_status(ws: &Path, name: &str) -> BinaryInfo {
    let release = ws.join("target").join("release").join(name);
    let debug = ws.join("target").join("debug").join(name);
    let (path, exists) = if release.exists() {
        (release, true)
    } else if debug.exists() {
        (debug, true)
    } else {
        (release, false)
    };
    let size_bytes = if exists {
        std::fs::metadata(&path).ok().map(|m| m.len())
    } else {
        None
    };
    let mtime_ms = std::fs::metadata(&path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64);
    BinaryInfo {
        name: name.to_string(),
        path: path.to_string_lossy().to_string(),
        exists,
        size_bytes,
        mtime_ms,
    }
}

fn read_norm_file(ws: &Path, file: &str) -> NormSnapshot {
    let path: PathBuf = ws.join("models").join(file);
    if !path.exists() {
        return NormSnapshot {
            kind: file.to_string(),
            path: path.to_string_lossy().to_string(),
            exists: false,
            data: None,
        };
    }
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => {
            return NormSnapshot {
                kind: file.to_string(),
                path: path.to_string_lossy().to_string(),
                exists: true,
                data: None,
            }
        }
    };
    let data = serde_json::from_str(&raw).ok();
    NormSnapshot {
        kind: file.to_string(),
        path: path.to_string_lossy().to_string(),
        exists: true,
        data,
    }
}

fn collect_host_info(ws: &Path) -> HostInfo {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_list(sysinfo::CpuRefreshKind::everything());
    sys.refresh_memory();
    let pid = std::process::id();
    let cpu_count = sys.cpus().len();
    let total_memory_bytes = sys.total_memory();
    HostInfo {
        workspace_root: ws.to_string_lossy().to_string(),
        pid,
        cpu_count,
        total_memory_bytes,
        rustc_version: "stable".into(),
    }
}

async fn ping_url(client: &Client, url: &str) -> BackendStatus {
    let start = SystemTime::now();
    let resp = client.get(url).send().await;
    let (reachable, latency, hint) = match resp {
        Ok(r) => {
            let latency = start.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let hint = if r.status().is_success() { "HTTP 200" } else { "HTTP non-2xx" };
            (true, Some(latency), hint.to_string())
        }
        Err(_) => (false, None, "unreachable".into()),
    };
    BackendStatus {
        url: url.to_string(),
        reachable,
        latency_ms: latency,
        last_checked_ms: now_ms(),
        version_hint: hint,
    }
}

#[derive(Deserialize, Default)]
struct SnapshotQuery {
    include_jobs: Option<bool>,
    testbench_backend_url: Option<String>,
}

async fn system_snapshot(Query(q): Query<SnapshotQuery>) -> Json<SystemSnapshot> {
    let ws = workspace_root();
    let models = scan_models(&ws);
    let datasets = scan_datasets(&ws);
    let binary_names = ["lnai", "lnai-gnn", "lnai-siren", "lunar-ai-cli", "lunar-backend", "lunar-testbench", "lunar-testbench-backend"];
    let binaries = binary_names
        .iter()
        .map(|n| binary_status(&ws, n))
        .collect::<Vec<_>>();

    let norms = vec![
        read_norm_file(&ws, "stellar_norm.json"),
        read_norm_file(&ws, "stellar_gnn_norm.json"),
        read_norm_file(&ws, "stellar_siren_norm.json"),
    ];

    let client = Client::new();
    let backend = ping_url(&client, "http://127.0.0.1:25255/").await;
    let testbench_backend = if let Some(url) = q.testbench_backend_url.as_deref() {
        Some(ping_url(&client, url).await)
    } else {
        None
    };

    let jobs: Vec<Job> = if q.include_jobs.unwrap_or(false) {
        if let Some(url) = q.testbench_backend_url.as_deref() {
            match client.get(format!("{}/jobs", url)).send().await {
                Ok(resp) => resp.json::<Vec<Job>>().await.unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    Json(SystemSnapshot {
        backend,
        testbench_backend,
        models,
        datasets,
        binaries,
        jobs,
        norms,
        host: collect_host_info(&ws),
    })
}

// =============================================================
// NEW: Pipeline proxy (calls lunar-backend for inference)
// =============================================================

fn stellar_metadata_from_star_lore(lore: StarLore) -> StellarMetadata {
    let (spectral_class, category) = if let Some((sc, cat)) = lore.category.split_once("-type ") {
        (sc.to_string(), cat.to_string())
    } else {
        ("G".into(), "Main Sequence".into())
    };
    StellarMetadata {
        spectral_class,
        category,
        designated_name: lore.designated_name,
        description: lore.visual_profile,
    }
}

fn mg_from_coords(g_mag: f32, x_pc: f32, y_pc: f32, z_pc: f32) -> f32 {
    let d_raw = (x_pc.powi(2) + y_pc.powi(2) + z_pc.powi(2)).sqrt();
    if d_raw < 0.1 { 4.67 } else { g_mag - 5.0 * d_raw.log10() + 5.0 }
}

async fn pipeline_handler(
    Json(payload): Json<PipelineRequest>,
) -> Json<PipelineResponse> {
    let backend_url = "http://127.0.0.1:25255";
    let client = Client::new();

    // 1. Call PINN
    let pinn_req = serde_json::json!({
        "x_pc": payload.x_pc,
        "y_pc": payload.y_pc,
        "z_pc": payload.z_pc,
        "bp_rp": payload.bp_rp,
        "g_mag": payload.g_mag,
    });

    let pinn = match client.post(format!("{}/pinn", backend_url)).json(&pinn_req).send().await {
        Ok(r) => r.json::<PinnResponse>().await.unwrap_or(PinnResponse {
            temperature_k: 5778.0, radius_solar: 1.0, mass_solar: 1.0, luminosity_solar: 1.0,
        }),
        Err(_) => PinnResponse { temperature_k: 5778.0, radius_solar: 1.0, mass_solar: 1.0, luminosity_solar: 1.0 },
    };

    let mg = mg_from_coords(payload.g_mag, payload.x_pc, payload.y_pc, payload.z_pc);
    let log_teff = if pinn.temperature_k > 0.0 { pinn.temperature_k.log10() } else { 3.76 };

    // 2. Call SIREN
    let siren_req = serde_json::json!({
        "width": payload.texture_size, "height": payload.texture_size,
        "bp_rp": payload.bp_rp, "m_g": mg, "log_teff": log_teff,
    });

    let siren = match client.post(format!("{}/siren/texture", backend_url)).json(&siren_req).send().await {
        Ok(r) => r.json::<SirenTextureResponse>().await.unwrap_or(SirenTextureResponse {
            width: payload.texture_size, height: payload.texture_size, pixels: vec![],
        }),
        Err(_) => SirenTextureResponse { width: payload.texture_size, height: payload.texture_size, pixels: vec![] },
    };

    // 3. Get metadata via /description
    let desc_payload = StarDescriptionPayload {
        pinn_payload: pinn.clone(),
        gnn_payload: lunar_structures::GnnResponse { stars: vec![] },
    };

    let lore = match client.post(format!("{}/description", backend_url)).json(&desc_payload).send().await {
        Ok(r) => r.json::<StarLore>().await.unwrap_or(StarLore {
            designated_name: "Lunar Star".into(),
            category: "G-type Main Sequence".into(),
            visual_profile: "A serene yellow star.".into(),
            system_lore: String::new(),
            metadata: lunar_structures::LoreMetadata {
                simulation_engine: "LunarSim v1.0".into(),
                data_source: "Procedurally Generated".into(),
                complexity_level: "High".into(),
            },
        }),
        Err(_) => StarLore {
            designated_name: "Lunar Star".into(),
            category: "G-type Main Sequence".into(),
            visual_profile: "A serene yellow star.".into(),
            system_lore: String::new(),
            metadata: lunar_structures::LoreMetadata {
                simulation_engine: "LunarSim v1.0".into(),
                data_source: "Procedurally Generated".into(),
                complexity_level: "High".into(),
            },
        },
    };
    let metadata = stellar_metadata_from_star_lore(lore);

    Json(PipelineResponse {
        pinn,
        siren,
        metadata,
    })
}

// =============================================================
// NEW: Pipeline PNG (proxy + encode PNG)
// =============================================================

fn encode_rgb_png(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;

    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw.push(0);
        for x in 0..w {
            let i = (y * w + x) * 3;
            raw.push(rgb[i]);
            raw.push(rgb[i + 1]);
            raw.push(rgb[i + 2]);
        }
    }

    let deflate = deflate_minimal(&raw);

    let mut png = Vec::new();
    png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    png.extend_from_slice(&ihdr_chunk(width, height));
    png.extend_from_slice(&idat_chunk(&deflate));
    png.extend_from_slice(&[0, 0, 0, 0, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82]);
    png
}

fn ihdr_chunk(width: u32, height: u32) -> [u8; 25] {
    let mut out = [0u8; 25];
    out[0..4].copy_from_slice(&13u32.to_be_bytes());
    out[4..8].copy_from_slice(b"IHDR");
    out[8..12].copy_from_slice(&width.to_be_bytes());
    out[12..16].copy_from_slice(&height.to_be_bytes());
    out[16] = 8;
    out[17] = 2;
    out[18..21].copy_from_slice(&[0, 0, 0]);
    let crc = crc32(&out[4..21]);
    out[21..25].copy_from_slice(&crc.to_be_bytes());
    out
}

fn idat_chunk(deflated: &[u8]) -> Vec<u8> {
    let len = deflated.len() as u32;
    let mut out = Vec::with_capacity(12 + deflated.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(b"IDAT");
    out.extend_from_slice(deflated);
    let crc = crc32(&out[4..4 + 4 + deflated.len()]);
    out.extend_from_slice(&crc.to_be_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn deflate_minimal(data: &[u8]) -> Vec<u8> {
    let max_block = 65535;
    let num_blocks = (data.len() + max_block - 1) / max_block;
    let mut compressed = Vec::with_capacity(data.len() + num_blocks * 5 + 6);
    compressed.push(0x78);
    compressed.push(0x01);

    let mut offset = 0;
    for i in 0..num_blocks {
        let end = (offset + max_block).min(data.len());
        let block_len = end - offset;
        let bfinal: u8 = if i == num_blocks - 1 { 1 } else { 0 };
        compressed.push(bfinal);
        compressed.extend_from_slice(&(block_len as u16).to_le_bytes());
        compressed.extend_from_slice(&(!(block_len as u16)).to_le_bytes());
        compressed.extend_from_slice(&data[offset..end]);
        offset = end;
    }

    compressed.extend_from_slice(&adler32(data).to_be_bytes());
    compressed
}

#[derive(Deserialize)]
struct PipelinePngParams {
    x_pc: Option<f32>,
    y_pc: Option<f32>,
    z_pc: Option<f32>,
    bp_rp: Option<f32>,
    g_mag: Option<f32>,
    size: Option<u32>,
}

async fn pipeline_png(Query(params): Query<PipelinePngParams>) -> Vec<u8> {
    let x = params.x_pc.unwrap_or(0.0);
    let y = params.y_pc.unwrap_or(0.0);
    let z = params.z_pc.unwrap_or(100.0);
    let bp_rp = params.bp_rp.unwrap_or(1.5);
    let g_mag = params.g_mag.unwrap_or(10.0);
    let size = params.size.unwrap_or(128).max(16).min(512);

    let backend_url = "http://127.0.0.1:25255";
    let client = Client::new();

    // 1. Call PINN
    let pinn_req = serde_json::json!({ "x_pc": x, "y_pc": y, "z_pc": z, "bp_rp": bp_rp, "g_mag": g_mag });

    let teff = match client.post(format!("{}/pinn", backend_url)).json(&pinn_req).send().await {
        Ok(r) => r.json::<PinnResponse>().await.map(|p| p.temperature_k).unwrap_or(5778.0),
        Err(_) => 5778.0,
    };

    let mg = mg_from_coords(g_mag, x, y, z);
    let log_teff = if teff > 0.0 { teff.log10() } else { 3.76 };

    // 2. Call SIREN texture
    let siren_req = serde_json::json!({
        "width": size, "height": size,
        "bp_rp": bp_rp, "m_g": mg, "log_teff": log_teff,
    });

    let pixels = match client.post(format!("{}/siren/texture", backend_url)).json(&siren_req).send().await {
        Ok(r) => r.json::<SirenTextureResponse>().await.map(|s| s.pixels).unwrap_or_default(),
        Err(_) => vec![],
    };

    if pixels.is_empty() {
        return vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    }

    encode_rgb_png(&pixels, size, size)
}

// =============================================================
// main
// =============================================================

#[tokio::main]
async fn main() -> Result<()> {
    let port: u16 = {
        let args: Vec<String> = std::env::args().collect();
        let from_args = args.windows(2)
            .find(|w| w[0] == "--port" || w[0] == "-p")
            .and_then(|w| w[1].parse().ok());
        from_args
            .or_else(|| std::env::var("LUNAR_TESTBENCH_BACKEND_PORT").ok().and_then(|s| s.parse().ok()))
            .unwrap_or(25256)
    };

    let registry = JobRegistry::new();
    let state = AppState { registry: registry.clone() };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(playground_ui))
        .route("/siren", get(siren_ui_page))
        .route("/system/snapshot", get(system_snapshot))
        .route("/pipeline", post(pipeline_handler))
        .route("/pipeline/png", get(pipeline_png))
        .route("/jobs", get(list_jobs))
        .route("/jobs/get", get(get_job_by_query).post(get_job))
        .route("/jobs/cancel", post(cancel_job).get(cancel_job_by_query))
        .route("/jobs/train", post(start_train))
        .route("/jobs/validate", post(start_validate))
        .with_state(state)
        .layer(cors);

    println!("lunar-testbench-backend listening on 127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
