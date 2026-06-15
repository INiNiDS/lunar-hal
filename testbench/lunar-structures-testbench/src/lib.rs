use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    Pinn,
    Gnn,
    Siren,
}

impl ModelKind {
    pub fn slug(&self) -> &'static str {
        match self {
            ModelKind::Pinn => "pinn",
            ModelKind::Gnn => "gnn",
            ModelKind::Siren => "siren",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ModelKind::Pinn => "PINN",
            ModelKind::Gnn => "GNN",
            ModelKind::Siren => "SIREN",
        }
    }

    pub fn binary_name(&self) -> &'static str {
        match self {
            ModelKind::Pinn => "lnai",
            ModelKind::Gnn => "lnai-gnn",
            ModelKind::Siren => "lnai-siren",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s {
            "pinn" => Some(Self::Pinn),
            "gnn" => Some(Self::Gnn),
            "siren" => Some(Self::Siren),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ModelArtifact {
    pub name: String,
    pub path: String,
    pub kind: String,
    pub size_bytes: u64,
    pub mtime_ms: u64,
    pub exists: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DatasetInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub mtime_ms: u64,
    pub kind: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct NormSnapshot {
    pub kind: String,
    pub path: String,
    pub exists: bool,
    pub data: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BinaryInfo {
    pub name: String,
    pub path: String,
    pub exists: bool,
    pub size_bytes: Option<u64>,
    pub mtime_ms: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HostInfo {
    pub workspace_root: String,
    pub pid: u32,
    pub cpu_count: usize,
    pub total_memory_bytes: u64,
    pub rustc_version: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BackendStatus {
    pub url: String,
    pub reachable: bool,
    pub latency_ms: Option<u64>,
    pub last_checked_ms: u64,
    pub version_hint: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SystemSnapshot {
    pub backend: BackendStatus,
    pub testbench_backend: Option<BackendStatus>,
    pub models: Vec<ModelArtifact>,
    pub datasets: Vec<DatasetInfo>,
    pub binaries: Vec<BinaryInfo>,
    pub jobs: Vec<Job>,
    pub norms: Vec<NormSnapshot>,
    pub host: HostInfo,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn tag(&self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }
}

impl JobKind {
    pub fn model_kind(&self) -> ModelKind {
        match self {
            JobKind::Train(t) => t.model.clone(),
            JobKind::Validate(v) => v.model.clone(),
            JobKind::Custom(_) => ModelKind::Pinn,
        }
    }
}

impl Job {
    pub fn new(kind: JobKind, title: String, total_epochs: u32) -> Self {
        let total = total_epochs.max(1);
        Self {
            id: new_job_id(),
            created_ms: now_ms(),
            started_ms: None,
            finished_ms: None,
            status: JobStatus::Queued,
            exit_code: None,
            spec: kind,
            title,
            last_metrics: Vec::new(),
            best_val_loss: None,
            log_tail: Vec::with_capacity(1024),
            error_summary: None,
            total_epochs_planned: total,
        }
    }

    pub fn progress(&self) -> f32 {
        if self.total_epochs_planned == 0 {
            return 0.0;
        }
        let last_epoch = self
            .last_metrics
            .last()
            .map(|m| m.epoch as f32)
            .unwrap_or(0.0);
        (last_epoch / self.total_epochs_planned as f32).clamp(0.0, 1.0)
    }
}

fn new_job_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let ctr = CTR.fetch_add(1, Ordering::Relaxed);
    format!("job-{nanos:x}-{ctr:x}")
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrainSpec {
    pub model: ModelKind,
    pub epochs: u32,
    pub batch_size: u32,
    pub lr: f64,
    pub physics_weight: f64,
    pub val_frac: f32,
    pub data_path: String,
    pub output_dir: String,
    pub resume_from: Option<String>,
    pub holdout: Option<String>,
    pub gpu_index: u32,
    pub knn_k: Option<u32>,
    pub hidden_dim: Option<u32>,
    pub texture_size: Option<u32>,
    pub max_stars: Option<u32>,
    pub patience: u32,
    pub grad_accum: u32,
    pub clip_grad_norm: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ValidateSpec {
    pub model: ModelKind,
    pub data_path: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub val_frac: f32,
    pub output_dir: String,
    pub hidden_dim: Option<u32>,
    pub knn_k: Option<u32>,
    pub texture_size: Option<u32>,
    pub max_stars: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobKind {
    Train(TrainSpec),
    Validate(ValidateSpec),
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LogLineKind {
    Header,
    Info,
    Warning,
    Error,
    Metric,
    EpochStart,
    EpochEnd,
    Checkpoint,
    Raw,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LogEntry {
    pub timestamp_ms: u64,
    pub line: String,
    pub kind: LogLineKind,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EpochMetric {
    pub epoch: u32,
    pub train_loss: f64,
    pub val_loss: f64,
    pub phys_loss: Option<f64>,
    pub lr: f64,
    pub timestamp_ms: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Job {
    pub id: String,
    pub created_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub status: JobStatus,
    pub exit_code: Option<i32>,
    pub spec: JobKind,
    pub title: String,
    pub last_metrics: Vec<EpochMetric>,
    pub best_val_loss: Option<f64>,
    pub log_tail: Vec<LogEntry>,
    pub error_summary: Option<String>,
    pub total_epochs_planned: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct JobIdPayload {
    pub id: String,
}
