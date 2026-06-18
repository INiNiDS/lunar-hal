use std::path::{Path, PathBuf};
use std::time::SystemTime;

use axum::Json;
use axum::extract::Query;
use reqwest::Client;
use serde::Deserialize;
use walkdir::WalkDir;

use lunar_structures_testbench::{
    BackendStatus, BinaryInfo, DatasetInfo, HostInfo, Job, ModelArtifact, NormSnapshot,
    SystemSnapshot,
};

use crate::jobs::{now_ms, workspace_root};

pub fn scan_models(ws: &Path) -> Vec<ModelArtifact> {
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

fn is_dataset_file(name: &str) -> bool {
    name.ends_with(".parquet") || name.ends_with(".csv")
}

fn mtime_ms_from_metadata(md: &std::fs::Metadata) -> u64 {
    md.modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn dataset_info_for(p: &Path) -> Option<DatasetInfo> {
    let name = p
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    if !is_dataset_file(&name) {
        return None;
    }
    let md = std::fs::metadata(p).ok()?;
    Some(DatasetInfo {
        name: name.clone(),
        path: p.to_string_lossy().to_string(),
        size_bytes: md.len(),
        mtime_ms: mtime_ms_from_metadata(&md),
        kind: detect_dataset_kind(&name),
    })
}

fn scan_directory(ws: &Path, subdir: &str) -> Vec<DatasetInfo> {
    let path = ws.join(subdir);
    if !path.exists() {
        return Vec::new();
    }
    WalkDir::new(&path)
        .max_depth(2)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let p = entry.path();
            if !p.is_file() {
                None
            } else {
                dataset_info_for(p)
            }
        })
        .collect()
}

pub fn scan_datasets(ws: &Path) -> Vec<DatasetInfo> {
    let dirs = ["ai_data", "data/chunks", "data"];
    let mut out = Vec::new();
    for d in dirs {
        out.extend(scan_directory(ws, d));
    }
    out
}

pub fn detect_dataset_kind(name: &str) -> String {
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

pub fn binary_status(ws: &Path, name: &str) -> BinaryInfo {
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

pub fn read_norm_file(ws: &Path, file: &str) -> NormSnapshot {
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
            };
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

pub fn collect_host_info(ws: &Path) -> HostInfo {
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

pub async fn ping_url(client: &Client, url: &str) -> BackendStatus {
    let start = SystemTime::now();
    let resp = client.get(url).send().await;
    let (reachable, latency, hint) = match resp {
        Ok(r) => {
            let latency = start
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            let hint = if r.status().is_success() {
                "HTTP 200"
            } else {
                "HTTP non-2xx"
            };
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
pub struct SnapshotQuery {
    pub include_jobs: Option<bool>,
    pub testbench_backend_url: Option<String>,
}

pub async fn system_snapshot(Query(q): Query<SnapshotQuery>) -> Json<SystemSnapshot> {
    let ws = workspace_root();
    let models = scan_models(&ws);
    let datasets = scan_datasets(&ws);
    let binary_names = [
        "lnai",
        "lnai-gnn",
        "lnai-siren",
        "lunar-ai-cli",
        "lunar-backend",
        "lunar-testbench",
        "lunar-testbench-backend",
    ];
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
