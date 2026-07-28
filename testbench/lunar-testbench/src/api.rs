use gloo_net::http::Request;
use gloo_net::http::Response;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use lunar_utils::env::{get_start_backend_url, get_testbench_url, get_url};

fn err_to_string(e: impl std::fmt::Display) -> String {
    e.to_string()
}

async fn decode_json<T: DeserializeOwned>(resp: Response) -> Result<T, String> {
    let status = resp.status();
    let text = resp.text().await.map_err(err_to_string)?;
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status}: {text}"));
    }
    serde_json::from_str(&text).map_err(err_to_string)
}

async fn get_json<T: DeserializeOwned>(base: &str, path: &str) -> Result<T, String> {
    let url = format!("{base}{path}");
    let resp = Request::get(&url).send().await.map_err(err_to_string)?;
    decode_json(resp).await
}

async fn post_json<B: Serialize, T: DeserializeOwned>(
    base: &str,
    path: &str,
    body: &B,
) -> Result<T, String> {
    let url = format!("{base}{path}");
    let resp = Request::post(&url)
        .json(body)
        .map_err(err_to_string)?
        .send()
        .await
        .map_err(err_to_string)?;
    decode_json(resp).await
}

async fn post_json_value<T: DeserializeOwned>(
    base: &str,
    path: &str,
    body: &Value,
) -> Result<T, String> {
    let url = format!("{base}{path}");
    let resp = Request::post(&url)
        .json(body)
        .map_err(err_to_string)?
        .send()
        .await
        .map_err(err_to_string)?;
    decode_json(resp).await
}

async fn get_ok(base: &str, path: &str) -> Result<(), String> {
    let url = format!("{base}{path}");
    let resp = Request::get(&url).send().await.map_err(err_to_string)?;
    if !(200..300).contains(&resp.status()) {
        return Err(format!("HTTP {}", resp.status()));
    }
    Ok(())
}

pub use lunar_structures_testbench::{
    Job, JobIdPayload, ModelArtifact, SystemSnapshot, TrainSpec, ValidateSpec,
};

/// Runtime execution state of a service (mirrors `lunar_start::backend::ServiceStatus`).
///
/// Deserializes from the tagged JSON shape emitted by `lunar-start-backend`,
/// e.g. `{ "kind": "stopped", "reason": "Pending" }`.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ServiceStatus {
    Starting,
    Running,
    Stopped { reason: String },
    Failed { reason: String },
}

impl ServiceStatus {
    /// Short lowercase tag suitable for CSS classes / LED color lookups.
    pub fn tag(&self) -> &'static str {
        match self {
            ServiceStatus::Starting => "starting",
            ServiceStatus::Running => "running",
            ServiceStatus::Stopped { .. } => "stopped",
            ServiceStatus::Failed { .. } => "failed",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            ServiceStatus::Stopped { reason } | ServiceStatus::Failed { reason } => Some(reason),
            ServiceStatus::Starting | ServiceStatus::Running => None,
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(self, ServiceStatus::Running)
    }
}

impl std::fmt::Display for ServiceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServiceStatus::Starting => write!(f, "starting"),
            ServiceStatus::Running => write!(f, "running"),
            ServiceStatus::Stopped { reason } => write!(f, "stopped ({reason})"),
            ServiceStatus::Failed { reason } => write!(f, "failed ({reason})"),
        }
    }
}

/// A service managed by `lunar-start-backend` (mirrors `ServiceInfo` there).
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceInfo {
    pub name: String,
    pub status: ServiceStatus,
    pub pid: Option<u32>,
}

/// Coarse severity of a streamed log line (mirrors `lunar_start::backend::LogLevel`).
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

/// A single streamed log line from `lunar-start-backend` (mirrors `lunar_start::backend::LogEvent`).
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceLogEvent {
    pub timestamp: String,
    pub service: String,
    pub text: String,
    pub is_stderr: bool,
    pub level: LogLevel,
}

/// Cheap liveness probe for `lunar-start-backend` (mirrors `HealthResponse` there).
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct HealthResponse {
    pub ok: bool,
    pub version: String,
    pub uptime_ms: u128,
}

/// Static per-service manifest entry (mirrors `lunar_start_backend::service_meta::ServiceMeta`).
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceMeta {
    pub name: String,
    pub title: String,
    pub icon: String,
    pub description: String,
    pub kind: String,
    pub url: Option<String>,
    pub provides: Vec<String>,
    pub depends_on: Vec<String>,
}

/// Aggregated warn/error counts for a service's buffered logs (mirrors `ServiceStats` there).
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceStats {
    pub name: String,
    pub warn_count: usize,
    pub err_count: usize,
    pub log_count: usize,
}

pub async fn system_snapshot() -> Result<SystemSnapshot, String> {
    let tb_url = get_testbench_url();
    get_json(
        &tb_url,
        &format!(
            "/system/snapshot?include_jobs=true&testbench_backend_url={}",
            urlencoding(&tb_url)
        ),
    )
    .await
}

pub async fn list_jobs() -> Result<Vec<Job>, String> {
    get_json(&get_testbench_url(), "/jobs").await
}

pub async fn start_train(spec: &TrainSpec) -> Result<Job, String> {
    post_json(&get_testbench_url(), "/jobs/train", spec).await
}

pub async fn start_validate(spec: &ValidateSpec) -> Result<Job, String> {
    post_json(&get_testbench_url(), "/jobs/validate", spec).await
}

pub async fn cancel_job(id: &str) -> Result<(), String> {
    let payload = JobIdPayload { id: id.to_string() };
    let tb_url = get_testbench_url();
    let resp = Request::post(&format!("{tb_url}/jobs/cancel"))
        .json(&payload)
        .map_err(err_to_string)?
        .send()
        .await
        .map_err(err_to_string)?;
    if !(200..300).contains(&resp.status()) {
        return Err(format!("HTTP {}", resp.status()));
    }
    Ok(())
}

/// List services managed by `lunar-start-backend` (testbench, testbench-backend, etc.).
pub async fn list_start_services() -> Result<Vec<ServiceInfo>, String> {
    get_json(&get_start_backend_url(), "/services").await
}

/// Cheap liveness probe used to detect when `lunar-start-backend` itself has come online.
pub async fn health() -> Result<HealthResponse, String> {
    get_json(&get_start_backend_url(), "/health").await
}

/// Static manifest of managed services (display metadata + dock app `provides`/`depends_on`).
pub async fn services_meta() -> Result<Vec<ServiceMeta>, String> {
    get_json(&get_start_backend_url(), "/services/meta").await
}

/// Last `tail` buffered log lines for a service, so a newly opened log window isn't empty.
pub async fn service_log_tail(name: &str, tail: usize) -> Result<Vec<ServiceLogEvent>, String> {
    get_json(
        &get_start_backend_url(),
        &format!("/services/{name}/logs?tail={tail}"),
    )
    .await
}

/// Aggregated warn/error counts for a service, used for LED badges on the rack.
pub async fn service_stats(name: &str) -> Result<ServiceStats, String> {
    get_json(&get_start_backend_url(), &format!("/services/{name}/stats")).await
}

pub async fn start_all_services() -> Result<(), String> {
    get_ok(&get_start_backend_url(), "/start").await
}

pub async fn stop_all_services() -> Result<(), String> {
    get_ok(&get_start_backend_url(), "/stop").await
}

pub async fn start_service(name: &str) -> Result<(), String> {
    get_ok(&get_start_backend_url(), &format!("/start/{name}")).await
}

pub async fn stop_service(name: &str) -> Result<(), String> {
    get_ok(&get_start_backend_url(), &format!("/stop/{name}")).await
}

pub async fn restart_service(name: &str) -> Result<(), String> {
    get_ok(&get_start_backend_url(), &format!("/restart/{name}")).await
}

/// SSE endpoint that streams `ServiceLogEvent`s for all services managed by `lunar-start-backend`.
pub fn start_backend_logs_url() -> String {
    format!("{}/logs", get_start_backend_url())
}

pub async fn pinn_infer(body: &Value) -> Result<Value, String> {
    post_json_value(&get_url(), "/pinn", body).await
}

pub async fn gnn_infer(body: &Value) -> Result<Value, String> {
    post_json_value(&get_url(), "/gnn", body).await
}

pub async fn siren_texture(body: &Value) -> Result<Value, String> {
    post_json_value(&get_url(), "/siren/texture", body).await
}

pub async fn random_star(body: &Value) -> Result<Value, String> {
    post_json_value(&get_url(), "/random_star", body).await
}

pub async fn description(body: &Value) -> Result<Value, String> {
    post_json_value(&get_url(), "/description", body).await
}

pub async fn pipeline(body: &Value) -> Result<Value, String> {
    post_json_value(&get_url(), "/pipeline", body).await
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct PipelinePngQuery {
    pub x_pc: f32,
    pub y_pc: f32,
    pub z_pc: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    pub size: u32,
}

pub async fn pipeline_png(q: &PipelinePngQuery) -> Result<(Vec<u8>, u32, u32), String> {
    let path = format!(
        "/pipeline/png?x_pc={}&y_pc={}&z_pc={}&bp_rp={}&g_mag={}&size={}",
        q.x_pc, q.y_pc, q.z_pc, q.bp_rp, q.g_mag, q.size
    );
    let url = format!("{}{}", get_url(), path);
    let resp = Request::get(&url).send().await.map_err(err_to_string)?;
    if !(200..300).contains(&resp.status()) {
        return Err(format!("HTTP {}", resp.status()));
    }
    let bytes = resp.binary().await.map_err(err_to_string)?;
    let (w, h) = read_png_dims(&bytes).unwrap_or((0, 0));
    Ok((bytes, w, h))
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct SirenPngQuery {
    pub width: u32,
    pub height: u32,
    pub bp_rp: f32,
    pub m_g: f32,
    pub temperature_k: f32,
}

pub async fn siren_png(q: &SirenPngQuery) -> Result<(Vec<u8>, u32, u32), String> {
    let path = format!(
        "/siren/png?width={}&height={}&bp_rp={}&m_g={}&temperature_k={}",
        q.width, q.height, q.bp_rp, q.m_g, q.temperature_k
    );
    let url = format!("{}{}", get_url(), path);
    let resp = Request::get(&url).send().await.map_err(err_to_string)?;
    if !(200..300).contains(&resp.status()) {
        return Err(format!("HTTP {}", resp.status()));
    }
    let bytes = resp.binary().await.map_err(err_to_string)?;
    let (w, h) = read_png_dims(&bytes).unwrap_or((0, 0));
    Ok((bytes, w, h))
}

pub async fn backend_proxy(
    path: &str,
    method: &str,
    body: Option<&Value>,
    query: Option<&str>,
) -> Result<Value, String> {
    let base_url = get_url();
    let url = match query {
        Some(q) if !q.is_empty() => format!("{base_url}{path}?{q}"),
        _ => format!("{base_url}{path}"),
    };
    let m = method.to_uppercase();
    let resp = if let Some(b) = body {
        match m.as_str() {
            "GET" => {
                Request::get(&url)
                    .json(b)
                    .map_err(err_to_string)?
                    .send()
                    .await
            }
            "POST" => {
                Request::post(&url)
                    .json(b)
                    .map_err(err_to_string)?
                    .send()
                    .await
            }
            "PUT" => {
                Request::put(&url)
                    .json(b)
                    .map_err(err_to_string)?
                    .send()
                    .await
            }
            "DELETE" => {
                Request::delete(&url)
                    .json(b)
                    .map_err(err_to_string)?
                    .send()
                    .await
            }
            _ => return Err(format!("unsupported method: {method}")),
        }
    } else {
        match m.as_str() {
            "GET" => Request::get(&url).send().await,
            "POST" => Request::post(&url).send().await,
            "PUT" => Request::put(&url).send().await,
            "DELETE" => Request::delete(&url).send().await,
            _ => return Err(format!("unsupported method: {method}")),
        }
    }
    .map_err(err_to_string)?;

    let status = resp.status();
    let text = resp.text().await.map_err(err_to_string)?;
    if !(200..300).contains(&status) {
        return Err(format!("HTTP {status}: {text}"));
    }
    if text.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&text).map_err(err_to_string)
}

pub fn read_png_dims(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    Some((w, h))
}

pub fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}
