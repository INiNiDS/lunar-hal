use std::collections::HashMap;

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

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ServiceApiError {
    pub status: u16,
    pub code: String,
    pub message: String,
    pub field_errors: HashMap<String, String>,
}

impl std::fmt::Display for ServiceApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.status == 0 {
            write!(f, "{}", self.message)
        } else {
            write!(f, "HTTP {}: {}", self.status, self.message)
        }
    }
}

impl std::error::Error for ServiceApiError {}

#[derive(serde::Deserialize)]
struct ServiceApiErrorBody {
    error: String,
    message: String,
    #[serde(default)]
    field_errors: HashMap<String, String>,
}

fn network_error(error: impl std::fmt::Display) -> ServiceApiError {
    ServiceApiError {
        status: 0,
        code: "network_error".to_string(),
        message: error.to_string(),
        field_errors: HashMap::new(),
    }
}

async fn decode_service_json<T: DeserializeOwned>(resp: Response) -> Result<T, ServiceApiError> {
    let status = resp.status();
    let text = resp.text().await.map_err(network_error)?;
    if !(200..300).contains(&status) {
        let body =
            serde_json::from_str::<ServiceApiErrorBody>(&text).unwrap_or(ServiceApiErrorBody {
                error: "http_error".to_string(),
                message: if text.is_empty() {
                    format!("Request failed with status {status}")
                } else {
                    text
                },
                field_errors: HashMap::new(),
            });
        return Err(ServiceApiError {
            status,
            code: body.error,
            message: body.message,
            field_errors: body.field_errors,
        });
    }
    serde_json::from_str(&text).map_err(|error| ServiceApiError {
        status,
        code: "invalid_response".to_string(),
        message: error.to_string(),
        field_errors: HashMap::new(),
    })
}

async fn service_get<T: DeserializeOwned>(path: &str) -> Result<T, ServiceApiError> {
    let url = format!("{}{path}", get_start_backend_url());
    let response = Request::get(&url).send().await.map_err(network_error)?;
    decode_service_json(response).await
}

async fn service_put<B: Serialize, T: DeserializeOwned>(
    path: &str,
    body: &B,
) -> Result<T, ServiceApiError> {
    let url = format!("{}{path}", get_start_backend_url());
    let response = Request::put(&url)
        .json(body)
        .map_err(network_error)?
        .send()
        .await
        .map_err(network_error)?;
    decode_service_json(response).await
}

async fn service_post<B: Serialize, T: DeserializeOwned>(
    path: &str,
    body: &B,
) -> Result<T, ServiceApiError> {
    let url = format!("{}{path}", get_start_backend_url());
    let response = Request::post(&url)
        .json(body)
        .map_err(network_error)?
        .send()
        .await
        .map_err(network_error)?;
    decode_service_json(response).await
}

async fn service_post_empty<T: DeserializeOwned>(path: &str) -> Result<T, ServiceApiError> {
    let url = format!("{}{path}", get_start_backend_url());
    let response = Request::post(&url).send().await.map_err(network_error)?;
    decode_service_json(response).await
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

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    String,
    Port,
    Path,
    Boolean,
    Select { options: Vec<String> },
    StringList,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceConfigField {
    pub key: String,
    pub label: String,
    pub description: Option<String>,
    pub field_type: FieldType,
    pub default_value: String,
    pub is_build_param: bool,
    pub required: bool,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub allowed_values: Option<Vec<String>>,
    pub read_only: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceConfigSchema {
    pub service: String,
    pub fields: Vec<ServiceConfigField>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq, Default)]
pub struct ServiceConfigValues {
    pub env: HashMap<String, String>,
    pub extra_args: Vec<String>,
    pub build_args: Vec<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceConfigState {
    pub defaults: ServiceConfigValues,
    pub saved: ServiceConfigValues,
    pub effective: Option<ServiceConfigValues>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq, Eq)]
pub struct FieldError {
    pub field: String,
    pub message: String,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq, Eq)]
pub struct ServiceValidationResult {
    pub ok: bool,
    pub field_errors: HashMap<String, String>,
}

impl ServiceValidationResult {
    pub fn errors(&self) -> Vec<FieldError> {
        let mut errors = self
            .field_errors
            .iter()
            .map(|(field, message)| FieldError {
                field: field.clone(),
                message: message.clone(),
            })
            .collect::<Vec<_>>();
        errors.sort_by(|left, right| left.field.cmp(&right.field));
        errors
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq, Default)]
pub struct StartServiceRequest {
    pub config: Option<ServiceConfigValues>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct ServiceActionResponse {
    pub service: ServiceInfo,
    pub config: ServiceConfigState,
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

pub async fn get_service_config_schema(name: &str) -> Result<ServiceConfigSchema, ServiceApiError> {
    service_get(&format!("/services/{name}/config/schema")).await
}

pub async fn get_service_config(name: &str) -> Result<ServiceConfigState, ServiceApiError> {
    service_get(&format!("/services/{name}/config")).await
}

pub async fn validate_service_config(
    name: &str,
    values: &ServiceConfigValues,
) -> Result<ServiceValidationResult, ServiceApiError> {
    service_post(&format!("/services/{name}/validate"), values).await
}

pub async fn save_service_config(
    name: &str,
    values: &ServiceConfigValues,
) -> Result<ServiceConfigState, ServiceApiError> {
    service_put(&format!("/services/{name}/config"), values).await
}

pub async fn start_service_request(
    name: &str,
    request: &StartServiceRequest,
) -> Result<ServiceActionResponse, ServiceApiError> {
    service_post(&format!("/services/{name}/start"), request).await
}

pub async fn start_service(name: &str) -> Result<ServiceActionResponse, ServiceApiError> {
    start_service_request(name, &StartServiceRequest::default()).await
}

pub async fn stop_service(name: &str) -> Result<ServiceActionResponse, ServiceApiError> {
    service_post_empty(&format!("/services/{name}/stop")).await
}

pub async fn restart_service_request(
    name: &str,
    request: &StartServiceRequest,
) -> Result<ServiceActionResponse, ServiceApiError> {
    service_post(&format!("/services/{name}/restart"), request).await
}

pub async fn restart_service(name: &str) -> Result<ServiceActionResponse, ServiceApiError> {
    restart_service_request(name, &StartServiceRequest::default()).await
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

#[cfg(test)]
mod config_api_tests {
    use super::*;

    #[test]
    fn config_api_models_match_server_json() {
        let schema: ServiceConfigSchema = serde_json::from_value(serde_json::json!({
            "service": "backend",
            "fields": [{
                "key": "COMPUTE_BACKEND",
                "label": "Compute backend",
                "description": null,
                "field_type": {"select": {"options": ["wgpu", "cuda"]}},
                "default_value": "wgpu",
                "is_build_param": true,
                "required": true,
                "min": null,
                "max": null,
                "allowed_values": ["wgpu", "cuda"],
                "read_only": false
            }]
        }))
        .unwrap();
        assert_eq!(schema.service, "backend");
        assert!(matches!(
            schema.fields[0].field_type,
            FieldType::Select { .. }
        ));
    }

    #[test]
    fn field_errors_are_normalized_for_ui_rendering() {
        let result = ServiceValidationResult {
            ok: false,
            field_errors: HashMap::from([
                ("port".to_string(), "Invalid port".to_string()),
                ("host".to_string(), "Required".to_string()),
            ]),
        };
        assert_eq!(
            result.errors(),
            [
                FieldError {
                    field: "host".to_string(),
                    message: "Required".to_string(),
                },
                FieldError {
                    field: "port".to_string(),
                    message: "Invalid port".to_string(),
                },
            ]
        );
    }

    #[test]
    fn empty_start_request_serializes_with_no_config() {
        let value = serde_json::to_value(StartServiceRequest::default()).unwrap();
        assert!(value["config"].is_null());
    }
}
