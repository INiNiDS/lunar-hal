use gloo_net::http::Request;
use gloo_net::http::Response;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

const BACKEND_URL: &str = match option_env!("LUNAR_BACKEND_URL") {
    Some(s) => s,
    None => "http://127.0.0.1:25255",
};

const TESTBENCH_BACKEND_URL: &str = match option_env!("LUNAR_TESTBENCH_BACKEND_URL") {
    Some(s) => s,
    None => "http://127.0.0.1:25256",
};

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
    let resp = Request::get(&url)
        .send()
        .await
        .map_err(err_to_string)?;
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
    body: Value,
) -> Result<T, String> {
    let url = format!("{base}{path}");
    let resp = Request::post(&url)
        .json(&body)
        .map_err(err_to_string)?
        .send()
        .await
        .map_err(err_to_string)?;
    decode_json(resp).await
}


pub use lunar_structures_testbench::{
    Job, JobIdPayload, ModelArtifact, SystemSnapshot, TrainSpec,
    ValidateSpec,
};

pub async fn system_snapshot() -> Result<SystemSnapshot, String> {
    get_json(
        TESTBENCH_BACKEND_URL,
        &format!(
            "/system/snapshot?include_jobs=true&testbench_backend_url={}",
            urlencoding(TESTBENCH_BACKEND_URL)
        ),
    )
    .await
}

pub async fn pinn_infer(body: Value) -> Result<Value, String> {
    post_json_value(BACKEND_URL, "/pinn", body).await
}

pub async fn gnn_infer(body: Value) -> Result<Value, String> {
    post_json_value(BACKEND_URL, "/gnn", body).await
}

pub async fn siren_texture(body: Value) -> Result<Value, String> {
    post_json_value(BACKEND_URL, "/siren/texture", body).await
}

pub async fn random_star(body: Value) -> Result<Value, String> {
    post_json_value(BACKEND_URL, "/random_star", body).await
}

pub async fn pipeline(body: Value) -> Result<Value, String> {
    post_json_value(TESTBENCH_BACKEND_URL, "/pipeline", body).await
}

pub async fn description(body: Value) -> Result<Value, String> {
    post_json_value(BACKEND_URL, "/description", body).await
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, PartialEq)]
pub struct SirenPngQuery {
    pub width: u32,
    pub height: u32,
    pub bp_rp: f32,
    pub m_g: f32,
    pub temperature_k: f32,
}

pub async fn siren_png(q: SirenPngQuery) -> Result<(Vec<u8>, u32, u32), String> {
    let path = format!(
        "/siren/png?width={}&height={}&bp_rp={}&m_g={}&temperature_k={}",
        q.width, q.height, q.bp_rp, q.m_g, q.temperature_k
    );
    let url = format!("{BACKEND_URL}{path}");
    let resp = Request::get(&url).send().await.map_err(err_to_string)?;
    if !(200..300).contains(&resp.status()) {
        return Err(format!("HTTP {}", resp.status()));
    }
    let bytes = resp.binary().await.map_err(err_to_string)?;
    let (w, h) = read_png_dims(&bytes).unwrap_or((0, 0));
    Ok((bytes, w, h))
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

pub async fn pipeline_png(q: PipelinePngQuery) -> Result<(Vec<u8>, u32, u32), String> {
    let path = format!(
        "/pipeline/png?x_pc={}&y_pc={}&z_pc={}&bp_rp={}&g_mag={}&size={}",
        q.x_pc, q.y_pc, q.z_pc, q.bp_rp, q.g_mag, q.size
    );
    let url = format!("{TESTBENCH_BACKEND_URL}{path}");
    let resp = Request::get(&url).send().await.map_err(err_to_string)?;
    if !(200..300).contains(&resp.status()) {
        return Err(format!("HTTP {}", resp.status()));
    }
    let bytes = resp.binary().await.map_err(err_to_string)?;
    let (w, h) = read_png_dims(&bytes).unwrap_or((0, 0));
    Ok((bytes, w, h))
}

pub async fn backend_proxy(path: String, method: String, body: Option<Value>, query: Option<String>) -> Result<Value, String> {
    let url = match &query {
        Some(q) if !q.is_empty() => format!("{BACKEND_URL}{path}?{q}"),
        _ => format!("{BACKEND_URL}{path}"),
    };
    let m = method.to_uppercase();
    let resp = if let Some(b) = body {
        match m.as_str() {
            "GET" => Request::get(&url).json(&b).map_err(err_to_string)?.send().await,
            "POST" => Request::post(&url).json(&b).map_err(err_to_string)?.send().await,
            "PUT" => Request::put(&url).json(&b).map_err(err_to_string)?.send().await,
            "DELETE" => Request::delete(&url).json(&b).map_err(err_to_string)?.send().await,
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


pub async fn list_jobs() -> Result<Vec<Job>, String> {
    get_json(TESTBENCH_BACKEND_URL, "/jobs").await
}

pub async fn start_train(spec: TrainSpec) -> Result<Job, String> {
    post_json(TESTBENCH_BACKEND_URL, "/jobs/train", &spec).await
}

pub async fn start_validate(spec: ValidateSpec) -> Result<Job, String> {
    post_json(TESTBENCH_BACKEND_URL, "/jobs/validate", &spec).await
}

pub async fn cancel_job(id: String) -> Result<(), String> {
    let payload = JobIdPayload { id };
    let resp = Request::post(&format!("{TESTBENCH_BACKEND_URL}/jobs/cancel"))
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
