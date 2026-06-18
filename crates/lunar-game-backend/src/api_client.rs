//! HTTP client for talking to the AI backend (`lunar-backend`).
//!
//! The game layer never talks to AI models directly; it always goes
//! through this client. The frontend (or test harness) supplies the
//! base URL, the client is constructed by [`Game`].

use lunar_structures::{
    CreateWorldRequest, GnnResponse, PipelineRequest, PipelineResponse, RandomStarRequest,
    RandomStarResponse, SectorRequest, World, WorldListResponse,
};
use lunar_utils::env::get_url;
use parking_lot::RwLock;
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur when communicating with the AI backend.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("JSON decode error: {0}")]
    Decode(String),
    #[error("Bad status {status}: {body}")]
    BadStatus { status: u16, body: String },
    #[error("Configuration error: {0}")]
    Config(String),
}

impl From<reqwest::Error> for ApiError {
    fn from(e: reqwest::Error) -> Self {
        Self::Http(e.to_string())
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(e: serde_json::Error) -> Self {
        Self::Decode(e.to_string())
    }
}

/// Lightweight async client. It owns a `reqwest::Client` for the
/// process lifetime and forwards calls to the AI backend endpoints.
pub struct ApiClient {
    base_url: RwLock<String>,
    http: reqwest::Client,
}

impl std::fmt::Debug for ApiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiClient")
            .field("base_url", &*self.base_url.read())
            .field("http", &"<reqwest::Client>")
            .finish()
    }
}

impl ApiClient {
    /// Create a new client with an explicit base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        let mut builder = reqwest::Client::builder();

        #[cfg(not(target_arch = "wasm32"))]
        {
            builder = builder.timeout(Duration::from_mins(1));
        }

        let http = builder.build().expect("reqwest client should build");

        Self {
            base_url: RwLock::new(base_url.into()),
            http,
        }
    }

    /// Create a client pointed at the URL described by `LUNAR_BACKEND_HOST`
    /// / `LUNAR_BACKEND_PORT` (or the default `127.0.0.1:25255`).
    pub fn from_env() -> Self {
        Self::new(get_url())
    }

    pub fn base_url(&self) -> String {
        self.base_url.read().clone()
    }

    pub fn set_base_url(&self, url: impl Into<String>) {
        *self.base_url.write() = url.into();
    }

    async fn post_json<B, R>(&self, path: &str, body: &B) -> Result<R, ApiError>
    where
        B: Serialize,
        R: serde::de::DeserializeOwned,
    {
        let url = format!("{}{}", self.base_url.read(), path);
        let builder = self.http.post(url).json(body);
        let resp = send_and_validate(builder).await?;
        Ok(resp.json().await?)
    }

    async fn get_json<R>(&self, path: &str) -> Result<R, ApiError>
    where
        R: serde::de::DeserializeOwned,
    {
        let url = format!("{}{}", self.base_url.read(), path);
        let builder = self.http.get(url);
        let resp = send_and_validate(builder).await?;
        Ok(resp.json().await?)
    }

    async fn delete(&self, path: &str) -> Result<(), ApiError> {
        let url = format!("{}{}", self.base_url.read(), path);
        let builder = self.http.delete(url);
        send_and_validate(builder).await?;
        Ok(())
    }

    pub async fn list_worlds(&self) -> Result<WorldListResponse, ApiError> {
        self.get_json("/worlds").await
    }

    pub async fn get_world(&self, id: &str) -> Result<World, ApiError> {
        self.get_json(&format!("/worlds/{id}")).await
    }

    pub async fn create_world(&self, req: CreateWorldRequest) -> Result<World, ApiError> {
        self.post_json("/worlds/create", &req).await
    }

    pub async fn delete_world(&self, id: &str) -> Result<(), ApiError> {
        self.delete(&format!("/worlds/{id}")).await
    }

    pub async fn sector_stars(&self, req: SectorRequest) -> Result<GnnResponse, ApiError> {
        self.post_json("/sector/stars", &req).await
    }

    pub async fn random_star(
        &self,
        req: RandomStarRequest,
    ) -> Result<RandomStarResponse, ApiError> {
        self.post_json("/random_star", &req).await
    }

    // === Pipeline ===

    pub async fn pipeline(&self, req: PipelineRequest) -> Result<PipelineResponse, ApiError> {
        self.post_json("/pipeline", &req).await
    }
}

/// Raw value passthrough for endpoints that should remain flexible
/// (e.g., testbench probes). The game layer itself does not use this.
pub async fn raw_post(client: &ApiClient, path: &str, body: &Value) -> Result<Value, ApiError> {
    let url = format!("{}{}", client.base_url(), path);
    let builder = client.http.post(url).json(body);
    let resp = send_and_validate(builder).await?;
    Ok(resp.json().await?)
}

/// Helper function to perform the request and validate the response status,
/// reducing duplicate error handling across different HTTP methods.
async fn send_and_validate(
    builder: reqwest::RequestBuilder,
) -> Result<reqwest::Response, ApiError> {
    let resp = builder.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(ApiError::BadStatus {
            status: status.as_u16(),
            body,
        });
    }
    Ok(resp)
}
