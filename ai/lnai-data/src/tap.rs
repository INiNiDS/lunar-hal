//! Production [`ShardFetcher`] backed by synchronous TAP-over-HTTP
//! (ESA Gaia archive). Works anonymously; if basic-auth credentials are
//! supplied by the caller they are sent once per request header and never
//! logged or persisted here.
//!
//! Row accounting streams the body straight to disk (no full-body buffering),
//! which keeps memory flat for very large shards.

use crate::collector::{FetchError, ShardFetcher};
use std::io::Write;
use std::path::Path;
use std::time::Duration;

pub const DEFAULT_GAIA_TAP_SYNC_URL: &str = "https://gea.esac.esa.int/tap-server/tap/sync";

#[derive(Debug, Clone)]
pub struct TapFetcher {
    sync_url: String,
    /// Caller-supplied basic-auth tuple; kept opaque and never printed.
    credentials: Option<(String, String)>,
    request_timeout_secs: u64,
}

impl TapFetcher {
    pub fn anonymous() -> Self {
        Self {
            sync_url: DEFAULT_GAIA_TAP_SYNC_URL.to_string(),
            credentials: None,
            request_timeout_secs: 3600,
        }
    }

    pub fn with_credentials(mut self, user: Option<String>, password: Option<String>) -> Self {
        // Both halves required for meaningful basic auth; partial config is
        // treated as anonymous rather than half-configured.
        match (
            user.filter(|u| !u.trim().is_empty()),
            password.filter(|p| !p.is_empty()),
        ) {
            (Some(u), Some(p)) => self.credentials = Some((u, p)),
            _ => {}
        }
        self
    }

    fn build_request(
        &self,
        client: &reqwest::blocking::Client,
        query: &str,
    ) -> Result<reqwest::blocking::RequestBuilder, FetchError> {
        let mut req = client.get(&self.sync_url).query(&[
            ("REQUEST", "doQuery"),
            ("LANG", "ADQL"),
            ("FORMAT", "csv"),
            ("QUERY", query),
        ]);
        if let Some((user, pass)) = &self.credentials {
            // NOTE: intentionally not logging any credential material.
            req = req.basic_auth(user.clone(), Some(pass.clone()));
        }
        Ok(req)
    }
}

impl ShardFetcher for TapFetcher {
    fn fetch(&self, query: &str, dest: &Path) -> Result<u64, FetchError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Some(Duration::from_secs(self.request_timeout_secs)))
            .connect_timeout(Some(Duration::from_secs(60)))
            .build()
            .map_err(|e| FetchError::Http(e.to_string()))?;

        let response = self
            .build_request(&client, query)?
            .send()
            .map_err(|e| FetchError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let code = status.as_u16();
            return Err(match code {
                // Gateway-ish transient statuses are retriable by callers.
                502 | 503 | 504 => FetchError::Http(format!("server busy: HTTP {code}")),
                _ => FetchError::Protocol(format!("TAP returned HTTP {code}")),
            });
        }

        let tmp_file = std::fs::File::create(dest)
            .map_err(|e| FetchError::Protocol(format!("temp file: {e}")))?;
        let mut writer = std::io::BufWriter::with_capacity(256 * 1024, tmp_file);

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();

        let mut rows = 0u64;
        let mut saw_header = false;
        let mut buffer = [0u8; 64 * 1024];
        let mut reader = response;
        {
            use std::io::Read;
            // Use a line-oriented protocol on top of the streaming body:
            // first non-empty CSV line is the header, then count data lines.
            loop {
                let n = match reader.read(&mut buffer) {
                    Ok(n) => n,
                    Err(e) => return Err(FetchError::Http(e.to_string())),
                };
                if n == 0 {
                    break;
                }
                writer
                    .write_all(&buffer[..n])
                    .map_err(|e| FetchError::Protocol(format!("disk write: {e}")))?;
                // Count newlines: every data row contains exactly one '\n'
                // after its header line; final row may lack trailing newline
                // which slightly undercounts -- acceptable because row-limit
                // detection uses a TOP budge of N+1 anyway.
                for &b in &buffer[..n] {
                    if b == b'\n' {
                        if !saw_header {
                            saw_header = true;
                        } else {
                            rows += 1;
                        }
                    }
                }
            }
        }
        writer
            .flush()
            .map_err(|e| FetchError::Protocol(format!("flush: {e}")))?;
        drop(writer);
        let _ = content_type;

        // An empty payload (no header) means the archive answered something
        // that is not CSV at all -- treat it as a protocol failure.
        if !saw_header && rows == 0 {
            return Err(FetchError::Protocol("empty TAP response".into()));
        }
        Ok(rows)
    }
}
