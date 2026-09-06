//! Stage 4A / пункт 6: per-source manifest with verifiable retrieval identity.
//!
//! Every raw recorded payload gets a `SourceManifestEntry`: the frozen query
//! hash, retrieval timestamp, uncompressed payload SHA-256 and row count.
//! This is the machine-readable half of "versioned subset" from the exit
//! gate; secret values can never appear here because entries only carry
//! names of adapters plus content hashes.

use crate::integrity::sha256_hex;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SourceManifestEntry {
    pub adapter_id: String,
    pub endpoint_url: String,
    /// Deterministic hash over the frozen request (query/body), not the URL.
    pub query_hash: String,
    /// ms since UNIX epoch of the real retrieval session that produced
    /// `payload_sha256`; frozen fixtures keep their original value.
    pub retrieved_ms: u64,
    pub payload_sha256: String,
    pub payload_bytes: u64,
    pub row_count: usize,
}

impl SourceManifestEntry {
    pub fn from_payload(
        adapter_id: &str,
        endpoint_url: &str,
        query_hash: &str,
        retrieved_ms: u64,
        payload: &str,
        row_count: usize,
    ) -> Self {
        Self {
            adapter_id: adapter_id.to_string(),
            endpoint_url: endpoint_url.to_string(),
            query_hash: query_hash.to_string(),
            retrieved_ms,
            payload_sha256: sha256_hex(payload.as_bytes()),
            payload_bytes: payload.len() as u64,
            row_count,
        }
    }
}

/// Top-level list, serialized as `source_manifest.json` next to the dataset.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SourceManifestV1 {
    pub version: String,
    pub entries: Vec<SourceManifestEntry>,
}

impl Default for SourceManifestV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceManifestV1 {
    pub const VERSION: &'static str = "1.0.0";

    pub fn new() -> Self {
        Self {
            version: Self::VERSION.into(),
            entries: Vec::new(),
        }
    }

    pub fn register(&mut self, entry: SourceManifestEntry) {
        if !self
            .entries
            .iter()
            .any(|e| e.payload_sha256 == entry.payload_sha256)
        {
            self.entries.push(entry);
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn load(path: &std::path::Path) -> Option<Self> {
        let raw = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&raw).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_is_idempotent_by_payload_hash_and_stable_on_save_load() {
        let mut m = SourceManifestV1::new();
        let make = || {
            SourceManifestEntry::from_payload(
                "nasa_exoplanet_tap_pscomppars_v1",
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                "qh",
                1_787_600_000_000,
                "col1,col2\n1,2\n",
                1,
            )
        };
        let first = make();
        m.register(make());
        m.register(make());
        assert_eq!(m.entries.len(), 1);

        // Byte changes must alter the payload hash (detectability contract).
        let mut mutated = make();
        mutated.payload_sha256 = "different".into();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source_manifest.json");
        m.save(&path).unwrap();
        let loaded = SourceManifestV1::load(&path).unwrap();
        assert_eq!(loaded, m);
        assert_eq!(first.row_count, 1);
    }
}
