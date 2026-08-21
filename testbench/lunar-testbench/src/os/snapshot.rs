//! Window snapshot envelope contract v1 (Stage 2). Apps adopt `AppSnapshot`
//! during the Stage 12 snapshot/restore migration.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use lunar_utils::time::current_time_ms;

/// Version of the snapshot envelope structure.
pub const SNAPSHOT_SCHEMA_VERSION: &str = "1.0.0";

/// Serializable state of an application window, used for minimize/restore lifecycle.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WindowSnapshotV1 {
    /// Snapshot schema version
    pub version: String,
    /// Application identifier (e.g., "terminal", "files")
    pub app_id: String,
    /// Window position and size
    pub window_geometry: WindowGeometry,
    /// Application-specific serialized state (JSON)
    pub app_state: serde_json::Value,
    /// SHA-256 checksum of the `app_state` to detect corruption
    pub checksum: String,
    /// Timestamp when the snapshot was taken (ms since UNIX_EPOCH)
    pub captured_ms: u64,
}

/// Window position and size.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy)]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl WindowSnapshotV1 {
    pub fn new(app_id: &str, geometry: WindowGeometry, app_state: serde_json::Value) -> Self {
        let checksum = calculate_checksum(&app_state);
        Self {
            version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            app_id: app_id.to_string(),
            window_geometry: geometry,
            app_state,
            checksum,
            captured_ms: current_time_ms(),
        }
    }

    /// Verifies snapshot integrity by recomputing the `app_state` checksum.
    pub fn verify_checksum(&self) -> Result<(), SnapshotError> {
        if calculate_checksum(&self.app_state) == self.checksum {
            Ok(())
        } else {
            Err(SnapshotError::ChecksumMismatch)
        }
    }
}

/// Trait that every Lunar-OS application must implement to support minimize/restore.
pub trait AppSnapshot: Send + Sync {
    /// Capture the current state into a serializable JSON value.
    fn capture_state(&self) -> serde_json::Value;

    /// Hydrate the application state from a snapshot.
    /// This is called *before* the first render.
    fn hydrate_state(&mut self, state: serde_json::Value) -> Result<(), SnapshotError>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotError {
    ChecksumMismatch,
    VersionMismatch,
    DeserializationFailed(String),
}

fn calculate_checksum(state: &serde_json::Value) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(state.to_string().as_bytes());
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> WindowSnapshotV1 {
        WindowSnapshotV1::new(
            "terminal",
            WindowGeometry {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            },
            serde_json::json!({ "cwd": "/home/star", "history": ["ls", "cargo test"] }),
        )
    }

    #[test]
    fn snapshot_round_trips_through_json() {
        let snap = sample_snapshot();
        let json = serde_json::to_string(&snap).expect("serialize snapshot");
        let back: WindowSnapshotV1 = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, snap);
    }

    #[test]
    fn checksum_is_sha256_hex_and_verifies() {
        let snap = sample_snapshot();
        assert_eq!(snap.checksum.len(), 64);
        assert!(snap.checksum.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(snap.verify_checksum(), Ok(()));
    }

    #[test]
    fn corrupted_app_state_is_detected() {
        let mut snap = sample_snapshot();
        snap.app_state["cwd"] = serde_json::Value::String("/elsewhere".into());
        assert_eq!(snap.verify_checksum(), Err(SnapshotError::ChecksumMismatch));
    }

    #[test]
    fn checksum_is_deterministic_for_equal_state() {
        let a = sample_snapshot();
        let b = sample_snapshot();
        assert_eq!(a.checksum, b.checksum);
    }

    #[test]
    fn snapshot_version_is_frozen() {
        assert_eq!(SNAPSHOT_SCHEMA_VERSION, "1.0.0");
        assert_eq!(sample_snapshot().version, "1.0.0");
    }
}
