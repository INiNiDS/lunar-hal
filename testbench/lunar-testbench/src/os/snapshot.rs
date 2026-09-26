//! Window snapshot envelope contract v1 (Stage 2 & Stage 11).
#![allow(dead_code)]

use lunar_utils::time::current_time_ms;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Version of the snapshot envelope structure.
pub const SNAPSHOT_SCHEMA_VERSION: &str = "1.0.0";

/// Window position and size.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Copy)]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// Computes the IEEE 802.3 CRC32 checksum for arbitrary byte slices.
pub fn calculate_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Versioned snapshot envelope for Lunar-OS window state (Stage 11).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AppSnapshotEnvelopeV1 {
    pub version: u32,
    pub instance_id: String,
    pub app_id: String,
    pub window_geometry: WindowGeometry,
    pub captured_at_ms: u64,
    pub checksum_crc32: u32,
    pub payload: serde_json::Value,
}

impl AppSnapshotEnvelopeV1 {
    pub fn new(
        instance_id: &str,
        app_id: &str,
        window_geometry: WindowGeometry,
        payload: serde_json::Value,
    ) -> Self {
        let serialized = payload.to_string();
        let checksum_crc32 = calculate_crc32(serialized.as_bytes());
        Self {
            version: 1,
            instance_id: instance_id.to_string(),
            app_id: app_id.to_string(),
            window_geometry,
            captured_at_ms: current_time_ms(),
            checksum_crc32,
            payload,
        }
    }

    /// Verifies snapshot integrity by validating schema version and CRC32 over the payload.
    pub fn verify_integrity(&self) -> Result<(), SnapshotError> {
        if self.version != 1 {
            return Err(SnapshotError::VersionMismatch);
        }
        self.verify_checksum()
    }

    /// Verifies snapshot integrity by recomputing CRC32 over the payload.
    pub fn verify_checksum(&self) -> Result<(), SnapshotError> {
        let serialized = self.payload.to_string();
        let expected = calculate_crc32(serialized.as_bytes());
        if self.checksum_crc32 == expected {
            Ok(())
        } else {
            Err(SnapshotError::ChecksumMismatch)
        }
    }
}

/// Serializable state of an application window, used for minimize/restore lifecycle (Legacy compatibility).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WindowSnapshotV1 {
    pub version: String,
    pub app_id: String,
    pub window_geometry: WindowGeometry,
    pub app_state: serde_json::Value,
    pub checksum: String,
    pub captured_ms: u64,
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

    pub fn verify_integrity(&self) -> Result<(), SnapshotError> {
        if self.version != SNAPSHOT_SCHEMA_VERSION && self.version != "1" && self.version != "1.0.0"
        {
            return Err(SnapshotError::VersionMismatch);
        }
        self.verify_checksum()
    }

    pub fn verify_checksum(&self) -> Result<(), SnapshotError> {
        if calculate_checksum(&self.app_state) == self.checksum {
            Ok(())
        } else {
            Err(SnapshotError::ChecksumMismatch)
        }
    }
}

impl From<AppSnapshotEnvelopeV1> for WindowSnapshotV1 {
    fn from(env: AppSnapshotEnvelopeV1) -> Self {
        let checksum = calculate_checksum(&env.payload);
        Self {
            version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            app_id: env.app_id,
            window_geometry: env.window_geometry,
            app_state: env.payload,
            checksum,
            captured_ms: env.captured_at_ms,
        }
    }
}

impl From<WindowSnapshotV1> for AppSnapshotEnvelopeV1 {
    fn from(snap: WindowSnapshotV1) -> Self {
        AppSnapshotEnvelopeV1::new(
            &snap.app_id,
            &snap.app_id,
            snap.window_geometry,
            snap.app_state,
        )
    }
}

/// Trait that every Lunar-OS application component must implement to support state hydration.
pub trait AppSnapshot: Send + Sync {
    /// Capture the current component state into a serializable JSON value.
    fn capture_snapshot(&self) -> serde_json::Value;

    /// Hydrate the component state from a snapshot payload.
    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String>;
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

    fn sample_envelope() -> AppSnapshotEnvelopeV1 {
        AppSnapshotEnvelopeV1::new(
            "inst-1",
            "datasets",
            WindowGeometry {
                x: 10,
                y: 20,
                width: 960,
                height: 640,
            },
            serde_json::json!({ "dir": "data/canonical-v1", "tab": "coverage" }),
        )
    }

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
    fn app_snapshot_envelope_crc32_and_round_trip() {
        let env = sample_envelope();
        let json = serde_json::to_string(&env).expect("serialize");
        let restored: AppSnapshotEnvelopeV1 = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, env);
        assert_eq!(env.verify_checksum(), Ok(()));
    }

    #[test]
    fn corrupted_envelope_payload_fails_crc32() {
        let mut env = sample_envelope();
        env.payload["dir"] = serde_json::json!("data/tampered");
        assert_eq!(env.verify_checksum(), Err(SnapshotError::ChecksumMismatch));
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

    #[test]
    fn envelope_schema_version_mismatch_rejected() {
        let mut env = sample_envelope();
        assert_eq!(env.verify_integrity(), Ok(()));
        env.version = 999;
        assert_eq!(env.verify_integrity(), Err(SnapshotError::VersionMismatch));
    }

    #[test]
    fn legacy_snapshot_schema_version_mismatch_rejected() {
        let mut snap = sample_snapshot();
        assert_eq!(snap.verify_integrity(), Ok(()));
        snap.version = "2.0.0".to_string();
        assert_eq!(snap.verify_integrity(), Err(SnapshotError::VersionMismatch));
    }
}
