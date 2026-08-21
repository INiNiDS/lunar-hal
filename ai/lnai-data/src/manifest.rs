use lunar_utils::time::current_time_ms;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Version of the manifest structure itself.
/// Tracks changes to how metadata is stored, independent of the data schema.
pub const MANIFEST_SCHEMA_VERSION: &str = "1.0.0";

/// Top-level manifest tracking the state of the entire dataset collection process.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DatasetManifestV1 {
    /// Version of this manifest structure
    pub version: String,
    /// Hash of the canonical schema (from schema.rs)
    pub schema_hash: String,
    /// Hash of the query parameters used for collection (e.g., ADQL query string)
    pub query_hash: String,
    /// Data source release identifier (e.g., "Gaia DR3")
    pub source_release: String,
    /// Total rows across all verified shards
    pub total_rows: u64,
    /// Global SHA-256 checksum of the final assembled dataset
    pub checksum: String,
    /// State of individual RA/Dec shards
    pub shards: Vec<ShardState>,
    /// Manifest creation timestamp (ms since UNIX_EPOCH)
    pub created_ms: u64,
    /// Last modification timestamp (ms since UNIX_EPOCH)
    pub updated_ms: u64,
}

impl DatasetManifestV1 {
    pub fn new(source_release: &str, query_hash: &str, schema_hash: &str) -> Self {
        let now_ms = current_time_ms();
        Self {
            version: MANIFEST_SCHEMA_VERSION.to_string(),
            schema_hash: schema_hash.to_string(),
            query_hash: query_hash.to_string(),
            source_release: source_release.to_string(),
            total_rows: 0,
            checksum: String::new(), // Empty until fully assembled
            shards: Vec::new(),
            created_ms: now_ms,
            updated_ms: now_ms,
        }
    }

    /// Updates the total row count and global checksum, typically called after
    /// all shards have successfully reached the `Verified` state.
    pub fn finalize(&mut self, global_checksum: &str) {
        self.total_rows = self.shards.iter().map(|s| s.row_count).sum();
        self.checksum = global_checksum.to_string();
        self.updated_ms = current_time_ms();
    }
}

/// State and metadata for an individual spatial shard (e.g., a specific RA/Dec tile).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ShardState {
    /// Unique identifier for the shard (e.g., "ra_000_010_dec_-90_-80")
    pub shard_id: String,
    /// Right Ascension range [start, end) in degrees
    pub ra_range: (f32, f32),
    /// Declination range [start, end) in degrees
    pub dec_range: (f32, f32),
    /// Number of rows downloaded/verified in this shard
    pub row_count: u64,
    /// SHA-256 checksum of the shard's raw data file
    pub checksum: String,
    /// Current state in the download/processing state machine
    pub status: ShardStatus,
    /// Number of failed attempts for resume/recovery logic
    pub retries: u32,
    /// Timestamp of the last state change (ms since UNIX_EPOCH)
    pub last_attempt_ms: Option<u64>,
    /// Bytes downloaded so far (for resume after interrupted writes)
    pub bytes_downloaded: u64,
}

impl ShardState {
    pub fn new(shard_id: String, ra_range: (f32, f32), dec_range: (f32, f32)) -> Self {
        Self {
            shard_id,
            ra_range,
            dec_range,
            row_count: 0,
            checksum: String::new(),
            status: ShardStatus::Pending,
            retries: 0,
            last_attempt_ms: None,
            bytes_downloaded: 0,
        }
    }

    /// Attempts to transition the shard to a new state.
    /// Enforces the state machine rules. Returns an error on invalid transitions.
    pub fn transition_to(&mut self, new_status: ShardStatus) -> Result<(), String> {
        if !self.status.can_transition_to(&new_status) {
            return Err(format!(
                "Invalid shard state transition: {:?} -> {:?} for shard {}",
                self.status, new_status, self.shard_id
            ));
        }

        // Increment retries if entering a Failed state
        if matches!(new_status, ShardStatus::Failed) {
            self.retries += 1;
        }

        // Update timestamp if entering an active state
        if matches!(
            new_status,
            ShardStatus::Downloading | ShardStatus::Verifying
        ) {
            self.last_attempt_ms = Some(current_time_ms());
        }

        self.status = new_status;
        Ok(())
    }
}

/// State machine for shard collection and processing.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ShardStatus {
    /// Initial state, waiting to be picked up by a worker
    Pending,
    /// Actively downloading data
    Downloading,
    /// Download finished, waiting for checksum verification
    Downloaded,
    /// Actively verifying checksum and row count
    Verifying,
    /// Successfully verified and ready for assembly
    Verified,
    /// Failed (network error, checksum mismatch, etc.), eligible for retry
    Failed,
}

impl ShardStatus {
    /// Defines the valid state transitions for the state machine.
    ///
    /// Allowed transitions:
    /// Pending -> Downloading
    /// Downloading -> Downloaded | Failed
    /// Downloaded -> Verifying | Failed
    /// Verifying -> Verified | Failed
    /// Failed -> Pending (retry logic)
    /// Verified -> Verifying (re-verification if needed, though rare)
    fn can_transition_to(&self, new_status: &ShardStatus) -> bool {
        use ShardStatus::*;
        match (self, new_status) {
            (Pending, Downloading) => true,
            (Downloading, Downloaded) => true,
            (Downloading, Failed) => true,
            (Downloaded, Verifying) => true,
            (Downloaded, Failed) => true,
            (Verifying, Verified) => true,
            (Verifying, Failed) => true,
            (Failed, Pending) => true,
            (Verified, Verifying) => true,
            _ => false,
        }
    }
}

impl fmt::Display for ShardStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardStatus::Pending => write!(f, "pending"),
            ShardStatus::Downloading => write!(f, "downloading"),
            ShardStatus::Downloaded => write!(f, "downloaded"),
            ShardStatus::Verifying => write!(f, "verifying"),
            ShardStatus::Verified => write!(f, "verified"),
            ShardStatus::Failed => write!(f, "failed"),
        }
    }
}

/// Checksum policy for shards and the final dataset.
/// Currently enforced as SHA-256.
pub const CHECKSUM_ALGORITHM: &str = "sha256";

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> DatasetManifestV1 {
        let mut manifest = DatasetManifestV1::new("gaia_dr3", "query-hash", "schema-hash");
        let mut shard =
            ShardState::new("ra_000_010_dec_-90_-80".into(), (0.0, 10.0), (-90.0, -80.0));
        shard.transition_to(ShardStatus::Downloading).unwrap();
        shard.transition_to(ShardStatus::Downloaded).unwrap();
        shard.transition_to(ShardStatus::Verifying).unwrap();
        shard.transition_to(ShardStatus::Verified).unwrap();
        shard.row_count = 42;
        shard.checksum = "abc123".into();
        manifest.shards.push(shard);
        manifest.finalize("global-checksum");
        manifest
    }

    #[test]
    fn dataset_manifest_v1_round_trips_through_json() {
        let manifest = sample_manifest();
        let json = serde_json::to_string(&manifest).expect("serialize manifest");
        let back: DatasetManifestV1 = serde_json::from_str(&json).expect("deserialize manifest");
        assert_eq!(back, manifest);
    }

    #[test]
    fn manifest_version_and_checksum_policy_are_frozen() {
        assert_eq!(MANIFEST_SCHEMA_VERSION, "1.0.0");
        assert_eq!(CHECKSUM_ALGORITHM, "sha256");
    }

    #[test]
    fn finalize_aggregates_rows_and_checksum() {
        let mut manifest = sample_manifest();
        let mut second = ShardState::new(
            "ra_010_020_dec_-90_-80".into(),
            (10.0, 20.0),
            (-90.0, -80.0),
        );
        second.row_count = 8;
        manifest.shards.push(second);
        manifest.finalize("global-2");
        assert_eq!(manifest.total_rows, 50);
        assert_eq!(manifest.checksum, "global-2");
    }

    #[test]
    fn shard_state_machine_accepts_only_contracted_transitions() {
        let valid: &[(ShardStatus, ShardStatus)] = &[
            (ShardStatus::Pending, ShardStatus::Downloading),
            (ShardStatus::Downloading, ShardStatus::Downloaded),
            (ShardStatus::Downloading, ShardStatus::Failed),
            (ShardStatus::Downloaded, ShardStatus::Verifying),
            (ShardStatus::Downloaded, ShardStatus::Failed),
            (ShardStatus::Verifying, ShardStatus::Verified),
            (ShardStatus::Verifying, ShardStatus::Failed),
            (ShardStatus::Failed, ShardStatus::Pending),
            (ShardStatus::Verified, ShardStatus::Verifying),
        ];
        for (from, to) in valid {
            let mut shard = ShardState::new("s".into(), (0.0, 1.0), (0.0, 1.0));
            shard.status = *from;
            assert!(
                shard.transition_to(*to).is_ok(),
                "{from:?} -> {to:?} must be allowed"
            );
        }
    }

    #[test]
    fn shard_state_machine_rejects_shortcuts_and_counts_retries() {
        let mut shard = ShardState::new("s".into(), (0.0, 1.0), (0.0, 1.0));
        assert!(shard.transition_to(ShardStatus::Verified).is_err());
        assert_eq!(shard.status, ShardStatus::Pending);

        shard.transition_to(ShardStatus::Downloading).unwrap();
        shard.transition_to(ShardStatus::Failed).unwrap();
        assert_eq!(shard.retries, 1);
        assert!(shard.last_attempt_ms.is_some());
        shard.transition_to(ShardStatus::Pending).unwrap();
        assert!(shard.transition_to(ShardStatus::Pending).is_err());
    }
}
