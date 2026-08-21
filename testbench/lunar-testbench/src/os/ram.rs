//! RAM-registry contract v1 for Lunar-OS window lifecycle (Stage 2).
//! Consumed by the window manager when minimize/restore lands (Stage 12).
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fmt;

use super::snapshot::WindowSnapshotV1;

/// An entry in the Lunar-OS RAM registry, tracking the lifecycle of an app instance.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LunarOsRamEntryV1 {
    /// Unique identifier for this specific window instance
    pub instance_id: String,
    /// The application ID (e.g., "files")
    pub app_id: String,
    /// Current lifecycle state
    pub state: RamEntryState,
    /// Generation counter to prevent duplicate restore attempts
    pub generation: u64,
    /// Snapshot data, present only when state is Minimized
    pub snapshot: Option<WindowSnapshotV1>,
}

impl LunarOsRamEntryV1 {
    pub fn new(instance_id: String, app_id: String) -> Self {
        Self {
            instance_id,
            app_id,
            state: RamEntryState::Live,
            generation: 0,
            snapshot: None,
        }
    }

    /// Transition to Minimized state, capturing a snapshot.
    pub fn minimize(&mut self, snapshot: WindowSnapshotV1) -> Result<(), String> {
        if !matches!(self.state, RamEntryState::Live) {
            return Err(format!(
                "Cannot minimize from state {:?} (instance: {})",
                self.state, self.instance_id
            ));
        }
        self.snapshot = Some(snapshot);
        self.state = RamEntryState::Minimized;
        self.generation += 1;
        Ok(())
    }

    /// Transition back to Live state, consuming the snapshot.
    pub fn restore(&mut self, expected_generation: u64) -> Result<WindowSnapshotV1, String> {
        if !matches!(self.state, RamEntryState::Minimized) {
            return Err(format!(
                "Cannot restore from state {:?} (instance: {})",
                self.state, self.instance_id
            ));
        }
        if self.generation != expected_generation {
            return Err(format!(
                "Generation mismatch for instance {}: expected {}, got {}",
                self.instance_id, expected_generation, self.generation
            ));
        }

        self.state = RamEntryState::Live;
        Ok(self.snapshot.take().expect("Snapshot must exist in Minimized state"))
    }

    /// Transition to Closed state. Idempotent.
    pub fn close(&mut self) {
        self.state = RamEntryState::Closed;
        self.snapshot = None; // Discard snapshot
    }
}

/// Lifecycle states for a Lunar-OS window instance.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RamEntryState {
    /// Window is mounted and actively rendering
    Live,
    /// Window is unmounted, state is persisted in RAM
    Minimized,
    /// Window has been closed, entry is kept for history/cleanup
    Closed,
}

impl fmt::Display for RamEntryState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RamEntryState::Live => write!(f, "live"),
            RamEntryState::Minimized => write!(f, "minimized"),
            RamEntryState::Closed => write!(f, "closed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::snapshot::{WindowGeometry, WindowSnapshotV1};

    fn snapshot() -> WindowSnapshotV1 {
        WindowSnapshotV1::new(
            "terminal",
            WindowGeometry {
                x: 10,
                y: 20,
                width: 640,
                height: 480,
            },
            serde_json::json!({ "scrollback": ["$ ls"] }),
        )
    }

    #[test]
    fn minimize_restore_round_trip_preserves_snapshot() {
        let mut entry = LunarOsRamEntryV1::new("inst-1".into(), "terminal".into());
        assert_eq!(entry.state, RamEntryState::Live);
        entry.minimize(snapshot()).expect("minimize from Live");
        assert_eq!(entry.state, RamEntryState::Minimized);
        assert_eq!(entry.generation, 1);

        let restored = entry.restore(1).expect("restore with fresh generation");
        assert_eq!(restored.app_id, "terminal");
        assert_eq!(restored.window_geometry.width, 640);
        assert_eq!(entry.state, RamEntryState::Live);
        assert!(entry.snapshot.is_none(), "snapshot must be consumed");
    }

    #[test]
    fn lifecycle_rejects_invalid_transitions() {
        let mut entry = LunarOsRamEntryV1::new("inst-2".into(), "files".into());
        assert!(
            entry.restore(0).is_err(),
            "restore is only valid from Minimized"
        );
        entry.minimize(snapshot()).unwrap();
        assert!(
            entry.minimize(snapshot()).is_err(),
            "double minimize must fail"
        );
    }

    #[test]
    fn stale_generation_cannot_restore() {
        let mut entry = LunarOsRamEntryV1::new("inst-3".into(), "models".into());
        entry.minimize(snapshot()).unwrap();
        assert!(entry.restore(0).is_err(), "stale generation must be rejected");
        assert_eq!(entry.state, RamEntryState::Minimized);
        assert!(entry.snapshot.is_some());
    }

    #[test]
    fn close_is_idempotent_and_discards_snapshot() {
        let mut entry = LunarOsRamEntryV1::new("inst-4".into(), "pipeline".into());
        entry.close();
        entry.close();
        assert_eq!(entry.state, RamEntryState::Closed);

        let mut live = LunarOsRamEntryV1::new("inst-5".into(), "pipeline".into());
        live.minimize(snapshot()).unwrap();
        live.close();
        assert_eq!(live.state, RamEntryState::Closed);
        assert!(live.snapshot.is_none());
    }

    #[test]
    fn ram_entry_round_trips_through_json() {
        let mut entry = LunarOsRamEntryV1::new("inst-6".into(), "sandbox".into());
        entry.minimize(snapshot()).unwrap();

        let json = serde_json::to_string(&entry).expect("serialize RAM entry");
        let back: LunarOsRamEntryV1 = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, entry);

        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["state"], "minimized");
    }
}
