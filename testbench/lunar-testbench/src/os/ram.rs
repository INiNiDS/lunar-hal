//! RAM-registry contract v1 for Lunar-OS window lifecycle (Stage 2 & Stage 11).
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::snapshot::{AppSnapshotEnvelopeV1, WindowSnapshotV1};

/// Lifecycle states for an application instance in Lunar-OS RAM.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RamLifecycleState {
    /// Window is mounted and actively rendering
    Active,
    /// Window is freezing state and computing snapshot
    Minimizing,
    /// Window is unmounted from DOM; snapshot is stored in RAM
    Minimized,
    /// Window is instantiating and hydrating state
    Restoring,
    /// Window is closed and entry is discarded
    Closed,
}

impl fmt::Display for RamLifecycleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RamLifecycleState::Active => write!(f, "active"),
            RamLifecycleState::Minimizing => write!(f, "minimizing"),
            RamLifecycleState::Minimized => write!(f, "minimized"),
            RamLifecycleState::Restoring => write!(f, "restoring"),
            RamLifecycleState::Closed => write!(f, "closed"),
        }
    }
}

/// An entry in the Lunar-OS RAM store tracking a single window instance lifecycle.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LunarOsRamEntry {
    pub instance_id: String,
    pub app_id: String,
    pub title: String,
    pub state: RamLifecycleState,
    pub restore_generation: u64,
    pub snapshot: Option<AppSnapshotEnvelopeV1>,
}

impl LunarOsRamEntry {
    pub fn new(instance_id: String, app_id: String, title: String) -> Self {
        Self {
            instance_id,
            app_id,
            title,
            state: RamLifecycleState::Active,
            restore_generation: 0,
            snapshot: None,
        }
    }

    /// Transitions to Minimizing state.
    pub fn begin_minimize(&mut self) -> Result<(), String> {
        if self.state != RamLifecycleState::Active {
            return Err(format!(
                "Cannot begin minimize from state {:?} (instance: {})",
                self.state, self.instance_id
            ));
        }
        self.state = RamLifecycleState::Minimizing;
        Ok(())
    }

    /// Verifies snapshot checksum and commits to Minimized state.
    pub fn commit_minimize(&mut self, snapshot: AppSnapshotEnvelopeV1) -> Result<(), String> {
        if self.state != RamLifecycleState::Minimizing {
            return Err(format!(
                "Cannot commit minimize from state {:?} (instance: {})",
                self.state, self.instance_id
            ));
        }
        snapshot
            .verify_integrity()
            .map_err(|e| format!("Snapshot integrity failed: {e:?}"))?;
        self.snapshot = Some(snapshot);
        self.state = RamLifecycleState::Minimized;
        self.restore_generation += 1;
        Ok(())
    }

    /// Rolls back minimize transition if snapshot failed.
    pub fn rollback_minimize(&mut self) {
        if self.state == RamLifecycleState::Minimizing {
            self.state = RamLifecycleState::Active;
        }
    }

    /// Transitions to Restoring state, claiming the current generation.
    pub fn begin_restore(&mut self) -> Result<u64, String> {
        if self.state != RamLifecycleState::Minimized {
            return Err(format!(
                "Cannot restore from state {:?} (instance: {})",
                self.state, self.instance_id
            ));
        }
        self.state = RamLifecycleState::Restoring;
        Ok(self.restore_generation)
    }

    /// Commits restore, ensuring generation is fresh, and returns snapshot for hydration.
    pub fn commit_restore(
        &mut self,
        expected_generation: u64,
    ) -> Result<AppSnapshotEnvelopeV1, String> {
        if self.state != RamLifecycleState::Restoring {
            return Err(format!(
                "Cannot commit restore from state {:?} (instance: {})",
                self.state, self.instance_id
            ));
        }
        if self.restore_generation != expected_generation {
            return Err(format!(
                "Stale generation for instance {}: expected {}, current {}",
                self.instance_id, expected_generation, self.restore_generation
            ));
        }
        let integrity = self
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.verify_integrity());
        match integrity {
            None => {
                self.state = RamLifecycleState::Minimized;
                return Err("Snapshot missing in restoring state".to_string());
            }
            Some(Err(error)) => {
                self.state = RamLifecycleState::Minimized;
                return Err(format!("Snapshot integrity failed: {error:?}"));
            }
            Some(Ok(())) => {}
        }
        let snap = self
            .snapshot
            .take()
            .expect("verified snapshot must still be present");
        self.state = RamLifecycleState::Active;
        Ok(snap)
    }

    pub fn rollback_restore(&mut self) {
        if self.state == RamLifecycleState::Restoring {
            self.state = RamLifecycleState::Minimized;
        }
    }

    /// Transitions to Closed and clears any snapshot data.
    pub fn close(&mut self) {
        self.state = RamLifecycleState::Closed;
        self.snapshot = None;
    }
}

/// The isolated RAM store maintaining all window instance lifecycle transactions.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct LunarOsRamStore {
    entries: HashMap<String, LunarOsRamEntry>,
}

impl LunarOsRamStore {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn register_active(&mut self, instance_id: &str, app_id: &str, title: &str) {
        let entry = self
            .entries
            .entry(instance_id.to_string())
            .or_insert_with(|| {
                LunarOsRamEntry::new(
                    instance_id.to_string(),
                    app_id.to_string(),
                    title.to_string(),
                )
            });
        if entry.state == RamLifecycleState::Closed {
            entry.state = RamLifecycleState::Active;
        }
    }

    pub fn begin_minimize(&mut self, instance_id: &str) -> Result<(), String> {
        let entry = self
            .entries
            .get_mut(instance_id)
            .ok_or_else(|| format!("Unknown instance {instance_id}"))?;
        entry.begin_minimize()
    }

    pub fn commit_minimize(
        &mut self,
        instance_id: &str,
        snapshot: AppSnapshotEnvelopeV1,
    ) -> Result<(), String> {
        let entry = self
            .entries
            .get_mut(instance_id)
            .ok_or_else(|| format!("Unknown instance {instance_id}"))?;
        entry.commit_minimize(snapshot)
    }

    pub fn rollback_minimize(&mut self, instance_id: &str) {
        if let Some(entry) = self.entries.get_mut(instance_id) {
            entry.rollback_minimize();
        }
    }

    pub fn begin_restore(&mut self, instance_id: &str) -> Result<u64, String> {
        let entry = self
            .entries
            .get_mut(instance_id)
            .ok_or_else(|| format!("Unknown instance {instance_id}"))?;
        entry.begin_restore()
    }

    pub fn commit_restore(
        &mut self,
        instance_id: &str,
        expected_generation: u64,
    ) -> Result<AppSnapshotEnvelopeV1, String> {
        let entry = self
            .entries
            .get_mut(instance_id)
            .ok_or_else(|| format!("Unknown instance {instance_id}"))?;
        entry.commit_restore(expected_generation)
    }

    pub fn rollback_restore(&mut self, instance_id: &str) {
        if let Some(entry) = self.entries.get_mut(instance_id) {
            entry.rollback_restore();
        }
    }

    pub fn close_instance(&mut self, instance_id: &str) {
        if let Some(entry) = self.entries.get_mut(instance_id) {
            entry.close();
        }
        self.entries.remove(instance_id);
    }

    pub fn get_entry(&self, instance_id: &str) -> Option<&LunarOsRamEntry> {
        self.entries.get(instance_id)
    }

    pub fn find_by_app(&self, app_id: &str) -> Option<&LunarOsRamEntry> {
        self.entries
            .values()
            .find(|e| e.app_id == app_id && e.state != RamLifecycleState::Closed)
    }

    pub fn find_minimized_by_app(&self, app_id: &str) -> Option<&LunarOsRamEntry> {
        self.entries
            .values()
            .find(|e| e.app_id == app_id && e.state == RamLifecycleState::Minimized)
    }

    pub fn find_by_app_mut(&mut self, app_id: &str) -> Option<&mut LunarOsRamEntry> {
        self.entries
            .values_mut()
            .find(|e| e.app_id == app_id && e.state != RamLifecycleState::Closed)
    }

    pub fn list_entries(&self) -> Vec<&LunarOsRamEntry> {
        self.entries
            .values()
            .filter(|e| e.state != RamLifecycleState::Closed)
            .collect()
    }

    pub fn is_app_minimized(&self, app_id: &str) -> bool {
        self.find_minimized_by_app(app_id).is_some()
    }

    pub fn is_instance_minimized(&self, instance_id: &str) -> bool {
        self.entries
            .get(instance_id)
            .is_some_and(|e| e.state == RamLifecycleState::Minimized)
    }

    pub fn is_instance_open(&self, instance_id: &str) -> bool {
        self.entries
            .get(instance_id)
            .is_some_and(|e| e.state != RamLifecycleState::Closed)
    }

    pub fn is_app_open(&self, app_id: &str) -> bool {
        self.find_by_app(app_id).is_some()
    }
}

// ---------------- Legacy compatibility adapter ----------------

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RamEntryState {
    Live,
    Minimized,
    Closed,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LunarOsRamEntryV1 {
    pub instance_id: String,
    pub app_id: String,
    pub state: RamEntryState,
    pub generation: u64,
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

    pub fn minimize(&mut self, snapshot: WindowSnapshotV1) -> Result<(), String> {
        if !matches!(self.state, RamEntryState::Live) {
            return Err(format!(
                "Cannot minimize from state {:?} (instance: {})",
                self.state, self.instance_id
            ));
        }
        snapshot
            .verify_integrity()
            .map_err(|e| format!("Snapshot integrity failed: {e:?}"))?;
        self.snapshot = Some(snapshot);
        self.state = RamEntryState::Minimized;
        self.generation += 1;
        Ok(())
    }

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
        Ok(self
            .snapshot
            .take()
            .expect("Snapshot must exist in Minimized state"))
    }

    pub fn close(&mut self) {
        self.state = RamEntryState::Closed;
        self.snapshot = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::snapshot::WindowGeometry;

    fn sample_envelope() -> AppSnapshotEnvelopeV1 {
        AppSnapshotEnvelopeV1::new(
            "inst-1",
            "datasets",
            WindowGeometry {
                x: 10,
                y: 20,
                width: 640,
                height: 480,
            },
            serde_json::json!({ "dir": "data/canonical-v1" }),
        )
    }

    #[test]
    fn ram_store_transaction_lifecycle() {
        let mut store = LunarOsRamStore::new();
        store.register_active("inst-1", "datasets", "Datasets");
        assert_eq!(store.is_app_minimized("datasets"), false);
        assert_eq!(store.is_app_open("datasets"), true);

        // Minimize transaction
        store.begin_minimize("inst-1").expect("begin minimize");
        let snap = sample_envelope();
        store
            .commit_minimize("inst-1", snap.clone())
            .expect("commit minimize");
        assert_eq!(store.is_app_minimized("datasets"), true);

        // Restore transaction
        let restore_gen = store.begin_restore("inst-1").expect("begin restore");
        let restored = store
            .commit_restore("inst-1", restore_gen)
            .expect("commit restore");
        assert_eq!(restored.app_id, "datasets");
        assert_eq!(store.is_app_minimized("datasets"), false);

        // Close
        store.close_instance("inst-1");
        assert_eq!(store.is_app_open("datasets"), false);
    }

    #[test]
    fn ram_store_rejects_stale_generation() {
        let mut store = LunarOsRamStore::new();
        store.register_active("inst-2", "datasets", "Datasets");
        store.begin_minimize("inst-2").unwrap();
        store.commit_minimize("inst-2", sample_envelope()).unwrap();

        let _gen = store.begin_restore("inst-2").unwrap();
        let res = store.commit_restore("inst-2", 999);
        assert!(res.is_err());
    }

    #[test]
    fn minimize_restore_round_trip_preserves_snapshot() {
        let mut entry = LunarOsRamEntryV1::new("inst-1".into(), "terminal".into());
        assert_eq!(entry.state, RamEntryState::Live);
        let snap = WindowSnapshotV1::new(
            "terminal",
            WindowGeometry {
                x: 10,
                y: 20,
                width: 640,
                height: 480,
            },
            serde_json::json!({ "scrollback": ["$ ls"] }),
        );
        entry.minimize(snap).expect("minimize from Live");
        assert_eq!(entry.state, RamEntryState::Minimized);
        assert_eq!(entry.generation, 1);

        let restored = entry.restore(1).expect("restore with fresh generation");
        assert_eq!(restored.app_id, "terminal");
        assert_eq!(restored.window_geometry.width, 640);
        assert_eq!(entry.state, RamEntryState::Live);
        assert!(entry.snapshot.is_none(), "snapshot must be consumed");
    }

    #[test]
    fn stale_generation_cannot_restore() {
        let mut entry = LunarOsRamEntryV1::new("inst-3".into(), "models".into());
        let snap = WindowSnapshotV1::new(
            "models",
            WindowGeometry {
                x: 10,
                y: 20,
                width: 640,
                height: 480,
            },
            serde_json::json!({}),
        );
        entry.minimize(snap).unwrap();
        assert!(
            entry.restore(0).is_err(),
            "stale generation must be rejected"
        );
        assert_eq!(entry.state, RamEntryState::Minimized);
        assert!(entry.snapshot.is_some());
    }

    #[test]
    fn checksum_failure_does_not_consume_minimized_snapshot() {
        let mut store = LunarOsRamStore::new();
        store.register_active("inst-1", "datasets", "Datasets");
        store.begin_minimize("inst-1").unwrap();
        store.commit_minimize("inst-1", sample_envelope()).unwrap();
        let entry = store.entries.get_mut("inst-1").unwrap();
        entry.snapshot.as_mut().unwrap().payload = serde_json::json!({"changed": true});
        let damaged = entry.snapshot.clone();

        let generation = store.begin_restore("inst-1").unwrap();
        assert!(store.commit_restore("inst-1", generation).is_err());
        let entry = store.get_entry("inst-1").unwrap();
        assert_eq!(entry.state, RamLifecycleState::Minimized);
        assert_eq!(entry.snapshot, damaged);
    }

    #[test]
    fn corrupted_crc32_snapshot_rejected_on_minimize() {
        let mut store = LunarOsRamStore::new();
        store.register_active("inst-corrupt", "datasets", "Datasets");
        store.begin_minimize("inst-corrupt").unwrap();
        let mut snap = sample_envelope();
        snap.checksum_crc32 = 0xdeadbeef;
        let res = store.commit_minimize("inst-corrupt", snap);
        assert!(res.is_err(), "corrupted CRC32 must be rejected");
        assert_eq!(store.is_instance_minimized("inst-corrupt"), false);
    }

    #[test]
    fn invalid_schema_version_rejected_on_minimize() {
        let mut store = LunarOsRamStore::new();
        store.register_active("inst-bad-ver", "datasets", "Datasets");
        store.begin_minimize("inst-bad-ver").unwrap();
        let mut snap = sample_envelope();
        snap.version = 999;
        let res = store.commit_minimize("inst-bad-ver", snap);
        assert!(res.is_err(), "unsupported schema version must be rejected");
        assert_eq!(store.is_instance_minimized("inst-bad-ver"), false);
    }

    #[test]
    fn multi_instance_isolation_in_ram_store() {
        let mut store = LunarOsRamStore::new();
        store.register_active("win-1", "log:backend", "Backend Log 1");
        store.register_active("win-2", "log:backend", "Backend Log 2");

        // Minimize win-1 only
        store.begin_minimize("win-1").unwrap();
        let mut snap1 = sample_envelope();
        snap1.instance_id = "win-1".into();
        snap1.app_id = "log:backend".into();
        store.commit_minimize("win-1", snap1).unwrap();

        assert_eq!(store.is_instance_minimized("win-1"), true);
        assert_eq!(store.is_instance_minimized("win-2"), false);
        assert_eq!(store.is_instance_open("win-1"), true);
        assert_eq!(store.is_instance_open("win-2"), true);

        // Closing win-2 does not affect win-1
        store.close_instance("win-2");
        assert_eq!(store.is_instance_open("win-2"), false);
        assert_eq!(store.is_instance_minimized("win-1"), true);
        assert_eq!(store.is_instance_open("win-1"), true);

        // Win-1 can still be restored cleanly
        let restore_gen = store.begin_restore("win-1").unwrap();
        let restored = store.commit_restore("win-1", restore_gen).unwrap();
        assert_eq!(restored.instance_id, "win-1");
        assert_eq!(store.is_instance_minimized("win-1"), false);
    }

    #[test]
    fn fifty_cycle_ram_minimize_restore_leak_test() {
        let mut store = LunarOsRamStore::new();
        let instance_id = "stress-test-instance-1";
        store.register_active(instance_id, "terminal", "Terminal");
        assert!(store.is_instance_open(instance_id));
        assert!(!store.is_instance_minimized(instance_id));
        assert_eq!(
            store.get_entry(instance_id).unwrap().state,
            RamLifecycleState::Active
        );

        for cycle in 1..=50 {
            // Verify Active before begin_minimize
            assert_eq!(
                store.get_entry(instance_id).unwrap().state,
                RamLifecycleState::Active
            );

            // Phase 1: Begin Minimize
            store.begin_minimize(instance_id).expect("begin minimize");
            assert_eq!(
                store.get_entry(instance_id).unwrap().state,
                RamLifecycleState::Minimizing
            );

            // Phase 2: Commit Minimize with CRC32 integrity verification
            let envelope = AppSnapshotEnvelopeV1::new(
                instance_id,
                "terminal",
                WindowGeometry {
                    x: 100 + (cycle % 50) as i32,
                    y: 100 + (cycle % 50) as i32,
                    width: 800,
                    height: 600,
                },
                serde_json::json!({
                    "cycle": cycle,
                    "history": vec![format!("cmd-{}", cycle)],
                    "session_key": "active-session-token",
                }),
            );
            assert!(envelope.verify_integrity().is_ok());

            store
                .commit_minimize(instance_id, envelope)
                .expect("commit minimize");
            assert!(store.is_instance_minimized(instance_id));
            assert_eq!(
                store.get_entry(instance_id).unwrap().state,
                RamLifecycleState::Minimized
            );
            assert_eq!(
                store.get_entry(instance_id).unwrap().restore_generation,
                cycle as u64
            );

            // Phase 3: Begin Restore
            let restore_gen = store.begin_restore(instance_id).expect("begin restore");
            assert_eq!(restore_gen, cycle as u64);
            assert_eq!(
                store.get_entry(instance_id).unwrap().state,
                RamLifecycleState::Restoring
            );

            // Phase 4: Commit Restore
            let restored = store
                .commit_restore(instance_id, restore_gen)
                .expect("commit restore");
            assert_eq!(restored.instance_id, instance_id);
            assert_eq!(restored.app_id, "terminal");
            assert!(restored.verify_integrity().is_ok());
            assert_eq!(restored.payload["cycle"], cycle);
            assert!(!store.is_instance_minimized(instance_id));
            assert_eq!(
                store.get_entry(instance_id).unwrap().state,
                RamLifecycleState::Active
            );
        }

        assert_eq!(store.get_entry(instance_id).unwrap().restore_generation, 50);

        // Verification of complete resource reclamation on close
        store.close_instance(instance_id);
        assert!(!store.is_instance_open(instance_id));
        assert!(!store.is_instance_minimized(instance_id));
        assert!(store.get_entry(instance_id).is_none());
        assert!(store.list_entries().is_empty());
    }

    #[test]
    fn multi_instance_isolation_50_cycles_test() {
        let mut store = LunarOsRamStore::new();
        let app_defs = [
            ("inst-0", "terminal", "Terminal"),
            ("inst-1", "editor", "Code Editor"),
            ("inst-2", "browser", "Web Browser"),
            ("inst-3", "files", "File Manager"),
            ("inst-4", "settings", "System Settings"),
        ];

        for (inst_id, app_id, title) in &app_defs {
            store.register_active(inst_id, app_id, title);
            assert!(store.is_instance_open(inst_id));
            assert!(!store.is_instance_minimized(inst_id));
        }
        assert_eq!(store.list_entries().len(), 5);

        let mut instance_generations = [0u64; 5];

        for round in 1..=50 {
            // Pick two distinct instances to toggle each round
            let min_idx = (round - 1) % 5;
            let (min_inst, min_app, _) = app_defs[min_idx];

            if !store.is_instance_minimized(min_inst) {
                store.begin_minimize(min_inst).expect("begin minimize");
                let snap = AppSnapshotEnvelopeV1::new(
                    min_inst,
                    min_app,
                    WindowGeometry {
                        x: 50 * (min_idx as i32 + 1),
                        y: 50 * (min_idx as i32 + 1),
                        width: 700,
                        height: 500,
                    },
                    serde_json::json!({
                        "round": round,
                        "instance": min_inst,
                        "payload": format!("state-{}-{}", min_inst, round),
                    }),
                );
                store
                    .commit_minimize(min_inst, snap)
                    .expect("commit minimize");
                instance_generations[min_idx] += 1;
                assert_eq!(
                    store.get_entry(min_inst).unwrap().restore_generation,
                    instance_generations[min_idx]
                );
                assert!(store.is_instance_minimized(min_inst));
            }

            // Restore an instance that was previously minimized
            let res_idx = (round + 2) % 5;
            let (res_inst, res_app, _) = app_defs[res_idx];
            if store.is_instance_minimized(res_inst) {
                let restore_gen = store.begin_restore(res_inst).expect("begin restore");
                assert_eq!(restore_gen, instance_generations[res_idx]);
                let restored = store
                    .commit_restore(res_inst, restore_gen)
                    .expect("commit restore");
                assert_eq!(restored.instance_id, res_inst);
                assert_eq!(restored.app_id, res_app);
                assert!(restored.verify_integrity().is_ok());
                assert!(!store.is_instance_minimized(res_inst));
            }

            // Validate that instances other than min_inst and res_inst retain their exact generations
            for (check_idx, (check_inst, _, _)) in app_defs.iter().enumerate() {
                let entry = store.get_entry(check_inst).expect("entry must exist");
                assert_eq!(entry.restore_generation, instance_generations[check_idx]);
            }
        }

        // Clean close for all instances in sequence
        for (idx, (inst_id, _, _)) in app_defs.iter().enumerate() {
            assert!(store.is_instance_open(inst_id));
            store.close_instance(inst_id);
            assert!(!store.is_instance_open(inst_id));
            assert!(store.get_entry(inst_id).is_none());
            assert_eq!(store.list_entries().len(), 5 - (idx + 1));
        }

        assert!(store.list_entries().is_empty());
    }
}
