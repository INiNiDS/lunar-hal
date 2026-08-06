//! Per-scene camera persistence for web and native frontend builds.
//!
//! The backend owns scene data. This module owns only the frontend camera view,
//! encoded through one versioned serde DTO so every platform has the same
//! compatibility and corruption policy.

use lunar_stellar_core::SceneCamera;
use serde::{Deserialize, Serialize};
use tracing::warn;

const CAMERA_STORAGE_SCHEMA_VERSION: u32 = 1;

#[cfg(feature = "web")]
const CAMERA_KEY_PREFIX: &str = "lunar.scene.camera.v1.";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraStorageRecord {
    pub schema_version: u32,
    pub scene_id: String,
    pub offset: [f32; 2],
    pub zoom: f32,
}

impl CameraStorageRecord {
    fn new(scene_id: &str, camera: SceneCamera) -> Self {
        Self {
            schema_version: CAMERA_STORAGE_SCHEMA_VERSION,
            scene_id: scene_id.to_string(),
            offset: [camera.offset.0, camera.offset.1],
            zoom: camera.zoom,
        }
    }

    fn camera_for_scene(&self, expected_scene_id: &str) -> Option<SceneCamera> {
        if self.schema_version != CAMERA_STORAGE_SCHEMA_VERSION
            || self.scene_id != expected_scene_id
            || !self.offset.iter().all(|value| value.is_finite())
            || !self.zoom.is_finite()
            || self.zoom <= 0.0
        {
            return None;
        }
        Some(SceneCamera::new((self.offset[0], self.offset[1]), self.zoom))
    }
}

fn decode_camera_record(raw: &str, scene_id: &str) -> Option<SceneCamera> {
    serde_json::from_str::<CameraStorageRecord>(raw)
        .ok()?
        .camera_for_scene(scene_id)
}

#[cfg(feature = "web")]
fn camera_key(scene_id: &str) -> String {
    format!("{CAMERA_KEY_PREFIX}{scene_id}")
}

#[cfg(feature = "web")]
pub fn save_scene_camera(scene_id: &str, camera: SceneCamera) {
    let Some(window) = web_sys::window() else { return };
    let Ok(Some(storage)) = window.local_storage() else { return };
    let record = CameraStorageRecord::new(scene_id, camera);
    match serde_json::to_string(&record) {
        Ok(value) => {
            if storage.set_item(&camera_key(scene_id), &value).is_err() {
                warn!(scene_id, "failed to persist scene camera in localStorage");
            }
        }
        Err(error) => warn!(%error, scene_id, "failed to serialize scene camera"),
    }
}

#[cfg(feature = "web")]
pub fn remove_scene_camera(scene_id: &str) {
    if let Some(window) = web_sys::window() {
        if let Ok(Some(storage)) = window.local_storage() {
            let _ = storage.remove_item(&camera_key(scene_id));
        }
    }
}

#[cfg(feature = "web")]
pub fn load_scene_camera(scene_id: &str) -> Option<SceneCamera> {
    let window = web_sys::window()?;
    let storage = window.local_storage().ok()??;
    let key = camera_key(scene_id);
    let raw = storage.get_item(&key).ok()??;
    match decode_camera_record(&raw, scene_id) {
        Some(camera) => Some(camera),
        None => {
            warn!(scene_id, "discarding corrupt or incompatible saved scene camera");
            let _ = storage.remove_item(&key);
            None
        }
    }
}

#[cfg(not(feature = "web"))]
fn camera_file_name(scene_id: &str) -> String {
    // Hex makes every scene identifier a single safe filename component and
    // avoids traversal/collision surprises without an extra dependency.
    let encoded: String = scene_id
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("camera-{encoded}.json")
}

#[cfg(not(feature = "web"))]
fn camera_path(scene_id: &str) -> std::path::PathBuf {
    lunar_utils::env::get_frontend_state_dir().join(camera_file_name(scene_id))
}

#[cfg(not(feature = "web"))]
pub fn save_scene_camera(scene_id: &str, camera: SceneCamera) {
    use std::fs;

    let path = camera_path(scene_id);
    let Some(parent) = path.parent() else { return };
    let record = CameraStorageRecord::new(scene_id, camera);
    let Ok(payload) = serde_json::to_vec(&record) else {
        warn!(scene_id, "failed to serialize scene camera");
        return;
    };
    if let Err(error) = fs::create_dir_all(parent) {
        warn!(%error, scene_id, "failed to create frontend state directory");
        return;
    }

    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let write_result = fs::write(&temporary, payload).and_then(|_| match fs::rename(&temporary, &path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(&path)?;
            fs::rename(&temporary, &path)
        }
        Err(error) => Err(error),
    });
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        warn!(%error, scene_id, "failed to persist scene camera");
    }
}

#[cfg(not(feature = "web"))]
pub fn remove_scene_camera(scene_id: &str) {
    let _ = std::fs::remove_file(camera_path(scene_id));
}

#[cfg(not(feature = "web"))]
pub fn load_scene_camera(scene_id: &str) -> Option<SceneCamera> {
    let path = camera_path(scene_id);
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            warn!(%error, scene_id, "failed to read saved scene camera");
            return None;
        }
    };
    match decode_camera_record(&raw, scene_id) {
        Some(camera) => Some(camera),
        None => {
            warn!(scene_id, "discarding corrupt or incompatible saved scene camera");
            let _ = std::fs::remove_file(path);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_storage_record_round_trips() {
        let camera = SceneCamera::new((12.5, -7.25), 1.75);
        let record = CameraStorageRecord::new("scene-alpha", camera);
        let json = serde_json::to_string(&record).expect("serialize camera record");

        assert_eq!(decode_camera_record(&json, "scene-alpha"), Some(camera));
    }

    #[test]
    fn camera_storage_rejects_corrupt_or_incompatible_records() {
        assert_eq!(decode_camera_record("not json", "scene-alpha"), None);
        assert_eq!(
            decode_camera_record(
                r#"{"schema_version":999,"scene_id":"scene-alpha","offset":[0.0,0.0],"zoom":1.0}"#,
                "scene-alpha",
            ),
            None
        );
        assert_eq!(
            decode_camera_record(
                r#"{"schema_version":1,"scene_id":"other","offset":[0.0,0.0],"zoom":1.0}"#,
                "scene-alpha",
            ),
            None
        );
    }
}
