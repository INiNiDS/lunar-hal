//! `localStorage` helpers for the web build, no-op stubs elsewhere.
//!
//! The game backend owns the in-memory scene camera cache; the
//! frontend is responsible for hydrating it from / persisting it to
//! the platform's local storage. This module is the only place that
//! knows about `web_sys` directly.

use lunar_stellar_core::SceneCamera;

#[cfg(feature = "web")]
const CAMERA_KEY_PREFIX: &str = "lunar.scene.camera.";

#[cfg(feature = "web")]
pub fn save_scene_camera(scene_id: &str, wc: SceneCamera) {
    if let Some(window) = web_sys::window() {
        if let Ok(Some(storage)) = window.local_storage() {
            let key = format!("{CAMERA_KEY_PREFIX}{scene_id}");
            let value = format!(
                r#"{{"offset":[{},{}],"zoom":{}}}"#,
                wc.offset.0, wc.offset.1, wc.zoom
            );
            let _ = storage.set_item(&key, &value);
        }
    }
}

#[cfg(feature = "web")]
pub fn load_scene_camera(scene_id: &str) -> Option<SceneCamera> {
    let window = web_sys::window()?;
    let storage = window.local_storage().ok()??;
    let key = format!("{CAMERA_KEY_PREFIX}{scene_id}");
    let raw = storage.get_item(&key).ok()??;
    let parsed: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let arr = parsed.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    let ox = arr[0].as_f64()? as f32;
    let oy = arr[1].as_f64()? as f32;
    let z = arr[2].as_f64()? as f32;
    Some(SceneCamera::new((ox, oy), z))
}

#[cfg(not(feature = "web"))]
pub fn save_scene_camera(_scene_id: &str, _wc: SceneCamera) {}

#[cfg(not(feature = "web"))]
pub fn load_scene_camera(_scene_id: &str) -> Option<SceneCamera> {
    None
}
