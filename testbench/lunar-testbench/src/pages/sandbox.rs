//! WebOS host for a *real* `lunar-frontend` editor page.
//!
//! The iframe remains an independent application. Every administrative action
//! below calls `lunar-backend` directly; no local `StellarScene`, renderer, or
//! parent-to-iframe mutation protocol exists in this page.

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::api::{
    self, ClearSceneRequest, CreateSceneStarRequest, CreateStarSceneRequest, GallerySource,
    GenerateSceneStarsRequest, ResponseStar, StarModelInputs, UpdateSceneStarRequest,
};
use crate::os::state::{
    is_window_lifecycle_visible, managed_web_frontend_url, use_window_lifecycle,
};
use crate::os::{
    AppSnapshot, WindowLifecycle, use_os_state, use_window_instance_id,
    use_window_instance_snapshot,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct SuspendReadySubscription {
    _inner: std::rc::Rc<SuspendReadySubscriptionInner>,
}

#[cfg(target_arch = "wasm32")]
struct SuspendReadySubscriptionInner {
    window: web_sys::Window,
    listener: std::rc::Rc<wasm_bindgen::closure::Closure<dyn FnMut(web_sys::MessageEvent)>>,
}

#[cfg(target_arch = "wasm32")]
impl Drop for SuspendReadySubscriptionInner {
    fn drop(&mut self) {
        let _ = self.window.remove_event_listener_with_callback(
            "message",
            self.listener.as_ref().as_ref().unchecked_ref(),
        );
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SandboxSnapshot {
    pub selected_scene: Option<String>,
    pub retained_iframe_src: Option<String>,
    pub snapshot: Option<api::LiveSceneSnapshot>,
    pub selected_star: Option<u32>,
    pub camera_offset: (f32, f32),
    pub camera_zoom: f32,
    pub selected_star_id: Option<u64>,
}

impl Default for SandboxSnapshot {
    fn default() -> Self {
        Self {
            selected_scene: None,
            retained_iframe_src: None,
            snapshot: None,
            selected_star: None,
            camera_offset: (0.0, 0.0),
            camera_zoom: 1.0,
            selected_star_id: None,
        }
    }
}

impl AppSnapshot for SandboxSnapshot {
    fn capture_snapshot(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    fn hydrate_snapshot(&mut self, payload: &serde_json::Value) -> Result<(), String> {
        let snap: SandboxSnapshot = serde_json::from_value(payload.clone())
            .map_err(|e| format!("Sandbox hydration failed: {e}"))?;
        *self = snap;
        Ok(())
    }
}

fn request_id(prefix: &str) -> String {
    format!("{prefix}-{}", js_sys::Date::now())
}

fn lifecycle_name(lifecycle: WindowLifecycle) -> &'static str {
    match lifecycle {
        WindowLifecycle::Visible => "visible",
        WindowLifecycle::Minimized => "minimized",
        WindowLifecycle::Blocked => "blocked",
    }
}

fn should_refresh_after_lifecycle_transition(
    previous: WindowLifecycle,
    current: WindowLifecycle,
) -> bool {
    previous != current && current == WindowLifecycle::Visible
}

fn post_iframe_lifecycle(lifecycle: WindowLifecycle) {
    let payload = serde_json::json!({
        "type": "lunar:sandbox-lifecycle",
        "state": lifecycle_name(lifecycle),
    })
    .to_string();
    let Ok(payload_literal) = serde_json::to_string(&payload) else {
        return;
    };
    let script = format!(
        r#"(() => {{
            const frame = document.querySelector('iframe[data-lunar-sandbox-frame="true"]');
            if (frame && frame.contentWindow) {{
                frame.contentWindow.postMessage({payload_literal}, '*');
            }}
        }})()"#
    );
    let _ = dioxus::document::eval(&script);
}

pub fn post_iframe_resume() {
    post_iframe_lifecycle(WindowLifecycle::Visible);
}

pub fn post_iframe_restore_state(snapshot: &SandboxSnapshot) {
    let scene_id = snapshot.selected_scene.clone().unwrap_or_default();
    let payload = serde_json::json!({
        "type": "lunar:restore-state",
        "snapshot": {
            "schema_version": 1,
            "scene_id": scene_id,
            "camera_offset": [snapshot.camera_offset.0, snapshot.camera_offset.1],
            "camera_zoom": snapshot.camera_zoom,
            "selected_star_id": snapshot.selected_star_id.or(snapshot.selected_star.map(|s| s as u64)),
            "last_event_id": null
        }
    })
    .to_string();
    let Ok(payload_literal) = serde_json::to_string(&payload) else {
        return;
    };
    let script = format!(
        r#"(() => {{
            const frame = document.querySelector('iframe[data-lunar-sandbox-frame="true"]');
            if (frame && frame.contentWindow) {{
                frame.contentWindow.postMessage({payload_literal}, '*');
            }}
        }})()"#
    );
    let _ = dioxus::document::eval(&script);
}

pub async fn prepare_iframe_suspend(
    instance_id: &str,
    request_id: &str,
    generation: u64,
) -> Result<serde_json::Value, String> {
    let payload = serde_json::json!({
        "type": "lunar:prepare-suspend",
        "request_id": request_id,
        "generation": generation,
    })
    .to_string();
    let payload_literal = serde_json::to_string(&payload).map_err(|e| e.to_string())?;
    let instance_literal = serde_json::to_string(instance_id).map_err(|e| e.to_string())?;
    let request_literal = serde_json::to_string(request_id).map_err(|e| e.to_string())?;
    let script = format!(
        r#"(() => {{
            const frame = [...document.querySelectorAll('iframe[data-lunar-sandbox-frame="true"]')]
                .find(frame => frame.getAttribute('data-lunar-instance-id') === {instance_literal});
            if (!frame || !frame.contentWindow) {{ dioxus.send(null); return; }}
            const url = new URL(frame.src, location.href);
            const requestId = {request_literal};
            let timer;
            const done = snapshot => {{
                window.removeEventListener('message', onMessage);
                clearTimeout(timer);
                dioxus.send(snapshot);
            }};
            const onMessage = event => {{
                if (event.source !== frame.contentWindow || event.origin !== url.origin) return;
                let data;
                try {{ data = JSON.parse(event.data); }} catch {{ return; }}
                const snapshot = data?.snapshot;
                if (data?.type !== 'lunar:suspend-ready' || data.request_id !== requestId ||
                    snapshot?.schema_version !== 1 || snapshot.scene_id !== url.searchParams.get('scene_id') ||
                    !Array.isArray(snapshot.camera_offset) || snapshot.camera_offset.length !== 2 ||
                    !snapshot.camera_offset.every(Number.isFinite) ||
                    !Number.isFinite(snapshot.camera_zoom) || snapshot.camera_zoom <= 0 ||
                    !(snapshot.selected_star_id === null || Number.isSafeInteger(snapshot.selected_star_id))) return;
                done(snapshot);
            }};
            window.addEventListener('message', onMessage);
            timer = setTimeout(() => done(null), 1500);
            frame.contentWindow.postMessage({payload_literal}, url.origin);
        }})()"#
    );
    dioxus::document::eval(&script)
        .recv::<Option<serde_json::Value>>()
        .await
        .map_err(|e| format!("Suspend handshake failed: {e}"))?
        .ok_or_else(|| "Iframe did not acknowledge suspend within 1.5 seconds".to_string())
}

pub fn apply_suspend_state(
    state: &mut SandboxSnapshot,
    received: &serde_json::Value,
) -> Result<(), String> {
    if received.get("schema_version").and_then(|v| v.as_u64()) != Some(1)
        || received.get("scene_id").and_then(|v| v.as_str()) != state.selected_scene.as_deref()
    {
        return Err("Suspend snapshot has an incompatible version or scene".into());
    }
    let offsets = received
        .get("camera_offset")
        .and_then(|v| v.as_array())
        .filter(|a| a.len() == 2)
        .ok_or("Invalid camera offset")?;
    let x = offsets[0].as_f64().ok_or("Invalid camera x")?;
    let y = offsets[1].as_f64().ok_or("Invalid camera y")?;
    let zoom = received
        .get("camera_zoom")
        .and_then(|v| v.as_f64())
        .ok_or("Invalid camera zoom")?;
    if !x.is_finite()
        || !y.is_finite()
        || !zoom.is_finite()
        || zoom < f32::MIN_POSITIVE as f64
        || [x, y, zoom]
            .iter()
            .any(|v| *v > f32::MAX as f64 || *v < -(f32::MAX as f64))
    {
        return Err("Camera values are out of range".into());
    }
    let selected = received
        .get("selected_star_id")
        .ok_or("Missing selected star id")?;
    let selected = if selected.is_null() {
        None
    } else {
        Some(selected.as_u64().ok_or("Invalid selected star id")?)
    };
    state.camera_offset = (x as f32, y as f32);
    state.camera_zoom = zoom as f32;
    state.selected_star_id = selected;
    Ok(())
}

/// Keep the exact iframe URL while dependencies are temporarily unavailable.
/// A replacement URL is applied only when a live web frontend supplies one.
fn retain_iframe_src(previous: Option<String>, candidate: Option<String>) -> Option<String> {
    candidate.or(previous)
}

#[cfg(test)]
mod tests {
    use super::{retain_iframe_src, should_refresh_after_lifecycle_transition};
    use crate::os::WindowLifecycle;

    #[test]
    fn lifecycle_refresh_runs_only_when_the_window_becomes_visible() {
        assert!(should_refresh_after_lifecycle_transition(
            WindowLifecycle::Minimized,
            WindowLifecycle::Visible,
        ));
        assert!(should_refresh_after_lifecycle_transition(
            WindowLifecycle::Blocked,
            WindowLifecycle::Visible,
        ));
        assert!(!should_refresh_after_lifecycle_transition(
            WindowLifecycle::Visible,
            WindowLifecycle::Visible,
        ));
        assert!(!should_refresh_after_lifecycle_transition(
            WindowLifecycle::Visible,
            WindowLifecycle::Minimized,
        ));
    }

    #[test]
    fn temporary_dependency_loss_keeps_the_existing_iframe_url() {
        let original = Some("http://127.0.0.1:8080/editor?scene_id=scene-a".to_string());
        assert_eq!(retain_iframe_src(original.clone(), None), original);
        assert_eq!(
            retain_iframe_src(
                Some("http://127.0.0.1:8080/editor?scene_id=scene-a".to_string()),
                Some("http://127.0.0.1:8081/editor?scene_id=scene-a".to_string()),
            ),
            Some("http://127.0.0.1:8081/editor?scene_id=scene-a".to_string())
        );
    }

    #[test]
    fn test_sandbox_snapshot_capture_and_hydration() {
        use crate::os::AppSnapshot;
        let original = super::SandboxSnapshot {
            selected_scene: Some("scene-xyz".into()),
            retained_iframe_src: Some("http://127.0.0.1:8080/editor".into()),
            snapshot: None,
            selected_star: Some(123),
            camera_offset: (150.0, -80.0),
            camera_zoom: 2.5,
            selected_star_id: Some(123),
        };
        let payload = original.capture_snapshot();
        let mut restored = super::SandboxSnapshot::default();
        restored.hydrate_snapshot(&payload).unwrap();
        assert_eq!(restored, original);
    }
}

#[component]
pub fn Sandbox() -> Element {
    let initial = use_window_instance_snapshot::<SandboxSnapshot>().unwrap_or_default();
    let instance_id = use_window_instance_id().unwrap_or_default();
    let mut os = use_os_state();
    let frontend_url = managed_web_frontend_url(&os.services.read());
    let lifecycle = use_window_lifecycle();
    let is_blocked = lifecycle
        .map(|signal| *signal.read() == WindowLifecycle::Blocked)
        .unwrap_or(false);

    let mut scenes = use_signal(Vec::new);
    let mut gallery = use_signal(Vec::new);
    let mut selected_scene = use_signal(|| initial.selected_scene.clone());
    // Do not conditionally remove the iframe when a dependency disappears.
    // Its URL is retained until a live replacement is available.
    let mut retained_iframe_src = use_signal(|| initial.retained_iframe_src.clone());
    let mut snapshot = use_signal(|| initial.snapshot.clone());
    let mut selected_star = use_signal(|| initial.selected_star);
    let camera_offset = use_signal(|| initial.camera_offset);
    let camera_zoom = use_signal(|| initial.camera_zoom);
    let selected_star_id = use_signal(|| initial.selected_star_id);
    let mut refresh_tick = use_signal(|| 0_u32);
    let initial_lifecycle = lifecycle
        .map(|signal| *signal.peek())
        .unwrap_or(WindowLifecycle::Visible);
    let mut previous_lifecycle = use_signal(|| initial_lifecycle);
    let mut status = use_signal(|| None::<String>);
    let mut busy = use_signal(|| false);

    use_effect(move || {
        let snap = SandboxSnapshot {
            selected_scene: selected_scene(),
            retained_iframe_src: retained_iframe_src(),
            snapshot: snapshot(),
            selected_star: selected_star(),
            camera_offset: camera_offset(),
            camera_zoom: camera_zoom(),
            selected_star_id: selected_star_id(),
        };
        if let Some(inst_id) = use_window_instance_id() {
            os.register_instance_snapshot(&inst_id, snap.capture_snapshot());
        }
        os.register_app_snapshot("sandbox", snap.capture_snapshot());
    });

    #[cfg(target_arch = "wasm32")]
    {
        let cam_off = camera_offset;
        let cam_zm = camera_zoom;
        let sel_star_id = selected_star_id;
        let sel_scene = selected_scene;
        let ret_src = retained_iframe_src;
        let sn_sig = snapshot;
        let sel_st = selected_star;

        use_hook(move || {
            let window = web_sys::window()?;
            let listener = std::rc::Rc::new(wasm_bindgen::closure::Closure::<
                dyn FnMut(web_sys::MessageEvent),
            >::new(
                move |event: web_sys::MessageEvent| {
                    let Some(raw) = event.data().as_string() else {
                        return;
                    };
                    let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) else {
                        return;
                    };
                    let Some(msg_type) = val.get("type").and_then(|t| t.as_str()) else {
                        return;
                    };

                    match msg_type {
                        "lunar:sandbox-ready" => {
                            let snap = SandboxSnapshot {
                                selected_scene: sel_scene.peek().clone(),
                                retained_iframe_src: ret_src.peek().clone(),
                                snapshot: sn_sig.peek().clone(),
                                selected_star: *sel_st.peek(),
                                camera_offset: *cam_off.peek(),
                                camera_zoom: *cam_zm.peek(),
                                selected_star_id: *sel_star_id.peek(),
                            };
                            post_iframe_restore_state(&snap);
                        }
                        _ => {}
                    }
                },
            ));

            window
                .add_event_listener_with_callback(
                    "message",
                    listener.as_ref().as_ref().unchecked_ref(),
                )
                .ok()?;

            Some(SuspendReadySubscription {
                _inner: std::rc::Rc::new(SuspendReadySubscriptionInner { window, listener }),
            })
        });
    }

    let mut create_name = use_signal(|| "Sandbox scene".to_string());
    let mut batch_count = use_signal(|| 1_u32);
    let mut custom_name = use_signal(|| "Custom star".to_string());
    let mut custom_x = use_signal(|| 0.0_f32);
    let mut custom_y = use_signal(|| 0.0_f32);
    let mut custom_z = use_signal(|| 0.0_f32);
    let mut custom_temp = use_signal(|| 5778.0_f32);
    let mut edit_name = use_signal(String::new);
    let mut import_gallery_id = use_signal(|| None::<String>);

    let listing_lifecycle = lifecycle;
    use_resource(move || {
        let tick = refresh_tick();
        let visible = is_window_lifecycle_visible(listing_lifecycle);
        async move {
            let _ = tick;
            if !visible {
                return;
            }
            match api::list_star_scenes().await {
                Ok(list) => {
                    let first = list.scenes.first().map(|scene| scene.id.clone());
                    scenes.set(list.scenes);
                    if selected_scene().is_none() {
                        selected_scene.set(first);
                    }
                }
                Err(error) => status.set(Some(error)),
            }
            match api::list_gallery_stars(None, 48, Some("updated_desc"), None).await {
                Ok(list) => {
                    let first = list.stars.first().map(|star| star.id.clone());
                    gallery.set(list.stars);
                    if import_gallery_id().is_none() {
                        import_gallery_id.set(first);
                    }
                }
                Err(error) => status.set(Some(error)),
            }
        }
    });

    let scene_lifecycle = lifecycle;
    use_resource(move || {
        let scene_id = selected_scene();
        let visible = is_window_lifecycle_visible(scene_lifecycle);
        async move {
            if !visible {
                return;
            }
            match scene_id {
                Some(id) => match api::get_live_scene(&id).await {
                    Ok(value) => snapshot.set(Some(value)),
                    Err(error) => status.set(Some(error)),
                },
                None => snapshot.set(None),
            }
        }
    });

    use_effect(move || {
        let current = lifecycle
            .map(|signal| *signal.read())
            .unwrap_or(WindowLifecycle::Visible);
        let previous = *previous_lifecycle.peek();
        if current == previous {
            return;
        }
        previous_lifecycle.set(current);
        if should_refresh_after_lifecycle_transition(previous, current) {
            // `peek` is deliberate: subscribing this effect to refresh_tick and
            // then writing it creates an unbounded render/effect loop.
            let next_tick = (*refresh_tick.peek()).wrapping_add(1);
            refresh_tick.set(next_tick);
        }
    });

    let iframe_lifecycle = lifecycle;
    use_effect(move || {
        let current = iframe_lifecycle
            .map(|signal| *signal.read())
            .unwrap_or(WindowLifecycle::Visible);
        post_iframe_lifecycle(current);
    });

    let control_disabled = is_blocked || busy();

    let iframe_src_candidate = if lifecycle
        .map(|signal| *signal.read() == WindowLifecycle::Visible)
        .unwrap_or(true)
    {
        frontend_url.as_ref().and_then(|base| {
            selected_scene()
                .as_ref()
                .map(|scene_id| format!("{base}/editor?embedded=sandbox&scene_id={scene_id}"))
        })
    } else {
        None
    };
    use_effect(move || {
        let previous = retained_iframe_src.peek().clone();
        let next = retain_iframe_src(previous.clone(), iframe_src_candidate.clone());
        if previous != next {
            retained_iframe_src.set(next);
        }
    });
    let iframe_src = retained_iframe_src();
    let lifecycle_for_iframe_load = lifecycle;

    let create_scene = move |_| {
        busy.set(true);
        status.set(None);
        let request = CreateStarSceneRequest {
            name: create_name().trim().to_string(),
            center_x: 0.0,
            center_y: 0.0,
            center_z: 0.0,
            temperature: 0.7,
        };
        spawn(async move {
            match api::create_star_scene(&request).await {
                Ok(created) => {
                    selected_scene.set(Some(created.scene.id));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                    status.set(Some("Live scene created.".into()));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let generate = move |_| {
        let Some(scene_id) = selected_scene() else {
            status.set(Some("Create or select a live scene first.".into()));
            return;
        };
        busy.set(true);
        status.set(None);
        let request = GenerateSceneStarsRequest {
            request_id: request_id("sandbox-generate"),
            count: batch_count().clamp(1, 32),
            entropy_temperature: None,
        };
        spawn(async move {
            match api::generate_scene_stars(&scene_id, &request).await {
                Ok(stars) => {
                    status.set(Some(format!(
                        "Generated {} star(s); the iframe updates from SSE.",
                        stars.len()
                    )));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let create_custom = move |_| {
        let Some(scene_id) = selected_scene() else {
            status.set(Some("Create or select a live scene first.".into()));
            return;
        };
        busy.set(true);
        let star = ResponseStar {
            id: 0,
            x: custom_x(),
            y: custom_y(),
            z: custom_z(),
            temperature_k: custom_temp().max(1.0),
            radius: 1.0,
            mass: 1.0,
            luminosity: 1.0,
            description: "Custom administrative stellar record".into(),
            name: custom_name().trim().to_string(),
            type_hint: "custom".into(),
            velocity_vector: [0.0; 3],
        };
        let request = CreateSceneStarRequest {
            request_id: request_id("sandbox-custom"),
            inputs: Some(StarModelInputs::from_star(&star)),
            star: Some(star),
            gallery_id: None,
            gallery_source: GallerySource::AdminGenerated,
        };
        spawn(async move {
            match api::create_scene_star(&scene_id, &request).await {
                Ok(star) => {
                    status.set(Some(format!(
                        "{} added to the live scene and Gallery.",
                        star.name
                    )));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let import_gallery = move |_| {
        let (Some(scene_id), Some(gallery_id)) = (selected_scene(), import_gallery_id()) else {
            status.set(Some("Select both a scene and a Gallery record.".into()));
            return;
        };
        busy.set(true);
        let request = CreateSceneStarRequest {
            request_id: request_id("sandbox-gallery-import"),
            star: None,
            inputs: None,
            gallery_id: Some(gallery_id),
            gallery_source: GallerySource::Imported,
        };
        spawn(async move {
            match api::create_scene_star(&scene_id, &request).await {
                Ok(star) => {
                    status.set(Some(format!(
                        "{} copied from Gallery into the live scene.",
                        star.name
                    )));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let rename_star = move |_| {
        let (Some(scene_id), Some(star_id)) = (selected_scene(), selected_star()) else {
            status.set(Some("Select a scene star to edit.".into()));
            return;
        };
        busy.set(true);
        let request = UpdateSceneStarRequest {
            name: Some(edit_name().trim().to_string()),
            ..Default::default()
        };
        spawn(async move {
            match api::update_scene_star(&scene_id, star_id, &request).await {
                Ok(star) => {
                    status.set(Some(format!("{} updated in the backend.", star.name)));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let delete_star = move |_| {
        let (Some(scene_id), Some(star_id)) = (selected_scene(), selected_star()) else {
            status.set(Some("Select a scene star to delete.".into()));
            return;
        };
        busy.set(true);
        spawn(async move {
            match api::delete_scene_star(&scene_id, star_id).await {
                Ok(()) => {
                    selected_star.set(None);
                    status.set(Some("Star deleted from the live scene.".into()));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let clear_scene = move |_| {
        let Some(scene_id) = selected_scene() else {
            return;
        };
        busy.set(true);
        let request = ClearSceneRequest {
            request_id: request_id("sandbox-clear"),
        };
        spawn(async move {
            match api::clear_live_scene(&scene_id, &request).await {
                Ok(()) => {
                    selected_star.set(None);
                    status.set(Some("Live scene cleared.".into()));
                    refresh_tick.set(refresh_tick().wrapping_add(1));
                }
                Err(error) => status.set(Some(error)),
            }
            busy.set(false);
        });
    };

    let visible_stars = snapshot()
        .as_ref()
        .map(|value| value.scene.stars.clone())
        .unwrap_or_default();

    rsx! {
        div {
            class: "sandbox-app",
            "data-testid": "sandbox-app",
            "data-camera-x": "{camera_offset().0}",
            "data-camera-y": "{camera_offset().1}",
            "data-camera-zoom": "{camera_zoom()}",
            "data-selected-star-id": "{selected_star_id().map(|id| id.to_string()).unwrap_or_default()}",
            if let Some(src) = iframe_src {
                iframe {
                    class: "sandbox-frame",
                    "data-lunar-sandbox-frame": "true",
                    "data-lunar-instance-id": "{instance_id}",
                    src: "{src}",
                    title: "Embedded lunar frontend editor",
                    onload: move |_| {
                        let current = lifecycle_for_iframe_load
                            .map(|signal| *signal.read())
                            .unwrap_or(WindowLifecycle::Visible);
                        post_iframe_lifecycle(current);
                        let snap = SandboxSnapshot {
                            selected_scene: selected_scene(),
                            retained_iframe_src: retained_iframe_src(),
                            snapshot: snapshot(),
                            selected_star: selected_star(),
                            camera_offset: camera_offset(),
                            camera_zoom: camera_zoom(),
                            selected_star_id: selected_star_id(),
                        };
                        post_iframe_restore_state(&snap);
                    },
                }
            } else {
                div { class: "sandbox-empty",
                    div { class: "rounded-2xl border border-white/10 bg-black/60 p-6 backdrop-blur-xl",
                        p { "A running web frontend with a public URL is required." }
                    }
                }
            }

            // This is the second independent layer: it never imports or mutates
            // iframe state. It only talks to `lunar-backend` through api.rs.
            aside { class: "sandbox-admin-overlay",
                div { class: "flex items-center justify-between gap-2",
                    div { h1 { class: "text-sm font-semibold", "Scene administration" } p { class: "text-[10px] uppercase tracking-widest text-white/40", "Backend-owned overlay" } }
                    if busy() { span { class: "text-xs text-amber-300", "Working…" } }
                }
                if let Some(message) = status() { p { class: "rounded-lg bg-white/10 px-2 py-1.5 text-xs text-white/70", "{message}" } }

                div { class: "space-y-2 rounded-xl border border-white/10 p-2",
                    label { class: "block text-[10px] uppercase tracking-wider text-white/45", "Live scene" }
                    select { class: "w-full rounded-lg bg-black/50 px-2 py-1.5 text-sm", value: selected_scene().unwrap_or_default(), onchange: move |event| { selected_scene.set(Some(event.value())); selected_star.set(None); },
                        option { value: "", "Select a scene" }
                        for scene in scenes() { option { value: "{scene.id}", "{scene.name} · {scene.star_count}" } }
                    }
                    div { class: "flex gap-2", input { class: "min-w-0 flex-1 rounded-lg bg-black/50 px-2 py-1.5 text-xs", "data-testid": "sandbox-scene-name", value: "{create_name()}", oninput: move |event| create_name.set(event.value()) } button { class: "rounded-lg bg-violet-500/30 px-2 text-xs hover:bg-violet-500/50", "data-testid": "sandbox-create-scene", disabled: control_disabled, onclick: create_scene, "New" } }
                }

                div { class: "space-y-2 rounded-xl border border-white/10 p-2",
                    label { class: "block text-[10px] uppercase tracking-wider text-white/45", "Generate one or batch" }
                    div { class: "flex gap-2", input { class: "w-16 rounded-lg bg-black/50 px-2 py-1.5 text-xs", r#type: "number", min: "1", max: "32", value: "{batch_count()}", oninput: move |event| if let Ok(value) = event.value().parse() { batch_count.set(value); } } button { class: "flex-1 rounded-lg bg-cyan-500/25 px-2 py-1.5 text-xs hover:bg-cyan-500/40", disabled: control_disabled, onclick: generate, "Generate & archive" } }
                }

                div { class: "space-y-2 rounded-xl border border-white/10 p-2",
                    label { class: "block text-[10px] uppercase tracking-wider text-white/45", "Custom star" }
                    input { class: "w-full rounded-lg bg-black/50 px-2 py-1.5 text-xs", "data-testid": "sandbox-custom-star-name", value: "{custom_name()}", oninput: move |event| custom_name.set(event.value()) }
                    div { class: "grid grid-cols-3 gap-1", input { class: "min-w-0 rounded bg-black/50 px-1.5 py-1 text-xs", placeholder: "x", value: "{custom_x()}", oninput: move |event| if let Ok(value) = event.value().parse() { custom_x.set(value); } } input { class: "min-w-0 rounded bg-black/50 px-1.5 py-1 text-xs", placeholder: "y", value: "{custom_y()}", oninput: move |event| if let Ok(value) = event.value().parse() { custom_y.set(value); } } input { class: "min-w-0 rounded bg-black/50 px-1.5 py-1 text-xs", placeholder: "z", value: "{custom_z()}", oninput: move |event| if let Ok(value) = event.value().parse() { custom_z.set(value); } } }
                    div { class: "flex gap-2", input { class: "min-w-0 flex-1 rounded bg-black/50 px-2 py-1 text-xs", r#type: "number", value: "{custom_temp()}", oninput: move |event| if let Ok(value) = event.value().parse() { custom_temp.set(value); } } button { class: "rounded-lg bg-violet-500/25 px-2 py-1 text-xs hover:bg-violet-500/40", "data-testid": "sandbox-add-custom-star", disabled: control_disabled, onclick: create_custom, "Add" } }
                }

                div { class: "space-y-2 rounded-xl border border-white/10 p-2",
                    label { class: "block text-[10px] uppercase tracking-wider text-white/45", "Import from Gallery" }
                    select { class: "w-full rounded-lg bg-black/50 px-2 py-1.5 text-xs", value: import_gallery_id().unwrap_or_default(), onchange: move |event| import_gallery_id.set(Some(event.value())),
                        option { value: "", "Choose saved star" }
                        for item in gallery() { option { value: "{item.id}", "{item.name.clone().unwrap_or_else(|| item.star.name.clone())}" } }
                    }
                    button { class: "w-full rounded-lg bg-emerald-500/25 px-2 py-1.5 text-xs hover:bg-emerald-500/40", disabled: control_disabled, onclick: import_gallery, "Copy to scene" }
                }

                div { class: "space-y-2 rounded-xl border border-white/10 p-2",
                    label { class: "block text-[10px] uppercase tracking-wider text-white/45", "Scene stars" }
                    div { class: "max-h-28 space-y-1 overflow-y-auto", for star in visible_stars { button { class: "block w-full truncate rounded px-2 py-1 text-left text-xs hover:bg-white/10", "data-testid": "sandbox-scene-star", "data-star-id": "{star.id}", disabled: control_disabled, onclick: move |_| { selected_star.set(Some(star.id)); edit_name.set(star.name.clone()); }, "{star.name} · {star.temperature_k:.0} K" } } }
                    if selected_star().is_some() { div { class: "flex gap-1", input { class: "min-w-0 flex-1 rounded bg-black/50 px-2 py-1 text-xs", value: "{edit_name()}", oninput: move |event| edit_name.set(event.value()) } button { class: "rounded bg-sky-500/25 px-2 text-xs", disabled: control_disabled, onclick: rename_star, "Save" } button { class: "rounded bg-red-500/25 px-2 text-xs", disabled: control_disabled, onclick: delete_star, "Delete" } } }
                    button { class: "w-full rounded bg-red-500/15 px-2 py-1 text-xs text-red-100 hover:bg-red-500/30", disabled: control_disabled, onclick: clear_scene, "Clear live scene" }
                }
            }
        }
    }
}
