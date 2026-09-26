//! Embedded, backend-driven editor used inside the WebOS Sandbox iframe.
//!
//! This module owns only local camera/selection UI state. Scene mutations are
//! received from the backend snapshot and SSE stream. The parent sends only a
//! UI lifecycle hint; there is deliberately no parent-to-iframe mutation protocol.

use dioxus::prelude::*;
use lunar_stellar_core::StellarScene;
use lunar_structures::ResponseStar;
#[cfg(feature = "web")]
use lunar_structures::SceneEvent;
use serde::{Deserialize, Serialize};

use crate::components::editor::star_map::StarMap;
use crate::stellar_state::{
    clear_selection, use_stellar_scene, use_stellar_scene_snapshot, use_stellar_scene_version,
};

const SANDBOX_LIFECYCLE_MESSAGE: &str = "lunar:sandbox-lifecycle";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SandboxSnapshotPayload {
    pub schema_version: u32,
    pub scene_id: String,
    pub camera_offset: (f32, f32),
    pub camera_zoom: f32,
    pub selected_star_id: Option<u64>,
    pub last_event_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum SandboxIpcMessage {
    #[serde(rename = "lunar:sandbox-ready")]
    SandboxReady { scene_id: String },

    #[serde(rename = "lunar:prepare-suspend")]
    PrepareSuspend { request_id: String, generation: u64 },

    #[serde(rename = "lunar:suspend-ready")]
    SuspendReady {
        request_id: String,
        snapshot: SandboxSnapshotPayload,
    },

    #[serde(rename = "lunar:restore-state")]
    RestoreState { snapshot: SandboxSnapshotPayload },

    #[serde(rename = "lunar:sandbox-lifecycle")]
    Lifecycle { state: String },
}

pub fn is_valid_ipc_origin(origin: &str, expected_origin: &str) -> bool {
    !origin.is_empty()
        && origin != "null"
        && expected_origin != "*"
        && (origin == expected_origin
            || expected_origin.trim_end_matches('/') == origin.trim_end_matches('/'))
}

fn origin_from_url(url: &str) -> Option<String> {
    let scheme_end = url.find("://")?;
    let scheme = &url[..scheme_end];
    if !matches!(scheme, "http" | "https") {
        return None;
    }
    let authority = url[scheme_end + 3..].split(['/', '?', '#']).next()?.trim();
    if authority.is_empty() {
        return None;
    }
    Some(format!("{scheme}://{authority}"))
}

#[cfg(feature = "web")]
pub fn post_sandbox_ipc_message(msg: &SandboxIpcMessage) {
    if let Some(window) = web_sys::window() {
        if let Ok(Some(parent)) = window.parent() {
            let parent_origin = window
                .document()
                .and_then(|document| origin_from_url(&document.referrer()));
            if let (Some(parent_origin), Ok(serialized)) =
                (parent_origin, serde_json::to_string(msg))
            {
                let _ = parent.post_message(
                    &wasm_bindgen::JsValue::from_str(&serialized),
                    &parent_origin,
                );
            }
        }
    }
}

fn parse_parent_activity(raw: &str) -> Option<bool> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    if value.get("type")?.as_str()? != SANDBOX_LIFECYCLE_MESSAGE {
        return None;
    }
    match value.get("state")?.as_str()? {
        "visible" => Some(true),
        "minimized" | "blocked" => Some(false),
        _ => None,
    }
}

/// Returns a scene id only for `/editor?embedded=sandbox&scene_id=<id>`.
/// Native targets never enter iframe composition mode.
pub fn embedded_scene_id() -> Option<String> {
    #[cfg(feature = "web")]
    {
        let search = web_sys::window()?.location().search().ok()?;
        let mut embedded = false;
        let mut scene_id = None;
        for pair in search.trim_start_matches('?').split('&') {
            let mut parts = pair.splitn(2, '=');
            match (parts.next(), parts.next()) {
                (Some("embedded"), Some("sandbox")) => embedded = true,
                (Some("scene_id"), Some(value)) if !value.trim().is_empty() => {
                    scene_id = Some(value.to_string())
                }
                _ => {}
            }
        }
        embedded.then_some(scene_id?).filter(|id| !id.is_empty())
    }
    #[cfg(not(feature = "web"))]
    {
        None
    }
}

#[cfg(feature = "web")]
#[derive(Clone)]
struct SceneEventSubscription {
    // `use_hook` stores a cloneable value. Keeping the non-cloneable callback
    // in an Rc also makes its lifetime match the EventSource subscription.
    _inner: std::rc::Rc<SceneEventSubscriptionInner>,
}

#[cfg(feature = "web")]
struct SceneEventSubscriptionInner {
    source: web_sys::EventSource,
    _listener: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::MessageEvent)>,
}

#[cfg(feature = "web")]
impl Drop for SceneEventSubscriptionInner {
    fn drop(&mut self) {
        self.source.close();
    }
}

#[cfg(feature = "web")]
#[derive(Clone)]
struct ParentLifecycleSubscription {
    _inner: std::rc::Rc<ParentLifecycleSubscriptionInner>,
}

#[cfg(feature = "web")]
struct ParentLifecycleSubscriptionInner {
    window: web_sys::Window,
    listener: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::MessageEvent)>,
}

#[cfg(feature = "web")]
impl Drop for ParentLifecycleSubscriptionInner {
    fn drop(&mut self) {
        use wasm_bindgen::JsCast;
        let _ = self
            .window
            .remove_event_listener_with_callback("message", self.listener.as_ref().unchecked_ref());
    }
}

#[cfg(feature = "web")]
fn use_sandbox_ipc(scene_id: String, game: Signal<StellarScene>, mut map_active: Signal<bool>) {
    use crate::stellar_state::{apply_camera_snapshot, export_camera_snapshot};
    use wasm_bindgen::JsCast;

    let _sub = use_hook(move || {
        let window = web_sys::window()?;
        let expected_parent_origin = window
            .document()
            .and_then(|document| origin_from_url(&document.referrer()));
        let current_scene_id = scene_id.clone();

        // Announce readiness to parent on mount
        post_sandbox_ipc_message(&SandboxIpcMessage::SandboxReady {
            scene_id: current_scene_id.clone(),
        });

        let listener = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
            move |event: web_sys::MessageEvent| {
                let origin = event.origin();
                let Some(expected_origin) = expected_parent_origin.as_deref() else {
                    return;
                };
                if !is_valid_ipc_origin(&origin, expected_origin) {
                    return;
                }

                let Some(raw) = event.data().as_string() else {
                    return;
                };

                let Ok(msg) = serde_json::from_str::<SandboxIpcMessage>(&raw) else {
                    return;
                };

                match msg {
                    SandboxIpcMessage::PrepareSuspend { request_id, .. } => {
                        map_active.set(false);
                        let g = game.read();
                        let (offset, zoom, selected_star_id) = export_camera_snapshot(&g);
                        let snapshot = SandboxSnapshotPayload {
                            schema_version: 1,
                            scene_id: current_scene_id.clone(),
                            camera_offset: offset,
                            camera_zoom: zoom,
                            selected_star_id,
                            last_event_id: None,
                        };
                        post_sandbox_ipc_message(&SandboxIpcMessage::SuspendReady {
                            request_id,
                            snapshot,
                        });
                    }
                    SandboxIpcMessage::RestoreState { snapshot } => {
                        if snapshot.schema_version == 1 {
                            let g = game.read().clone();
                            apply_camera_snapshot(&g, snapshot.camera_offset, snapshot.camera_zoom);
                            if let Some(star_id) = snapshot.selected_star_id {
                                let snap = g.snapshot();
                                if let Some(found) = snap
                                    .sector_stars
                                    .iter()
                                    .find(|s| s.id as u64 == star_id)
                                    .cloned()
                                {
                                    g.select_star(Some(found));
                                } else if let Some(active) = &snap.active_scene {
                                    if let Some(found) = active
                                        .stars
                                        .iter()
                                        .find(|s| s.id as u64 == star_id)
                                        .cloned()
                                    {
                                        g.select_star(Some(found));
                                    }
                                }
                            }
                        }
                        map_active.set(true);
                    }
                    SandboxIpcMessage::Lifecycle { state } => match state.as_str() {
                        "visible" => map_active.set(true),
                        "minimized" | "blocked" => map_active.set(false),
                        _ => {}
                    },
                    _ => {}
                }
            },
        );

        window
            .add_event_listener_with_callback("message", listener.as_ref().unchecked_ref())
            .ok()?;

        Some(ParentLifecycleSubscription {
            _inner: std::rc::Rc::new(ParentLifecycleSubscriptionInner { window, listener }),
        })
    });
}

#[cfg(not(feature = "web"))]
fn use_sandbox_ipc(_scene_id: String, _game: Signal<StellarScene>, _map_active: Signal<bool>) {}

#[cfg(feature = "web")]
fn use_scene_event_stream(scene_id: String, game: Signal<StellarScene>, enabled: Signal<bool>) {
    use wasm_bindgen::JsCast;

    let mut current_subscription = use_signal(|| Option::<SceneEventSubscription>::None);

    use_effect(move || {
        let is_enabled = enabled();
        if !is_enabled {
            current_subscription.set(None);
            return;
        }

        if current_subscription.peek().is_none() {
            let url = format!("{}/scenes/{scene_id}/events", game.read().backend_url());
            if let Ok(source) = web_sys::EventSource::new(&url) {
                let stream_game = game;
                let listener =
                    wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
                        move |message: web_sys::MessageEvent| {
                            let Some(data) = message.data().as_string() else {
                                return;
                            };
                            let Ok(event) = serde_json::from_str::<SceneEvent>(&data) else {
                                return;
                            };
                            stream_game.read().clone().apply_scene_event(event);
                        },
                    );
                source.set_onmessage(Some(listener.as_ref().unchecked_ref()));
                current_subscription.set(Some(SceneEventSubscription {
                    _inner: std::rc::Rc::new(SceneEventSubscriptionInner {
                        source,
                        _listener: listener,
                    }),
                }));
            }
        }
    });
}

#[cfg(not(feature = "web"))]
fn use_scene_event_stream(_scene_id: String, _game: Signal<StellarScene>, _enabled: Signal<bool>) {
    let _subscription = use_hook(|| ());
}

#[component]
pub fn EmbeddedSandbox(scene_id: String) -> Element {
    let game = use_stellar_scene();
    let _version = use_stellar_scene_version();
    let map_active = use_signal(|| true);

    use_sandbox_ipc(scene_id.clone(), game, map_active);

    let load_scene_id = scene_id.clone();
    use_resource(move || {
        let game = game;
        let scene_id = load_scene_id.clone();
        async move {
            game.read()
                .clone()
                .load_scene(&scene_id)
                .await
                .map_err(|error| error.to_string())
        }
    });

    use_scene_event_stream(scene_id.clone(), game, map_active);

    let snapshot = use_stellar_scene_snapshot();
    let active = snapshot.active_scene.filter(|scene| scene.id == scene_id);
    let scene_stars = active
        .as_ref()
        .map(|scene| scene.stars.clone())
        .unwrap_or_default();
    let (center_x, center_y) = active
        .as_ref()
        .map(|scene| (scene.center_x, scene.center_y))
        .unwrap_or((0.0, 0.0));
    let selected = snapshot.selected_star.clone();
    let selected_id = selected.as_ref().map(|star| star.id);

    let on_select = move |star: ResponseStar| {
        game.read().clone().select_star(Some(star));
    };
    let clear_selection = move |_| clear_selection(&game.read().clone());

    let root_class = if map_active() {
        "embedded-sandbox h-screen w-screen overflow-hidden bg-[#050505] relative select-none"
    } else {
        "embedded-sandbox embedded-sandbox--suspended h-screen w-screen overflow-hidden bg-[#050505] relative select-none"
    };

    rsx! {
        div { class: "{root_class}",
            StarMap {
                game,
                scene_stars,
                center_x,
                center_y,
                selected_id,
                active: map_active,
                on_select,
            }
            if let Some(scene) = active {
                div { class: "absolute left-4 top-4 rounded-xl border border-white/10 bg-black/55 px-3 py-2 text-[10px] uppercase tracking-[0.16em] text-white/65 backdrop-blur-xl pointer-events-none",
                    "Embedded scene · {scene.name}"
                }
            } else {
                div { class: "absolute inset-0 grid place-items-center pointer-events-none",
                    p { class: "rounded-xl border border-white/10 bg-black/60 px-4 py-3 text-xs text-white/60 backdrop-blur-xl", "Loading backend scene…" }
                }
            }
            if let Some(star) = selected {
                div { class: "absolute right-4 top-4 flex items-center gap-3 rounded-xl border border-white/10 bg-black/60 px-3 py-2 text-xs text-white/75 backdrop-blur-xl",
                    div { class: "min-w-0", span { class: "block font-semibold", "{star.name}" } span { class: "text-white/45", "{star.temperature_k:.0} K" } }
                    button { class: "rounded-md px-2 py-1 text-[10px] uppercase tracking-wider text-white/60 hover:bg-white/10", onclick: clear_selection, "Clear" }
                }
            }
        }
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn parses_only_the_sandbox_lifecycle_protocol() {
        assert_eq!(
            parse_parent_activity(r#"{"type":"lunar:sandbox-lifecycle","state":"visible"}"#),
            Some(true)
        );
        assert_eq!(
            parse_parent_activity(r#"{"type":"lunar:sandbox-lifecycle","state":"minimized"}"#),
            Some(false)
        );
        assert_eq!(
            parse_parent_activity(r#"{"type":"other","state":"visible"}"#),
            None
        );
    }

    #[test]
    fn sandbox_ipc_message_round_trips() {
        let msg = SandboxIpcMessage::PrepareSuspend {
            request_id: "req-123".into(),
            generation: 1,
        };
        let serialized = serde_json::to_string(&msg).unwrap();
        let parsed: SandboxIpcMessage = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed, msg);

        let snap_msg = SandboxIpcMessage::SuspendReady {
            request_id: "req-123".into(),
            snapshot: SandboxSnapshotPayload {
                schema_version: 1,
                scene_id: "scene-alpha".into(),
                camera_offset: (10.0, -20.0),
                camera_zoom: 2.5,
                selected_star_id: Some(42),
                last_event_id: None,
            },
        };
        let serialized_snap = serde_json::to_string(&snap_msg).unwrap();
        let parsed_snap: SandboxIpcMessage = serde_json::from_str(&serialized_snap).unwrap();
        assert_eq!(parsed_snap, snap_msg);
    }

    #[test]
    fn origin_validation_rules() {
        assert!(is_valid_ipc_origin(
            "http://localhost:25256",
            "http://localhost:25256"
        ));
        assert!(!is_valid_ipc_origin("http://remote:25256", "*"));
        assert!(!is_valid_ipc_origin("", "http://localhost:25256"));
        assert!(!is_valid_ipc_origin("null", "http://localhost:25256"));
        assert!(!is_valid_ipc_origin(
            "http://attacker.com",
            "http://localhost:25256"
        ));
    }

    #[test]
    fn parent_origin_is_parsed_from_referrer_url() {
        assert_eq!(
            origin_from_url("http://127.0.0.1:8080/testbench/path?x=1"),
            Some("http://127.0.0.1:8080".into())
        );
        assert_eq!(
            origin_from_url("https://testbench.example:9443/"),
            Some("https://testbench.example:9443".into())
        );
        assert_eq!(origin_from_url(""), None);
        assert_eq!(origin_from_url("file:///tmp/testbench.html"), None);
    }
}
