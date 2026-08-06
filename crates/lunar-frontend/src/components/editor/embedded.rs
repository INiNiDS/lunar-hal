//! Embedded, backend-driven editor used inside the WebOS Sandbox iframe.
//!
//! This module owns only local camera/selection UI state. Scene mutations are
//! received from the backend snapshot and SSE stream; it deliberately exposes
//! no parent-to-iframe command receiver.

use dioxus::prelude::*;
use lunar_stellar_core::StellarScene;
use lunar_structures::ResponseStar;
#[cfg(feature = "web")]
use lunar_structures::SceneEvent;

use crate::components::editor::star_map::StarMap;
use crate::stellar_state::{
    clear_selection,
    use_stellar_scene, use_stellar_scene_snapshot, use_stellar_scene_version,
};

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
    inner: std::rc::Rc<SceneEventSubscriptionInner>,
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
fn use_scene_event_stream(scene_id: String, game: Signal<StellarScene>) {
    use wasm_bindgen::JsCast;

    let _subscription = use_hook(move || {
        let url = format!("{}/scenes/{scene_id}/events", game.read().backend_url());
        let source = web_sys::EventSource::new(&url).ok()?;
        let stream_game = game;
        let listener = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
            move |message| {
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
        Some(SceneEventSubscription {
            inner: std::rc::Rc::new(SceneEventSubscriptionInner {
                source,
                _listener: listener,
            }),
        })
    });
}

#[cfg(not(feature = "web"))]
fn use_scene_event_stream(_scene_id: String, _game: Signal<StellarScene>) {
    let _subscription = use_hook(|| ());
}

#[component]
pub fn EmbeddedSandbox(scene_id: String) -> Element {
    let game = use_stellar_scene();
    let _version = use_stellar_scene_version();
    let load_scene_id = scene_id.clone();
    use_resource(move || {
        let game = game;
        let scene_id = load_scene_id.clone();
        async move {
            game.read().clone().load_scene(&scene_id).await.map_err(|error| error.to_string())
        }
    });
    use_scene_event_stream(scene_id.clone(), game);

    let snapshot = use_stellar_scene_snapshot();
    let active = snapshot
        .active_scene
        .filter(|scene| scene.id == scene_id);
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

    rsx! {
        div { class: "h-screen w-screen overflow-hidden bg-[#050505] relative select-none",
            StarMap {
                game,
                scene_stars,
                center_x,
                center_y,
                selected_id,
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
