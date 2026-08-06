//! Reactivity bridge between [`StellarScene`] and Dioxus.
//!
//! The stellar-scene client is framework-agnostic. To make it play nicely with
//! Dioxus's reactive system, this module installs a version signal
//! in the current context and pipes the client’s internal change
//! notifications (delivered via `tokio::sync::watch`) into that
//! signal. Components that need to re-render read the version (or a
//! [`StellarSceneSnapshot`]) and Dioxus tracks the dependency for us.

use dioxus::prelude::*;
use lunar_stellar_core::{StellarScene, StellarSceneConfig, StellarSceneSnapshot};
use tracing::warn;

use crate::local_storage;
use crate::runtime_config::RuntimeConfig;

/// Private helper hook to eliminate code duplication for Tokio channel subscription.
fn use_setup_stellar_scene_listener(game: Signal<StellarScene>, version: Signal<u64>) {
    use_hook(|| {
        let mut changes = game.read().subscribe();
        let mut version_for_task = version;
        spawn(async move {
            while changes.changed().await.is_ok() {
                let v = *changes.borrow();
                let current = version_for_task.peek().saturating_add(1);
                version_for_task.set(current.max(v));
            }
        });
    });
}

/// Provide a [`StellarScene`] instance to the component subtree. Wraps the
/// stellar scene in a Dioxus context, installs a `Signal<u64>` version
/// counter, and starts a coroutine that bumps the counter on every
/// scene mutation.
#[allow(clippy::future_not_send, clippy::used_underscore_binding)]
pub fn use_provide_stellar_scene() -> Signal<StellarScene> {
    let version = use_signal(|| 0u64);
    let game = use_signal(|| {
        let runtime = RuntimeConfig::from_environment();
        StellarScene::with_config(StellarSceneConfig::new(runtime.backend_url))
    });

    use_setup_stellar_scene_listener(game, version);

    use_context_provider(|| game);
    use_context_provider(|| version);
    game
}

/// Provide a pre-configured [`StellarScene`] instance. Useful when a host
/// application already constructed a stellar-scene client and wants to inject it
/// into the Dioxus context.
#[allow(dead_code)]
pub fn use_provide_stellar_scene_with(initial: StellarScene) -> Signal<StellarScene> {
    let version = use_signal(|| 0u64);
    let game = use_signal(|| initial);

    use_setup_stellar_scene_listener(game, version);

    use_context_provider(|| game);
    use_context_provider(|| version);
    game
}

/// Borrow the shared [`StellarScene`] from the current context.
pub fn use_stellar_scene() -> Signal<StellarScene> {
    use_context::<Signal<StellarScene>>()
}

/// Read the shared version signal. Components that need to re-render
/// after a scene mutation should call this first to subscribe.
pub fn use_stellar_scene_version() -> Signal<u64> {
    use_context::<Signal<u64>>()
}

/// Reactive snapshot of the entire game state. Subscribes to the
/// version signal so the calling component re-renders on every
/// mutation.
pub fn use_stellar_scene_snapshot() -> StellarSceneSnapshot {
    let version = use_stellar_scene_version();
    let game = use_stellar_scene();
    let _ = version();
    game.read().snapshot()
}

/// Install a per-scene camera persistence hook that mirrors the
/// client's `scene_cameras` map into `localStorage` (web) or a no-op
/// stub (desktop). Call this once near the top of the editor.
pub fn use_provide_scene_camera_persistence() {
    let game = use_stellar_scene();
    let version = use_stellar_scene_version();

    use_effect(move || {
        let _ = version();
        let snap = game.read().snapshot();
        if let Some(id) = snap.active_scene_id() {
            if let Some(wc) = snap.scene_cameras.get(id).copied() {
                local_storage::save_scene_camera(id, wc);
            }
        }
    });
}

pub fn hydrate_scene_camera_from_storage(game: &StellarScene, scene_id: &str) {
    if let Some(wc) = local_storage::load_scene_camera(scene_id) {
        if let Err(e) = game.set_scene_camera(scene_id, wc) {
            warn!(error = %e, scene_id, "ignoring invalid saved camera");
        }
    }
}

/// Helper for components that want to know whether the scene id
/// they last saw has changed (used to apply persisted cameras).
pub fn use_scene_id_change<F>(mut on_change: F)
where
    F: FnMut(Option<&str>, Option<&str>) + 'static,
{
    let game = use_stellar_scene();
    let version = use_stellar_scene_version();
    let mut prev = use_signal(|| Option::<String>::None);
    use_effect(move || {
        let _ = version();
        let current = game.read().active_scene().map(|w| w.id.clone());
        let prev_val = prev.peek().clone();
        let curr_val = current.clone();
        if prev_val != curr_val {
            on_change(prev_val.as_deref(), curr_val.as_deref());
            prev.set(current);
        }
    });
}

/// Keep the per-scene camera persistence in sync after the camera
/// moves. Records the current camera into the client’s in-memory
/// `scene_cameras` map on every version bump. The actual
/// `localStorage` writing is handled by
/// [`use_provide_scene_camera_persistence`].
pub fn use_persist_scene_camera() {
    let game = use_stellar_scene();
    let version = use_stellar_scene_version();

    use_effect(move || {
        let _ = version();
        let g = game.read();
        if let Some(w) = g.active_scene() {
            if let Err(e) = g.remember_current_camera_for(&w.id) {
                warn!(error = %e, "remember_current_camera_for failed");
            }
        }
    });
}

/// Pipeline response for the currently selected star. Re-renders
/// the calling component on every scene mutation.
pub fn use_pipeline_snapshot() -> Option<lunar_structures::PipelineResponse> {
    let game = use_stellar_scene();
    let version = use_stellar_scene_version();
    let _ = version();
    game.read().pipeline()
}

/// Fetch the pipeline for the currently selected star, if any. The
/// actual work is done inside the stellar-scene client; this is a thin helper
/// for components that want to fire-and-forget.
#[allow(dead_code)]
pub fn fetch_pipeline_for_selected(game: &StellarScene) {
    if let Some(star) = game.selected_star() {
        let game = game.clone();
        spawn(async move {
            let _ = game.fetch_pipeline(star).await;
        });
    }
}
