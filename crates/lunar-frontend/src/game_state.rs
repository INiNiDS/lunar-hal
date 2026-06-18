//! Reactivity bridge between [`lunar_game_backend::Game`] and Dioxus.
//!
//! The game layer is framework-agnostic. To make it play nicely with
//! Dioxus's reactive system, this module installs a version signal
//! in the current context and pipes the game's internal change
//! notifications (delivered via `tokio::sync::watch`) into that
//! signal. Components that need to re-render read the version (or a
//! [`GameSnapshot`]) and Dioxus tracks the dependency for us.

use dioxus::prelude::*;
use lunar_game_backend::{Game, GameSnapshot};

use crate::local_storage;

/// Provide a [`Game`] instance to the component subtree. Wraps the
/// game in a Dioxus context, installs a `Signal<u64>` version
/// counter, and starts a coroutine that bumps the counter on every
/// game mutation.
///
/// All hooks are called at the top level of this function (never
/// nested inside another hook's closure) to comply with the rules of
/// hooks. The version-bumping task is started exactly once via
/// [`use_hook`], whose initializer only schedules the future and does
/// not itself call any hooks.
#[allow(clippy::future_not_send, clippy::used_underscore_binding)]
pub fn provide_game() -> Signal<Game> {
    let version = use_signal(|| 0u64);
    let game = use_signal(Game::new);
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
    use_context_provider(|| game);
    use_context_provider(|| version);
    game
}

/// Provide a pre-configured [`Game`] instance. Useful when a host
/// application already constructed a game and wants to inject it
/// into the Dioxus context.
#[allow(
    dead_code,
    clippy::future_not_send,
    clippy::used_underscore_binding
)]
pub fn provide_game_with(initial: Game) -> Signal<Game> {
    let version = use_signal(|| 0u64);
    let game = use_signal(|| initial);
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
    use_context_provider(|| game);
    use_context_provider(|| version);
    game
}

/// Borrow the shared [`Game`] from the current context.
pub fn use_game() -> Signal<Game> {
    use_context::<Signal<Game>>()
}

/// Read the shared version signal. Components that need to re-render
/// after a game mutation should call this first to subscribe.
pub fn use_game_version() -> Signal<u64> {
    use_context::<Signal<u64>>()
}

/// Reactive snapshot of the entire game state. Subscribes to the
/// version signal so the calling component re-renders on every
/// mutation.
pub fn use_game_snapshot() -> GameSnapshot {
    let version = use_game_version();
    let game = use_game();
    let _ = version();
    game.read().snapshot()
}

/// Install a per-world camera persistence hook that mirrors the
/// game's `world_cameras` map into `localStorage` (web) or a no-op
/// stub (desktop). Call this once near the top of the editor.
pub fn provide_world_camera_persistence() {
    let game = use_game();
    let version = use_game_version();

    use_effect(move || {
        let _ = version();
        let snap = game.read().snapshot();
        if let Some(id) = snap.active_world_id()
            && let Some(wc) = snap.world_cameras.get(id).copied()
        {
            local_storage::save_world_camera(id, wc);
        }
    });
}

pub fn hydrate_world_camera_from_storage(game: &Game, world_id: &str) {
    if let Some(wc) = local_storage::load_world_camera(world_id) {
        if let Err(e) = game.set_world_camera(world_id, wc) {
            tracing::warn!(error = %e, world_id, "ignoring invalid saved camera");
        }
    }
}

/// Helper for components that want to know whether the world id
/// they last saw has changed (used to apply persisted cameras).
pub fn use_world_id_change<F>(mut on_change: F)
where
    F: FnMut(Option<&str>, Option<&str>) + 'static,
{
    let game = use_game();
    let version = use_game_version();
    let mut prev = use_signal(|| Option::<String>::None);
    use_effect(move || {
        let _ = version();
        let current = game.read().active_world().map(|w| w.id.clone());
        let prev_val = prev.peek().clone();
        let curr_val = current.clone();
        if prev_val != curr_val {
            on_change(prev_val.as_deref(), curr_val.as_deref());
            prev.set(current);
        }
    });
}

/// Keep the per-world camera persistence in sync after the camera
/// moves. Records the current camera into the game's in-memory
/// `world_cameras` map on every version bump. The actual
/// `localStorage` write is handled by
/// [`provide_world_camera_persistence`].
pub fn use_persist_world_camera() {
    let game = use_game();
    let version = use_game_version();

    use_effect(move || {
        let _ = version();
        let g = game.read();
        if let Some(w) = g.active_world()
            && let Err(e) = g.remember_current_camera_for(&w.id)
        {
            tracing::warn!(error = %e, "remember_current_camera_for failed");
        }
    });
}

/// Pipeline response for the currently selected star. Re-renders
/// the calling component on every game mutation.
pub fn use_pipeline_snapshot() -> Option<lunar_structures::PipelineResponse> {
    let game = use_game();
    let version = use_game_version();
    let _ = version();
    game.read().pipeline()
}

/// Fetch the pipeline for the currently selected star, if any. The
/// actual work is done inside the game layer; this is a thin helper
/// for components that want to fire-and-forget.
#[allow(dead_code)]
pub fn fetch_pipeline_for_selected(game: &Game) {
    if let Some(star) = game.selected_star() {
        let game = game.clone();
        spawn(async move {
            let _ = game.fetch_pipeline(star).await;
        });
    }
}
