
use dioxus::prelude::*;
use lunar_stellar_core::{StellarScene, GameSnapshot};
use tracing::warn;

use crate::local_storage;

fn use_setup_game_listener(game: Signal<StellarScene>, version: Signal<u64>) {
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

#[allow(clippy::future_not_send, clippy::used_underscore_binding)]
pub fn use_provide_game() -> Signal<StellarScene> {
    let version = use_signal(|| 0u64);
    let game = use_signal(StellarScene::new);

    use_setup_game_listener(game, version);

    use_context_provider(|| game);
    use_context_provider(|| version);
    game
}

#[allow(dead_code)]
pub fn use_provide_game_with(initial: StellarScene) -> Signal<StellarScene> {
    let version = use_signal(|| 0u64);
    let game = use_signal(|| initial);

    use_setup_game_listener(game, version);

    use_context_provider(|| game);
    use_context_provider(|| version);
    game
}

pub fn use_game() -> Signal<StellarScene> {
    use_context::<Signal<StellarScene>>()
}

pub fn use_game_version() -> Signal<u64> {
    use_context::<Signal<u64>>()
}

pub fn use_game_snapshot() -> GameSnapshot {
    let version = use_game_version();
    let game = use_game();
    let _ = version();
    game.read().snapshot()
}

pub fn use_provide_world_camera_persistence() {
    let game = use_game();
    let version = use_game_version();

    use_effect(move || {
        let _ = version();
        let snap = game.read().snapshot();
        if let Some(id) = snap.active_world_id() {
            if let Some(wc) = snap.world_cameras.get(id).copied() {
                local_storage::save_world_camera(id, wc);
            }
        }
    });
}

pub fn hydrate_world_camera_from_storage(game: &StellarScene, world_id: &str) {
    if let Some(wc) = local_storage::load_world_camera(world_id) {
        if let Err(e) = game.set_world_camera(world_id, wc) {
            warn!(error = %e, world_id, "ignoring invalid saved camera");
        }
    }
}

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

pub fn use_persist_world_camera() {
    let game = use_game();
    let version = use_game_version();

    use_effect(move || {
        let _ = version();
        let g = game.read();
        if let Some(w) = g.active_world() {
            if let Err(e) = g.remember_current_camera_for(&w.id) {
                warn!(error = %e, "remember_current_camera_for failed");
            }
        }
    });
}

pub fn use_pipeline_snapshot() -> Option<lunar_structures::PipelineResponse> {
    let game = use_game();
    let version = use_game_version();
    let _ = version();
    game.read().pipeline()
}

#[allow(dead_code)]
pub fn fetch_pipeline_for_selected(game: &StellarScene) {
    if let Some(star) = game.selected_star() {
        let game = game.clone();
        spawn(async move {
            let _ = game.fetch_pipeline(star).await;
        });
    }
}
