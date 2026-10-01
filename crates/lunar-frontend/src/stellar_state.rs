
use dioxus::prelude::*;
use lunar_stellar_core::{SceneCamera, StellarScene, StellarSceneConfig, StellarSceneSnapshot};
use tracing::warn;

use crate::local_storage;
use crate::runtime_config::RuntimeConfig;

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

pub fn use_provide_stellar_scene_with(initial: StellarScene) -> Signal<StellarScene> {
    let version = use_signal(|| 0u64);
    let game = use_signal(|| initial);

    use_setup_stellar_scene_listener(game, version);

    use_context_provider(|| game);
    use_context_provider(|| version);
    game
}

pub fn use_stellar_scene() -> Signal<StellarScene> {
    use_context::<Signal<StellarScene>>()
}

pub fn use_stellar_scene_version() -> Signal<u64> {
    use_context::<Signal<u64>>()
}

pub fn use_stellar_scene_snapshot() -> StellarSceneSnapshot {
    let version = use_stellar_scene_version();
    let game = use_stellar_scene();
    let _ = version();
    game.read().snapshot()
}

const CAMERA_PERSIST_DEBOUNCE_MS: u32 = 350;

async fn wait_for_camera_persist_debounce() {
    #[cfg(feature = "web")]
    gloo_timers::future::TimeoutFuture::new(CAMERA_PERSIST_DEBOUNCE_MS).await;

    #[cfg(not(feature = "web"))]
    tokio::time::sleep(std::time::Duration::from_millis(
        CAMERA_PERSIST_DEBOUNCE_MS.into(),
    ))
    .await;
}

fn persist_camera_debounced(
    scene_id: String,
    camera: SceneCamera,
    generation: u64,
    latest_generation: Signal<u64>,
) {
    spawn(async move {
        wait_for_camera_persist_debounce().await;
        if *latest_generation.peek() == generation {
            local_storage::save_scene_camera(&scene_id, camera);
        }
    });
}

pub fn use_provide_scene_camera_persistence() {
    let game = use_stellar_scene();
    let version = use_stellar_scene_version();
    let mut latest_generation = use_signal(|| 0_u64);

    use_effect(move || {
        let _ = version();
        let snap = game.read().snapshot();
        let Some(scene_id) = snap.active_scene_id().map(str::to_owned) else {
            return;
        };
        let camera = SceneCamera::new(snap.camera.offset, snap.camera.zoom);
        let generation = latest_generation.peek().wrapping_add(1);
        latest_generation.set(generation);
        persist_camera_debounced(scene_id, camera, generation, latest_generation);
    });
}

pub fn restore_camera(game: &StellarScene, scene_id: &str) {
    let Some(camera) = local_storage::load_scene_camera(scene_id) else {
        return;
    };
    if let Err(error) = game.set_scene_camera(scene_id, camera) {
        warn!(%error, scene_id, "discarding invalid saved camera");
        local_storage::remove_scene_camera(scene_id);
        return;
    }
    if let Err(error) = game.apply_scene_camera(scene_id) {
        warn!(%error, scene_id, "failed to apply restored scene camera");
    }
}

pub fn clear_selection(game: &StellarScene) {
    game.select_star(None);
}

pub fn export_camera_snapshot(game: &StellarScene) -> ((f32, f32), f32, Option<u64>) {
    let cam = game.camera();
    let star_id = game.selected_star().map(|s| s.id as u64);
    (cam.offset, cam.zoom, star_id)
}

pub fn apply_camera_snapshot(game: &StellarScene, offset: (f32, f32), zoom: f32) {
    let mut cam = game.camera();
    cam.offset = offset;
    cam.zoom = zoom;
    game.set_camera(cam);
}

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

pub fn use_pipeline_snapshot() -> Option<lunar_structures::PipelineResponse> {
    let game = use_stellar_scene();
    let version = use_stellar_scene_version();
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
