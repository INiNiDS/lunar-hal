use lunar_stellar_core::sector::{
    CHUNK_SIZE_PC, INNER_EXCLUSION_PC, MAX_CACHED_CHUNKS, PX_PER_PC, chunk_center,
    chunk_distance_sq, evict_excess_cache, sectors_to_fetch, visible_chunks,
};
use lunar_stellar_core::{StellarScene, StellarSceneConfig};
use lunar_structures::ResponseStar;
use std::collections::{HashMap, HashSet};

fn star_at(x: f32, y: f32, z: f32) -> ResponseStar {
    ResponseStar {
        id: 0,
        x,
        y,
        z,
        temperature_k: 5778.0,
        radius: 1.0,
        mass: 1.0,
        luminosity: 1.0,
        description: String::new(),
        name: String::new(),
        type_hint: String::new(),
        velocity_vector: [0.0, 0.0, 0.0],
    }
}

#[test]
fn visible_chunks_returns_empty_for_zero_viewport() {
    let chunks = visible_chunks((0.0, 0.0), 1.0, (0.0, 0.0), (0.0, 0.0));
    assert!(chunks.is_empty());
}

#[test]
fn visible_chunks_centers_around_camera() {
    // Camera at origin looking at scene (0,0); zoom 1.0; viewport 4000x4000.
    // We should see one chunk (the one at the origin).
    let chunks = visible_chunks((0.0, 0.0), 1.0, (4000.0, 4000.0), (0.0, 0.0));
    assert!(!chunks.is_empty());
    assert!(chunks.contains(&(0, 0)));
}

#[test]
fn visible_chunks_excludes_chunks_inside_inner_radius() {
    // StarScene center at (0,0). Chunks within INNER_EXCLUSION_PC of the
    // center are skipped by `sectors_to_fetch`. The chunk at (0,0)
    // has its center at CHUNK_SIZE_PC/2 = 200, which is <
    // INNER_EXCLUSION_PC = 450, so it is excluded.
    let center_x = 0.5 * CHUNK_SIZE_PC;
    let center_y = 0.5 * CHUNK_SIZE_PC;
    assert!(center_x < INNER_EXCLUSION_PC);
    assert!(center_y < INNER_EXCLUSION_PC);
    let mut cache = HashMap::new();
    let loading = HashSet::new();
    cache.insert((10, 10), vec![star_at(0.0, 0.0, 0.0)]);
    let request = lunar_stellar_core::sector::SectorFetchRequest {
        viewport: (10000.0, 10000.0),
        cam_offset: (0.0, 0.0),
        cam_zoom: 1.0,
        scene_center: (0.0, 0.0, 0.0),
    };
    let to_fetch = sectors_to_fetch(request, &cache, &loading);
    assert!(!to_fetch.iter().any(|(c, _)| *c == (0, 0)));
}

#[test]
fn chunk_center_is_offset_by_half_chunk() {
    let (cx, cy) = chunk_center((0, 0));
    assert!((cx - 0.5 * CHUNK_SIZE_PC).abs() < 1e-3);
    assert!((cy - 0.5 * CHUNK_SIZE_PC).abs() < 1e-3);
}

#[test]
fn chunk_distance_sq_matches_euclidean() {
    let a = (1, 2);
    let target = (4500.0_f32, -1200.0_f32);
    let (cx, cy) = chunk_center(a);
    let dx = cx - target.0;
    let dy = cy - target.1;
    let expected = dx * dx + dy * dy;
    let actual = chunk_distance_sq(a, target);
    assert!((actual - expected).abs() < 1e-3);
    assert!(actual > 0.0);
}

#[test]
fn sectors_to_fetch_skips_cached_and_loading() {
    let mut cache = HashMap::new();
    cache.insert((10, 10), vec![star_at(0.0, 0.0, 0.0)]);
    let mut loading = HashSet::new();
    loading.insert((20, 20));

    let request = lunar_stellar_core::sector::SectorFetchRequest {
        viewport: (5000.0, 5000.0),
        cam_offset: (0.0, 0.0),
        cam_zoom: 1.0,
        scene_center: (0.0, 0.0, 0.0),
    };

    let to_fetch = sectors_to_fetch(request, &cache, &loading);
    // We can't assert specific chunk coordinates because of the
    // inner-exclusion filter, but the cached/loading chunks must be
    // absent.
    assert!(!to_fetch.iter().any(|(c, _)| *c == (10, 10)));
    assert!(!to_fetch.iter().any(|(c, _)| *c == (20, 20)));
}

#[test]
fn evict_excess_cache_keeps_only_closest_chunks() {
    let mut cache = HashMap::new();
    for i in 0..(MAX_CACHED_CHUNKS + 10) {
        cache.insert((i as i32, 0), vec![star_at(0.0, 0.0, 0.0)]);
    }
    let cam_pos = chunk_center((0, 0));
    evict_excess_cache(&mut cache, cam_pos);
    assert_eq!(cache.len(), MAX_CACHED_CHUNKS);
    // The farthest chunks must have been removed.
    assert!(!cache.contains_key(&((MAX_CACHED_CHUNKS as i32) + 5, 0)));
}

#[test]
fn px_per_pc_is_positive() {
    assert!(PX_PER_PC > 0.0);
}

#[test]
fn scene_can_be_constructed_with_default_config() {
    let _game = StellarScene::new();
}

#[test]
fn scene_can_be_constructed_with_explicit_config() {
    let _game = StellarScene::with_config(StellarSceneConfig::new("http://127.0.0.1:1"));
}

#[test]
fn scene_snapshot_reflects_camera_changes() {
    let game = StellarScene::new();
    assert_eq!(game.snapshot().camera.zoom, 1.0);
    game.set_camera(lunar_stellar_core::Camera {
        offset: (10.0, 20.0),
        zoom: 2.0,
        dragging: false,
    });
    let snap = game.snapshot();
    assert_eq!(snap.camera.offset, (10.0, 20.0));
    assert!((snap.camera.zoom - 2.0).abs() < 1e-6);
}

#[test]
fn scene_camera_zoom_around_center_preserves_scene_point() {
    let camera = lunar_stellar_core::Camera {
        offset: (0.0, 0.0),
        zoom: 1.0,
        dragging: false,
    };
    let new_camera = camera.zoom_around_center((1000.0, 1000.0), 2.0);
    assert!((new_camera.zoom - 2.0).abs() < 1e-6);

    // The scene point that was under the viewport center before the
    // zoom must still be under the center afterward.
    let viewport_center = (500.0_f32, 500.0_f32);
    let before_x = (viewport_center.0 - camera.offset.0) / (camera.zoom * PX_PER_PC);
    let before_y = (viewport_center.1 - camera.offset.1) / (camera.zoom * PX_PER_PC);
    let after_x = (viewport_center.0 - new_camera.offset.0) / (new_camera.zoom * PX_PER_PC);
    let after_y = (viewport_center.1 - new_camera.offset.1) / (new_camera.zoom * PX_PER_PC);
    assert!((before_x - after_x).abs() < 1e-3);
    assert!((before_y - after_y).abs() < 1e-3);
}

#[test]
fn adopt_scene_clears_sector_cache() {
    let game = StellarScene::new();
    let chunk = (5, 5);
    game.apply_sector(chunk, vec![star_at(1.0, 2.0, 3.0)]);
    assert!(game.snapshot().sector_cache.contains_key(&chunk));

    let scene = lunar_structures::StarScene {
        id: "test".into(),
        name: "Test".into(),
        created_at: 0,
        center_x: 0.0,
        center_y: 0.0,
        center_z: 0.0,
        temperature: 0.5,
        bp_rp: 1.0,
        g_mag: 5.0,
        stars: vec![],
    };
    game.adopt_scene(Some(scene));
    assert!(!game.snapshot().sector_cache.contains_key(&chunk));
}

#[test]
fn scene_camera_persists_in_memory() {
    let game = StellarScene::new();
    game.set_scene_camera("foo", lunar_stellar_core::SceneCamera::new((1.0, 2.0), 1.5))
        .unwrap();
    let snap = game.snapshot();
    let wc = snap.scene_cameras.get("foo").copied();
    assert!(wc.is_some());
    let wc = wc.unwrap();
    assert_eq!(wc.offset, (1.0, 2.0));
    assert!((wc.zoom - 1.5).abs() < 1e-6);
}

#[test]
fn apply_scene_camera_copies_saved_camera_to_current() {
    let game = StellarScene::new();
    game.set_scene_camera("foo", lunar_stellar_core::SceneCamera::new((3.0, 4.0), 2.5))
        .unwrap();
    assert!(game.apply_scene_camera("foo").unwrap());
    let cam = game.camera();
    assert_eq!(cam.offset, (3.0, 4.0));
    assert!((cam.zoom - 2.5).abs() < 1e-6);
}

#[test]
fn apply_scene_camera_returns_false_for_unknown_scene() {
    let game = StellarScene::new();
    assert!(!game.apply_scene_camera("does-not-exist").unwrap());
}
