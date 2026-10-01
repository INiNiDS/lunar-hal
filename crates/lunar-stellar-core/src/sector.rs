
use std::collections::{HashMap, HashSet};

use lunar_structures::ResponseStar;

pub const CHUNK_SIZE_PC: f32 = 400.0;

pub const PX_PER_PC: f32 = 15.0;

pub const INNER_EXCLUSION_PC: f32 = 450.0;

pub const MAX_CACHED_CHUNKS: usize = 64;

pub const PREFETCH_PAD_CHUNKS: i32 = 1;

pub const MAX_CONCURRENT_FETCHES: usize = 3;

pub const FETCH_COOLDOWN_MS: u32 = 120;

pub const MIN_FETCH_COOLDOWN_MS: u32 = FETCH_COOLDOWN_MS;

pub const MIN_FETCH_OBJECTS: usize = 10;

pub const MIN_FETCH_RECORDS: usize = 10;

pub const FIELD_HALF: i32 = 18000;

pub type SectorKey = (i32, i32);

pub fn chunk_center(chunk: SectorKey) -> (f32, f32) {
    (
        (chunk.0 as f32 + 0.5) * CHUNK_SIZE_PC,
        (chunk.1 as f32 + 0.5) * CHUNK_SIZE_PC,
    )
}

pub fn chunk_distance_sq(chunk: SectorKey, target: (f32, f32)) -> f32 {
    let (cx, cy) = chunk_center(chunk);
    let dx = cx - target.0;
    let dy = cy - target.1;
    dx * dx + dy * dy
}

fn is_excluded(chunk: SectorKey, center: (f32, f32)) -> bool {
    chunk_distance_sq(chunk, center) < INNER_EXCLUSION_PC * INNER_EXCLUSION_PC
}

pub fn scene_point_under_center(
    camera: (f32, f32),
    zoom: f32,
    scene_origin: (f32, f32),
) -> (f32, f32) {
    if zoom <= 0.0 {
        return scene_origin;
    }
    let cam_scene_x = scene_origin.0 - camera.0 / (zoom * PX_PER_PC);
    let cam_scene_y = scene_origin.1 - camera.1 / (zoom * PX_PER_PC);
    (cam_scene_x, cam_scene_y)
}

pub fn visible_chunks(
    cam_offset: (f32, f32),
    cam_zoom: f32,
    viewport: (f32, f32),
    scene_center: (f32, f32),
) -> Vec<SectorKey> {
    if viewport.0 <= 0.0 || viewport.1 <= 0.0 || cam_zoom <= 0.0 {
        return Vec::new();
    }

    let center_x = scene_center.0 - cam_offset.0 / (cam_zoom * PX_PER_PC);
    let center_y = scene_center.1 - cam_offset.1 / (cam_zoom * PX_PER_PC);
    let half_w = (viewport.0 * 0.5) / (cam_zoom * PX_PER_PC);
    let half_h = (viewport.1 * 0.5) / (cam_zoom * PX_PER_PC);

    let min_cx = ((center_x - half_w) / CHUNK_SIZE_PC).floor() as i32 - PREFETCH_PAD_CHUNKS;
    let max_cx = ((center_x + half_w) / CHUNK_SIZE_PC).floor() as i32 + PREFETCH_PAD_CHUNKS;
    let lo_cy = ((center_y - half_h) / CHUNK_SIZE_PC).floor() as i32 - PREFETCH_PAD_CHUNKS;
    let hi_cy = ((center_y + half_h) / CHUNK_SIZE_PC).floor() as i32 + PREFETCH_PAD_CHUNKS;

    let mut chunks = Vec::new();
    for cx in min_cx..=max_cx {
        for cy in lo_cy..=hi_cy {
            chunks.push((cx, cy));
        }
    }
    chunks
}

pub fn evict_excess_cache_preserving(
    cache: &mut HashMap<SectorKey, Vec<ResponseStar>>,
    cam_pos: (f32, f32),
    protected: &HashSet<SectorKey>,
) {
    if cache.len() <= MAX_CACHED_CHUNKS {
        return;
    }

    let mut removable: Vec<SectorKey> = cache
        .keys()
        .copied()
        .filter(|key| !protected.contains(key))
        .collect();
    removable.sort_unstable_by(|&a, &b| {
        let dist_a = chunk_distance_sq(a, cam_pos);
        let dist_b = chunk_distance_sq(b, cam_pos);
        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let to_remove = cache.len() - MAX_CACHED_CHUNKS;
    for key in removable.iter().rev().take(to_remove) {
        cache.remove(key);
    }
}

pub fn evict_excess_cache(cache: &mut HashMap<SectorKey, Vec<ResponseStar>>, cam_pos: (f32, f32)) {
    evict_excess_cache_preserving(cache, cam_pos, &HashSet::new());
}

#[derive(Clone, Copy, Debug)]
pub struct SectorFetchRequest {
    pub viewport: (f32, f32),
    pub cam_offset: (f32, f32),
    pub cam_zoom: f32,
    pub scene_center: (f32, f32, f32),
}

pub fn streaming_chunks(req: SectorFetchRequest) -> Vec<SectorKey> {
    let scene_center = (req.scene_center.0, req.scene_center.1);
    let viewport_center = scene_point_under_center(req.cam_offset, req.cam_zoom, scene_center);
    let mut visible = visible_chunks(req.cam_offset, req.cam_zoom, req.viewport, scene_center);
    visible.retain(|&chunk| !is_excluded(chunk, scene_center));
    visible.sort_unstable_by(|&a, &b| {
        chunk_distance_sq(a, viewport_center)
            .partial_cmp(&chunk_distance_sq(b, viewport_center))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    visible.truncate(MAX_CACHED_CHUNKS);
    visible
}

pub fn sectors_to_fetch(
    req: SectorFetchRequest,
    cache: &HashMap<SectorKey, Vec<ResponseStar>>,
    loading: &HashSet<SectorKey>,
) -> Vec<(SectorKey, (f32, f32, f32))> {
    let available_slots = MAX_CONCURRENT_FETCHES.saturating_sub(loading.len());
    if available_slots == 0 {
        return Vec::new();
    }

    let mut visible = streaming_chunks(req);
    visible.retain(|chunk| !cache.contains_key(chunk) && !loading.contains(chunk));
    visible.truncate(available_slots);

    visible
        .into_iter()
        .map(|chunk| {
            let (cx, cy) = chunk_center(chunk);
            (chunk, (cx, cy, req.scene_center.2))
        })
        .collect()
}

pub fn eviction_cam_pos(
    cam_offset: (f32, f32),
    cam_zoom: f32,
    scene_center: (f32, f32),
) -> (f32, f32) {
    scene_point_under_center(cam_offset, cam_zoom, scene_center)
}

pub fn chunk_at_scene_point(point: (f32, f32)) -> Option<SectorKey> {
    if !point.0.is_finite() || !point.1.is_finite() {
        return None;
    }
    let cx = (point.0 / CHUNK_SIZE_PC).floor() as i32;
    let cy = (point.1 / CHUNK_SIZE_PC).floor() as i32;
    if cx < crate::validation::limits::SECTOR_KEY_MIN
        || cx > crate::validation::limits::SECTOR_KEY_MAX
        || cy < crate::validation::limits::SECTOR_KEY_MIN
        || cy > crate::validation::limits::SECTOR_KEY_MAX
    {
        return None;
    }
    Some((cx, cy))
}
