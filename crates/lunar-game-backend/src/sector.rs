//! Sector/chunk streaming rules.
//!
//! The world is divided into fixed-size square chunks (in parsecs).
//! `Game` uses the functions in this module to decide which chunks
//! must be fetched, which are in-flight, and which can be evicted.
//! None of this logic lives in the frontend anymore.

use std::collections::{HashMap, HashSet};

use lunar_structures::ResponseStar;

/// Size of one streaming chunk along each axis (in parsecs).
pub const CHUNK_SIZE_PC: f32 = 400.0;

/// Pixels per parsec at zoom = 1.0. Frontends read this to scale
/// world coordinates into the viewport.
pub const PX_PER_PC: f32 = 15.0;

/// Chunks whose center is closer than this to the world origin are
/// skipped — the world itself occupies that space.
pub const INNER_EXCLUSION_PC: f32 = 450.0;

/// Hard upper bound on simultaneously cached chunks.
pub const MAX_CACHED_CHUNKS: usize = 64;

/// Extra chunks of padding around the visible viewport that should
/// also be prefetched (in chunk units).
pub const PREFETCH_PAD_CHUNKS: i32 = 1;

/// Maximum number of concurrent in-flight sector requests.
pub const MAX_CONCURRENT_FETCHES: usize = 3;

/// Minimum delay between dispatched fetches, in milliseconds. Used
/// by the frontend to throttle reflows.
pub const FETCH_COOLDOWN_MS: u32 = 120;

/// Same as [`FETCH_COOLDOWN_MS`] but named in PascalCase for
/// re-export consistency.
pub const MIN_FETCH_COOLDOWN_MS: u32 = FETCH_COOLDOWN_MS;

pub const FIELD_HALF: i32 = 18000;

/// Integer coordinate of a chunk in chunk-space.
pub type SectorKey = (i32, i32);

/// Center of a chunk in world coordinates (parsecs).
pub fn chunk_center(chunk: SectorKey) -> (f32, f32) {
    (
        (chunk.0 as f32 + 0.5) * CHUNK_SIZE_PC,
        (chunk.1 as f32 + 0.5) * CHUNK_SIZE_PC,
    )
}

/// Squared distance from a chunk's center to a target point.
pub fn chunk_distance_sq(chunk: SectorKey, target: (f32, f32)) -> f32 {
    let (cx, cy) = chunk_center(chunk);
    let dx = cx - target.0;
    let dy = cy - target.1;
    dx * dx + dy * dy
}

fn is_excluded(chunk: SectorKey, center: (f32, f32)) -> bool {
    chunk_distance_sq(chunk, center) < INNER_EXCLUSION_PC * INNER_EXCLUSION_PC
}

/// Compute the world-space point currently under the viewport center
/// for a given camera and world origin.
pub fn world_point_under_center(
    camera: (f32, f32),
    zoom: f32,
    world_origin: (f32, f32),
) -> (f32, f32) {
    if zoom <= 0.0 {
        return world_origin;
    }
    let cam_world_x = world_origin.0 - camera.0 / (zoom * PX_PER_PC);
    let cam_world_y = world_origin.1 - camera.1 / (zoom * PX_PER_PC);
    (cam_world_x, cam_world_y)
}

/// Compute the set of chunks visible in the viewport, with optional
/// padding for prefetching.
pub fn visible_chunks(
    cam_offset: (f32, f32),
    cam_zoom: f32,
    viewport: (f32, f32),
    world_center: (f32, f32),
) -> Vec<SectorKey> {
    if viewport.0 <= 0.0 || viewport.1 <= 0.0 || cam_zoom <= 0.0 {
        return Vec::new();
    }

    let center_x = world_center.0 - cam_offset.0 / (cam_zoom * PX_PER_PC);
    let center_y = world_center.1 - cam_offset.1 / (cam_zoom * PX_PER_PC);
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

/// Evict the farthest cached chunks until the cache is at or below
/// [`MAX_CACHED_CHUNKS`]. Eviction is by squared distance from the
/// camera's current world position.
pub fn evict_excess_cache(
    cache: &mut HashMap<SectorKey, Vec<ResponseStar>>,
    cam_pos: (f32, f32),
) {
    if cache.len() <= MAX_CACHED_CHUNKS {
        return;
    }
    let mut keys: Vec<SectorKey> = cache.keys().copied().collect();
    keys.sort_unstable_by(|&a, &b| {
        let dist_a = chunk_distance_sq(a, cam_pos);
        let dist_b = chunk_distance_sq(b, cam_pos);
        dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let to_remove = keys.len() - MAX_CACHED_CHUNKS;
    for key in keys.iter().rev().take(to_remove) {
        cache.remove(key);
    }
}

/// Inputs for deciding which sectors to fetch this frame.
#[derive(Clone, Copy, Debug)]
pub struct SectorFetchRequest {
    pub viewport: (f32, f32),
    pub cam_offset: (f32, f32),
    pub cam_zoom: f32,
    /// World-space center of the visible region (typically the active
    /// world's center, or the last pregen sector center).
    pub world_center: (f32, f32, f32),
}

/// Decide which sectors are currently visible but neither cached
/// nor in-flight. Returns at most `MAX_CONCURRENT_FETCHES` chunks to
/// keep the request rate bounded.
pub fn sectors_to_fetch(
    req: SectorFetchRequest,
    cache: &HashMap<SectorKey, Vec<ResponseStar>>,
    loading: &HashSet<SectorKey>,
) -> Vec<(SectorKey, (f32, f32, f32))> {
    let mut visible = visible_chunks(
        req.cam_offset,
        req.cam_zoom,
        req.viewport,
        (req.world_center.0, req.world_center.1),
    );

    visible.retain(|&chunk| {
        !is_excluded(chunk, (req.world_center.0, req.world_center.1))
            && !cache.contains_key(&chunk)
            && !loading.contains(&chunk)
    });

    visible.truncate(MAX_CONCURRENT_FETCHES);

    visible
        .into_iter()
        .map(|chunk| {
            let (cx, cy) = chunk_center(chunk);
            (chunk, (cx, cy, req.world_center.2))
        })
        .collect()
}

/// Compute the world-space camera position used for cache eviction.
pub fn eviction_cam_pos(
    cam_offset: (f32, f32),
    cam_zoom: f32,
    world_center: (f32, f32),
) -> (f32, f32) {
    world_point_under_center(cam_offset, cam_zoom, world_center)
}

/// Map a world-space point (in parsecs) to the [`SectorKey`] that
/// contains it. Returns `None` when the key is outside the playable
/// map or the coordinate is not finite.
pub fn chunk_at_world_point(point: (f32, f32)) -> Option<SectorKey> {
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
