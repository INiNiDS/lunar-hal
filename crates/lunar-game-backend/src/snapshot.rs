//! Read-only view of the game state.
//!
//! Frontends get a [`GameSnapshot`] from [`crate::Game::snapshot`]
//! and render based on it. Snapshots are lightweight to clone (no Arc) and
//! immutable, so they can be passed freely between components.

use std::collections::{HashMap, HashSet};

use lunar_structures::{GnnResponse, PipelineResponse, ResponseStar, World, WorldSummary};

use crate::attention::AttentionEntry;
use crate::camera::Camera;
use crate::sector::SectorKey;

/// Immutable view of the world as the game currently sees it.
#[derive(Clone, Debug, Default)]
pub struct GameSnapshot {
    /// All worlds the game knows about (server-authoritative list).
    pub worlds: Vec<WorldSummary>,
    /// The world currently being explored, if any.
    pub active_world: Option<World>,
    /// Cached sector stars, flattened for rendering.
    pub sector_stars: Vec<ResponseStar>,
    /// Chunks that are currently in flight.
    pub sector_loading: HashSet<SectorKey>,
    /// Per-chunk sector stars (the raw cache, in case the renderer
    /// wants to differentiate by chunk).
    pub sector_cache: HashMap<SectorKey, Vec<ResponseStar>>,
    /// Current camera.
    pub camera: Camera,
    /// The currently selected star, if any.
    pub selected_star: Option<ResponseStar>,
    /// Pre-generated sector used when no world is active.
    pub pregen: Option<GnnResponse>,
    /// Entropy / temperature used for the current pregen.
    pub temperature: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    /// World-space center of the currently visible region.
    pub sector_center: Option<(f32, f32, f32)>,
    /// Pipeline response for the selected star, if any.
    pub pipeline: Option<PipelineResponse>,
    /// Per-world camera persistence entries.
    pub world_cameras: HashMap<String, crate::camera::WorldCamera>,
    /// Карта внимания игрока: для каждой звезды — время невнимания
    /// и расстояние до курсора мыши.
    pub attention_map: HashMap<u32, AttentionEntry>,
}

impl GameSnapshot {
    /// Effective world-space center: prefer the active world, fall
    /// back to the sector center, then to the origin.
    pub fn effective_center(&self) -> (f32, f32, f32) {
        if let Some(w) = &self.active_world {
            (w.center_x, w.center_y, w.center_z)
        } else if let Some(c) = self.sector_center {
            c
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// World center projected to XY for sector streaming.
    pub fn effective_center_xy(&self) -> (f32, f32) {
        let c = self.effective_center();
        (c.0, c.1)
    }

    pub fn active_world_id(&self) -> Option<&str> {
        self.active_world.as_ref().map(|w| w.id.as_str())
    }
}
