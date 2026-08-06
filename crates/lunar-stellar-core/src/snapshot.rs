//! Read-only view of stellar-scene state.
//!
//! Frontends get a [`StellarSceneSnapshot`] from [`crate::StellarScene::snapshot`]
//! and render based on it. Snapshots are lightweight to clone (no Arc) and
//! immutable, so they can be passed freely between components.

use std::collections::{HashMap, HashSet};

use lunar_structures::{GnnResponse, PipelineResponse, ResponseStar, StarScene, StarSceneSummary};

use crate::camera::Camera;
use crate::sector::SectorKey;

/// Immutable view of the scene as the stellar-scene client currently sees it.
#[derive(Clone, Debug, Default)]
pub struct StellarSceneSnapshot {
    /// All scenes the client knows about (server-authoritative list).
    pub scenes: Vec<StarSceneSummary>,
    /// The scene currently being explored, if any.
    pub active_scene: Option<StarScene>,
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
    /// Pre-generated sector used when no scene is active.
    pub pregen: Option<GnnResponse>,
    /// Entropy / temperature used for the current pregen.
    pub temperature: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    /// StarScene-space center of the currently visible region.
    pub sector_center: Option<(f32, f32, f32)>,
    /// Pipeline response for the selected star, if any.
    pub pipeline: Option<PipelineResponse>,
    /// Per-scene camera persistence entries.
    pub scene_cameras: HashMap<String, crate::camera::SceneCamera>,
}

impl StellarSceneSnapshot {
    /// Effective scene-space center: prefer the active scene, fall
    /// back to the sector center, then to the origin.
    pub fn effective_center(&self) -> (f32, f32, f32) {
        if let Some(w) = &self.active_scene {
            (w.center_x, w.center_y, w.center_z)
        } else if let Some(c) = self.sector_center {
            c
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// StarScene center projected to XY for sector streaming.
    pub fn effective_center_xy(&self) -> (f32, f32) {
        let c = self.effective_center();
        (c.0, c.1)
    }

    pub fn active_scene_id(&self) -> Option<&str> {
        self.active_scene.as_ref().map(|w| w.id.as_str())
    }
}
