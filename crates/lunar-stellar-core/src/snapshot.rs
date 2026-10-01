
use std::collections::{HashMap, HashSet};

use lunar_structures::{GnnResponse, PipelineResponse, ResponseStar, StarScene, StarSceneSummary};

use crate::camera::Camera;
use crate::sector::SectorKey;

#[derive(Clone, Debug, Default)]
pub struct StellarSceneSnapshot {
    pub scenes: Vec<StarSceneSummary>,
    pub active_scene: Option<StarScene>,
    pub sector_stars: Vec<ResponseStar>,
    pub sector_loading: HashSet<SectorKey>,
    pub sector_cache: HashMap<SectorKey, Vec<ResponseStar>>,
    pub camera: Camera,
    pub selected_star: Option<ResponseStar>,
    pub pregen: Option<GnnResponse>,
    pub temperature: f32,
    pub bp_rp: f32,
    pub g_mag: f32,
    pub sector_center: Option<(f32, f32, f32)>,
    pub pipeline: Option<PipelineResponse>,
    pub scene_cameras: HashMap<String, crate::camera::SceneCamera>,
}

impl StellarSceneSnapshot {
    pub fn effective_center(&self) -> (f32, f32, f32) {
        if let Some(w) = &self.active_scene {
            (w.center_x, w.center_y, w.center_z)
        } else if let Some(c) = self.sector_center {
            c
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    pub fn effective_center_xy(&self) -> (f32, f32) {
        let c = self.effective_center();
        (c.0, c.1)
    }

    pub fn active_scene_id(&self) -> Option<&str> {
        self.active_scene.as_ref().map(|w| w.id.as_str())
    }
}
