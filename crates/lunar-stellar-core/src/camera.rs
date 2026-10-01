use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub offset: (f32, f32),
    pub zoom: f32,
    pub dragging: bool,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            offset: (0.0, 0.0),
            zoom: 1.0,
            dragging: false,
        }
    }

    pub fn zoom_around_center(&self, viewport: (f32, f32), factor: f32) -> Self {
        let (vp_w, vp_h) = viewport;
        let (cx, cy) = (vp_w * 0.5, vp_h * 0.5);
        let new_zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let wx = (cx - self.offset.0) / self.zoom;
        let wy = (cy - self.offset.1) / self.zoom;
        let new_off_x = cx - wx * new_zoom;
        let new_off_y = cy - wy * new_zoom;
        Self {
            offset: (new_off_x, new_off_y),
            zoom: new_zoom,
            dragging: self.dragging,
        }
    }

    pub fn zoom_around(&self, viewport: (f32, f32), anchor: (f32, f32), factor: f32) -> Self {
        let (vp_w, vp_h) = viewport;
        let (cx, cy) = (vp_w * 0.5, vp_h * 0.5);
        let new_zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let wx = (anchor.0 - cx - self.offset.0) / self.zoom;
        let wy = (anchor.1 - cy - self.offset.1) / self.zoom;
        let new_off_x = anchor.0 - cx - wx * new_zoom;
        let new_off_y = anchor.1 - cy - wy * new_zoom;
        Self {
            offset: (new_off_x, new_off_y),
            zoom: new_zoom,
            dragging: self.dragging,
        }
    }

    pub fn pan(&self, delta: (f32, f32)) -> Self {
        Self {
            offset: (self.offset.0 + delta.0, self.offset.1 + delta.1),
            zoom: self.zoom,
            dragging: self.dragging,
        }
    }

    pub fn reset(&self) -> Self {
        Self {
            offset: (0.0, 0.0),
            zoom: 1.0,
            dragging: false,
        }
    }

    pub fn delta(&self, prev: &Self) -> (f32, f32) {
        (self.offset.0 - prev.offset.0, self.offset.1 - prev.offset.1)
    }
}

pub const MIN_ZOOM: f32 = 0.05;
pub const MAX_ZOOM: f32 = 15.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneCamera {
    pub offset: (f32, f32),
    pub zoom: f32,
}

impl SceneCamera {
    pub const fn new(offset: (f32, f32), zoom: f32) -> Self {
        Self { offset, zoom }
    }

    pub fn to_camera(&self) -> Camera {
        Camera {
            offset: self.offset,
            zoom: self.zoom,
            dragging: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SceneCameraStore {
    pub entries: HashMap<String, SceneCamera>,
}
