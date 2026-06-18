use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// 2D pan/zoom camera state owned by the game (not the renderer).
///
/// The frontend reads these values to position its viewport and writes
/// them back when the user pans or zooms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    /// Pixel offset of the world origin in viewport space.
    pub offset: (f32, f32),
    /// Zoom multiplier (`1.0` = identity).
    pub zoom: f32,
    /// Whether the user is currently dragging the view.
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

    /// Compute a new camera state that zooms by `factor` around the
    /// viewport center, preserving the world point under the center.
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

    /// Compute a new camera state that zooms by `factor` around a
    /// specific viewport-space anchor point.
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

    /// Pan the camera by a viewport-space delta (in pixels).
    pub fn pan(&self, delta: (f32, f32)) -> Self {
        Self {
            offset: (self.offset.0 + delta.0, self.offset.1 + delta.1),
            zoom: self.zoom,
            dragging: self.dragging,
        }
    }

    /// Reset to the identity camera.
    pub fn reset(&self) -> Self {
        Self {
            offset: (0.0, 0.0),
            zoom: 1.0,
            dragging: false,
        }
    }

    /// Pixel displacement since `prev`.
    pub fn delta(&self, prev: &Self) -> (f32, f32) {
        (self.offset.0 - prev.offset.0, self.offset.1 - prev.offset.1)
    }
}

pub const MIN_ZOOM: f32 = 0.05;
pub const MAX_ZOOM: f32 = 15.0;

/// Per-world camera state remembered between sessions. Frontends
/// implement the actual persistence (e.g. `localStorage`); the game
/// stores and retrieves it via [`crate::Game::world_camera`] /
/// [`crate::Game::set_world_camera`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldCamera {
    pub offset: (f32, f32),
    pub zoom: f32,
}

impl WorldCamera {
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

/// In-memory camera persistence for worlds. Frontends can hydrate this
/// from `localStorage`, a server, or any other source.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorldCameraStore {
    pub entries: HashMap<String, WorldCamera>,
}
