
use instant::Instant;
use std::time::Duration;

use crate::camera::Camera;
use crate::sector::SectorKey;
use lunar_structures::ResponseStar;

#[derive(Clone, Debug, PartialEq)]
pub enum PlayerAction {
    Pan { delta: (f32, f32) },
    Zoom { factor: f32 },
    SelectStar { star_id: Option<u32> },
    SetTemperature { value: f32 },
    SetBpRp { value: f32 },
    SetGMag { value: f32 },
    RecenterCamera,
    LoadWorld { world_id: String },
}

#[derive(Clone, Debug)]
pub struct ActionRecord {
    pub action: PlayerAction,
    pub when: Option<Instant>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CameraMovement {
    pub offset_delta: (f32, f32),
    pub zoom: f32,
    pub dragging: bool,
}

#[derive(Clone, Debug)]
pub struct UpdatePayload {
    pub camera_movement: CameraMovement,
    pub recent_actions: Vec<ActionRecord>,
    pub current_sector: Option<SectorKey>,
    pub sector_stars: Vec<ResponseStar>,
    pub session_duration: Duration,
}

#[derive(Debug)]
pub struct ActionBuffer {
    records: Vec<ActionRecord>,
    camera_snapshots: Vec<(Camera, Instant)>,
    started_at: Instant,
    window: Duration,
}

impl ActionBuffer {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            camera_snapshots: Vec::new(),
            started_at: Instant::now(),
            window: Duration::from_secs(60),
        }
    }

    pub fn push(&mut self, action: PlayerAction) {
        self.records.push(ActionRecord {
            action,
            when: Some(Instant::now()),
        });
    }

    pub fn push_camera(&mut self, camera: Camera) {
        self.camera_snapshots.push((camera, Instant::now()));
    }

    pub fn build_update(&self) -> UpdatePayload {
        let elapsed = self.started_at.elapsed();
        let now = Instant::now();

        UpdatePayload {
            camera_movement: self.calculate_camera_movement(now, elapsed),
            recent_actions: self.filter_recent_actions(now, elapsed),
            current_sector: None,
            sector_stars: Vec::new(),
            session_duration: elapsed,
        }
    }

    pub fn prune(&mut self) {
        let Some(cutoff) = Instant::now().checked_sub(self.window) else {
            return;
        };
        self.records
            .retain(|r| r.when.map_or(true, |t| t >= cutoff));
        let first_in_window = self.camera_snapshots.iter().position(|(_, t)| *t >= cutoff);
        if let Some(pos) = first_in_window {
            let keep_from = pos.saturating_sub(1);
            self.camera_snapshots.drain(..keep_from);
        } else if let Some(last) = self.camera_snapshots.last() {
            let last_instant = last.1;
            self.camera_snapshots.retain(|(_, t)| *t == last_instant);
        }
    }

    fn calculate_camera_movement(&self, now: Instant, elapsed: Duration) -> CameraMovement {
        if elapsed < self.window {
            return CameraMovement::default();
        }

        let cutoff = now - self.window;
        let oldest = self
            .camera_snapshots
            .iter()
            .find(|(_, t)| *t >= cutoff)
            .map(|(c, _)| c);
        let newest = self.camera_snapshots.last().map(|(c, _)| c);

        match (oldest, newest) {
            (Some(old), Some(new)) => CameraMovement {
                offset_delta: (new.offset.0 - old.offset.0, new.offset.1 - old.offset.1),
                zoom: new.zoom,
                dragging: new.dragging,
            },
            _ => CameraMovement::default(),
        }
    }

    fn filter_recent_actions(&self, now: Instant, elapsed: Duration) -> Vec<ActionRecord> {
        if elapsed < self.window {
            self.records
                .iter()
                .map(|r| ActionRecord {
                    action: r.action.clone(),
                    when: None,
                })
                .collect()
        } else {
            let cutoff = now - self.window;
            self.records
                .iter()
                .filter(|r| r.when.map_or(false, |t| t >= cutoff))
                .cloned()
                .collect()
        }
    }
}

impl Default for ActionBuffer {
    fn default() -> Self {
        Self::new()
    }
}
