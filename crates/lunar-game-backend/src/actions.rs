//! Player action recording with a one-minute rolling window.
//!
//! Every user interaction (pan, zoom, selection, parameter changes)
//! flows through [`ActionBuffer`]. The [`Game::update`](crate::Game::update)
//! method records a camera snapshot, computes the accumulated camera
//! delta over the rolling window, drains recent actions, and prunes
//! the buffer.

use std::time::{Duration, Instant};

use crate::attention::AttentionEntry;
use crate::camera::Camera;
use crate::sector::SectorKey;
use lunar_structures::ResponseStar;

/// Everything a player can do that we want to track.
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

/// A single recorded action together with the moment it happened.
///
/// `when` is `None` when the buffer has not yet accumulated a full
/// window of history — the timestamp is replaced with a default
/// sentinel to signal "not enough data".
#[derive(Clone, Debug)]
pub struct ActionRecord {
    pub action: PlayerAction,
    pub when: Option<Instant>,
}

/// Accumulated camera displacement over the rolling window.
///
/// `offset_delta` is the **total** pixel displacement between the
/// oldest camera snapshot still inside the 60 s window and the
/// current camera — not just the last frame.
///
/// When the session is shorter than the window, every field is
/// [`Default`] (zero offset, zero zoom, not dragging).
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraMovement {
    /// Total offset delta over the rolling window (pixels).
    pub offset_delta: (f32, f32),
    /// Current zoom level.
    pub zoom: f32,
    /// Whether the player is currently dragging.
    pub dragging: bool,
}

/// What [`Game::update`](crate::Game::update) delivers to its caller.
///
/// All fields are *owned* so the payload can be sent across threads
/// or serialized independently of the game lock.
#[derive(Clone, Debug)]
pub struct UpdatePayload {
    /// Total camera displacement over the rolling window (default:
    /// 60 s). Zero when the session is shorter than the window.
    pub camera_movement: CameraMovement,
    /// Every action the player performed inside the rolling window.
    ///
    /// If the session is shorter than the window (default: 60 s),
    /// every [`ActionRecord::when`] will be `None` — the timestamps
    /// are replaced with a default sentinel.
    pub recent_actions: Vec<ActionRecord>,
    /// The sector under the camera center (viewport midpoint mapped
    /// to chunk coordinates). `None` when no world is loaded or the
    /// viewport cannot be resolved.
    pub current_sector: Option<SectorKey>,
    /// Stars belonging to whichever chunk is the current sector.
    /// Empty when `current_sector` is `None` or the chunk hasn't
    /// been fetched yet.
    pub sector_stars: Vec<ResponseStar>,
    /// Карта внимания игрока для текущего сектора. Ключ — star.id.
    pub attention_map: std::collections::HashMap<u32, AttentionEntry>,
}

/// Accumulates [`PlayerAction`]s and camera snapshots and exposes
/// them through [`ActionBuffer::build_update`].
///
/// The buffer *owns* its backing [`Vec`]s and is stored inside
/// [`GameState`](crate::game::GameState). Pruning happens on every
/// update so memory stays bounded.
#[derive(Debug)]
pub struct ActionBuffer {
    records: Vec<ActionRecord>,
    /// Camera snapshots keyed by timestamp, used to compute the
    /// total displacement over the rolling window.
    camera_snapshots: Vec<(Camera, Instant)>,
    started_at: Instant,
    window: Duration,
}

impl ActionBuffer {
    /// Fresh buffer with a 60-second rolling window.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            camera_snapshots: Vec::new(),
            started_at: Instant::now(),
            window: Duration::from_secs(60),
        }
    }

    /// Append an action to the buffer with a current timestamp.
    pub fn push(&mut self, action: PlayerAction) {
        self.records.push(ActionRecord {
            action,
            when: Some(Instant::now()),
        });
    }

    /// Record a camera snapshot for later delta computation.
    pub fn push_camera(&mut self, camera: Camera) {
        self.camera_snapshots.push((camera, Instant::now()));
    }

    /// Produce an [`UpdatePayload`] with:
    ///
    /// * **camera_movement** — total offset delta from the oldest
    ///   camera snapshot still inside the rolling window to the
    ///   newest. When the session is shorter than the window, the
    ///   entire [`CameraMovement`] is [`Default`].
    /// * **recent_actions** — actions that fall inside the window
    ///   (timestamps set to `None` when < window).
    pub fn build_update(&self) -> UpdatePayload {
        let elapsed = self.started_at.elapsed();
        let now = Instant::now();

        UpdatePayload {
            camera_movement: self.calculate_camera_movement(now, elapsed),
            recent_actions: self.filter_recent_actions(now, elapsed),
            current_sector: None,
            sector_stars: Vec::new(),
            attention_map: std::collections::HashMap::new(),
        }
    }

    /// Remove records and camera snapshots older than the rolling
    /// window.
    pub fn prune(&mut self) {
        let cutoff = Instant::now() - self.window;
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
