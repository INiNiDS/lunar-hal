//! Framework-agnostic gameplay layer.
//!
//! This crate owns *all* gameplay state and rules: which world is loaded,
//! which sectors are visible, where the camera is, which star is selected,
//! and which sectors must be fetched from the AI backend to satisfy the
//! current viewport. Frontends (Dioxus, anything else) only render what
//! [`Game`] exposes and forward user input back through its methods.
//!
//! The split is intentional:
//!
//! * [`lunar_backend`] (the AI server) provides physics/AI inference over HTTP.
//! * [`lunar_game_backend`] (this crate) decides *what to ask for* and *where
//!   things are*, persisting and exposing the resulting state.
//! * Frontends stay swappable — swap the UI, keep the gameplay.

#![forbid(unsafe_code)]

pub mod actions;
pub mod api_client;
pub mod attention;
pub mod camera;
pub mod error;
pub mod game;
pub mod sector;
pub mod snapshot;
pub mod validation;
pub mod enemy;

pub use actions::{
    ActionBuffer, ActionRecord, CameraMovement, PlayerAction, UpdatePayload,
};
pub use attention::{AttentionEntry, AttentionMap};
pub use camera::{Camera, WorldCamera, WorldCameraStore, MAX_ZOOM, MIN_ZOOM};
pub use error::GameError;
pub use game::{Game, GameConfig};
pub use sector::{
    chunk_at_world_point, chunk_center, SectorFetchRequest, SectorKey, CHUNK_SIZE_PC,
    INNER_EXCLUSION_PC, MAX_CACHED_CHUNKS, MAX_CONCURRENT_FETCHES, MIN_FETCH_COOLDOWN_MS,
    PX_PER_PC,
};
pub use snapshot::GameSnapshot;
pub use validation::{
    limits, validate_bp_rp, validate_center_x, validate_center_y, validate_center_z,
    validate_entropy, validate_g_mag, validate_pipeline, validate_response_star,
    validate_response_stars, validate_search_radius, validate_sector_key, validate_temperature,
    validate_world, validate_world_id, validate_world_name, validate_world_summary, validate_zoom,
    ValidationError, ValidationResult,
};
