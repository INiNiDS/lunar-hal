#![forbid(unsafe_code)]

pub mod actions;
pub mod api_client;
pub mod attention;
pub mod camera;
pub mod enemy;
pub mod error;
pub mod game;
pub mod sector;
pub mod snapshot;
pub mod validation;

pub use actions::{ActionBuffer, ActionRecord, CameraMovement, PlayerAction, UpdatePayload};
pub use attention::{AttentionEntry, AttentionMap};
pub use camera::{Camera, MAX_ZOOM, MIN_ZOOM, WorldCamera, WorldCameraStore};
pub use enemy::{Enemy, EnemyAction, EnemyDamage, EnemyType, Projectile, STAR_MAX_HP};
pub use error::GameError;
pub use game::{Game, GameConfig};
pub use sector::{
    CHUNK_SIZE_PC, CHUNK_SIZE_PC as CHUNK_SIZE, INNER_EXCLUSION_PC, MAX_CACHED_CHUNKS,
    MAX_CONCURRENT_FETCHES, MIN_FETCH_OBJECTS, MIN_FETCH_RECORDS, PX_PER_PC, SectorKey,
    chunk_center,
};
pub use snapshot::GameSnapshot;
pub use validation::{
    ValidationError, ValidationResult, limits, validate_bp_rp, validate_center_x,
    validate_center_y, validate_center_z, validate_entropy, validate_g_mag, validate_pipeline,
    validate_response_star, validate_response_stars, validate_search_radius, validate_sector_key,
    validate_temperature, validate_world, validate_world_id, validate_world_name,
    validate_world_summary, validate_zoom,
};
